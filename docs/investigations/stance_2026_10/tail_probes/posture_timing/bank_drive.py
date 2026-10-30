"""TEST 1, the policy's own commands at ONE FIXED observation (policy_probe bank_action_now, bank row
'statue_settled', recorded at every PPO update).  The state is held fixed, so a change here is a change in the policy's
mapping, not a response to a changed posture.  Does the policy's tail_3 output at the fixed state start rising before,
after, or with its posture-related outputs (hip pitch, knee, ankle, neck, head) at the same fixed state?
Read only.  Output: bank_drive.txt (stdout).
Action order (probe_env.txt in clamp/): 0 neck_pitch, 1 neck_yaw, 2 head_pitch, 3 r_hip_pitch, 4 r_hip_roll, 5 r_knee,
6 r_ankle, 7 l_hip_pitch, 8 l_hip_roll, 9 l_knee, 10 l_ankle, 11 tail_1_pitch, 12 tail_1_yaw, 13 tail_2_pitch,
14 tail_3_pitch.  Legs are shown as the left/right mean.
"""
import numpy as np

import pt_common as C

SRC = {"r10 s42": C.f2_file("s42", "policy_probe.npz"), "r10 s60": C.f2_file("s60", "policy_probe.npz"),
       "r10 s80": C.f2_file("s80", "policy_probe.npz")}
SRC.update({f"r9 {k.split('_')[-1]}": v for k, v in C.R9_PROBE.items()})
COLS = {"tail3": [14], "tail2": [13], "tail1": [11], "neck": [0], "head": [2], "hip": [3, 7], "knee": [5, 9], "ankle": [6, 10]}


def main():
    print("TEST 1 / policy commands at the fixed statue_settled observation (bank_drive.py; read only)\n")
    edges = np.arange(0, 12_058_624 + 1, 262_144)
    for lab, path in SRC.items():
        d = np.load(path, allow_pickle=False)
        i0 = list(d["bank_labels"]).index("statue_settled")
        t = d["update_timesteps"]
        a = d["bank_action_now"][:, i0, :].astype(float)
        cols = {k: C.bin_series(t, a[:, idx].mean(1), edges) for k, idx in COLS.items()}
        print(f"  {lab}: 262,144-step bin medians of the deterministic command at statue_settled")
        print("     step(M)  " + " ".join(f"{k:>7s}" for k in cols))
        for i, m in enumerate(edges[1:]):
            if m <= 5_000_000 or (m <= 9_000_000 and i % 4 == 3):
                print(f"     {m/1e6:6.3f}  " + " ".join(f"{cols[k][i]:+7.3f}" for k in cols))
        # first sustained change of 0.05 command units from the [524,288, 1,048,576) reference, per output
        sm = {k: C.smooth(a[:, idx].mean(1), 16) for k, idx in COLS.items()}
        ref = {k: float(np.median(v[(t >= 524_288) & (t < 1_048_576)])) for k, v in sm.items()}
        aft = t >= 1_048_576
        cells = []
        for k in cols:
            up = C.first_sustained(t[aft], sm[k][aft], ref[k] + 0.05, True, 16)
            dn = C.first_sustained(t[aft], sm[k][aft], ref[k] - 0.05, False, 16)
            cells.append(f"{k} ref {ref[k]:+.3f} up {C.fmt(up)} down {C.fmt(dn)}")
        print("     first sustained 0.05 change from the reference (16-update mean, 16 updates): " + "; ".join(cells))
        print()


if __name__ == "__main__":
    main()
