"""TEST 1, r9 tail screen (tailround/screen/runs, 18 continuations of stance5 s42 / s80 from 4,194,304 for 1,048,576
steps; arms control, b90, b60 = a training-loss bound on the tail_3 mean at 0.9 / 0.6; 3 seeds each).  Do posture and
the tail command move together, including where the bound held the tail down?  Read only.  No simulation.
Output: ts_comovement.txt (stdout).

Posture: per 131,072-step interval, the training observation means recovered from the saved VecNormalize statistics
(start pkl in tailround/fetch/<state>/A, then each checkpoint pkl): pelvis pitch from projected gravity, tail_3 /
neck / head joints.  Tail: telemetry.npz per rollout cmd_mean[:, 14] (the sampled command) averaged per interval;
panels.csv per checkpoint (deterministic, 20 episodes) cmd.
The pkls are the screen's own outputs (stable_baselines3 VecNormalize, unpickled with the venv's SB3; no env is built).
"""
import csv
import glob
import os
import pickle

import numpy as np

import pt_common as C

CK = [4_325_376, 4_456_448, 4_587_520, 4_718_592, 4_849_664, 4_980_736, 5_111_808, 5_242_880]


def rms_of(path):
    with open(path, "rb") as fh:
        v = pickle.load(fh)
    return np.asarray(v.obs_rms.mean, dtype=np.float64), float(v.obs_rms.count)


def main():
    panels = {}
    for r in csv.DictReader(open(f"{C.S2}/tailround/screen/panels.csv")):
        panels[(r["name"], r["label"])] = r
    print("TEST 1 / r9 tail screen co-movement (ts_comovement.py; read only)\n")
    print("Per run: intervals ending at each checkpoint; cmd = telemetry mean sampled tail_3 command over the interval; "
          "pcmd = panel deterministic tail_3 command at the checkpoint; q3 = training tail_3 angle (deg); pitch (deg, + "
          "nose up), neck, head joints (deg) from the recovered observation means.\n")
    res = {}
    for p in sorted(glob.glob(f"{C.TS_RUNS}/*")):
        name = os.path.basename(p)
        state = name.split("_")[0]
        m0, c0 = rms_of(f"{C.S2}/tailround/fetch/stance5_LTSB_{state}/A/stage1_vecnormalize_4194304_steps.pkl")
        tel = np.load(f"{p}/telemetry.npz", allow_pickle=False)
        tt, cmd = tel["num_timesteps"], tel["cmd_mean"][:, C.TAIL3]
        rows = []
        prev_m, prev_c, prev_t = m0, c0, 4_194_304
        for ck in CK:
            m, c = rms_of(f"{p}/stage1_vecnormalize_{ck}_steps.pkl")
            bm = (m * c - prev_m * prev_c) / (c - prev_c)
            sel = (tt > prev_t) & (tt <= ck)
            pc = panels.get((name, str(ck)))
            rows.append({"t": ck, "dc": c - prev_c, "cmd": float(cmd[sel].mean()) if sel.any() else float("nan"),
                         "pcmd": float(pc["cmd"]) if pc else float("nan"),
                         "q3": bm[C.OBS["tail_3_pitch"]] * C.R2D, "pitch": float(C.pitch_from_gx(bm[C.OBS["g_x"]])),
                         "neck": bm[C.OBS["neck_pitch"]] * C.R2D, "head": bm[C.OBS["head_pitch"]] * C.R2D})
            prev_m, prev_c, prev_t = m, c, ck
        res[name] = rows
        print(f"  {name}   (count increments {sorted({int(r['dc']) for r in rows})})")
        print("     step(M)    cmd    pcmd     q3     pitch    neck    head")
        for r in rows:
            print(f"     {r['t']/1e6:6.3f}  {r['cmd']:+6.3f}  {r['pcmd']:+6.3f}  {r['q3']:+6.2f}  {r['pitch']:+7.3f}  {r['neck']:+6.2f}  {r['head']:+6.2f}")
    print()
    print("SUMMARY PER STATE AND ARM (first interval -> last interval; mean of the 3 seeds; per-seed signs in brackets)")
    for state in ("s42", "s80"):
        for arm in ("control", "b90", "b60"):
            names = [n for n in res if n.startswith(f"{state}_{arm}_")]
            d = {k: [res[n][-1][k] - res[n][0][k] for n in names] for k in ("cmd", "q3", "pitch", "neck", "head")}
            lv = {k: [res[n][-1][k] for n in names] for k in ("cmd", "pitch")}
            print(f"  {state} {arm:7s}: end cmd {np.mean(lv['cmd']):+.3f} [{', '.join(f'{x:+.2f}' for x in lv['cmd'])}]; "
                  f"end pitch {np.mean(lv['pitch']):+.3f} [{', '.join(f'{x:+.2f}' for x in lv['pitch'])}]; "
                  + "; ".join(f"d {k} {np.mean(v):+.3f} [{', '.join(f'{x:+.2f}' for x in v)}]" for k, v in d.items()))
    print()
    print("ACROSS THE 9 CONTINUATIONS OF EACH STATE: Spearman rho, end-interval tail command vs end-interval posture "
          "(descriptive; the arms set the tail level by a training penalty)")
    for state in ("s42", "s80"):
        names = [n for n in res if n.startswith(state)]
        c = [res[n][-1]["cmd"] for n in names]
        for k in ("pitch", "neck", "head"):
            print(f"  {state}: tail cmd vs {k}: rho {C.spearman(c, [res[n][-1][k] for n in names]):+.2f}  "
                  f"(values: {', '.join(f'{n.split(chr(95))[1]}:{res[n][-1][k]:+.2f}' for n in names)})")
    print()
    print("WITHIN EACH CONTINUATION OVER ITS 8 INTERVALS: Spearman rho, interval tail command vs pitch / head joint")
    for state in ("s42", "s80"):
        for arm in ("control", "b90", "b60"):
            names = [n for n in res if n.startswith(f"{state}_{arm}_")]
            rp = [C.spearman([r["cmd"] for r in res[n]], [r["pitch"] for r in res[n]]) for n in names]
            rh = [C.spearman([r["cmd"] for r in res[n]], [r["head"] for r in res[n]]) for n in names]
            print(f"  {state} {arm:7s}: pitch {', '.join(f'{x:+.2f}' for x in rp)};  head {', '.join(f'{x:+.2f}' for x in rh)}")


if __name__ == "__main__":
    main()
