"""PS design input (POST HOC on BS2's finished, verified records; not an outcome of PS): how much do the OTHER 14
actuators' on-policy commands move over BS2's window, per continuation and arm?  Used to choose and size the
'habit elsewhere' secondary (criteria_v1 SEC-ELSEWHERE).  Read only (allow_pickle False); prints to stdout.

Per continuation (BS2 runs/<name>/01_stance/policy_probe.npz) and per actuator j (0..14):
  m_j = mean of rollout_mu_mean[:, j] over the 8 probe rows with update_timesteps 3,473,408 ... 3,530,752 (the
        primary's rows); b_j = the parent's mean over its rows 248-255 (update_timesteps 2,039,808 ... 2,097,152; the
        F2 policy_probe.npz.copy1), the same start for every continuation (B_S's rows; b_14 = B_S);
  rise_j = m_j - b_j;  arise_j = |m_j| - |b_j| (movement toward a rail, either sign).
  ELSE = max over j != 14 of arise_j (the candidate SEC-ELSEWHERE value) and its argmax; ELSE_P = the same over the
  four midline pitch actuators other than tail_3 (neck_pitch 0, head_pitch 2, tail_1_pitch 11, tail_2_pitch 13).
Also the parent's own m_j at E (its rows 423-430) for reference.
Usage: cd /tmp && PY -I -B other_actuators_bs2.py > other_actuators_bs2.out 2> other_actuators_bs2.err
"""
import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np

import sys as _sys
# S2 is the venv owner: PY = S2/venv/bin/python (run with PY -I -B); S1 is its sibling session scratchpad.
S2 = Path(_sys.executable).parents[2]
S1 = S2.parent.parent / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad"
assert (S2 / "r10round/bias_screen2").is_dir() and (S1 / "foot/stance6").is_dir(), "run with S2/venv/bin/python"
RUNS = S2 / "r10round/bias_screen2/runs"
F2P = S1 / "foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1"
ACTS = ("neck_pitch", "neck_yaw", "head_pitch", "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "l_hip_pitch",
        "l_hip_roll", "l_knee", "l_ankle", "tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch")
MIDLINE_PITCH = (0, 2, 11, 13)
PRIMARY_STEPS = tuple(3_473_408 + 8_192 * i for i in range(8))
S_STEPS = tuple(2_039_808 + 8_192 * i for i in range(8))
E_STEPS_PARENT = PRIMARY_STEPS  # the parent's rows at the same steps (F2 has rows every 8,192 steps)
SEEDS = (2501, 2511, 2521, 2531)
ARMS = ("control", "critic_blind_tail3")


def rows_mean(ut, mu, steps):
    idx = [int(np.flatnonzero(ut == s)[0]) for s in steps]
    assert len(idx) == 8 and all(len(np.flatnonzero(ut == s)) == 1 for s in steps)
    return mu[idx].astype(np.float64).mean(axis=0)


with np.load(F2P, allow_pickle=False) as z:
    put, pmu = z["update_timesteps"], z["rollout_mu_mean"]
    b = rows_mean(put, pmu, S_STEPS)
    pe = rows_mean(put, pmu, E_STEPS_PARENT)
print("parent (F2 probe): b_j (rows 248-255 ending at S) and m_j at E (rows at 3,473,408 ... 3,530,752)")
for j, a in enumerate(ACTS):
    print(f"  {j:>2} {a:<13} b {b[j]:+.4f}  parent at E {pe[j]:+.4f}  rise {pe[j] - b[j]:+.4f}  "
          f"|.| rise {abs(pe[j]) - abs(b[j]):+.4f}")
print(f"  check b_14 = B_S: {b[14]!r} (criteria B_S 0.1265954738482833)")

vals = {}
for arm in ARMS:
    for s in SEEDS:
        name = f"{arm}_{s}"
        with np.load(RUNS / name / "01_stance/policy_probe.npz", allow_pickle=False) as z:
            m = rows_mean(z["update_timesteps"], z["rollout_mu_mean"], PRIMARY_STEPS)
        arise = np.abs(m) - np.abs(b)
        others = [j for j in range(15) if j != 14]
        jmax = max(others, key=lambda j: arise[j])
        jmp = max(MIDLINE_PITCH, key=lambda j: arise[j])
        vals[name] = {"m": m, "arise": arise, "else": arise[jmax], "jmax": jmax, "else_p": arise[jmp], "jmp": jmp}

print("\nper continuation: rise_j = m_j - b_j for every actuator (tail_3 = the primary's rise)")
print("  " + " " * 24 + "".join(f"{a[:9]:>10}" for a in ACTS))
for name, v in vals.items():
    print(f"  {name:<24}" + "".join(f"{x:>+10.3f}" for x in v["m"] - b))
print("\nper continuation: arise_j = |m_j| - |b_j|")
for name, v in vals.items():
    print(f"  {name:<24}" + "".join(f"{x:>+10.3f}" for x in v["arise"]))
print("\nELSE = max over the 14 others of arise_j (argmax); ELSE_P = over neck_pitch, head_pitch, tail_1_pitch, "
      "tail_2_pitch")
for name, v in vals.items():
    print(f"  {name:<24} ELSE {v['else']:+.4f} ({ACTS[v['jmax']]})   ELSE_P {v['else_p']:+.4f} ({ACTS[v['jmp']]})   "
          f"tail_2 arise {v['arise'][13]:+.4f}  rise {v['m'][13] - b[13]:+.4f}")
for key in ("else", "else_p"):
    for arm in ARMS:
        xs = [vals[f"{arm}_{s}"][key] for s in SEEDS]
        print(f"  {key:<7} {arm:<20} mean {np.mean(xs):+.4f} median {np.median(xs):+.4f} SD {np.std(xs, ddof=1):.4f}")


def perm_p_higher(arm_hi, arm_lo):
    """exact one-sided p that arm_hi's mean is higher (ties against), 70 relabelings."""
    x = [Fraction(float(v)) for v in arm_hi] + [Fraction(float(v)) for v in arm_lo]
    obs = sum(x[:4])
    cnt = sum(1 for c in itertools.combinations(range(8), 4) if sum(x[i] for i in c) >= obs)
    return Fraction(cnt, 70)


for key in ("else", "else_p"):
    hi = [vals[f"critic_blind_tail3_{s}"][key] for s in SEEDS]
    lo = [vals[f"control_{s}"][key] for s in SEEDS]
    print(f"  POST HOC (BS2 blinded vs control, blinded HIGHER): {key} exact p = {perm_p_higher(hi, lo)}")
hi = [vals[f"critic_blind_tail3_{s}"]["m"][13] - b[13] for s in SEEDS]
lo = [vals[f"control_{s}"]["m"][13] - b[13] for s in SEEDS]
print(f"  POST HOC tail_2 rise, blinded HIGHER: exact p = {perm_p_higher(hi, lo)}")
print("END")
