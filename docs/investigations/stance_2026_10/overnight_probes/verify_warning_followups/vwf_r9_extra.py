"""Verifier extras (read only): reason rule vs saturated_actuator_fraction > 0.10 on r10; S5 on r10; how long r9 s80's
P1 firing at 1.38M stayed above 0.150; r9 positives' trail8 just before their alarm."""
import numpy as np, pandas as pd
F2 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2"
TR = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/fetch"
tot = dis = 0
for r in ("s42", "s60", "s80"):
    d = pd.read_csv(f"{F2}/stance6_r10LTSB_{r}/copies/01_stance/eval_episodes.csv.copy1")
    a = d["screen_v2_reasons"].astype(str).str.contains("max_actuator_saturation_fraction", regex=False)
    b = d["saturated_actuator_fraction"] > 0.10
    tot += len(d); dis += int((a != b).sum())
    acts = sorted(set(d.loc[b, "saturated_actuator"].astype(str)))
    z = np.load(f"{F2}/stance6_r10LTSB_{r}/copies/01_stance/policy_probe.npz.copy1")
    t = z["update_timesteps"]; mu = pd.Series(z["rollout_mu_mean"][:, 14].astype(float)).rolling(8, min_periods=1).mean().to_numpy()
    m = (mu > 0.638) & (t <= 11927552)
    print(f"r10 {r}: episodes {len(d)}, over-bar {int(b.sum())}, actuators {acts}; S5 trail8(rollout_mu)>0.638 first {int(t[np.argmax(m)]) if m.any() else None}")
print(f"r10 reason rule vs fraction>0.10: {dis} disagreements in {tot} episodes")
for r, alarm in (("s80", 4456448), ("s42", 3670016)):
    z = np.load(f"{TR}/stance5_LTSB_{r}/A/policy_probe.npz")
    lab = z["bank_labels"].astype(str); t = z["update_timesteps"]
    ss = z["bank_action_now"][:, lab == "statue_settled", 14].astype(float).mean(axis=1)
    tr = pd.Series(ss).rolling(8, min_periods=1).mean().to_numpy()
    above = tr > 0.150
    i0 = int(np.argmax(above)); j = i0
    while j < len(above) and above[j]:
        j += 1
    pre = (t < alarm)
    k = np.flatnonzero(pre)[-1]
    frac_pre = float(above[(t >= 1_048_576) & pre].mean())
    print(f"r9 {r}: P1 first above at {int(t[i0])}, stays above for {j - i0} rows ({(j - i0) * 8192} steps), max in that run of rows {tr[i0:j].max():.4f}; "
          f"trail8 at the last row before alarm ({int(t[k])}) {tr[k]:+.4f}; share of rows above 0.150 between 1.05M and the alarm {frac_pre:.3f}")
