"""Verifier (read only, no network): z-scores of each probe bank's statue_settled raw state under each run's own
running normaliser (probe obs_rms_mean / obs_rms_var rows), and of the r9 bank under the r10 normalisers (the situation
procedure X evaluates).  Clip at |z| = 10 as VecNormalize does."""
import numpy as np
F2 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2"
TR = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/fetch"
def load(p):
    z = np.load(p); lab = z["bank_labels"].astype(str)
    return z["update_timesteps"], z["obs_rms_mean"].astype(float), z["obs_rms_var"].astype(float), z["bank_raw_obs"][lab == "statue_settled"]
P10 = {r: load(f"{F2}/stance6_r10LTSB_{r}/copies/01_stance/policy_probe.npz.copy1") for r in ("s42", "s60", "s80")}
P9 = {r: load(f"{TR}/stance5_LTSB_{r}/A/policy_probe.npz") for r in ("s42", "s52", "s60", "s70", "s80")}
X10 = P10["s42"][3]; X9 = P9["s42"][3]
def zs(X, m, v):
    return (X - m) / np.sqrt(v + 1e-8)
print("mark   | case                         | tail_3 z (2 rows)   | median |z| all ch | channels |z|>3 | channels clipped |z|>=10")
for mk in (1_048_576, 2_097_152, 4_194_304, 8_388_608):
    for name, P, X in (("r10 bank / r10 s42 norm", P10["s42"], X10), ("r10 bank / r10 s80 norm", P10["s80"], X10),
                       ("r9 bank / r9 s42 norm", P9["s42"], X9), ("r9 bank / r9 s70 norm", P9["s70"], X9),
                       ("r9 bank / r10 s42 norm (X)", P10["s42"], X9), ("r9 bank / r10 s80 norm (X)", P10["s80"], X9)):
        t, m, v, _ = P
        i = np.flatnonzero(t <= mk)[-1]
        z = zs(X, m[i], v[i])
        print(f"{mk / 1e6:5.2f}M | {name:28s} | {np.round(z[:, 6], 1).tolist()!s:19s} | {np.median(np.abs(z)):17.2f} | "
              f"{int((np.abs(z).max(axis=0) > 3).sum()):15d} | {int((np.abs(z).max(axis=0) >= 10).sum())}")
print()
t, m, v, _ = P10["s80"]; i = np.flatnonzero(t <= 2_097_152)[-1]
z = zs(X9, m[i], v[i]); zr = zs(X10, m[i], v[i])
big = np.flatnonzero(np.abs(z).max(axis=0) > 3)
print("channels with |z| > 3 for the r9 bank under the r10 s80 normaliser at 2.10M:", big.tolist())
for c in big:
    print(f"  ch {c}: r9 raw {np.round(X9[:, c], 4).tolist()} r10 raw {np.round(X10[:, c], 4).tolist()} z(r9) {np.round(z[:, c], 1).tolist()} z(r10) {np.round(zr[:, c], 1).tolist()}")
