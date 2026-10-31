"""The window rows' ent_coef / learning_rate against the parent's F2 rows at the same steps (float32 equality), and
the primary's rank agreement with CMD_L at E across the 8 continuations (a field-identity sanity check)."""
import json
import numpy as np
from pathlib import Path
_BS2 = Path(__file__).resolve().parents[3]                 # work -> stat -> check -> bias_screen2
S2 = str(_BS2.parents[1])                                   # bias_screen2 -> r10round -> the session scratchpad
S1 = str(_BS2.parents[3] / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad")
BS2 = str(_BS2)
F2 = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1"
S = 2_097_152
with open(F2, "rb") as fh:
    z = np.load(fh, allow_pickle=False)
    pu, pe, pl = np.array(z["update_timesteps"]), np.array(z["ent_coef"]), np.array(z["learning_rate"])
pidx = {int(s): i for i, s in enumerate(pu.tolist())}
names = [f"{a}_{s}" for s in (2501, 2511, 2521, 2531) for a in ("control", "critic_blind_tail3")]
prim, cmd = {}, {}
for n in names:
    with open(f"{BS2}/runs/{n}/01_stance/policy_probe.npz", "rb") as fh:
        z = np.load(fh, allow_pickle=False)
        u, e, l = np.array(z["update_timesteps"]), np.array(z["ent_coef"]), np.array(z["learning_rate"])
    w = np.nonzero(u > S)[0]
    eq_e = all(e[i] == pe[pidx[int(u[i])]] for i in w)
    eq_l = all(l[i] == pl[pidx[int(u[i])]] for i in w)
    print(f"{n:24s} window rows {len(w)}: ent_coef == parent (float32) {eq_e}; learning_rate == parent {eq_l}; "
          f"lr at first/last window row {float(l[w[0]]):.4e}/{float(l[w[-1]]):.4e}; ent {float(e[w[0]]):.5f}/{float(e[w[-1]]):.5f}")
sc = json.load(open(BS2 + "/score/score.json"))
for c in sc["classes"]:
    prim[c["name"]] = c["primary"]
for s in sc["secondaries"]:
    if s["key"] == "SEC-CMD":
        cmd = s["values"]
rp = sorted(names, key=lambda n: prim[n])
rc = sorted(names, key=lambda n: cmd[n])
print("rank by primary:", rp)
print("rank by CMD_L(E):", rc)
print("identical ranking:", rp == rc)
