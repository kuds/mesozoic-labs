"""Independent probe audit (numpy only, allow_pickle False): window rows, ent_coef / learning_rate vs the parent's F2 probe,
the primary in exact rationals, B_S, the exact permutation p; plus seeds / settings equal across the 8 run_metas."""
import json, os, sys, itertools
from fractions import Fraction
import numpy as np
BS2 = sys.argv[1]
F2P = (os.environ["S1"] + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1")
S, E, R = 2097152, 3538944, 8192
win = [S + R * k for k in range(1, 176)]
last8 = win[-8:]
with np.load(F2P, allow_pickle=False) as z:
    pst = np.asarray(z["update_timesteps"]).astype(np.int64)
    pent = np.asarray(z["ent_coef"]); plr = np.asarray(z["learning_rate"])
    pmu = np.asarray(z["rollout_mu_mean"])
idx_bs = [int(np.where(pst == s)[0][0]) for s in range(S - 7 * R, S + 1, R)]
B_S = sum(Fraction(float(np.float64(pmu[i, 14]))) for i in idx_bs) / 8
print("B_S rows", idx_bs, "B_S", float(B_S), B_S == Fraction(135930855, 1073741824))
prim = {}
QUEUE = ["control_2501", "critic_blind_tail3_2501", "control_2511", "critic_blind_tail3_2511",
         "control_2521", "critic_blind_tail3_2521", "control_2531", "critic_blind_tail3_2531"]
settings = {}
for name in QUEUE:
    with np.load(os.path.join(BS2, "runs", name, "01_stance", "policy_probe.npz"), allow_pickle=False) as z:
        st = np.asarray(z["update_timesteps"]).astype(np.int64)
        ent = np.asarray(z["ent_coef"]); lr = np.asarray(z["learning_rate"]); mu = np.asarray(z["rollout_mu_mean"])
    inwin = [int(s) for s in st if S < s <= E]
    ok_rows = inwin == win
    mism_ent = mism_lr = 0
    for s in win:
        i = int(np.where(st == s)[0][0]); j = np.where(pst == s)[0]
        if len(j) != 1:
            mism_ent += 1; continue
        j = int(j[0])
        mism_ent += int(np.float32(ent[i]) != np.float32(pent[j]))
        mism_lr += int(np.float32(lr[i]) != np.float32(plr[j]))
    vals = []
    for s in last8:
        ii = np.where(st == s)[0]
        assert len(ii) == 1
        v = float(np.float64(mu[int(ii[0]), 14]))
        assert np.isfinite(v)
        vals.append(Fraction(v))
    prim[name] = sum(vals) / 8
    pre = [int(s) for s in st if s <= S]
    print(f"{name}: window rows {len(inwin)} at S+8192k: {ok_rows}; rows <= S {len(pre)}; ent/lr mismatches vs parent {mism_ent}/{mism_lr}; primary {float(prim[name]):+.5f}")
    m = json.load(open(os.path.join(BS2, "runs", name, "run_meta.json")))
    settings[name] = (json.dumps(m["ppo_applied"], sort_keys=True), m["recipe_v5"]["stage_config_sha256"],
                      json.dumps(m["recipe_v5"]["recomputed_hyperparameters_sha256"], sort_keys=True), json.dumps(m["stack"], sort_keys=True),
                      m["torch_threads"], json.dumps({k: v for k, v in m["environment"].items() if "THREADS" in k}, sort_keys=True),
                      m["argv"][1:])
print("ppo_applied / stage_config / hyperparameters / stack / threads equal across 8:",
      len({v[:6] for v in settings.values()}) == 1)
for n, v in settings.items():
    print("  argv", n, v[6])
ctrl = [prim[n] for n in QUEUE if n.startswith("control")]
blind = [prim[n] for n in QUEUE if not n.startswith("control")]
allv = ctrl + blind
obs = sum(ctrl)
cnt = sum(1 for c in itertools.combinations(range(8), 4) if sum(allv[i] for i in c) >= obs)
sc = sorted(ctrl)
med = (sc[1] + sc[2]) / 2
print("control median", float(med), "rise", float(med - B_S), ">= 0.20:", med - B_S >= Fraction(1, 5))
print("exact one-sided permutation p =", cnt, "/ 70")
sf = sum(1 for signs in itertools.product((1, -1), repeat=4)
         if sum(sg * (c - b) for sg, c, b in zip(signs, ctrl, blind)) >= sum(c - b for c, b in zip(ctrl, blind)))
print("sign-flip p =", sf, "/ 16")
