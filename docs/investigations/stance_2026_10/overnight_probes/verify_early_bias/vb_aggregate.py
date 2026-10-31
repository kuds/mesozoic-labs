"""Aggregate vb_out/*.json (the verifier's rollouts): mean ± SE over rollouts.  Writes vb_aggregate.txt / .json."""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

V = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias")
groups = defaultdict(list)
for f in sorted(glob.glob(str(V / "vb_out" / "*.json"))):
    d = json.load(open(f))
    groups[(d["run"], d["steps"])].append(d)


def ms(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return m, se, (m / se if se and se > 0 else float("nan")), n


def f(x, p=5):
    m, se, t, n = ms(x)
    return f"{m:+.{p}f}±{se:.{p}f} (t{t:+.1f})"


L, J = [], {}
order = sorted(groups, key=lambda k: (k[0], k[1]))
for key in order:
    D = groups[key]
    run, st = key
    n = len(D)
    o = {}
    ck = defaultdict(list)
    for d in D:
        for k, v in d["checks"].items():
            if isinstance(v, bool):
                ck[k].append(0.0 if v else 1.0)
            elif isinstance(v, list):
                ck[k].append(abs(v[0] - v[1]) / max(abs(v[0]), 1e-12))
            elif isinstance(v, dict):
                ck[k].append(max(v.values()))
            else:
                ck[k].append(float(v))
    L.append("=" * 110)
    L.append(f"{run} {st:,}  n={n} rollouts (seeds {min(d['seed'] for d in D)}-{max(d['seed'] for d in D)}); lr {D[0]['lr']:.3e} "
             f"ent {D[0]['ent_coef']:.5f}; adam t {D[0]['adam_t']:.0f}; n_updates {D[0]['n_updates_ckpt']}; hp mismatch {D[0]['hp_mismatch']}")
    L.append("  checks (max over rollouts; list entries = relative A-vs-B gap): " + "; ".join(f"{k} {max(v):.2g}" for k, v in ck.items()))
    o["checks"] = {k: max(v) for k, v in ck.items()}
    for tn in ("roll", "bank", "ss"):
        L.append(f"  [{tn:4s}] Adam per epoch: full {f([d['J'][f'adam_{tn}_full'] for d in D])} | pg {f([d['J'][f'adam_{tn}_pg'] for d in D])} | "
                 f"mirror {f([d['J'][f'adam_{tn}_kl'] for d in D])} | ent {f([d['J'][f'adam_{tn}_ent'] for d in D], 7)} | "
                 f"vf {f([d['J'][f'adam_{tn}_vf'] for d in D], 7)} | plain full {f([d['J'][f'plain_{tn}_full'] * 1e6 for d in D], 3)}e-6")
        o[tn] = {c: ms([d["J"][f"adam_{tn}_{c}"] for d in D]) for c in ("full", "pg", "kl")}
    for tn in ("roll", "ss"):
        S = [d["split"][f"adam_{tn}"] for d in D]
        L.append(f"  [{tn}] policy term by noise carrier: tail3 {f([s['pg_tail3_noise'] for s in S])} legs {f([s['pg_legs_noise'] for s in S])} "
                 f"other {f([s['pg_other_noise'] for s in S])}")
        keys = sorted({k for s_ in S for k in s_["src"]})
        rew = [sum(v for k, v in s["src"].items() if k != "BOOT") for s in S]
        boot = [s["src"]["BOOT"] for s in S]
        L.append(f"  [{tn}] raw source split (no fitted baselines): rewards within GAE horizon {f(rew)}; critic/bootstrap remainder {f(boot)}")
        sig = []
        for k in keys:
            if k == "BOOT":
                continue
            m, se, t, _ = ms([s["src"].get(k, 0.0) for s in S])
            if abs(t) >= 2:
                sig.append(f"{k.replace('reward_', '')} {m:+.5f} (t{t:+.1f})")
        L.append(f"  [{tn}]   terms |t|>=2: " + ", ".join(sig))
        g = [s["gae_keep"] for s in S]
        mc = [s["mc_keep"] for s in S]
        dl = [a - b for a, b in zip(g, mc)]
        g3 = [s["gae_keep_t3"] for s in S]
        mc3 = [s["mc_keep_t3"] for s in S]
        L.append(f"  [{tn}] CRITIC-FREE TEST (rows >= 200 from a cut): GAE {f(g)} vs MC(0.98) {f(mc)}; GAE-MC {f(dl)} | tail3-noise only: "
                 f"GAE {f(g3)} MC {f(mc3)} diff {f([a - b for a, b in zip(g3, mc3)])}")
        o[f"split_{tn}"] = {"tail3": ms([s["pg_tail3_noise"] for s in S]), "rew": ms(rew), "boot": ms(boot), "gae_keep": ms(g),
                            "mc_keep": ms(mc), "gae_minus_mc": ms(dl), "gae_t3": ms(g3), "mc_t3": ms(mc3),
                            "gae_minus_mc_t3": ms([a - b for a, b in zip(g3, mc3)])}
        S2 = [s for s in S if "mc_terms_t3" in s]
        if len(S2) >= 3:
            for lab in ("t3", "all"):
                for fam in ("gae_terms", "mc_terms"):
                    kk = sorted({k for s_ in S2 for k in s_[f"{fam}_{lab}"]})
                    rs = [sum(v for k, v in s[f"{fam}_{lab}"].items() if k not in ("BOOT", "MINUS_V")) for s in S2]
                    sig = []
                    for k in kk:
                        m, se, t, _ = ms([s[f"{fam}_{lab}"].get(k, 0.0) for s in S2])
                        if abs(t) >= 2 or k in ("BOOT", "MINUS_V"):
                            sig.append(f"{k.replace('reward_', '')} {m:+.5f}±{se:.5f} (t{t:+.1f})")
                    L.append(f"  [{tn}] {fam} noise={lab} (n {len(S2)}): reward-terms sum {f(rs)}; |t|>=2: " + ", ".join(sig))
                    o[f"split_{tn}"][f"{fam}_{lab}"] = {k: ms([s[f'{fam}_{lab}'].get(k, 0.0) for s in S2]) for k in kk}
                    o[f"split_{tn}"][f"{fam}_{lab}_rewsum"] = ms(rs)
    G = [d["mb"]["G14_mean"] for d in D]
    L.append(f"  r9-style dL_pg/d bias14 per mini-batch {f(G, 4)}, {sum(1 for x in G if x < 0)}/{n} negative; RMS(SD) {np.mean([d['mb']['G14_sd'] for d in D]):.3f}")
    o["G14"] = ms(G)
    L.append(f"  grad norms (median): full {np.median([d['mb']['norm_med'] for d in D]):.1f}, pg {np.median([d['mb']['comp_norm_med']['pg'] for d in D]):.1f}, "
             f"mirror {np.median([d['mb']['comp_norm_med']['kl'] for d in D]):.2f}, vf {np.median([d['mb']['comp_norm_med']['vf'] for d in D]):.3f}, "
             f"ent {np.median([d['mb']['comp_norm_med']['ent'] for d in D]):.4f}; clip factor {np.median([d['mb']['c_med'] for d in D]):.4f}")
    R = [d["rollout"] for d in D]
    eps = [e for d in D for e in d["rollout"]["eps"]]
    L.append(f"  rollout: mu14 {f([r['mu14_mean'] for r in R], 4)}; tail angle {np.mean([r['tail_deg_mean'] for r in R]):+.2f} deg; "
             f"dV/dq3 per deg {f([r['dV_dq3_deg'] for r in R], 4)}; G=5.16*dmu/dq3 {f([5.16 * r['dmu_dq3_deg'] for r in R], 3)}; "
             f"corr(tail, ep step) {np.mean([r['corr_tail_tep'] for r in R]):+.2f}; episodes {len(eps)} return {np.mean([e[0] for e in eps]):.1f} "
             f"falls {sum(1 for e in eps if e[1] < 1000)}")
    o["rollout"] = {"mu14": ms([r["mu14_mean"] for r in R]), "dV": ms([r["dV_dq3_deg"] for r in R]),
                    "G": ms([5.16 * r["dmu_dq3_deg"] for r in R]), "ret": float(np.mean([e[0] for e in eps])), "n_eps": len(eps)}
    J[f"{run}_{st}"] = o
(V / "vb_aggregate.txt").write_text("\n".join(L) + "\n")
(V / "vb_aggregate.json").write_text(json.dumps(J, indent=1))
print("\n".join(L))
