"""verify_why POST HOC PH5 (not pre-registered): the tail joints' critic slopes split into the part the phase proxy
predicts (kappa x b_j, kappa fitted through the origin over the 21 joint angles, obs 6 excluded from the fit) and the
remainder, per degree.  B-apparent (a height valuation) predicts positive remainders on tail_1, tail_2 and tail_3,
in proportion to their tip lift (23.6 / 18.3 / 13.1 mm per degree).  Reads redo_phase.json and the probe's npz meta."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vw_common as C
R = json.loads((C.HERE / "redo_phase.json").read_text())
out, L = {}, []
for lab in ("P2", "C11a", "C11E", "C31a", "C31E"):
    r = R[lab]
    meta = json.loads(str(np.load(f"{C.PP}/rollouts/{lab}.npz", allow_pickle=False)["meta_json"]))
    sd = np.sqrt(np.asarray(meta["obs_var"]) + meta["epsilon"])[:21]
    c = np.asarray(r["K6_c"]); b = np.array([np.nan if x is None else x for x in r["K6_b"]])
    m = np.isfinite(b); m[C.OBS_T3] = False
    kappa = float((c[m] * b[m]).sum() / (b[m] ** 2).sum())
    perdeg = lambda v, j: v * (math.pi / 180) / sd[j]
    row = {}
    for name, j in (("tail_1_pitch", 3), ("tail_2_pitch", 5), ("tail_3_pitch", 6)):
        row[name] = {"critic": perdeg(c[j], j), "proxy_pred": perdeg(kappa * b[j], j), "remainder": perdeg(c[j] - kappa * b[j], j)}
    out[lab] = {"kappa": kappa, "joints": row}
    L.append(f"PH5 {lab:5s} kappa {kappa:+.3f}: " + "; ".join(f"{n} critic {v['critic']:+.4f} proxy-predicted {v['proxy_pred']:+.4f} remainder {v['remainder']:+.4f}" for n, v in row.items()) + " (per degree)")
(C.HERE / "posthoc_ph5.json").write_text(json.dumps(out, indent=1))
(C.HERE / "posthoc_ph5.txt").write_text("POST HOC PH5 (not pre-registered)\n" + "\n".join(L) + "\n")
print("\n".join(L))
