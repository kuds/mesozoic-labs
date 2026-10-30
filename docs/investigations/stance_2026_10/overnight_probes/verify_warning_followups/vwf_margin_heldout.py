"""Verifier: held-out-seed (4001-4020) C1 errors for every predictor, and pooled 40-seed margins (20th-21st order
statistic: smallest value parking >= 21 of 40 seeds), for the 4 near-rail checkpoints.  No simulation."""
import json
from pathlib import Path
import numpy as np
N = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes")
pred = json.loads((N / "warning_margin/wm_predict.json").read_text())
def T(rows):
    out = {}
    for s in sorted({r["seed"] for r in rows}):
        rr = [r for r in rows if r["seed"] == s]
        if any(r["kind"] == "none" and r["parked"] for r in rr):
            out[s] = 0.0; continue
        p = [r["value"] for r in rr if r["kind"] == "bias" and r["parked"]]
        out[s] = min(p) if p else 2.0
    return out
orig = [json.loads(x) for x in (N / "warning_margin/episodes.jsonl").read_text().splitlines()]
mine = [json.loads(x) for x in (N / "verify_warning_followups/vwf_margin_episodes.jsonl").read_text().splitlines()]
for run, st in (("s80", 11534336), ("s80", 12001280), ("s60", 7340032), ("s42", 3145728)):
    To = T([r for r in orig if r["run"] == run and r["ckpt_steps"] == st])
    Th = T([r for r in mine if r["run"] == run and r["ckpt"] == st and r["seed"] >= 4001])
    Mo, Mh = sorted(To.values())[10], sorted(Th.values())[10]
    pooled = sorted(list(To.values()) + list(Th.values()))[20]
    p = pred[f"{run}_{st}"]
    cells = []
    for k in ("bstar_P", "bstar_L", "bstar_S", "gap_open"):
        v = p[k]; tol = max(0.05, 0.2 * v)
        cells.append(f"{k} {v:.3f}: err orig {Mo - v:+.3f} {'Y' if abs(Mo - v) <= tol else 'n'} / held {Mh - v:+.3f} {'Y' if abs(Mh - v) <= tol else 'n'} / pooled {pooled - v:+.3f} {'Y' if abs(pooled - v) <= tol else 'n'}")
    print(f"{run} {st}: M orig {Mo:.4f} held-out {Mh:.4f} pooled-40 {pooled:.4f} | " + " | ".join(cells))
