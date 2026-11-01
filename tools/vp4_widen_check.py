"""PC VP4 (part) dry run, report only: the manipulation check and the phase-fed critic feed on a STANCE6 pair widened
in memory with zero phase columns (pc/manip_check.widen_stock_policy), on the HEAD measuring copy.  Expected: the
widened pair reads NOT IN FORCE (failing (i) only: zero phase weights, every |dV| 0), the actor is unchanged (its means
equal the stock policy's bitwise), and V with the phase fed equals the stock critic's V (bitwise, or the largest |dV|
printed as found).  Reads the F2 copies read only (their sha256 checked against the fetch manifest); writes stdout only.

    PY -I -B vp4_widen_check.py --copy COPY [--run stance6_r10LTSB_s80] [--label final]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True
T = Path(__file__).resolve().parent
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
FETCH = S1 / "foot/stance6/fetch"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", required=True)
    ap.add_argument("--run", default="stance6_r10LTSB_s80")
    ap.add_argument("--label", default="final")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("manip_check", T / "pc" / "manip_check.py")
    MC = importlib.util.module_from_spec(spec)
    sys.modules["manip_check"] = MC
    spec.loader.exec_module(MC)
    rels = (("01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl") if a.label == "final"
            else (f"01_stance/models/stage1_{a.label}_steps.zip", f"01_stance/models/stage1_vecnormalize_{a.label}_steps.pkl"))
    with open(FETCH / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["fetch"] == "F2" and r["run_id"] == a.run and r["status"] == "COPY"]
    paths = []
    for rel in rels:
        p = FETCH / "F2" / a.run / "copies" / f"{rel}.copy1"
        h = sha(p)
        ok = any(r["path"] == rel and r["sha256"] == h for r in rows)
        print(f"{rel}.copy1 sha256 {h} in the F2 fetch manifest: {ok}")
        if not ok:
            return 2
        paths.append(p)
    copy = Path(a.copy).resolve()
    sys.path.insert(0, str(copy))
    import numpy as np
    from environments.shared.config import load_all_stages
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.reporting.episode_trace import load_trace_policy
    from environments.shared.species_registry import env_class as model_env_class
    from environments.shared.stage_manifest import load_stage_manifest

    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    stage = load_stage_manifest(model_id).resolve_behavior("stance").reference
    env_kwargs = dict(load_all_stages(model_id)[stage]["env_kwargs"])
    env_class = model_env_class(model_id)
    tp = load_trace_policy(str(paths[0]), str(paths[1]), lambda: env_class(**env_kwargs),
                           plant_identity=current_plant_identity(model_id))
    env = env_class(**env_kwargs)
    try:
        bank, ks, per = MC.collect_bank(tp, env)
    finally:
        env.close()
    stock = tp.model.policy
    print(f"stock policy {type(stock).__name__}; bank {bank.shape}; episodes {json.dumps(per)}")
    res_stock = MC.manipulation_checks(stock, bank, ks)
    print(f"STOCK (no phase input): verdict {res_stock['verdict']} {json.dumps(res_stock['checks'])}")
    wide = MC.widen_stock_policy(stock, 200, 0.98)
    res = MC.manipulation_checks(wide, bank, ks)
    print(f"WIDENED (zero phase columns): verdict {res['verdict']}, failing {res['failing']}")
    print(json.dumps(res["checks"], indent=1))
    v_stock = MC.critic_values(stock, bank)
    for mode in ("effective", "settled"):
        v_wide = MC.critic_values(wide, bank, ks, mode=mode)
        d = np.abs(v_wide - v_stock)
        print(f"phase-fed V ({mode}) vs the stock critic's V: bitwise equal {bool(np.array_equal(v_wide, v_stock))}; "
              f"max |dV| {float(d.max()):.3e}; rows differing {int(np.count_nonzero(d))} of {len(d)}")
    mu_stock = MC.actor_means(stock, bank)
    mu_wide = MC.actor_means(wide, bank, MC.phase_matrix(wide.critic_phase, ks))
    print(f"actor means widened (phase appended) vs stock: bitwise equal {bool(np.array_equal(mu_stock, mu_wide))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
