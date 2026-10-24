"""TRACK TAIL (scratch, report only): the judge's balance readouts per checkpoint, deterministic and under joint noise.

    S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
    T=$S/foot/stance5/tailexp
    env ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $S/foot/stance3/tools/iso.py $T/balance.py \
        --models-dir DIR --label LABEL --steps STEPS [STEPS ...]

measure.py keeps only the falls of the H4 noise tests.  This script calls judge_stance5.noise_matched (imported from its
path, never edited; arithmetic unchanged) twice per checkpoint, both in joint space on the judge's noise spawns
3042-3061 (numpy default_rng(episode seed), a = clip(clip(mu) + sigma z)):
  * sigma = 0 (the 15-vector of zeros): a = clip(mu), the deterministic policy through the judge's noise path;
  * sigma = frozen_params criteria.H4.control_sigma["42"].vector: the K3 joint-noise test itself.
It keeps summarize_episodes' balance readouts of each (falls, K2a / K2b counts, the whole-foot CoP x_c and COM medians,
the pad gate CoP median and its count over 0.8, the whole-foot rail, the roll rail, per-foot CoP and digit share, the
full-horizon mean reward) and per episode length, reward and max_sole_cop_fore_aft.  Writes
tailexp/balance/<label>__<steps>.json (never overwritten).  Runs under the clean band clone at its pinned HEAD.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
JUDGE = SCRATCH + "/foot/stance5/judge/judge_stance5.py"
TAILEXP = SCRATCH + "/foot/stance5/tailexp"
RUN_SEED = 42
KEEP = ("n", "n_full_horizon", "falls", "fall_seeds", "fall_lengths", "k2a_count", "k2a_seeds", "k2b_count", "k2b_seeds",
        "xc_median_full_horizon_m", "xc_mean_full_horizon_m", "com_mtp3_head_median_full_horizon_m",
        "com_fore_aft_footsite_median_full_horizon_m", "cop_gt_bar_gate", "cop_unmeasured", "gate_cop_median",
        "foot_cop_gt_bar", "roll_rail_full_horizon", "reward_mean_full", "per_foot", "sigma_mean", "space", "panel_seconds")


def load_judge():
    spec = importlib.util.spec_from_file_location("judge_stance5", JUDGE)
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge_stance5"] = J
    spec.loader.exec_module(J)
    return J


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--steps", nargs="+", required=True)
    a = ap.parse_args()
    out_dir = os.path.join(TAILEXP, "balance")
    os.makedirs(out_dir, exist_ok=True)
    J = load_judge()
    spec = J.ARMS["LTSB"]
    clone = spec["clone"]
    if os.path.realpath(os.environ.get("ISO_WT", clone)) != os.path.realpath(clone):
        raise SystemExit(f"ISO_WT {os.environ.get('ISO_WT')} is not arm LTSB's clone {clone}")
    modules = J.isolate(clone)
    git = J.git_state(clone)
    if git["status"] or git["head"] != spec["head"]:
        raise SystemExit(f"balance.py runs the clean clone at {spec['head']} (got {git['head']}, dirty={bool(git['status'])})")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    with open(J.FROZEN_PARAMS, "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != J.FROZEN_PARAMS_SHA256:
            raise SystemExit("frozen_params.json is not the pinned one")
    ctrl = J.read_json(J.FROZEN_PARAMS)["criteria"]["H4"]["control_sigma"][str(RUN_SEED)]
    J.install_trace_capture()
    ctx = J.Ctx("LTSB")
    for st in a.steps:
        path = os.path.join(out_dir, f"{a.label}__{st}.json")
        if os.path.exists(path):
            print(json.dumps({"skip_existing": path}), flush=True)
            continue
        t0 = time.monotonic()
        stem, zp, vp = J.checkpoint_paths(os.path.abspath(a.models_dir), J.parse_steps(st))
        model, normalizer = J.load_checkpoint(ctx, zp, vp)
        nu = int(model.policy.log_std.shape[0])
        rec = {"tool": os.path.realpath(__file__), "tool_sha256": J.sha256_file(os.path.realpath(__file__)),
               "judge_sha256": J.sha256_file(JUDGE), "report_only": True, "label": a.label, "steps": st,
               "checkpoint": stem, "model": zp, "model_sha256": J.sha256_file(zp), "num_timesteps": int(model.num_timesteps),
               "clone": clone, "git": git, "modules": modules, "tests": {}}
        for key, sigma, src in (("det_sigma0_joint", np.zeros(nu), "zeros"), ("noise_control_joint", ctrl["vector"], ctrl["source"])):
            nm = J.noise_matched(ctx, model, normalizer, sigma_arg="control" if src != "zeros" else "zero", sigma=sigma,
                                 sigma_source=src, space="joint")
            out = {k: nm.get(k) for k in KEEP}
            out["episodes"] = [{k: e.get(k) for k in ("seed", "length", "reward", "max_sole_cop_fore_aft", "max_foot_cop_fore_aft",
                                                      "max_sole_tilt_deg", "xc_both_feet_m", "com_fore_aft_footsite_m",
                                                      "k2a_over", "k2b_outside")} for e in nm["episodes_detail"]]
            rec["tests"][key] = out
        J.TRACES.clear()
        rec["seconds"] = round(time.monotonic() - t0, 1)
        rec["modules_all_under_clone"] = J.modules_under(clone)
        with open(path, "x") as fh:
            json.dump(J.jsonable(rec), fh, indent=1)
        d, n = rec["tests"]["det_sigma0_joint"], rec["tests"]["noise_control_joint"]

        def mm(v):
            return None if v is None else round(1000 * v, 1)

        print(json.dumps({"label": a.label, "steps": st, "det_falls": d["falls"], "det_k2a": d["k2a_count"],
                          "det_xc_mm": mm(d["xc_median_full_horizon_m"]), "det_gate_cop_med": round(d["gate_cop_median"] or float("nan"), 3),
                          "det_reward": round(d["reward_mean_full"] or float("nan"), 1),
                          "noise_falls": n["falls"], "noise_k2a": n["k2a_count"], "noise_xc_mm": mm(n["xc_median_full_horizon_m"]),
                          "noise_gate_cop_med": round(n["gate_cop_median"] or float("nan"), 3),
                          "noise_reward_full": None if n["reward_mean_full"] is None else round(n["reward_mean_full"], 1),
                          "seconds": rec["seconds"]}), flush=True)
        del model, normalizer
    return 0


if __name__ == "__main__":
    sys.exit(main())
