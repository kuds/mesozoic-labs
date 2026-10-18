"""Offline six-heading stance certificate (stance_quality_heading/v1) on one checkpoint.

Report-only. Calls main's own build_stance_heading_report / write_stance_heading_report
(the function the notebook's end-of-stage certificate calls) on main's stance stage config,
optionally with #624's arm L env options turned on. No training.
"""
import argparse, json, sys, time, hashlib, logging
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--vecnorm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--options", choices=["off", "armL"], default="off")
ap.add_argument("--episodes", type=int, default=None)
ap.add_argument("--seed", type=int, default=None, help="first seed; default the certification panel (PUBLICATION_SEED_START)")
args = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
import environments
assert "main_ro" in environments.__file__, environments.__file__
from environments.shared.model_id import ModelId
from environments.shared.config import load_stage_config
from environments.shared.reporting.stance_report import build_stance_heading_report, write_stance_heading_report

mid = ModelId("tyrannosaurus_rex", "anatomical")
cfg = load_stage_config(mid, 1)
assert cfg["curriculum_kwargs"]["gate_kind"] == "stance_quality_heading/v1"
if args.options == "armL":
    cfg["env_kwargs"] = dict(cfg["env_kwargs"])
    cfg["env_kwargs"].update(lateral_lean_weight=0.25, lateral_lean_tolerance_m=0.015, leg_home_pose_reference="settled")
t0 = time.time()
rep = build_stance_heading_report(mid, 1, stage_config=cfg, model_path=args.model, vecnorm_path=args.vecnorm,
                                  episodes=args.episodes, **({} if args.seed is None else {"seed": args.seed}))
dt = time.time() - t0
out = Path(args.out)
written = write_stance_heading_report(out, rep)
r = rep["result"]
summary = {
    "first_seed": rep["seed"], "seeds": [rep["seeds"][0], rep["seeds"][-1]],
    "environments_file": environments.__file__,
    "options": args.options,
    "env_overrides": ({"lateral_lean_weight": 0.25, "lateral_lean_tolerance_m": 0.015, "leg_home_pose_reference": "settled"}
                      if args.options == "armL" else {}),
    "model": args.model, "model_sha256": sha(args.model),
    "vecnorm": args.vecnorm, "vecnorm_sha256": sha(args.vecnorm),
    "seconds": dt, "passed": rep["passed"], "failures": rep["failures"],
    "n_clean_seeds": r.get("n_clean_seeds"), "n_seeds": r.get("n_seeds"), "clean_seed_lcb": r.get("clean_seed_lcb"),
    "task_sha256": rep["task_sha256"], "gate_sha256": rep["gate_sha256"],
    "headings": [{"yaw_deg": h["yaw_deg"], **{k: h["metrics"][k] for k in ("reward_mean", "reward_std", "episode_length_mean",
                  "full_horizon_fraction", "n_clean", "clean_lcb")},
                  "statue_reward_mean": (h["statue"] or {}).get("reference", {}).get("mean_reward") if h["statue"] else None}
                 for h in rep["headings"]],
    "written": {k: str(v) for k, v in written.items()},
}
(out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: summary[k] for k in ("options", "seconds", "passed", "n_clean_seeds", "n_seeds", "clean_seed_lcb")}))
