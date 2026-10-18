"""The repository's impulse-recovery probe, run offline on one role (a checkpoint or the statue).

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen). Report only; certifies
nothing; no training.

This calls main's own probe machinery exactly as environments/shared/reporting/stage_artifacts.py:_write_impulse_probe
does for a stage that sets stance_probe_impulse_speeds (configs/tyrannosaurus_rex/anatomical/stance.toml: [0.5, 1.0,
2.0]): stance_report.impulse_variants(speeds, step=settle_steps) -> build_stance_gate_report(..., episodes=
_IMPULSE_EPISODES, impulse=variant) for the policy and for the zero-action statue, then write_impulse_probe.

The one extension: main's impulse_variants() builds the zero control plus lateral +y/-y only. Forward (+x) and
backward (-x) rows are added as plain stance_report.RootImpulse(step, (+-speed, 0, 0)) objects, the same dataclass
the lateral rows are, and go through the same build_stance_gate_report call (_apply_root_impulse adds delta_v to
qvel[0:3] at that step, before the action). The stance stage spawns facing +x, so +x is forward and +y is left.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--vecnorm", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(1)

import environments  # noqa: E402

assert "main_ro" in environments.__file__, environments.__file__
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.reporting import stage_artifacts  # noqa: E402
from environments.shared.reporting.stance_report import (  # noqa: E402
    PUBLICATION_SEED_START,
    RootImpulse,
    build_stance_gate_report,
    impulse_variants,
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


MID = ModelId("tyrannosaurus_rex", "anatomical")
CFG = load_stage_config(MID, 1)
CUR = CFG["curriculum_kwargs"]
assert CUR["gate_kind"] == "stance_quality_heading/v1"
SPEEDS = [float(v) for v in CUR["stance_probe_impulse_speeds"]]
SETTLE = int(CUR["settle_steps"])
EPISODES = int(stage_artifacts._IMPULSE_EPISODES)

variants = impulse_variants(SPEEDS, step=SETTLE)
for speed in sorted(set(SPEEDS)):
    variants.append(RootImpulse(step=SETTLE, delta_v=(speed, 0.0, 0.0), axis_label="forward +x"))
    variants.append(RootImpulse(step=SETTLE, delta_v=(-speed, 0.0, 0.0), axis_label="backward -x"))

zero = args.model is None
reports = []
t0 = time.time()
for variant in variants:
    t1 = time.time()
    rep = build_stance_gate_report(
        MID,
        1,
        stage_config=CFG,
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        zero_action=zero,
        episodes=EPISODES,
        impulse=variant,
    )
    reports.append(rep)
    print(json.dumps({"role": args.role, "impulse": variant.as_dict(), "len_mean": rep["metrics"]["episode_length_mean"],
                      "full_horizon": rep["metrics"]["full_horizon_fraction"], "terminations": rep["terminations"],
                      "seconds": round(time.time() - t1, 1)}), flush=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / f"sweep_{args.role}.pkl", "wb") as f:
    pickle.dump(reports, f)
meta = {
    "label": "OFFLINE EVALUATION of an uncertified scratch-recipe checkpoint; report only, certifies nothing",
    "role": args.role,
    "environments_file": environments.__file__,
    "model": args.model, "model_sha256": None if zero else sha(args.model),
    "vecnorm": args.vecnorm, "vecnorm_sha256": None if zero else sha(args.vecnorm),
    "speeds": SPEEDS, "impulse_step": SETTLE, "episodes_per_row": EPISODES, "first_seed": PUBLICATION_SEED_START,
    "variants": [v.as_dict() for v in variants],
    "seconds": time.time() - t0,
}
(out / f"sweep_{args.role}_meta.json").write_text(json.dumps(meta, indent=1))
print("done", args.role, round(meta["seconds"], 1))
