"""Measure a foot's toe line by the held-offset rule, and record it.

A foot that declares an ankle and a support span in ``FOOT_SUPPORT``
(``ankle``, ``heel_line``, ``toe_line``; ``gait/morphology.py``) reads its
whole-foot centre of pressure in half-lengths of that span, so the toe line
is where a reading of 1 sits.  This script places it by one fixed rule,
measured on the zero-action statue of the stance stage, so that two plants
measured with it are read on the same instrument:

* hold one ankle's target at an offset toward the toe (ramped in over
  control steps 0-199 and held; every other actuator at the statue's zero
  action), on a uniform grid from -2.5 deg in 0.5 deg steps, until the first
  level at which at least one episode falls;
* five episodes per level, seeds 3042-3046, the stance [env] (reset noise
  included), 1000 steps;
* per level, the OTHER foot's whole-foot centre of pressure
  (``EpisodeTrace.foot_cop``, m ahead of its ankle axis), its mean over the
  window (control steps 200 to the end), and the median of those means over
  the episodes that do not fall;
* the ankle's onset is the first level with a fall, and its value the median
  at that level;
* the toe line is the smaller onset value of the two ankles, floored to 1 mm.

Flooring, not rounding, moves the line less than 1 mm heel-ward of the
measured onset on every plant, so it biases no plant against another.  The
heel line is not measured here: it is the sole's own heel edge.

Run from the repository root (about two minutes on one CPU)::

    python -m environments.shared.scripts.foot_toe_line --model-id tyrannosaurus_rex anatomical

It writes ``environments/<species>/<embodiment>/data/foot_toe_line.json``
(``--output`` elsewhere): the plant's physics digest, the rule, every
episode's reading, both onsets and the line.  The model's ``FOOT_SUPPORT``
``toe_line`` must equal the record's line, and the record's physics digest
the committed plant's (``test_gait_morphology.py``), so a physics revision
re-runs this script.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT

#: The record's schema.
TOE_LINE_SCHEMA = "mesozoic.foot-toe-line/v1"

#: The rule, fixed: the seeds per level, the grid, the ramp and the resolution.
SEEDS: tuple[int, ...] = (3042, 3043, 3044, 3045, 3046)
START_DEG = -2.5
STEP_DEG = 0.5
RAMP_STEPS = 200
RESOLUTION_M = 0.001
#: A guard, not part of the rule: an ankle that has not fallen this far toe-ward is reported, not extrapolated.
MAX_LEVELS = 24

STAGE = "stance"


def record_path(model_id: ModelId) -> Path:
    """Where the model's record lives: ``environments/<species>/<embodiment>/data/foot_toe_line.json``."""
    return model_id.path(ENVIRONMENTS_ROOT) / "data" / "foot_toe_line.json"


def level_median(episodes: Sequence[Mapping[str, Any]]) -> float | None:
    """The median of the window means of the episodes that did not fall at one level, m; ``None`` if all fell."""
    values = [float(episode["other_foot_cop_m"]) for episode in episodes if not episode["fell"]]
    return float(np.median(values)) if values else None


def ankle_onset(levels: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The first level (in grid order) with at least one fall, or ``None`` when no level fell."""
    return next((level for level in levels if any(episode["fell"] for episode in level["episodes"])), None)


def toe_line_from_onsets(onset_values_m: Sequence[float]) -> float:
    """The toe line: the smallest onset value, floored to :data:`RESOLUTION_M` (in m)."""
    smallest = min(onset_values_m)
    millimetres = math.floor(round(smallest / RESOLUTION_M, 6))
    return round(millimetres * RESOLUTION_M, 6)


def toe_line_of_record(record: Mapping[str, Any]) -> float:
    """Re-apply the rule to a record's own episodes: the line it must state.

    Raises ``ValueError`` when an ankle never fell or its onset level has no
    standing episode, since then the rule gives no line.
    """
    values = []
    for label, ankle in record["ankles"].items():
        onset = ankle_onset(ankle["levels"])
        if onset is None:
            raise ValueError(f"ankle {label} did not fall on the recorded grid")
        median = level_median(onset["episodes"])
        if median is None:
            raise ValueError(f"ankle {label}'s onset level has no standing episode")
        values.append(median)
    return toe_line_from_onsets(values)


def _action_for(scale: Callable[[np.ndarray], np.ndarray], nu: int, index: int, target: float) -> float:
    """The action component whose control equals *target*, by bisection on the env's own action mapping."""

    def ctrl(value: float) -> float:
        action = np.zeros(nu)
        action[index] = value
        return float(np.asarray(scale(action)).ravel()[index])

    low_ctrl, high_ctrl = ctrl(-1.0), ctrl(1.0)
    if not min(low_ctrl, high_ctrl) <= target <= max(low_ctrl, high_ctrl):
        raise ValueError(f"control {target} lies outside actuator {index}'s range [{low_ctrl}, {high_ctrl}]")
    increasing = high_ctrl >= low_ctrl
    low, high = -1.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if (ctrl(middle) < target) == increasing:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def measure(model_id: ModelId) -> dict[str, Any]:
    """Roll the rule on *model_id*'s stance statue and return the record."""
    import mujoco

    from environments.shared.config import build_env, load_stage_config
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder
    from environments.shared.plant_contract import current_plant_identity, validate_environment_plant

    identity = current_plant_identity(model_id)
    config = load_stage_config(model_id, STAGE)
    window_start = int(config["curriculum_kwargs"].get("settle_steps", 200))
    env = build_env(model_id, STAGE)
    validate_environment_plant(env, identity, artifact=f"{model_id} toe-line environment")
    try:
        morphology = Morphology.from_env(env, model_id)
        if morphology.n_feet != 2 or any(foot.ankle_joint < 0 for foot in morphology.feet):
            raise SystemExit(f"{model_id}: the rule needs two feet that each declare an ankle in FOOT_SUPPORT")
        model = env.model
        nu = int(model.nu)
        horizon = int(env.max_episode_steps)
        home = np.asarray(env._scale_action(np.zeros(nu)), dtype=np.float64).ravel()
        recorder = SubstepContactRecorder(env, morphology).attach()
        ankles: dict[str, Any] = {}
        try:
            for index, foot in enumerate(morphology.feet):
                joint = int(foot.ankle_joint)
                actuators = [a for a in range(nu) if int(model.actuator_trnid[a, 0]) == joint]
                if len(actuators) != 1:
                    raise SystemExit(f"{model_id}: ankle {model.joint(joint).name} has {len(actuators)} actuators")
                actuator = actuators[0]
                other = 1 - index
                levels = []
                for step in range(MAX_LEVELS):
                    degrees = START_DEG - STEP_DEG * step
                    hold = np.zeros(nu)
                    hold[actuator] = _action_for(
                        env._scale_action, nu, actuator, home[actuator] + math.radians(degrees)
                    )
                    episodes: list[dict[str, Any]] = []
                    for seed in SEEDS:
                        env.reset(seed=seed)
                        terminated = truncated = False
                        reason = ""
                        for t in range(horizon):
                            action = hold * min(1.0, (t + 1) / RAMP_STEPS)
                            _, reward, terminated, truncated, info = env.step(action)
                            recorder.end_step(action, float(reward))
                            if terminated or truncated:
                                reason = str(info.get("termination_reason", ""))
                                break
                        trace = recorder.end_episode(
                            terminated=bool(terminated), truncated=bool(truncated), termination_reason=reason
                        )
                        window = np.asarray(trace.foot_cop[window_start:, other], dtype=np.float64)
                        mean = float(np.nanmean(window)) if np.isfinite(window).any() else None
                        fell = int(trace.length) < horizon
                        episodes.append(
                            {
                                "seed": seed,
                                "length": int(trace.length),
                                "fell": fell,
                                "termination_reason": reason if fell else "",
                                "other_foot_cop_m": None if mean is None else round(mean, 9),
                            }
                        )
                    median = level_median(episodes)
                    levels.append(
                        {
                            "deg": degrees,
                            "falls": sum(bool(episode["fell"]) for episode in episodes),
                            "other_foot_cop_median_m": None if median is None else round(median, 9),
                            "episodes": episodes,
                        }
                    )
                    if any(episode["fell"] for episode in episodes):
                        break
                onset = ankle_onset(levels)
                ankles[foot.label] = {
                    "joint": str(model.joint(joint).name),
                    "actuator": str(model.actuator(actuator).name),
                    "other_foot": morphology.feet[other].label,
                    "onset_deg": None if onset is None else onset["deg"],
                    "onset_m": None if onset is None else onset["other_foot_cop_median_m"],
                    "levels": levels,
                }
        finally:
            recorder.detach()
    finally:
        env.close()
    record: dict[str, Any] = {
        "schema": TOE_LINE_SCHEMA,
        "species": model_id.species,
        "embodiment": model_id.embodiment,
        "physics_revision": int(identity.physics_revision),
        "physics_sha256": identity.physics_sha256,
        "mujoco_version": mujoco.__version__,
        "rule": {
            "stage": STAGE,
            "seeds": list(SEEDS),
            "start_deg": START_DEG,
            "step_deg": STEP_DEG,
            "ramp_steps": RAMP_STEPS,
            "window_start_step": window_start,
            "horizon": horizon,
            "resolution_m": RESOLUTION_M,
            "statistic": "the other foot's window-mean whole-foot CoP (m ahead of its ankle axis), median over "
            "the episodes that do not fall; the onset is the first level with a fall; the line is the smaller "
            "onset of the two ankles, floored to the resolution",
        },
        "ankles": ankles,
    }
    record["toe_line_m"] = toe_line_of_record(record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    from environments.shared.species_names import add_model_id_argument, parsed_model_ids

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_model_id_argument(parser, default=(ModelId("tyrannosaurus_rex", "anatomical"),), what="measure")
    parser.add_argument("--output", type=Path, default=None, help="the record's path (default: the model's data/)")
    args = parser.parse_args(argv)
    model_ids = parsed_model_ids(parser, args)
    if len(model_ids) != 1:
        parser.error("name exactly one model with --model-id SPECIES EMBODIMENT")
    model_id = model_ids[0]
    record = measure(model_id)
    path = args.output or record_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for label, ankle in record["ankles"].items():
        print(f"{label} ankle: onset {ankle['onset_deg']} deg, other foot's CoP {ankle['onset_m']} m")
    print(f"toe line {record['toe_line_m']} m -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
