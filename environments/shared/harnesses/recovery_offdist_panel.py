"""Hand-run harness: recovery certification panels on and off distribution.

Runs the §6.1 off-distribution generalization test and the §6.2 checkpoint
comparison from docs/investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md:
rolls a controller (statue null, brace null, or a trained SB3 policy) over
the seeded pushed panel at the training schedule or at schedules the policy
never trained on, judged under either the provisional safe set or the P3
calibrated posture-only set.

Like everything in harnesses/, this is hand-run instrumentation, not part
of the gated pipeline: it re-rolls panels on demand, while the frozen
gate_resolution.json — written once by ``freeze_recovery_gate.py`` — is
what any gate actually consumes.  The calibrated safe set and the fixed
height reference are IMPORTED from ``recovery_evaluation`` rather than
restated here, so a hand roll and the frozen record can never disagree
about what "calibrated" means.

Examples (repo root, trex):

    # Reproduce the frozen statue null at the training schedule
    python -m environments.shared.harnesses.recovery_offdist_panel \
        --controller statue --schedule on --safe-set provisional

    # §6.1 timing variant for a trained policy
    python -m environments.shared.harnesses.recovery_offdist_panel \
        --controller policy --schedule timing --safe-set calibrated \
        --policy-zip <robust_best_model.zip> --vecnorm <..._vecnorm.pkl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.curriculum.recovery_gate import binomial_lcb, binomial_ucb

# The panel geometry, the recovery clock, and the checkpoint forward pass
# are the P5-frozen ones, taken from the freeze producer under this
# harness's historical names: a hand panel that used a different dwell,
# seed block, or inference path would not be comparable with the frozen
# record it exists to probe — §9's panels ran the producer's NumPy
# fallback, which is why the loader lives there and is shared.  (The
# producer imports this module's brace_controller lazily, so the
# dependency stays one-way.)
from environments.shared.harnesses.freeze_recovery_gate import (
    DWELL_STEPS,
    T_RECOVER_STEPS,
    policy_controller,
)
from environments.shared.harnesses.freeze_recovery_gate import MIN_EVAL_EPISODES as PANEL_EPISODES
from environments.shared.harnesses.freeze_recovery_gate import PANEL_SEED_START as PANEL_SEED

# The P3-calibrated posture-only judge (first-runs record §4.1/§4.3),
# re-exported under the names this harness has always used.  Its
# derivation record — including why there is no per-step support term —
# lives with the definition in recovery_evaluation.
from environments.shared.recovery_evaluation import (
    CALIBRATED_HEIGHT_REFERENCE_M,
    CALIBRATED_POSTURE_ONLY,
    DEFAULT_SAFE_SET,
    roll_recovery_panel,
    zero_action_controller,
)

#: §6.1 schedules.  "on" is the training distribution (recovery.toml);
#: the off-distribution rows change exactly one thing each.  The magnitude
#: rows are expressed as capture-velocity multiples because that is the
#: env's parameter; on the r7 plant multiple 1.5 derives 165.5 N, so the
#: 120 N / 210 N targets of §6.1 are 1.5 * (target / 165.501).
SCHEDULES = {
    "on": {},
    "timing": {"perturbation_interval": 3.5, "perturbation_jitter": 1.5},
    "mag120": {"perturbation_capture_velocity_multiple": 1.5 * 120.0 / 165.501},
    "mag210": {"perturbation_capture_velocity_multiple": 1.5 * 210.0 / 165.501},
}

BRACE_SETTLE_SEEDS = (5042, 5043, 5044, 5045, 5046)
BRACE_SETTLE_STEPS = 200


def build_env(schedule: str) -> Any:
    from environments.trex.envs.trex_env import TRexEnv

    config = load_stage_config("trex", "recovery")
    env_kwargs = dict(config["env_kwargs"])
    env_kwargs.update(SCHEDULES[schedule])
    return TRexEnv(**env_kwargs)


def brace_controller(env: Any, predict: Callable[[Any], np.ndarray]) -> Callable[[Any], np.ndarray]:
    """The policy's post-settle mean action, held (first-runs record §3.1)."""
    from environments.shared.recovery_evaluation import constant_action_controller

    actions: list[np.ndarray] = []
    for seed in BRACE_SETTLE_SEEDS:
        obs, _ = env.reset(seed=seed)
        for step in range(BRACE_SETTLE_STEPS * 2):
            action = predict(obs)
            if step >= BRACE_SETTLE_STEPS:
                actions.append(np.asarray(action, dtype=np.float64))
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    return constant_action_controller(np.mean(np.stack(actions), axis=0))


def roll_panel(
    env: Any,
    predict: Callable[[Any], np.ndarray],
    *,
    controller_id: str,
    safe_set: dict[str, float],
    height_reference: "float | None",
) -> dict[str, Any]:
    """Roll the seeded panel; a fixed height_reference selects the calibrated judge.

    Delegates the rolling and the judging to
    ``recovery_evaluation.roll_recovery_panel`` — since P5 the stock roller
    takes the ``height_reference`` this harness used to need its own loop
    for — and adds only the summary statistics a hand run wants to read
    (the bounds, the per-shove totals, the schedule the panel actually
    ran).  One roller means a hand panel and the frozen record cannot drift
    apart in their event logic.
    """
    evidence = roll_recovery_panel(
        env,
        predict,
        controller_id=controller_id,
        episodes=PANEL_EPISODES,
        seed=PANEL_SEED,
        t_recover_steps=T_RECOVER_STEPS,
        dwell_steps=DWELL_STEPS,
        safe_set=safe_set,
        height_reference=height_reference,
    )
    episodes: list[dict[str, Any]] = [
        {
            "seed": record.panel_seed,
            "length": record.length,
            "full_horizon": record.full_horizon,
            "n_pushes": record.n_pushes,
            "n_recovered": record.n_recovered,
            "success": record.success,
            "reward": record.reward,
        }
        for record in evidence.episodes
    ]
    shove_total = len(evidence.shoves)
    shove_recovered = sum(1 for shove in evidence.shoves if shove.recovered)

    successes = sum(1 for e in episodes if e["success"])
    rewards = np.array([e["reward"] for e in episodes])
    lengths = np.array([e["length"] for e in episodes])
    return {
        "controller": controller_id,
        "push_force_n": float(env._push_force_n),
        "interval_s": float(env.perturbation_interval),
        "jitter_s": float(env.perturbation_jitter),
        "safe_set": dict(safe_set),
        "height_reference": height_reference,
        "successes": successes,
        "episodes": PANEL_EPISODES,
        "success_lcb95": binomial_lcb(successes, PANEL_EPISODES),
        "success_ucb95": binomial_ucb(successes, PANEL_EPISODES),
        "full_horizon": sum(1 for e in episodes if e["full_horizon"]),
        "shoves_judged": shove_total,
        "shoves_recovered": shove_recovered,
        "mean_length": float(lengths.mean()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "per_episode": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["statue", "brace", "policy"], required=True)
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), default="on")
    parser.add_argument("--safe-set", choices=["provisional", "calibrated"], default="provisional")
    parser.add_argument("--policy-zip", help="SB3 checkpoint (.zip); required for policy and brace")
    parser.add_argument("--vecnorm", help="matching VecNormalize stats (.pkl)")
    parser.add_argument(
        "--inference",
        choices=("auto", "sb3", "numpy"),
        default="auto",
        help="forward pass for the checkpoint; 'numpy' skips PPO.load entirely (the §9 path)",
    )
    parser.add_argument("--out", help="write the full result (per-episode rows included) as JSON")
    args = parser.parse_args()

    env = build_env(args.schedule)
    if args.controller == "statue":
        predict = zero_action_controller(env.action_space.shape[0])
    else:
        if not (args.policy_zip and args.vecnorm):
            parser.error("--policy-zip and --vecnorm are required for policy/brace controllers")
        predict = policy_controller(
            args.policy_zip, args.vecnorm, action_space=env.action_space, inference=args.inference
        )
        if args.controller == "brace":
            predict = brace_controller(env, predict)

    if args.safe_set == "provisional":
        safe_set, height_reference = dict(DEFAULT_SAFE_SET), None
    else:
        safe_set, height_reference = dict(CALIBRATED_POSTURE_ONLY), CALIBRATED_HEIGHT_REFERENCE_M

    result = roll_panel(
        env,
        predict,
        controller_id=f"{args.controller}/{args.schedule}/{args.safe_set}",
        safe_set=safe_set,
        height_reference=height_reference,
    )

    summary = {k: v for k, v in result.items() if k != "per_episode"}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
