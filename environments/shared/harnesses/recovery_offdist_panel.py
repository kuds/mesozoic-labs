"""Hand-run harness: recovery certification panels on and off distribution.

Runs the §6.1 off-distribution generalization test and the §6.2 checkpoint
comparison from docs/investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md:
rolls a controller (statue null, brace null, or a trained SB3 policy) over
the seeded pushed panel at the training schedule or at schedules the policy
never trained on, judged under either the provisional safe set or the P3
calibrated posture-only set.

Like everything in harnesses/, this is hand-run instrumentation, not part
of the gated pipeline: the calibrated safe set and the fixed height
reference below restate the P3 record and are superseded the moment a
frozen gate_resolution.json exists.

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
from environments.shared.curriculum.recovery_gate import (
    binomial_lcb,
    binomial_ucb,
    episode_recovery_success,
    per_push_recovery,
)
from environments.shared.recovery_evaluation import DEFAULT_SAFE_SET, zero_action_controller

# P3-calibrated posture-only safe set (first-runs record §4.1/§4.3).  The
# support term is deliberately absent: quiet certified stance itself reads
# 0.0 N on a foot during weight shifts, so per-step support fails the
# certification target (§4.2).  min_foot_force_n = 0.0 keeps the stock
# predicate shape while making the support clause vacuous for the two-foot
# trex plant (its _foot_contact_forces always returns both feet).
CALIBRATED_POSTURE_ONLY = {
    "height_error_max_m": 0.0168,
    "tilt_max_rad": 0.0825,
    "planar_speed_max_mps": 0.3203,
    "min_foot_force_n": 0.0,
}
#: §4.1: the measured settled median pelvis height of certified stance —
#: the calibrated set judges height against this fixed reference, not the
#: per-episode reset stamp the provisional set uses.
CALIBRATED_HEIGHT_REFERENCE_M = 0.9267

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

PANEL_SEED = 3042
PANEL_EPISODES = 40
T_RECOVER_STEPS = 100
DWELL_STEPS = 50
BRACE_SETTLE_SEEDS = (5042, 5043, 5044, 5045, 5046)
BRACE_SETTLE_STEPS = 200


def build_env(schedule: str) -> Any:
    from environments.trex.envs.trex_env import TRexEnv

    config = load_stage_config("trex", "recovery")
    env_kwargs = dict(config["env_kwargs"])
    env_kwargs.update(SCHEDULES[schedule])
    return TRexEnv(**env_kwargs)


def policy_controller(policy_zip: str, vecnorm_pkl: str) -> Callable[[Any], np.ndarray]:
    """Deterministic SB3 policy with the checkpoint's frozen obs normalization."""
    import pickle

    from stable_baselines3 import PPO

    model = PPO.load(policy_zip, device="cpu")
    with open(vecnorm_pkl, "rb") as fh:
        vecnorm = pickle.load(fh)
    obs_rms = vecnorm.obs_rms
    clip_obs = float(getattr(vecnorm, "clip_obs", 10.0))
    epsilon = float(getattr(vecnorm, "epsilon", 1e-8))

    def predict(obs: Any) -> np.ndarray:
        normalized = np.clip(
            (np.asarray(obs) - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
            -clip_obs,
            clip_obs,
        )
        action, _ = model.predict(normalized.astype(np.float32), deterministic=True)
        return np.asarray(action, dtype=np.float64)

    return predict


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

    Mirrors recovery_evaluation.roll_recovery_panel (structural pairing via
    the seed-derived schedule, identical judged-push filter) with the one
    P3 change the stock roller cannot express: judging height against the
    plant's measured settled reference instead of the reset stamp.
    """
    from environments.shared.recovery_evaluation import _safe_step

    horizon = int(env.max_episode_steps)
    episodes: list[dict[str, Any]] = []
    shove_total = 0
    shove_recovered = 0
    for index in range(PANEL_EPISODES):
        panel_seed = PANEL_SEED + index
        obs, _ = env.reset(seed=panel_seed)
        env._recovery_height_reference = (
            float(env.data.qpos[2]) if height_reference is None else float(height_reference)
        )
        starts = np.asarray(env._push_schedule_starts, dtype=int)
        duration = int(env._push_duration_steps)

        safe_mask: list[bool] = []
        total_reward = 0.0
        steps = 0
        truncated = False
        while True:
            obs, reward, terminated, truncated, _info = env.step(predict(obs))
            total_reward += float(reward)
            safe_mask.append(_safe_step(env, safe_set))
            steps += 1
            if terminated or truncated:
                break
        full_horizon = bool(truncated and steps >= horizon)

        judged = [
            k for k, start in enumerate(starts) if int(start) < steps and int(start) + duration + DWELL_STEPS <= horizon
        ]
        recovered_flags = []
        for k in judged:
            recovered, _recovery_step = per_push_recovery(
                safe_mask,
                push_end_step=int(starts[k]) + duration,
                t_recover_steps=T_RECOVER_STEPS,
                dwell_steps=DWELL_STEPS,
            )
            recovered_flags.append(recovered)
        shove_total += len(judged)
        shove_recovered += sum(recovered_flags)
        episodes.append(
            {
                "seed": panel_seed,
                "length": steps,
                "full_horizon": full_horizon,
                "n_pushes": len(judged),
                "n_recovered": sum(recovered_flags),
                "success": episode_recovery_success(full_horizon, recovered_flags),
                "reward": total_reward,
            }
        )

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
    parser.add_argument("--out", help="write the full result (per-episode rows included) as JSON")
    args = parser.parse_args()

    env = build_env(args.schedule)
    if args.controller == "statue":
        predict = zero_action_controller(env.action_space.shape[0])
    else:
        if not (args.policy_zip and args.vecnorm):
            parser.error("--policy-zip and --vecnorm are required for policy/brace controllers")
        predict = policy_controller(args.policy_zip, args.vecnorm)
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
