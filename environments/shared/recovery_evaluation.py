"""Pushed-panel evaluation for the recovery stage (stage 1b, W4 part 2).

Rolls a policy (or a null controller) over the standard seeded panel on a
push-enabled environment, judges every scheduled push with the recovery
gate's event logic, and produces the evidence the split plan makes part of
the estimand: one row per episode AND one row per shove, each carrying the
controller ID, seed, push vector/timing, and recovery outcome, so a gate
verdict can be audited shove by shove.

Pairing is structural, not procedural: the environment derives its push
schedule from the reset seed, so rolling the policy and every null over
the same seeds guarantees identical pushes without any coordination here.
A determinism test pins that property end to end.

The safe set is measured from PHYSICAL state (pelvis height vs the plant's
settled height, tilt, planar speed, bilateral foot load), not from reward
terms — the same measurement-over-proxy principle as the stance gate.  Its
thresholds are provisional (plan P3 calibrates them) and are recorded in
the evidence so no row can outlive the definition that produced it.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .curriculum.recovery_gate import episode_recovery_success, per_push_recovery

#: Provisional safe-set thresholds (plan §6 / P3).  Recorded per evidence
#: file; a calibration run replaces them WITH the re-derivation, never
#: silently.
DEFAULT_SAFE_SET = {
    "height_error_max_m": 0.10,
    "tilt_max_rad": 0.35,
    "planar_speed_max_mps": 0.30,
    "min_foot_force_n": 0.1,
}


@dataclass(frozen=True)
class ShoveRecord:
    """One scheduled push and its outcome."""

    controller_id: str
    episode: int
    panel_seed: int
    push_index: int
    start_step: int
    end_step: int
    direction_x: float
    direction_y: float
    force_n: float
    recovered: bool
    recovery_step: int


@dataclass(frozen=True)
class EpisodeRecord:
    """One panel episode's outcome under the pushed task."""

    controller_id: str
    episode: int
    panel_seed: int
    length: int
    full_horizon: bool
    n_pushes: int
    n_recovered: int
    success: bool
    reward: float


@dataclass(frozen=True)
class RecoveryPanelEvidence:
    controller_id: str
    episodes: tuple[EpisodeRecord, ...]
    shoves: tuple[ShoveRecord, ...]
    safe_set: dict[str, float]

    def successes_by_seed(self) -> dict[int, bool]:
        return {record.panel_seed: record.success for record in self.episodes}

    def episode_successes(self) -> tuple[bool, ...]:
        return tuple(record.success for record in self.episodes)


def zero_action_controller(action_dim: int) -> Callable[[Any], np.ndarray]:
    """The statue: commands the home keyframe and cannot respond to a push."""
    zero = np.zeros(action_dim, dtype=np.float64)

    def predict(_obs: Any) -> np.ndarray:
        return zero

    return predict


def constant_action_controller(action: Sequence[float]) -> Callable[[Any], np.ndarray]:
    """A brace: holds one fixed command (e.g. the policy's post-settle mean)."""
    held = np.asarray(action, dtype=np.float64).ravel()

    def predict(_obs: Any) -> np.ndarray:
        return held

    return predict


def _safe_step(env: Any, safe_set: dict[str, float]) -> bool:
    """The per-step safe predicate, from physical state.

    Height is judged against the panel-start root height stamped by
    ``roll_recovery_panel`` immediately after reset.  Provisional: the P3
    calibration replaces this reference with the plant's measured settled
    pelvis height (the reset height differs from the settled stance by the
    settle transient plus reset noise), together with the numeric
    thresholds — recorded per evidence file so no row outlives its
    definition.
    """
    data = env.data
    height_target = getattr(env, "_recovery_height_reference", None)
    if height_target is None:
        raise RuntimeError("roll_recovery_panel must stamp _recovery_height_reference before stepping")
    height_ok = abs(float(data.qpos[2]) - height_target) <= safe_set["height_error_max_m"]
    tilt_ok = float(env._quat_to_tilt(data.qpos[3:7])) <= safe_set["tilt_max_rad"]
    speed_ok = float(np.linalg.norm(data.qvel[0:2])) <= safe_set["planar_speed_max_mps"]
    forces = env._foot_contact_forces()
    support_ok = len(forces) >= 2 and all(f >= safe_set["min_foot_force_n"] for f in forces)
    return height_ok and tilt_ok and speed_ok and support_ok


def roll_recovery_panel(
    env: Any,
    predict: Callable[[Any], np.ndarray],
    *,
    controller_id: str,
    episodes: int,
    seed: int,
    t_recover_steps: int,
    dwell_steps: int,
    safe_set: "dict[str, float] | None" = None,
) -> RecoveryPanelEvidence:
    """Roll one controller over the seeded pushed panel and judge every shove.

    The environment must have perturbation enabled — a push-free env would
    produce a panel that vacuously recovers nothing, so it is refused.
    """
    if getattr(env, "_push_schedule_starts", None) is None and env.perturbation_capture_velocity_multiple <= 0.0:
        raise ValueError("roll_recovery_panel requires a perturbation-enabled environment")
    active_safe_set = dict(DEFAULT_SAFE_SET if safe_set is None else safe_set)
    episode_records: list[EpisodeRecord] = []
    shove_records: list[ShoveRecord] = []
    horizon = int(env.max_episode_steps)

    for index in range(episodes):
        panel_seed = seed + index
        obs, _ = env.reset(seed=panel_seed)
        env._recovery_height_reference = float(env.data.qpos[2])
        starts = np.asarray(env._push_schedule_starts, dtype=int)
        directions = np.asarray(env._push_schedule_directions, dtype=float)
        duration = int(env._push_duration_steps)
        force_n = float(env._push_force_n)

        safe_mask: list[bool] = []
        total_reward = 0.0
        steps = 0
        truncated = False
        while True:
            action = np.asarray(predict(obs), dtype=np.float64).ravel()
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            safe_mask.append(_safe_step(env, active_safe_set))
            steps += 1
            if terminated or truncated:
                break
        full_horizon = bool(truncated and steps >= horizon)

        # A push is JUDGED only when (a) it was actually delivered — its
        # window opened before the episode ended — and (b) a full recovery
        # window (push + dwell) fits inside the horizon.  A horizon-adjacent
        # push that cannot be recovered by construction is excluded, not
        # counted as a failure (it would cap achievable success below any
        # threshold for every controller); an undelivered push is a phantom
        # and pollutes per-shove analysis if recorded.
        judged = [
            k for k, start in enumerate(starts) if int(start) < steps and int(start) + duration + dwell_steps <= horizon
        ]
        recovered_flags: list[bool] = []
        for k in judged:
            start = int(starts[k])
            end = start + duration
            recovered, recovery_step = per_push_recovery(
                safe_mask,
                push_end_step=end,
                t_recover_steps=t_recover_steps,
                dwell_steps=dwell_steps,
            )
            recovered_flags.append(recovered)
            shove_records.append(
                ShoveRecord(
                    controller_id=controller_id,
                    episode=index + 1,
                    panel_seed=panel_seed,
                    push_index=k,
                    start_step=start,
                    end_step=end,
                    direction_x=float(directions[k][0]),
                    direction_y=float(directions[k][1]),
                    force_n=force_n,
                    recovered=recovered,
                    recovery_step=recovery_step,
                )
            )
        episode_records.append(
            EpisodeRecord(
                controller_id=controller_id,
                episode=index + 1,
                panel_seed=panel_seed,
                length=steps,
                full_horizon=full_horizon,
                n_pushes=len(judged),
                n_recovered=sum(recovered_flags),
                success=episode_recovery_success(full_horizon, recovered_flags),
                reward=total_reward,
            )
        )
    return RecoveryPanelEvidence(
        controller_id=controller_id,
        episodes=tuple(episode_records),
        shoves=tuple(shove_records),
        safe_set=active_safe_set,
    )


def paired_success_differences(
    policy: RecoveryPanelEvidence,
    null: RecoveryPanelEvidence,
) -> tuple[float, ...]:
    """Per-seed policy-minus-null success differences, aligned by panel seed.

    Fail-closed: a seed present on one side and missing on the other is a
    pairing defect, not a skippable row.
    """
    policy_by_seed = policy.successes_by_seed()
    null_by_seed = null.successes_by_seed()
    if set(policy_by_seed) != set(null_by_seed):
        raise ValueError(
            "paired panels do not share their seed set: "
            f"policy-only {sorted(set(policy_by_seed) - set(null_by_seed))}, "
            f"null-only {sorted(set(null_by_seed) - set(policy_by_seed))}"
        )
    return tuple(
        float(policy_by_seed[panel_seed]) - float(null_by_seed[panel_seed]) for panel_seed in sorted(policy_by_seed)
    )


def write_recovery_evidence(stage_dir: "str | Path", evidence: RecoveryPanelEvidence) -> dict[str, Path]:
    """Write the per-episode and per-shove CSVs for one controller."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, rows, fields in (
        ("episodes", evidence.episodes, EpisodeRecord),
        ("shoves", evidence.shoves, ShoveRecord),
    ):
        path = stage_dir / f"recovery_{name}_{evidence.controller_id}.csv"
        fieldnames = list(fields.__dataclass_fields__)
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        paths[name] = path
    return paths
