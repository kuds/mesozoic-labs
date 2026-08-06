"""Score a saved checkpoint against its stage's ``stance_quality/v1`` gate.

Answers one question: **would this policy advance?** It rolls the checkpoint
on the stage's own configured environment, reduces the episodes with the same
:mod:`~environments.shared.curriculum.stance_gate` code the training loop runs,
and prints the verdict criterion by criterion.

Why this exists as a script rather than a notebook cell: the gate's verdict is
the thing runs are judged on, and a verdict computed by hand-rolled
reimplementation is not the same verdict.  Everything here funnels through
``summarize_stance_panel`` and ``evaluate_stance_gate``, so what this prints is
what the curriculum would decide.

It also covers the case the training loop cannot: **an already-finished run**.
Run ``20260801_203206`` completed stage 1 under the old ``reward_and_length/v1``
gate and advanced on reward 2313.2 and length 1000 -- both of which a
zero-action statue clears.  Whether it would have passed the stance gate is a
question only a rollout can answer, and the checkpoint is a 4 MB binary that
lives next to the run, not in the repository.

Usage::

    python environments/shared/scripts/stance_gate_report.py trex \\
        --model  path/to/robust_best_model.zip \\
        --vecnorm path/to/robust_best_model_vecnorm.pkl

    # the do-nothing reference, for the same panel and seeds
    python environments/shared/scripts/stance_gate_report.py trex --zero-action

    # probe: does the pose this policy holds need feedback, or only a constant?
    python environments/shared/scripts/stance_gate_report.py trex \\
        --model path/to/robust_best_model.zip \\
        --vecnorm path/to/robust_best_model_vecnorm.pkl \\
        --hold-constant

Thresholds come from the stage TOML, so the report re-anchors automatically
when the gate is retuned.  ``--episodes`` defaults to the stage's own
``min_eval_episodes``, because that is the panel size the bound's power is
specified at; overriding it downward makes the bound weaker than the gate
claims, and the report says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.curriculum.stance_gate import (  # noqa: E402
    STANCE_GATE_KIND,
    StanceGateThresholds,
    evaluate_stance_gate,
    stance_panel_from_episode_duties,
)
from environments.shared.species_registry import SPECIES_FACTORIES  # noqa: E402
from environments.shared.stance_diagnostics import derive_stance_info  # noqa: E402


class StanceGateReportError(RuntimeError):
    """The report could not be produced for a reason the caller should see.

    Deliberately a plain ``Exception`` subclass rather than ``SystemExit``,
    which is what these paths used to raise.  ``SystemExit`` derives from
    ``BaseException``, so it sailed straight through the ``except Exception``
    guard in ``reporting.stage_artifacts._write_stance_gate_report`` that
    exists precisely so a diagnostic cannot sink a finished training run --
    a truncated VecNormalize ``.pkl`` aborted artifact generation before the
    graphs and videos were written.  ``main`` converts this to ``SystemExit``
    for CLI use, which is where that behaviour belongs.
    """


def _load_policy(
    model_path: str,
    vecnorm_path: str | None,
    env_factory: Any,
    *,
    plant_identity: Any = None,
    allow_legacy_plant: bool = False,
):
    """Return ``(predict_fn, describe)`` for a saved SB3 checkpoint.

    Observation normalisation is applied from the saved ``VecNormalize``
    statistics rather than re-estimated: a policy evaluated on unnormalised
    observations is a *different policy*, and would score arbitrarily badly
    for reasons that have nothing to do with stance.  That is why a missing
    or unreadable file is fatal here instead of a warning.

    Loaded through ``VecNormalize.load`` with a throwaway ``DummyVecEnv``,
    matching the sibling report scripts -- the loader needs a live env to
    rebuild the wrapper, and hand-unpickling reconstructs a partial object
    whose ``__setstate__`` expectations drift with the SB3 version.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(model_path, device="cpu")
    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(model, plant_identity, artifact=model_path, allow_legacy=allow_legacy_plant)

    if vecnorm_path is None:
        return (lambda obs: model.predict(obs, deterministic=True)[0]), "no obs normalisation"

    try:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([env_factory]))
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        # A truncated or text-mode-copied .pkl is the usual cause, and the raw
        # UnpicklingError/KeyError gives no hint that the file rather than the
        # code is at fault.
        raise StanceGateReportError(
            f"cannot read VecNormalize statistics from {vecnorm_path}: "
            f"{type(exc).__name__}: {exc}. Re-copy the file in binary mode. "
            "Running without --vecnorm would evaluate the policy on unnormalised "
            "observations — a different policy — so this is fatal, not a warning."
        ) from exc

    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(normalizer, plant_identity, artifact=vecnorm_path, allow_legacy=allow_legacy_plant)

    obs_rms = normalizer.obs_rms
    if isinstance(obs_rms, dict):
        # SB3 keeps a per-key RunningMeanStd for Dict observation spaces. Every
        # species here exposes a flat Box, so a dict means the checkpoint was
        # trained against a different observation contract than the stage
        # config builds — normalising with the wrong statistics would silently
        # score a different policy.
        raise StanceGateReportError(
            f"{vecnorm_path} holds per-key statistics for a Dict observation space "
            f"(keys: {sorted(obs_rms)}), but this stage builds a flat Box observation."
        )
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    epsilon = float(normalizer.epsilon)
    clip = float(normalizer.clip_obs)

    def predict(obs: np.ndarray) -> np.ndarray:
        normalized = np.clip((obs - mean) / np.sqrt(var + epsilon), -clip, clip).astype(np.float32)
        # np.asarray, not a bare return: SB3's predict() is untyped when
        # stable-baselines3 is absent (as it is in the lint job), so the value
        # is Any there and typed here. Coercing makes the annotation true in
        # both environments instead of only the one that happens to run mypy.
        return np.asarray(model.predict(normalized, deterministic=True)[0])

    return predict, f"VecNormalize stats applied (clip {clip:g})"


def run_panel(
    species: str,
    *,
    predict: Any,
    episodes: int,
    seed: int,
    settle_steps: int,
    horizon: int,
    env_kwargs: dict[str, Any],
    plant_identity: Any = None,
    control_dt: float = 0.01,
) -> dict[str, Any]:
    """Roll ``episodes`` deterministic episodes and reduce them for the gate.

    The rollout environment is validated against *plant_identity* when one is
    supplied, and always closed -- a MuJoCo env holds native handles, and the
    artifact path builds one of these per stage.

    Returns the panel plus the per-episode evidence it was reduced from, so a
    caller can serialize the measurements rather than only their summary.
    """
    env_class = SPECIES_FACTORIES[species]().env_class
    env = env_class(**env_kwargs)
    if plant_identity is not None:
        from environments.shared.plant_contract import validate_environment_plant

        try:
            validate_environment_plant(env, plant_identity, artifact=f"{species} stance gate report environment")
        except BaseException:
            env.close()
            raise

    lengths: list[float] = []
    rewards: list[float] = []
    duties: list[float] = []
    bilateral_duties: list[float] = []
    single_duties: list[float] = []
    terminations: dict[str, int] = {}
    # Per-term totals, so a failing panel can be read for *why*. A policy that
    # keeps paying the airborne penalty is being funded by some other term;
    # which one is not guessable from the aggregate score.
    components: dict[str, float] = {}
    # Per-actuator action statistics. Which actuators carry a static offset
    # decides whether a leg-pose weight can reach it at all: `leg_home_pose`
    # covers only the leg joints, so a displacement living in the tail or neck
    # is invisible to it and unfixable by it (issue #489).
    action_stats = _new_action_stats()
    joint_names: list[str] = []

    try:
        joint_names = _actuator_joint_names(env)
        _roll_episodes(
            env,
            predict=predict,
            episodes=episodes,
            seed=seed,
            settle_steps=settle_steps,
            lengths=lengths,
            rewards=rewards,
            duties=duties,
            bilateral_duties=bilateral_duties,
            single_duties=single_duties,
            terminations=terminations,
            components=components,
            action_stats=action_stats,
        )
    finally:
        env.close()

    panel = stance_panel_from_episode_duties(
        episode_lengths=lengths,
        episode_duties=duties,
        episode_rewards=rewards,
        horizon=horizon,
    )

    def _full_horizon_mean(values: list[float]) -> float:
        """Mean over the same episodes the gate measures duty on.

        The ungated shares used to be averaged over every episode that
        measured anything, while the gated duty uses full-horizon episodes
        only. The three then did not sum to 1 whenever an episode failed
        early -- measured 0.90 with 5 failures in 40 -- which is exactly the
        identity the report cites as the reason for printing bilateral at
        all. Worse, the drift ran in the direction that hides the problem:
        the flailing episodes were folded into bilateral and single but
        excluded from unsupported.
        """
        kept = [value for value, length in zip(values, lengths) if length >= horizon and not np.isnan(value)]
        return float(np.mean(kept)) if kept else float("nan")

    return {
        "panel": panel,
        "lengths": np.asarray(lengths),
        "rewards": np.asarray(rewards),
        "terminations": terminations,
        "components": {key: value / episodes for key, value in components.items()},
        "bilateral_duty": _full_horizon_mean(bilateral_duties),
        "single_duty": _full_horizon_mean(single_duties),
        "action": _reduce_action_stats(action_stats, joint_names, control_dt),
        # The measurements the panel was reduced from. Kept rather than
        # discarded so the report can serialize the evidence, not only its
        # summary: an aggregate cannot be re-checked, and duty is the one
        # column the publication evidence CSV does not carry.
        "episodes": [
            {
                "episode": index,
                "seed": seed + index,
                "length": length,
                "reward": reward,
                "reached_horizon": length >= horizon,
                "unsupported_duty": duty,
                "bilateral_support_duty": bilateral,
                "single_support_duty": single,
            }
            for index, (length, reward, duty, bilateral, single) in enumerate(
                zip(lengths, rewards, duties, bilateral_duties, single_duties)
            )
        ],
    }


def _new_action_stats() -> dict[str, Any]:
    return {
        "sum": None,
        "sq_sum": None,
        "count": 0,
        "delta": 0.0,
        "jerk": 0.0,
        # Counted separately: a difference needs two samples and a second
        # difference three, and episode boundaries reset the history, so
        # dividing all three by the sample count biases the ratio -- and the
        # ratio is what becomes a frequency.
        "delta_count": 0,
        "jerk_count": 0,
        "prev": None,
        "prev_prev": None,
    }


def _accumulate_action(stats: dict[str, Any], action: np.ndarray) -> None:
    """Accumulate one post-settle action into the per-actuator statistics."""
    if stats["sum"] is None or stats["sum"].shape != action.shape:
        stats["sum"] = np.zeros_like(action)
        stats["sq_sum"] = np.zeros_like(action)
    stats["sum"] += action
    stats["sq_sum"] += action * action
    stats["count"] += 1
    prev, prev_prev = stats["prev"], stats["prev_prev"]
    if prev is not None and prev.shape == action.shape:
        # Summed over actuators, matching the `Sum` in the reward terms so
        # these invert straight back through `reward_action_smoothness` and
        # `reward_action_jerk`.
        stats["delta"] += float(np.sum((action - prev) ** 2))
        stats["delta_count"] += 1
        if prev_prev is not None and prev_prev.shape == action.shape:
            stats["jerk"] += float(np.sum((action - 2.0 * prev + prev_prev) ** 2))
            stats["jerk_count"] += 1
    stats["prev_prev"] = prev
    stats["prev"] = action


def _actuator_joint_names(env: Any) -> list[str]:
    """Names of the joint each actuator drives, or ``[]`` if unavailable.

    Unlike the training callback -- which reaches the environment only through
    VecEnv wrappers and so cannot resolve these without a best-effort lookup
    that silently returns nothing -- this script holds the env directly, so the
    mapping is exact and the per-actuator report can be read by joint rather
    than by index.
    """
    try:
        import mujoco

        model = env.unwrapped.model
        names = []
        for index in range(model.nu):
            joint_id = int(model.actuator_trnid[index, 0])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            names.append(name or f"actuator_{index}")
        return names
    except Exception:  # noqa: BLE001 - names are a convenience, not the measurement
        return []


def _reduce_action_stats(stats: dict[str, Any], names: list[str], control_dt: float) -> dict[str, Any]:
    """Reduce the accumulators into the report's ``action`` block.

    The DC/AC split is the point. ``mean(a)`` per actuator is the static pose
    the policy commands -- under ``home-keyframe-residual/v1`` the distance
    from the home keyframe, since ``action = 0`` *is* that keyframe -- and the
    spread about it is what the policy does on top. They have different causes
    and different fixes, and a single pooled number cannot tell them apart
    (issue #489).
    """
    count = stats["count"]
    if not count or stats["sum"] is None:
        return {}
    mean = stats["sum"] / count
    var = np.maximum(stats["sq_sum"] / count - mean * mean, 0.0)
    ac = np.sqrt(var)
    delta = stats["delta"] / stats["delta_count"] if stats["delta_count"] else 0.0
    jerk = stats["jerk"] / stats["jerk_count"] if stats["jerk_count"] else 0.0
    # For a narrowband action signal `jerk/delta = (2 sin(pi f dt))^2`, blind
    # to any constant offset. At the white-noise limit the ratio saturates,
    # which means broadband rather than a tone at that frequency.
    freq = float("nan")
    if delta > 0.0:
        freq = math.asin(min(1.0, math.sqrt(jerk / delta) / 2.0)) / (math.pi * control_dt)
    return {
        "dc_rms": float(np.sqrt(np.mean(mean * mean))),
        "ac_rms": float(np.sqrt(np.mean(var))),
        "delta_per_step": float(delta),
        "jerk_per_step": float(jerk),
        "effective_freq_hz": freq,
        "per_actuator": [
            {
                "index": index,
                "joint": names[index] if index < len(names) else f"actuator_{index}",
                "dc": float(mean[index]),
                "ac_rms": float(ac[index]),
            }
            for index in range(mean.size)
        ],
    }


def _roll_episodes(
    env: Any,
    *,
    predict: Any,
    episodes: int,
    seed: int,
    settle_steps: int,
    lengths: list[float],
    rewards: list[float],
    duties: list[float],
    bilateral_duties: list[float],
    single_duties: list[float],
    terminations: dict[str, int],
    components: dict[str, float],
    action_stats: dict[str, Any],
) -> None:
    """Roll the panel, appending per-episode measurements to the given lists.

    Split out only so ``run_panel`` can wrap it in the ``finally`` that closes
    the environment; the accumulation is unchanged.
    """
    for index in range(episodes):
        obs, _ = env.reset(seed=seed + index)
        # A stateful predict (the low-pass probe) must not carry one episode's
        # tail into the next; plain policies have no reset and are untouched.
        reset_predict = getattr(predict, "reset", None)
        if callable(reset_predict):
            reset_predict()
        total = 0.0
        steps = 0
        unsupported = 0
        bilateral = 0
        single = 0
        measured = 0
        while True:
            action = np.asarray(predict(obs), dtype=np.float64).ravel()
            # Post-settle only, the same window the duty uses, so the reset
            # transient does not read as tremor.
            if steps >= settle_steps:
                _accumulate_action(action_stats, action)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            for key, value in info.items():
                if key.startswith("reward_") and key != "reward_total":
                    components[key] = components.get(key, 0.0) + float(value)
            stance = derive_stance_info(info)
            if stance and steps >= settle_steps:
                measured += 1
                unsupported += int(stance["unsupported_duty"])
                # The gate bounds unsupported duty only. Reporting the full
                # three-way split shows where recovered airborne time actually
                # went: a policy that trades flight for SINGLE support drives
                # the gated number down without ever planting both feet.
                bilateral += int(stance["bilateral_support_duty"])
                single += int(stance["single_support_duty"])
            steps += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "terminated" if terminated else "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                break
        # An episode boundary is not an action difference.
        action_stats["prev"] = None
        action_stats["prev_prev"] = None
        lengths.append(float(steps))
        rewards.append(total)
        # All three shares stay positionally aligned with `lengths`, so the
        # full-horizon filter below applies to them identically.
        duties.append(unsupported / measured if measured else float("nan"))
        bilateral_duties.append(bilateral / measured if measured else float("nan"))
        single_duties.append(single / measured if measured else float("nan"))


#: Bumped when the JSON report's field meanings change.
REPORT_SCHEMA = "mesozoic.stance-gate-report/v1"


def thresholds_from_curriculum(curriculum: dict[str, Any]) -> StanceGateThresholds:
    """Read the stance thresholds a stage declares.

    Ceilings default to ``+inf`` and floors to ``0.0`` so a stage that does not
    gate on stance still produces a readable report rather than a spuriously
    strict one -- ``0.0`` would be the tightest possible ceiling, not "absent".
    """
    return StanceGateThresholds(
        min_full_horizon_fraction=float(curriculum.get("min_full_horizon_fraction", 0.0)),
        max_unsupported_duty=float(curriculum.get("max_unsupported_duty", float("inf"))),
        max_unsupported_duty_ucb=float(curriculum.get("max_unsupported_duty_ucb", float("inf"))),
        settle_steps=int(curriculum.get("settle_steps", 0)),
        min_eval_episodes=int(curriculum.get("min_eval_episodes", 40)),
        min_avg_reward=float(curriculum.get("min_avg_reward", -float("inf"))),
        required_consecutive=int(curriculum.get("required_consecutive", 3)),
    )


def _low_pass_predict(predict: Any, cutoff_hz: float, control_dt: float) -> Any:
    """Wrap *predict* so its action is low-passed before reaching the plant.

    A probe, not a feature: it answers whether a policy's high-frequency
    action content is load-bearing or waste. Run ``20260805_011234`` passed
    the stance gate carrying a tremor of rms 0.277 at an effective 22.6 Hz,
    and whether that tremor is closed-loop stabilisation or a free-running
    buzz decides whether the fix belongs on the pose or on the action path
    (issue #489). Open-loop experiments on the statue bound what a tremor
    *can* do; only filtering the real policy answers it for that policy.

    First-order IIR, ``y += alpha * (x - y)`` with
    ``alpha = dt / (RC + dt)`` and ``RC = 1 / (2*pi*fc)`` -- the discrete
    equivalent of an RC filter, chosen because it is the one a plant-side
    implementation would most plausibly use. Filtering happens between the
    policy and the environment, so the policy still sees unfiltered
    observations and is genuinely being asked to stand without its own
    high-frequency component.

    The state is per-call-sequence and reset by the caller between episodes
    via :func:`reset`, so one episode's tail cannot leak into the next.
    """
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = control_dt / (rc + control_dt)
    state: dict[str, Any] = {"y": None}

    def filtered(obs: np.ndarray) -> np.ndarray:
        action = np.asarray(predict(obs), dtype=np.float64)
        previous = state["y"]
        # Seeded with the first action rather than zeros: starting from zero
        # would inject a step transient that is an artefact of the probe, not
        # a property of the policy.
        state["y"] = (
            action if previous is None or previous.shape != action.shape else previous + alpha * (action - previous)
        )
        return np.asarray(state["y"], dtype=np.float32)

    filtered.reset = lambda: state.update(y=None)  # type: ignore[attr-defined]
    return filtered


@dataclass(frozen=True)
class ConstantHold:
    """A constant command to substitute for the policy, and when to substitute it.

    The probe behind ``docs/investigations/TREX_STAGE1_BOUNCE_2026_08.md`` §5:
    the passing T-Rex stage-1 policy stands with a 0.366 rms tremor at ~22.7 Hz,
    and the low-pass probe shows that tremor is load-bearing. That leaves two
    readings, which the filter probe cannot separate:

    * the *pose* the policy holds needs continuous active stabilisation, or
    * the pose is holdable by a constant and the tremor is waste the action
      penalties failed to suppress.

    Replacing the policy with the constant it commands **on average** decides
    it. If the animal keeps standing, the smooth solution existed and the
    optimiser did not find it. If it falls, the pose genuinely requires
    feedback, and the question moves to why the policy chose a pose it then has
    to fight -- which points at the 13 of 21 actuators no stage-1 term
    constrains.

    Attributes:
        actions: The constant command, one entry per actuator.
        handoff_steps: Steps of real policy before the substitution. ``0``
            commands the constant from reset; a value at or above the horizon
            never substitutes at all, which is how the probe rolls its own
            unmodified control panel without producing a report that could be
            mistaken for a gate verdict.
        ramp_steps: Steps to blend from the last commanded action into
            *actions*. ``0`` switches in one step. The ramp exists to answer
            the obvious objection to a hard switch -- that the animal fell from
            the step transient rather than from losing feedback -- with a
            second measurement instead of an argument.
        label: Short name for this variant in the probe artifact.
        source: Where *actions* came from, recorded so a reader can tell a
            measured hold from a hand-supplied one.
    """

    actions: tuple[float, ...]
    handoff_steps: int
    ramp_steps: int = 0
    label: str = "hold"
    source: str = "measured"
    #: Set for the control variant that never substitutes, so the renderer can
    #: say "this row is the unmodified policy" rather than implying a hold.
    holds: bool = field(default=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "holds": self.holds,
            "handoff_steps": self.handoff_steps,
            "ramp_steps": self.ramp_steps,
            "actions": list(self.actions),
        }


def _hold_constant_predict(predict: Any, hold: ConstantHold) -> Any:
    """Wrap *predict* so the plant receives a constant command after handoff.

    Feedback is cut, not attenuated: past ``handoff_steps`` the returned action
    no longer depends on the observation at all. That is the point -- a
    low-pass still lets the policy respond slowly, so it cannot distinguish
    "needs bandwidth" from "needs feedback", and this can.

    The step counter is per episode and reset by the caller through
    :func:`reset`, exactly like the low-pass probe: without it the second
    episode would start already past handoff and the panel would measure
    something different from the first.
    """
    constant = np.asarray(hold.actions, dtype=np.float64)
    state: dict[str, Any] = {"step": 0, "last": None}

    def held(obs: np.ndarray) -> np.ndarray:
        step = state["step"]
        state["step"] = step + 1
        if step < hold.handoff_steps:
            action = np.asarray(predict(obs), dtype=np.float64).ravel()
            state["last"] = action
            return np.asarray(action, dtype=np.float32)
        elapsed = step - hold.handoff_steps
        if hold.ramp_steps > 0 and elapsed < hold.ramp_steps:
            # Blend from whatever was last commanded. At handoff that is the
            # policy's own last action; from reset there is none, and zero is
            # the right start because `action = 0` IS the home keyframe the
            # episode begins at -- so the ramp starts from the actual pose
            # rather than jumping to one.
            previous = state["last"]
            if previous is None or previous.shape != constant.shape:
                previous = np.zeros_like(constant)
            weight = (elapsed + 1) / hold.ramp_steps
            return np.asarray((1.0 - weight) * previous + weight * constant, dtype=np.float32)
        return np.asarray(constant, dtype=np.float32)

    held.reset = lambda: state.update(step=0, last=None)  # type: ignore[attr-defined]
    return held


def constant_hold_actions(report: dict[str, Any]) -> tuple[float, ...]:
    """The per-actuator DC a report measured, as a constant command.

    DC is the time-mean of the commanded action over the post-settle window --
    under ``home-keyframe-residual/v1`` the static pose the policy holds, since
    ``action = 0`` is the home keyframe. Holding exactly that is what makes the
    probe a fair test of the pose rather than of some nearby pose: the animal
    is asked to stand where it was already standing, with the variation about
    it removed and nothing else changed.
    """
    per_actuator = (report.get("action") or {}).get("per_actuator") or []
    if not per_actuator:
        raise StanceGateReportError(
            "the report carries no per-actuator action statistics, so there is no "
            "measured pose to hold. It was produced before that block existed, or "
            "from a panel in which no episode survived the settling window."
        )
    return tuple(float(entry["dc"]) for entry in sorted(per_actuator, key=lambda e: e["index"]))


def build_stance_gate_report(
    species: str,
    stage: int,
    *,
    stage_config: dict[str, Any],
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    zero_action: bool = False,
    episodes: int | None = None,
    seed: int = 3042,
    allow_legacy_plant: bool = False,
    filter_actions_hz: float | None = None,
    hold_constant: ConstantHold | None = None,
) -> dict[str, Any]:
    """Roll a policy and return its gate verdict as a serializable dict.

    Importable so the training pipeline can emit the same report it would
    produce by hand, rather than a second implementation that could disagree.

    The checkpoint and the rollout environment are validated against the
    species' current plant identity: a verdict measured on a different plant
    than the one in the tree is not a verdict about this stage, and every
    other artifact path in the repository already refuses that pairing. Pass
    *allow_legacy_plant* to score a checkpoint that predates the contract --
    which is a real use for this script, since it exists partly to judge
    already-finished runs -- and the report records that it was done.

    ``episodes`` of ``None`` means the stage's own ``min_eval_episodes``,
    the panel size the bound's power is specified at. ``0`` is rejected
    rather than silently treated as absent.
    """
    if episodes is not None and episodes < 1:
        raise ValueError(f"episodes must be at least 1 if given, got {episodes}")
    if filter_actions_hz is not None and filter_actions_hz <= 0:
        raise ValueError(f"filter_actions_hz must be positive if given, got {filter_actions_hz}")
    curriculum = stage_config["curriculum_kwargs"]
    env_kwargs = dict(stage_config["env_kwargs"])
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    control_dt = float(env_kwargs.get("timestep", 0.002)) * int(env_kwargs.get("frame_skip", 5))
    thresholds = thresholds_from_curriculum(curriculum)
    panel_episodes = thresholds.min_eval_episodes if episodes is None else episodes

    from environments.shared.plant_contract import current_plant_identity

    plant_identity = current_plant_identity(species)

    env_class = SPECIES_FACTORIES[species]().env_class
    if zero_action:
        probe = env_class(**env_kwargs)
        zero = np.zeros(probe.action_space.shape[0], dtype=np.float32)
        probe.close()

        def predict(_obs: np.ndarray) -> np.ndarray:
            return zero

        description = "zero action (do-nothing reference)"
    else:
        if model_path is None:
            raise ValueError("model_path is required unless zero_action is set")
        predict, note = _load_policy(
            model_path,
            vecnorm_path,
            lambda: env_class(**env_kwargs),
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
        description = f"{Path(model_path).name} — {note}"

    if filter_actions_hz is not None:
        predict = _low_pass_predict(predict, filter_actions_hz, control_dt)
        description += f" — actions low-passed at {filter_actions_hz:g} Hz"

    if hold_constant is not None:
        expected = int(plant_identity.action_dim)
        if len(hold_constant.actions) != expected:
            # Checked before the rollout rather than at the first step: numpy
            # broadcasts a mismatched length silently, which would score a
            # policy nobody asked for and report it as this one.
            raise ValueError(
                f"hold_constant carries {len(hold_constant.actions)} actions but "
                f"{species} has action_dim {expected}"
            )
        predict = _hold_constant_predict(predict, hold_constant)
        description += (
            f" — {hold_constant.label}"
            if not hold_constant.holds
            else f" — commanded action held constant from step {hold_constant.handoff_steps}"
        )

    result = run_panel(
        species,
        predict=predict,
        episodes=panel_episodes,
        seed=seed,
        settle_steps=thresholds.settle_steps,
        horizon=horizon,
        env_kwargs=env_kwargs,
        plant_identity=plant_identity,
        control_dt=control_dt,
    )
    panel = result["panel"]
    passed, failures = evaluate_stance_gate(panel, thresholds)

    return {
        "schema": REPORT_SCHEMA,
        "species": species,
        "stage": stage,
        "gate_kind": curriculum.get("gate_kind"),
        "policy": description,
        "episodes": panel_episodes,
        "seed": seed,
        "settle_steps": thresholds.settle_steps,
        "horizon": horizon,
        "passed": passed,
        "failures": failures,
        "thresholds": {
            "min_full_horizon_fraction": thresholds.min_full_horizon_fraction,
            "max_unsupported_duty": thresholds.max_unsupported_duty,
            "max_unsupported_duty_ucb": thresholds.max_unsupported_duty_ucb,
            "min_avg_reward": thresholds.min_avg_reward,
            "min_eval_episodes": thresholds.min_eval_episodes,
        },
        "metrics": {
            "reward_mean": float(result["rewards"].mean()),
            "reward_std": float(result["rewards"].std()),
            "episode_length_mean": float(result["lengths"].mean()),
            "full_horizon_fraction": panel.full_horizon_fraction,
            "mean_unsupported_duty": panel.mean_unsupported_duty,
            "unsupported_duty_ucb": panel.unsupported_duty_ucb,
            "n_duty_episodes": panel.n_duty_episodes,
            # Not gated. Reported because the three shares sum to 1, so a
            # falling unsupported duty does not by itself mean the feet are
            # being planted -- the time can land in single support instead.
            "bilateral_support_duty": result["bilateral_duty"],
            "single_support_duty": result["single_duty"],
        },
        # Whether the CHECKPOINT's plant identity was checked against the
        # tree. `None` for --zero-action, which has no checkpoint to check.
        # The rollout ENVIRONMENT is always validated regardless, so this
        # field is deliberately narrow rather than a blanket "plant_validated"
        # that would claim something false in the zero-action case.
        "checkpoint_plant_validated": None if zero_action else not allow_legacy_plant,
        # `None` for an ordinary run. Recorded because a filtered rollout is
        # NOT a verdict about the policy: it is a probe of a modified one, and
        # a PASS produced this way must never be mistakable for a real PASS in
        # a stored report. `render_stance_gate_report` prints it for the same
        # reason.
        "filter_actions_hz": filter_actions_hz,
        # `None` for an ordinary run. Same contract as `filter_actions_hz`: a
        # held rollout scored a policy whose feedback was cut, so it is a probe
        # and never a verdict, and `_PROBE_MARKERS` keys the filenames off this.
        "hold_constant": None if hold_constant is None else hold_constant.as_dict(),
        # Where the commanded pose sits and what the policy does around it,
        # per actuator. `{}` when no post-settle step was measured.
        "action": result.get("action", {}),
        "terminations": result["terminations"],
        "reward_components": result["components"],
        # The per-episode measurements the panel was reduced from. This is
        # the duty evidence `result_bundle.evidence` refuses stance-gated
        # bundles for lacking -- note it does NOT by itself lift that
        # refusal, which reads `evaluation_selected.csv`; teaching that
        # function to consume this is the remaining step.
        "episode_evidence": result["episodes"],
    }


#: Report keys that mark a rollout as a PROBE rather than a verdict, mapped to
#: the filename stem that probe writes under. Each names a modification applied
#: between the policy and the plant, so a report carrying any of them scored
#: something other than the checkpoint -- and must never land on the real
#: report's filenames or supply the evidence a bundle is certified from.
#:
#: A registry rather than a chain of ``if``s because the failure mode is
#: forgetting to extend the check: the filter probe's guard was written as a
#: single ``is not None`` test, and a second probe added beside it would have
#: silently inherited the certification filenames.
_PROBE_MARKERS: dict[str, str] = {
    "filter_actions_hz": "stance_gate_probe_filtered",
    "hold_constant": "stance_gate_probe_constant",
}


def probe_stem(report: dict[str, Any]) -> str | None:
    """The filename stem a probe report writes under, or ``None`` if it is a verdict."""
    for key, stem in _PROBE_MARKERS.items():
        if report.get(key) is not None:
            return stem
    return None


def _probe_banner(report: dict[str, Any]) -> list[str]:
    """The warning a probe report must carry above its verdict line."""
    if report.get("filter_actions_hz") is not None:
        what = f"actions low-passed at {report['filter_actions_hz']:g} Hz before reaching the plant."
    elif report.get("hold_constant") is not None:
        hold = report["hold_constant"]
        what = (
            "the policy ran the whole episode; this row is the unmodified control."
            if not hold.get("holds", True)
            else (
                f"the commanded action was frozen to a constant from step {hold['handoff_steps']}"
                + (f", ramped in over {hold['ramp_steps']} steps." if hold.get("ramp_steps") else ".")
            )
        )
    else:
        return []
    return [
        f"PROBE: {what}",
        "This scores a MODIFIED policy. The verdict below is not a gate result for this checkpoint.",
        "",
    ]


def render_stance_gate_report(report: dict[str, Any]) -> str:
    """Render a report dict as the human-readable text form."""
    thresholds = report["thresholds"]
    metrics = report["metrics"]
    lines: list[str] = []
    if report["gate_kind"] != STANCE_GATE_KIND:
        lines.append(
            f"NOTE: {report['species']} stage {report['stage']} declares gate_kind "
            f"{report['gate_kind']!r}, not {STANCE_GATE_KIND!r}. The stance criteria "
            "below are reported but are not what this stage advances on."
        )
        lines.append("")
    # Before the verdict, not after: a probe rollout scores a MODIFIED policy,
    # so its PASS/FAIL is not a statement about the checkpoint and a reader must
    # not reach the verdict line without knowing that.
    lines += _probe_banner(report)
    measured = report["horizon"] - report["settle_steps"]
    lines += [
        f"policy              {report['policy']}",
        f"stage               {report['species']} stage {report['stage']} ({report['gate_kind']})",
        f"panel               {report['episodes']} episodes, "
        f"seeds {report['seed']}-{report['seed'] + report['episodes'] - 1}",
        f"settle_steps        {report['settle_steps']} (duty measured over the remaining {measured})",
        "",
        f"reward                 {metrics['reward_mean']:9.1f} +/- {metrics['reward_std']:.1f}",
        f"episode length         {metrics['episode_length_mean']:9.1f}",
        f"full_horizon_fraction  {metrics['full_horizon_fraction']:9.4f}   "
        f"(>= {thresholds['min_full_horizon_fraction']:.4f})",
        f"mean_unsupported_duty  {metrics['mean_unsupported_duty']:9.4f}   "
        f"(<= {thresholds['max_unsupported_duty']:.4f})",
        f"unsupported_duty_ucb   {metrics['unsupported_duty_ucb']:9.4f}   "
        f"(<= {thresholds['max_unsupported_duty_ucb']:.4f})",
        f"duty episodes          {metrics['n_duty_episodes']:9d}   (full-horizon episodes only)",
        f"  bilateral support    {metrics['bilateral_support_duty']:9.4f}   (statue 0.998, not gated)",
        f"  single support       {metrics['single_support_duty']:9.4f}   (statue 0.002, not gated)",
        f"terminations           {report['terminations']}",
    ]
    components = report["reward_components"]
    if components:
        lines += ["", "reward per episode, by term (largest magnitude first):"]
        for key in sorted(components, key=lambda k: -abs(components[k])):
            if abs(components[key]) > 0.05:
                lines.append(f"  {key:28s} {components[key]:10.2f}")
    action = report.get("action") or {}
    if action:
        lines += [
            "",
            "commanded action, post-settle "
            f"(DC {action['dc_rms']:.3f} rms, AC {action['ac_rms']:.3f} rms, "
            f"~{action['effective_freq_hz']:.1f} Hz):",
            "  DC is the distance from the home keyframe (action = 0 IS that pose);",
            "  AC is what the policy does around it. Different causes, different fixes.",
        ]
        per_actuator = action.get("per_actuator") or []
        # Largest offset first: the question this answers is WHICH joints are
        # displaced, because a leg-pose weight can only reach the leg ones.
        for entry in sorted(per_actuator, key=lambda e: -abs(e["dc"]))[:12]:
            lines.append(f"  {entry['joint']:28s} DC {entry['dc']:+7.3f}   AC {entry['ac_rms']:6.3f}")
        if len(per_actuator) > 12:
            lines.append(f"  ... {len(per_actuator) - 12} more (full list in the JSON)")
    lines += ["", f"GATE: {'PASS' if report['passed'] else 'FAIL'}"]
    lines += [f"  - {failure}" for failure in report["failures"]]
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``null``, recursively.

    ``json.dumps`` writes bare ``NaN`` and ``Infinity`` tokens, which Python
    reads back but RFC 8259 does not permit -- ``jq``, ``JSON.parse`` and Go's
    ``encoding/json`` all reject them.  That bit exactly when it hurt most:
    the gate's "unmeasurable" sentinel is ``+inf`` and the ungated shares are
    ``NaN``, so it was the FAILING panel, the one worth inspecting, that
    produced the unparseable file -- while the docstring below promises this
    form exists so later tooling can read it without parsing prose.

    ``null`` rather than a string, so a consumer's numeric parse fails loudly
    instead of silently comparing ``"inf"``.  The human-readable text form
    still prints ``inf``, and ``failures`` says which criterion was
    unmeasurable.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_stance_gate_report(stage_dir: "str | Path", report: dict[str, Any]) -> dict[str, Path]:
    """Write the report beside the stage's other artifacts.

    Two forms on purpose: the text is what a human opens next to
    ``stage_summary.txt``, and the JSON is what later tooling can read without
    parsing prose.

    Three forms, and the third is load-bearing. ``stance_panel_selected.csv``
    is the per-episode duty evidence ``result_bundle.evidence`` needs to
    certify a stance-gated stage: it used to refuse such bundles outright
    because ``evaluation_selected.csv`` records reward and length but no duty,
    and certifying on the reward rail alone would pass the statue. The refusal
    was correct and it also made the milestone unreachable -- a run that
    genuinely cleared the stance gate still could not produce a publishable
    bundle. This file is what lifts it, and it is a CSV rather than the JSON's
    ``episode_evidence`` so the auditor recomputes the panel from the same
    kind of per-episode record it already uses for reward and length.

    The panel is a different evaluation from ``evaluation_selected.csv``: 40
    episodes at the panel seeds versus 30 at the publication seed. That is
    deliberate -- ``min_eval_episodes`` is the sample size the bound's power
    is specified at -- so it gets its own file rather than columns bolted
    onto a file with a different episode count.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # A filtered rollout scored a MODIFIED policy, so it gets its own filenames
    # and never the certification ones. Derived here rather than passed in so a
    # caller cannot land a probe on top of the real verdict by forgetting an
    # argument -- and `stance_panel_selected.csv` is skipped entirely below,
    # because that file is the per-episode evidence `result_bundle.evidence`
    # certifies a stance-gated stage from. Writing a filtered panel there would
    # certify a policy that was never run.
    stem = probe_stem(report)
    probe = stem is not None
    stem = stem or "stance_gate_report"
    text_path = directory / f"{stem}.txt"
    json_path = directory / f"{stem}.json"
    text_path.write_text(render_stance_gate_report(report) + "\n", encoding="utf-8")
    # allow_nan=False turns any non-finite value _json_safe missed into a
    # ValueError here rather than an unparseable artifact on Drive.
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written = {"stance_gate_report_txt": text_path, "stance_gate_report_json": json_path}
    if probe:
        return written
    panel_path = write_stance_panel_evidence(directory, report)
    if panel_path is not None:
        written["stance_panel_csv"] = panel_path
    return written


def write_action_filter_sweep(
    stage_dir: "str | Path", reports: list[dict[str, Any]], *, probe_episodes: int
) -> dict[str, Path]:
    """Write the action-filter probe sweep as one curve, not N verdicts.

    Each entry scored a MODIFIED policy, so none of them is a gate result and
    the artifact says so before it says anything else. What is worth reading is
    the trend: how long the filtered policy survives against cutoff. On T-Rex
    stage 1 today that is 96 steps at 5 Hz rising to 351 at 35 Hz, against a
    1000-step horizon -- the policy needs essentially its full command
    bandwidth to stand (issue #491), and a future policy that reaches the
    horizon at a low cutoff is one that could survive a real actuator.

    Deliberately does not write ``stance_panel_selected.csv``: that file is the
    per-episode evidence a stance-gated bundle is certified from, and none of
    these panels scored the policy that would be certified.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (
            {
                "filter_actions_hz": report["filter_actions_hz"],
                "episode_length_mean": report["metrics"]["episode_length_mean"],
                "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
                "reward_mean": report["metrics"]["reward_mean"],
                "mean_unsupported_duty": report["metrics"]["mean_unsupported_duty"],
                "terminations": report["terminations"],
            }
            for report in reports
        ),
        key=lambda row: row["filter_actions_hz"],
    )
    horizon = reports[0]["horizon"]
    # Each entry's `policy` carries its own cutoff suffix, so the first one
    # would label the whole sweep "low-passed at 5 Hz" -- wrong for every other
    # row. Strip it; the per-cutoff values are the table's first column.
    policy = str(reports[0]["policy"]).split(" — actions low-passed at ")[0]
    payload = {
        "schema": "mesozoic.action-filter-sweep/v1",
        "species": reports[0]["species"],
        "stage": reports[0]["stage"],
        "policy": policy,
        "horizon": horizon,
        "episodes_per_cutoff": probe_episodes,
        "note": (
            "Each row scored a MODIFIED policy (actions low-passed before reaching the "
            "plant). No row is a gate verdict. The metric to read is episode_length_mean "
            "against filter_actions_hz."
        ),
        "sweep": rows,
        "reports": reports,
    }
    lines = [
        "PROBE: actions low-passed before reaching the plant, at several cutoffs.",
        "Every row scores a MODIFIED policy. None of them is a gate verdict.",
        "",
        f"policy              {policy}",
        f"stage               {reports[0]['species']} stage {reports[0]['stage']}",
        f"panel               {probe_episodes} episodes per cutoff, horizon {horizon}",
        "",
        f"  {'cutoff':>9}{'ep length':>12}{'full-horiz':>12}{'reward':>10}   terminations",
    ]
    for row in rows:
        lines.append(
            f"  {row['filter_actions_hz']:>7.4g} Hz{row['episode_length_mean']:>12.1f}"
            f"{row['full_horizon_fraction']:>12.4f}{row['reward_mean']:>10.1f}   {row['terminations']}"
        )
    lines += [
        "",
        "Read the trend, not the rows: episode_length_mean against cutoff measures how",
        "much of its own high-frequency command content the policy needs in order to",
        "stand. Balance correction on this plant is ~1.1-1.4 Hz, which every cutoff here",
        "passes essentially untouched, so a short episode means the policy depends on",
        "content well above the frequency band the task itself requires.",
    ]
    text_path = directory / "stance_gate_probe_filtered.txt"
    json_path = directory / "stance_gate_probe_filtered.json"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"action_filter_sweep_txt": text_path, "action_filter_sweep_json": json_path}


def render_constant_hold_probe(
    reports: list[dict[str, Any]], *, probe_episodes: int
) -> tuple[str, dict[str, Any]]:
    """Reduce the constant-hold variants to their comparison table and payload.

    Split from the writer so the CLI can print the table without depositing
    files in whatever directory it was run from.

    The rows are only meaningful against each other, which is why they share a
    file. Two of them are controls that bracket the measurement:

    * the **unmodified policy**, which must reach the horizon or the probe is
      measuring a broken harness rather than a property of the pose, and
    * the **zero hold**, the statue -- a constant that is already known to
      stand all 1000 steps in 40 of 40 episodes.

    Between them sit the held variants. Read them as a single yes/no:

    * held rollouts reach the horizon -> the pose is holdable by a constant.
      The tremor is not stabilisation; the action penalties simply failed to
      find the smooth solution, and this is an optimisation problem.
    * held rollouts fall while both controls stand -> the pose the policy chose
      genuinely requires feedback. The question then moves to *why* it chose a
      pose it has to fight, which points at the actuators no stage-1 term
      constrains rather than at the action penalties.

    The ramped variant exists to close the obvious objection to the hard
    switch -- that a step transient, not the loss of feedback, is what knocked
    the animal over -- with a measurement instead of an argument. If the hard
    switch falls and the ramped one stands, the transient was the cause and
    neither row says anything about the pose.

    Deliberately does not write ``stance_panel_selected.csv``: no row here
    scored the policy that would be certified.
    """
    rows = [
        {
            "label": report["hold_constant"]["label"],
            "holds": report["hold_constant"]["holds"],
            "source": report["hold_constant"]["source"],
            "handoff_steps": report["hold_constant"]["handoff_steps"],
            "ramp_steps": report["hold_constant"]["ramp_steps"],
            "episode_length_mean": report["metrics"]["episode_length_mean"],
            "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
            "reward_mean": report["metrics"]["reward_mean"],
            "mean_unsupported_duty": report["metrics"]["mean_unsupported_duty"],
            "terminations": report["terminations"],
        }
        for report in reports
    ]
    horizon = reports[0]["horizon"]
    policy = str(reports[0]["policy"]).split(" — commanded action held constant")[0].split(" — ")[0]
    # The rows that hold the POLICY'S measured pose. Deliberately not "every
    # row that holds something": `hold_zero` holds a constant too, and it is a
    # control whose standing is already known and proves nothing about this
    # policy. Folding it in would make an all-falling result read as mixed.
    held = [row for row in rows if row["source"] == "measured"]
    hold_actions = next(
        (report["hold_constant"]["actions"] for report in reports if report["hold_constant"]["source"] == "measured"),
        [],
    )
    payload = {
        "schema": "mesozoic.constant-hold-probe/v1",
        "species": reports[0]["species"],
        "stage": reports[0]["stage"],
        "policy": policy,
        "horizon": horizon,
        "episodes_per_variant": probe_episodes,
        "hold_actions": hold_actions,
        "note": (
            "Each row scored a MODIFIED policy (its commanded action replaced by a "
            "constant partway through the episode). No row is a gate verdict. The "
            "question is whether the held rows reach the horizon: if they do, the "
            "pose is holdable without feedback and the tremor is waste; if they "
            "fall while both controls stand, the pose requires active stabilisation."
        ),
        "variants": rows,
        "reports": reports,
    }
    lines = [
        "PROBE: the policy's commanded action replaced by a constant partway through the episode.",
        "Every held row scores a MODIFIED policy. None of them is a gate verdict.",
        "",
        f"policy              {policy}",
        f"stage               {reports[0]['species']} stage {reports[0]['stage']}",
        f"panel               {probe_episodes} episodes per variant, horizon {horizon}",
        f"hold                the policy's own post-settle mean action, {len(hold_actions)} actuators",
        "",
        f"  {'variant':<26}{'handoff':>8}{'ramp':>6}{'ep length':>11}{'full-horiz':>12}{'reward':>10}   terminations",
    ]
    for row in rows:
        handoff = "never" if not row["holds"] else str(row["handoff_steps"])
        lines.append(
            f"  {row['label']:<26}{handoff:>8}{row['ramp_steps']:>6}"
            f"{row['episode_length_mean']:>11.1f}{row['full_horizon_fraction']:>12.4f}"
            f"{row['reward_mean']:>10.1f}   {row['terminations']}"
        )
    lines += ["", "How to read it:"]
    if held and all(row["full_horizon_fraction"] >= 0.95 for row in held):
        lines += [
            "  Every held variant reached the horizon. The pose the policy holds does NOT",
            "  need feedback to stand -- a constant command suffices. The tremor is therefore",
            "  not closed-loop stabilisation, and the gap to the statue is an optimisation",
            "  failure rather than a missing reward term.",
        ]
    elif held and all(row["full_horizon_fraction"] <= 0.0 for row in held):
        lines += [
            "  No held variant reached the horizon. The pose the policy holds REQUIRES",
            "  continuous feedback. The tremor is stabilisation, not waste, and penalising",
            "  it harder would remove the only thing keeping the animal up. The question",
            "  becomes why the policy settled on a pose it has to fight -- look at the",
            "  actuators no stage-1 reward term constrains.",
        ]
    else:
        lines += [
            "  Mixed. Compare the ramped variant against the hard switch: if only the hard",
            "  switch falls, the step transient caused it and neither row is evidence about",
            "  the pose. Re-run with more episodes before drawing a conclusion.",
        ]
    lines += [
        "",
        "  The two controls bracket the measurement. 'policy' is the unmodified checkpoint",
        "  and must reach the horizon; 'hold_zero' is the statue, a constant already known",
        "  to stand. If either misbehaves, the harness is at fault, not the pose.",
    ]
    return "\n".join(lines) + "\n", payload


#: A variant is treated as standing when it clears this share of full-horizon
#: episodes, and as falling at or below :data:`_ABLATION_FALLS`. Two thresholds
#: with a gap between them on purpose: an ablation that lands in the gap is
#: reported as inconclusive rather than rounded to whichever side is nearer,
#: because the conclusions on the two sides are opposite.
_ABLATION_STANDS = 0.95
_ABLATION_FALLS = 0.05


def _ablation_verdicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify each ablated group by necessity and sufficiency.

    Necessity and sufficiency are asked separately because either alone
    misleads. A group that is merely *correlated* with the fall -- the
    saturated set is the obvious candidate, since saturation is visible and
    dramatic -- shows up as necessary-and-not-sufficient or as neither, and
    only the pair distinguishes that from a cause.
    """
    by_label = {row["label"]: row for row in rows}
    stands = lambda row: row is not None and row["full_horizon_fraction"] >= _ABLATION_STANDS  # noqa: E731
    falls = lambda row: row is not None and row["full_horizon_fraction"] <= _ABLATION_FALLS  # noqa: E731
    verdicts = []
    for label in by_label:
        if not label.startswith("release_"):
            continue
        group = label[len("release_") :]
        release, only = by_label.get(label), by_label.get(f"only_{group}")
        necessary = stands(release)
        sufficient = falls(only)
        if release is None or only is None:
            verdict = "incomplete (a variant did not run)"
        elif not (stands(release) or falls(release)) or not (stands(only) or falls(only)):
            verdict = "inconclusive (a variant landed between the thresholds)"
        elif necessary and sufficient:
            verdict = "CAUSE — necessary and sufficient"
        elif necessary:
            verdict = "contributes — necessary, not sufficient alone"
        elif sufficient:
            verdict = "sufficient alone, but not necessary"
        else:
            verdict = "bystander — neither"
        verdicts.append(
            {
                "group": group,
                "necessary": necessary,
                "sufficient": sufficient,
                "verdict": verdict,
                "release_full_horizon": None if release is None else release["full_horizon_fraction"],
                "only_full_horizon": None if only is None else only["full_horizon_fraction"],
            }
        )
    return verdicts


def render_constant_hold_ablation(
    reports: list[dict[str, Any]], *, probe_episodes: int
) -> tuple[str, dict[str, Any]]:
    """Reduce the release ablation to its table, per-group verdicts, and payload.

    The ablation walks the path between two already-measured endpoints: the
    policy's full held pose, which falls, and the statue, which stands. Each
    row moves one actuator group between them, so the table asks which
    coordinate carries the effect rather than merely restating that the pose
    is unholdable.
    """
    rows = [
        {
            "label": report["hold_constant"]["label"],
            "holds": report["hold_constant"]["holds"],
            "source": report["hold_constant"]["source"],
            "episode_length_mean": report["metrics"]["episode_length_mean"],
            "full_horizon_fraction": report["metrics"]["full_horizon_fraction"],
            "reward_mean": report["metrics"]["reward_mean"],
            "terminations": report["terminations"],
            "n_released": sum(1 for value in report["hold_constant"]["actions"] if value == 0.0),
        }
        for report in reports
    ]
    verdicts = _ablation_verdicts(rows)
    horizon = reports[0]["horizon"]
    policy = str(reports[0]["policy"]).split(" — ")[0]
    payload = {
        "schema": "mesozoic.constant-hold-ablation/v1",
        "species": reports[0]["species"],
        "stage": reports[0]["stage"],
        "policy": policy,
        "horizon": horizon,
        "episodes_per_variant": probe_episodes,
        "stands_threshold": _ABLATION_STANDS,
        "falls_threshold": _ABLATION_FALLS,
        "note": (
            "Each row scored a MODIFIED policy: its commanded action frozen to a constant "
            "with one actuator group returned to the home keyframe (released) or with only "
            "that group held. No row is a gate verdict. release_G tests whether G is "
            "NECESSARY for the fall; only_G tests whether G is SUFFICIENT to cause it."
        ),
        "variants": rows,
        "verdicts": verdicts,
        "reports": reports,
    }
    lines = [
        "PROBE: a held constant pose, ablated one actuator group at a time.",
        "Every row scores a MODIFIED policy. None of them is a gate verdict.",
        "",
        f"policy              {policy}",
        f"stage               {reports[0]['species']} stage {reports[0]['stage']}",
        f"panel               {probe_episodes} episodes per variant, horizon {horizon}",
        "released            commanded 0, which IS the home keyframe under home-keyframe-residual/v1",
        "",
        f"  {'variant':<28}{'released':>9}{'ep length':>11}{'full-horiz':>12}{'reward':>10}   terminations",
    ]
    for row in rows:
        released = "-" if not row["holds"] else str(row["n_released"])
        lines.append(
            f"  {row['label']:<28}{released:>9}{row['episode_length_mean']:>11.1f}"
            f"{row['full_horizon_fraction']:>12.4f}{row['reward_mean']:>10.1f}   {row['terminations']}"
        )
    if verdicts:
        lines += [
            "",
            "Per group — release_G asks if G is NECESSARY (does removing it rescue the pose?),",
            "only_G asks if G is SUFFICIENT (does it alone break the statue?):",
            "",
            f"  {'group':<14}{'release stands':>16}{'only falls':>12}   verdict",
        ]
        for entry in verdicts:
            release_text = "-" if entry["release_full_horizon"] is None else f"{entry['release_full_horizon']:.4f}"
            only_text = "-" if entry["only_full_horizon"] is None else f"{entry['only_full_horizon']:.4f}"
            lines.append(f"  {entry['group']:<14}{release_text:>16}{only_text:>12}   {entry['verdict']}")
    if verdicts and not any(entry["necessary"] for entry in verdicts):
        sufficient = [entry["group"] for entry in verdicts if entry["sufficient"]]
        lines += ["", "Overall:"]
        if sufficient:
            lines += [
                f"  No group is necessary, but {', '.join(sufficient)} " + ("is" if len(sufficient) == 1 else "are"),
                "  sufficient. The pose is OVER-DETERMINED: more than one subset of it breaks",
                "  the animal independently, so removing any single group leaves another still",
                "  able to. 'Which joint is the cause' is therefore the wrong question -- but the",
                "  sufficient groups are where the effect is concentrated, and a group that holds",
                "  the full horizon on its own is exonerated no matter how extreme its DC looks.",
            ]
        else:
            lines += [
                "  No group is necessary and none is sufficient. Whatever makes the pose",
                "  unholdable is not separated by these groupings -- it may be a combination",
                "  that crosses them, or a joint outside every group. Regroup before concluding.",
            ]
    lines += [
        "",
        "Read the pair, never one side. A group can look implicated because it is",
        "conspicuous -- twelve actuators pinned at exactly +/-1.000 are hard to ignore --",
        "and still be a bystander. Only necessary AND sufficient identifies a cause;",
        "necessary alone means it contributes with others, and neither means the",
        "stabilisation load lives somewhere this ablation did not separate.",
        "",
        "The endpoints bound the path: 'hold_all' is the full pose and must fall,",
        "'hold_zero' is the statue and must stand. If either misbehaves the harness is",
        "at fault and no row below it means anything.",
    ]
    return "\n".join(lines) + "\n", payload


def write_constant_hold_ablation(
    stage_dir: "str | Path", reports: list[dict[str, Any]], *, probe_episodes: int
) -> dict[str, Path]:
    """Write the release ablation beside the stage's other artifacts.

    Its own filenames, and never ``stance_panel_selected.csv`` -- same
    discipline as the other two probes, for the same reason.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text, payload = render_constant_hold_ablation(reports, probe_episodes=probe_episodes)
    text_path = directory / "stance_gate_probe_release.txt"
    json_path = directory / "stance_gate_probe_release.json"
    text_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"constant_hold_ablation_txt": text_path, "constant_hold_ablation_json": json_path}


def write_constant_hold_probe(
    stage_dir: "str | Path", reports: list[dict[str, Any]], *, probe_episodes: int
) -> dict[str, Path]:
    """Write the constant-hold probe beside the stage's other artifacts.

    Its own filenames, and never ``stance_panel_selected.csv`` -- the same
    discipline the filtered probe follows, for the same reason: no row here
    scored the policy a bundle would be certified from.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text, payload = render_constant_hold_probe(reports, probe_episodes=probe_episodes)
    text_path = directory / "stance_gate_probe_constant.txt"
    json_path = directory / "stance_gate_probe_constant.json"
    text_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"constant_hold_probe_txt": text_path, "constant_hold_probe_json": json_path}


#: Anatomical actuator groups, as substrings matched against joint names. Used
#: to ablate a held pose one region at a time. Substrings rather than indices so
#: the same groups resolve on any species whose joints follow the naming
#: convention, and so adding a joint does not silently fall out of its group.
_ACTUATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "tail": ("tail",),
    "toes": ("toe",),
    "head_neck": ("head", "neck"),
    "hip_rolls": ("hip_roll",),
}


def actuator_indices_matching(report: dict[str, Any], patterns: "tuple[str, ...] | list[str]") -> tuple[int, ...]:
    """Indices whose joint name contains any of *patterns*."""
    per_actuator = (report.get("action") or {}).get("per_actuator") or []
    return tuple(
        int(entry["index"])
        for entry in sorted(per_actuator, key=lambda e: e["index"])
        if any(pattern in str(entry["joint"]) for pattern in patterns)
    )


def saturated_actuator_indices(report: dict[str, Any], threshold: float = 0.99) -> tuple[int, ...]:
    """Indices whose commanded pose sits at or beyond *threshold* of the action range.

    A saturated actuator has no headroom in one direction, so it cannot
    contribute correction on that axis -- and on the T-Rex stage-1 checkpoint
    the saturated set is also exactly the set with zero AC, i.e. the joints the
    policy pins and then never moves.
    """
    per_actuator = (report.get("action") or {}).get("per_actuator") or []
    return tuple(
        int(entry["index"])
        for entry in sorted(per_actuator, key=lambda e: e["index"])
        if abs(float(entry["dc"])) >= threshold
    )


def constant_hold_released(hold: tuple[float, ...], release: "tuple[int, ...] | set[int]") -> tuple[float, ...]:
    """*hold* with the given actuators returned to the home keyframe.

    "Released" means commanded ``0``, which under ``home-keyframe-residual/v1``
    is exactly the home pose -- so releasing every actuator gives the statue,
    and releasing none gives the policy's own pose. Every ablation below is a
    point on the path between a pose that falls and one that stands.
    """
    released = set(int(index) for index in release)
    return tuple(0.0 if index in released else value for index, value in enumerate(hold))


def constant_hold_release_variants(
    hold: tuple[float, ...],
    report: dict[str, Any],
    *,
    horizon: int,
    ramp_steps: int = 50,
) -> list[ConstantHold]:
    """Ablate the held pose one actuator group at a time, both ways round.

    Every ablated variant is commanded **from reset**, not from a handoff
    partway through. That is load-bearing and was got wrong the first time:
    handing off at ``settle_steps`` means the policy first establishes its own
    displaced pose and velocities, and the ablation then snaps some subset of
    joints from ``±1.000`` to ``0`` in one step. The transient dominates, every
    variant lands within noise of every other (311-349 steps across all
    thirteen), and -- decisively -- the all-released endpoint falls too, at 323
    steps, when the statue it is supposed to be stands for 1000. Both endpoints
    on the same side of the answer means the path has no signal to locate.

    From reset the endpoints are the two known measurements: releasing every
    actuator commands ``0`` throughout, which *is* the statue and stands the
    full horizon, and releasing none reproduces the probe's own
    ``hold_from_reset`` at 133 steps. The path then genuinely runs from falling
    to standing, and an ablation on it locates which coordinate carries the
    effect.

    Each group gets two variants, because "which joints cause the fall" needs
    two different questions answered and one of them alone is not evidence:

    * ``release_G`` holds everything *except* G, testing **necessity** -- does
      taking G away rescue a pose that otherwise falls?
    * ``only_G`` holds *only* G, testing **sufficiency** -- does adding G alone
      break the statue, which is known to stand?

    A group that is both necessary and sufficient is the cause. A group that is
    necessary but not sufficient contributes with others. A group that is
    neither is a bystander, whatever its DC looks like -- and the saturated
    joints look dramatic (twelve pinned at exactly +/-1.000) precisely because
    saturation is visible, not because it was shown to matter.

    The two endpoints of the path are included as controls: ``hold_all`` is the
    full pose, already measured to fall, and ``hold_zero`` is the statue,
    already measured to stand. Every ablation sits between them.
    """
    zero = tuple(0.0 for _ in hold)
    groups: dict[str, tuple[int, ...]] = {"saturated": saturated_actuator_indices(report)}
    for name, patterns in _ACTUATOR_GROUPS.items():
        indices = actuator_indices_matching(report, patterns)
        if indices:
            groups[name] = indices
    everything = tuple(range(len(hold)))

    variants = [
        ConstantHold(hold, handoff_steps=horizon, label="policy (control)", source="unmodified", holds=False),
        ConstantHold(hold, handoff_steps=0, ramp_steps=ramp_steps, label="hold_all"),
    ]
    for name, indices in groups.items():
        if not indices or len(indices) == len(hold):
            # Releasing everything is the statue and releasing nothing is
            # `hold_all`; neither is an ablation, and both are already rows.
            continue
        complement = tuple(index for index in everything if index not in set(indices))
        variants.append(
            ConstantHold(
                constant_hold_released(hold, indices),
                handoff_steps=0,
                ramp_steps=ramp_steps,
                label=f"release_{name}",
            )
        )
        variants.append(
            ConstantHold(
                constant_hold_released(hold, complement),
                handoff_steps=0,
                ramp_steps=ramp_steps,
                label=f"only_{name}",
            )
        )
    variants.append(
        ConstantHold(zero, handoff_steps=0, ramp_steps=ramp_steps, label="hold_zero (statue control)", source="zeros")
    )
    return variants


def constant_hold_variants(
    hold: tuple[float, ...], *, settle_steps: int, horizon: int, ramp_steps: int = 50
) -> list[ConstantHold]:
    """The standard variant set: two controls bracketing three held rollouts.

    ``handoff_steps = horizon`` is how the unmodified control is expressed. It
    rolls the real policy for the whole episode yet still carries the
    ``hold_constant`` marker, so it can never be written to the certification
    filenames by a caller who forgets which report is which.
    """
    zero = tuple(0.0 for _ in hold)
    return [
        ConstantHold(hold, handoff_steps=horizon, label="policy (control)", source="unmodified", holds=False),
        ConstantHold(hold, handoff_steps=settle_steps, ramp_steps=0, label="hold_after_settle"),
        ConstantHold(hold, handoff_steps=settle_steps, ramp_steps=ramp_steps, label="hold_after_settle_ramped"),
        ConstantHold(hold, handoff_steps=0, ramp_steps=ramp_steps, label="hold_from_reset"),
        ConstantHold(zero, handoff_steps=0, ramp_steps=0, label="hold_zero (statue control)", source="zeros"),
    ]


#: Column order of ``stance_panel_selected.csv``.  ``unsupported_duty`` is
#: empty for an episode whose duty could not be measured (one shorter than the
#: settling window); the auditor treats empty as unmeasured rather than zero,
#: because zero is the statue's value and the best possible score.
STANCE_PANEL_FIELDNAMES = (
    "episode",
    "panel_seed",
    "length",
    "reward",
    "reached_horizon",
    "unsupported_duty",
    "bilateral_support_duty",
    "single_support_duty",
)


def write_stance_panel_evidence(stage_dir: "str | Path", report: dict[str, Any]) -> "Path | None":
    """Write the panel's per-episode measurements as publication evidence.

    Returns ``None`` when the report carries no ``episode_evidence`` -- an
    older report, or one built before the field existed. The caller records
    only what was written, and the auditor's own refusal covers the absence,
    so a missing file fails the bundle rather than passing it.
    """
    episodes = report.get("episode_evidence")
    if not episodes:
        return None
    output = Path(stage_dir) / "stance_panel_selected.csv"
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(STANCE_PANEL_FIELDNAMES))
        writer.writeheader()
        for episode in episodes:
            duty = episode.get("unsupported_duty")
            writer.writerow(
                {
                    "episode": episode["episode"],
                    "panel_seed": episode["seed"],
                    "length": int(episode["length"]),
                    "reward": float(episode["reward"]),
                    "reached_horizon": bool(episode["reached_horizon"]),
                    "unsupported_duty": "" if duty is None else float(duty),
                    "bilateral_support_duty": _optional_float_cell(episode.get("bilateral_support_duty")),
                    "single_support_duty": _optional_float_cell(episode.get("single_support_duty")),
                }
            )
    return output


def _optional_float_cell(value: Any) -> str | float:
    return "" if value is None else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("species", help="Species id, e.g. trex")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--model", help="Path to a saved SB3 checkpoint (.zip)")
    parser.add_argument("--vecnorm", help="Path to the checkpoint's VecNormalize .pkl")
    parser.add_argument(
        "--zero-action",
        action="store_true",
        help="Score the do-nothing policy instead of a checkpoint (the reference the gate is calibrated against)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Defaults to the stage's min_eval_episodes")
    parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="Score a checkpoint that predates the plant contract. The verdict is then about a plant "
        "that may differ from the one in the tree, and the report records plant_validated=false.",
    )
    parser.add_argument("--seed", type=int, default=3042, help="First evaluation seed (default: publication seed)")
    parser.add_argument("--config", help="Explicit stage TOML path")
    parser.add_argument("--out-dir", help="Also write stance_gate_report.{txt,json} to this directory")
    parser.add_argument(
        "--filter-actions",
        type=float,
        metavar="HZ",
        help="Low-pass the policy's actions at HZ before they reach the plant, and score THAT. "
        "A probe for whether a policy's high-frequency action content is load-bearing: if it "
        "still stands, the tremor was waste and belongs on the action path; if it falls, the "
        "tremor is closed-loop stabilisation and the fix belongs elsewhere (issue #489).",
    )
    parser.add_argument(
        "--hold-constant",
        action="store_true",
        help="Replace the policy's action with the constant it commands ON AVERAGE partway through "
        "each episode, and score THAT, over a standard variant set with two controls. Separates "
        "'the pose needs feedback' from 'the tremor is waste the action penalties missed' -- the "
        "one question the low-pass probe cannot answer, because a filtered policy still responds.",
    )
    parser.add_argument(
        "--hold-from-report",
        metavar="JSON",
        help="Take the constant from an existing stance_gate_report.json instead of measuring it. "
        "Skips the measurement panel when you already have one for this checkpoint.",
    )
    parser.add_argument(
        "--hold-episodes",
        type=int,
        default=10,
        help="Episodes per hold variant (default 10). The probe certifies nothing and the effect "
        "it measures is a fall or a full horizon, so it does not need the gate's panel size.",
    )
    parser.add_argument(
        "--hold-ramp-steps",
        type=int,
        default=50,
        help="Steps to blend into the constant in the ramped variants (default 50).",
    )
    parser.add_argument(
        "--hold-release-ablation",
        action="store_true",
        help="Instead of the standard hold variants, ablate the held pose one actuator group at a "
        "time -- releasing each group back to the home keyframe, and holding each group alone. "
        "Answers WHICH joints make the pose unholdable, by testing necessity and sufficiency "
        "separately. Implies --hold-constant.",
    )
    args = parser.parse_args()
    args.hold_constant = args.hold_constant or args.hold_release_ablation

    if not args.zero_action and not args.model:
        parser.error("pass --model, or --zero-action for the do-nothing reference")
    if args.hold_constant and args.filter_actions is not None:
        parser.error("--hold-constant and --filter-actions are different probes; run them separately")
    if args.hold_episodes < 1:
        parser.error(f"--hold-episodes must be at least 1, got {args.hold_episodes}")
    if args.hold_ramp_steps < 0:
        parser.error(f"--hold-ramp-steps cannot be negative, got {args.hold_ramp_steps}")
    if args.filter_actions is not None and args.filter_actions <= 0:
        parser.error(f"--filter-actions must be positive, got {args.filter_actions}")
    # `--episodes 0` used to fall through `episodes or min_eval_episodes` to the
    # stage default AND skip the under-powered-panel warning below, so it
    # silently produced a full-size panel while reading as an override.
    if args.episodes is not None and args.episodes < 1:
        parser.error(f"--episodes must be at least 1, got {args.episodes}")

    stage_config = load_stage_config(args.species, args.stage, config_path=args.config)
    try:
        report = build_stance_gate_report(
            args.species,
            args.stage,
            stage_config=stage_config,
            model_path=args.model,
            vecnorm_path=args.vecnorm,
            zero_action=args.zero_action,
            episodes=args.episodes,
            seed=args.seed,
            allow_legacy_plant=args.allow_legacy_plant,
            filter_actions_hz=args.filter_actions,
        )
    except StanceGateReportError as exc:
        # The CLI boundary is where a diagnosable failure becomes an exit
        # status; inside the library it stays an ordinary exception.
        raise SystemExit(str(exc)) from exc
    print()
    print(render_stance_gate_report(report))

    if args.out_dir:
        for path in write_stance_gate_report(args.out_dir, report).values():
            print(f"written: {path}")

    if args.hold_constant:
        try:
            hold = (
                constant_hold_actions(json.loads(Path(args.hold_from_report).read_text(encoding="utf-8")))
                if args.hold_from_report
                else constant_hold_actions(report)
            )
        except StanceGateReportError as exc:
            raise SystemExit(str(exc)) from exc
        source_report = (
            json.loads(Path(args.hold_from_report).read_text(encoding="utf-8"))
            if args.hold_from_report
            else report
        )
        variants = (
            constant_hold_release_variants(
                hold,
                source_report,
                horizon=report["horizon"],
                ramp_steps=args.hold_ramp_steps,
            )
            if args.hold_release_ablation
            else constant_hold_variants(
                hold,
                settle_steps=report["settle_steps"],
                horizon=report["horizon"],
                ramp_steps=args.hold_ramp_steps,
            )
        )
        probes = [
            build_stance_gate_report(
                args.species,
                args.stage,
                stage_config=stage_config,
                model_path=args.model,
                vecnorm_path=args.vecnorm,
                zero_action=args.zero_action,
                episodes=args.hold_episodes,
                seed=args.seed,
                allow_legacy_plant=args.allow_legacy_plant,
                hold_constant=variant,
            )
            for variant in variants
        ]
        render = render_constant_hold_ablation if args.hold_release_ablation else render_constant_hold_probe
        write = write_constant_hold_ablation if args.hold_release_ablation else write_constant_hold_probe
        text, _ = render(probes, probe_episodes=args.hold_episodes)
        print()
        print(text, end="")
        if args.out_dir:
            for path in write(args.out_dir, probes, probe_episodes=args.hold_episodes).values():
                print(f"written: {path}")

    declared = report["thresholds"]["min_eval_episodes"]
    if args.episodes is not None and args.episodes != declared:
        print()
        print(
            f"WARNING: --episodes {args.episodes} differs from the stage's min_eval_episodes "
            f"{declared}. The bound's power is specified at the latter; this panel does not "
            "certify what the gate claims."
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
