"""JAX-compatible curriculum manager.

Mirrors the stage-gating logic from ``curriculum.py`` but works with
the JAX training path.  Stage configs are loaded from the same TOML
files used by the SB3 path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import Any

from .config import load_stage_config
from .curriculum.gate_schema import (
    BACKEND_OVERRIDABLE_KEYS,
    GateSchemaError,
    apply_backend_overrides,
    finite_gate_metric,
    has_backend_overrides,
    validate_gate_config,
)
from .curriculum.recovery_gate import RECOVERY_GATE_KIND
from .curriculum.stance_gate import (
    STANCE_GATE_KIND,
    StanceGateThresholds,
    StancePanel,
    evaluate_stance_gate,
)

_logger = logging.getLogger(__name__)

#: The historical gate kind :func:`check_stage_gate`'s reward-and-length arm
#: evaluates.  Named so the dispatch and the pre-flight check cannot drift.
_REWARD_AND_LENGTH_GATE_KIND = "reward_and_length/v1"

#: Gate kinds the in-training JAX path has an evaluator for.  Schema-valid is
#: NOT sufficient: recovery_quality/v1 validates, but its verdict comes only
#: from the frozen gate resolver
#: (curriculum.gate_resolver.evaluate_recovery_gate_from_resolution), so
#: dispatching it here would either crash on a missing threshold key after the
#: stage's whole budget or advance the pushed stage on its optional reward
#: rail alone.  P5 re-audited this and left it out deliberately — see
#: :func:`_require_evaluable_gate_kind` for the three inputs this path cannot
#: obtain.
_EVALUATABLE_GATE_KINDS = frozenset({_REWARD_AND_LENGTH_GATE_KIND, STANCE_GATE_KIND})


def _require_evaluable_gate_kind(stage: int | str, gate_kind: str) -> None:
    """Reject a schema-valid gate kind this module cannot evaluate.

    ``recovery_quality/v1`` is the audited case (plan P5).  Its verdict is a
    function of three things this path does not have and cannot derive from
    what it is given:

    * the **stage directory** holding the frozen ``gate_resolution.json`` —
      :func:`check_stage_gate` receives a loaded TOML config and a metrics
      dict, neither of which knows where the run writes;
    * the stage's **current task fingerprint**, which the resolution's
      staleness check compares against;
    * a **pairable pushed panel** — per-episode successes on the registered
      panel seeds, aligned seed-by-seed against the frozen null manifest.
      ``eval_metrics`` carries reduced aggregates, never per-seed outcomes,
      and no MJX pushed-panel roller exists: every recovery panel to date is
      SB3 (first-runs record §8), which is also the plan's decision (§6,
      "SB3 first for all 1b evidence").

    So the refusal stands, and it is raised rather than logged because
    :func:`run_curriculum` pre-flights it *before* spending a stage's budget.

    Args:
        stage: Stage identifier, used only in the error message.
        gate_kind: A kind already validated by
            :func:`~environments.shared.curriculum.gate_schema.validate_gate_config`.

    Raises:
        GateSchemaError: If no arm of :func:`check_stage_gate` evaluates the
            kind.  Falling through to the reward gate instead would advance
            the stage on return alone — the fall-through
            ``reporting/gates.py`` refused for exactly this reason before it
            gained its resolver wiring.
    """
    if gate_kind not in _EVALUATABLE_GATE_KINDS:
        detail = ""
        if gate_kind == RECOVERY_GATE_KIND:
            detail = (
                " A recovery verdict needs the stage directory's frozen gate_resolution.json, "
                "the stage's current task_sha256, and per-seed episode successes on the "
                "registered panel seeds paired against the frozen null manifest; this path "
                "receives a TOML config and reduced eval metrics, so it has none of them, and "
                "no MJX pushed-panel roller exists (every recovery panel is SB3). The verdict "
                "is produced after the stage by reporting.gates.evaluate_stage_gate."
            )
        raise GateSchemaError(
            f"stage {stage} declares gate_kind {gate_kind!r}, which the in-training "
            f"JAX curriculum cannot evaluate; kinds evaluated here: "
            f"{sorted(_EVALUATABLE_GATE_KINDS)}. Recovery verdicts come only from "
            "the gate resolver (curriculum.gate_resolver."
            "evaluate_recovery_gate_from_resolution); falling through to the reward "
            f"gate would advance the stage on return alone.{detail}"
        )


#: TOML ``[jax]`` keys mapped to :func:`~environments.shared.jax_training.train_jax`
#: parameter names.  Module-level so the mapping loop in :func:`run_curriculum`
#: and :func:`validate_jax_kwargs`'s known-key set cannot drift apart — a key
#: added to one without the other now fails a test instead of going silently
#: inert (``ramp_attr`` and ``obs_rms_decay_on_resume`` were both dropped this
#: way while eight TOML comments asserted they worked).
_JAX_KEY_MAP = {
    "num_envs": "num_envs",
    "rollout_len": "rollout_len",
    "num_updates": "num_updates",
    "learning_rate": "learning_rate",
    "learning_rate_end": "learning_rate_end",
    "max_grad_norm": "max_grad_norm",
    "gamma": "gamma",
    "gae_lambda": "gae_lambda",
    "clip_range": "clip_range",
    "vf_clip_range": "vf_clip_range",
    "ent_coef": "ent_coef",
    "vf_coef": "vf_coef",
    "ppo_epochs": "n_epochs",
    "target_kl": "target_kl",
    "minibatch_size": "minibatch_size",
    "warmup_updates": "warmup_updates",
    "warmup_clip_range": "warmup_clip_range",
    "warmup_ent_coef": "warmup_ent_coef",
    "ramp_updates": "ramp_updates",
    "ramp_start_fraction": "ramp_start_fraction",
    "ramp_attr": "ramp_attr",
    # ``[jax.policy_kwargs]`` — net_arch sizes the actor-critic backbone
    # (train_jax -> make_actor_critic(hidden_dims=...)).  It was declared in
    # every stage TOML as "Match SB3 PPO architecture" while every JAX call
    # site built the hardcoded (512, 256) default.
    "policy_kwargs": "policy_kwargs",
}

#: Keys ``[jax.policy_kwargs]`` may carry.  Only the architecture is wired;
#: an SB3-only key (activation_fn, ortho_init, ...) would be silently inert.
KNOWN_JAX_POLICY_KWARGS = frozenset({"net_arch"})

#: ``[jax]`` keys applied as ``[env]`` overrides rather than train kwargs.
_JAX_ENV_OVERRIDE_KEYS = ("fall_penalty", "reset_noise_scale", "init_qpos_noise", "init_yaw_noise")

#: Every ``[jax]`` key some consumer actually reads.  ``[curriculum]`` keys get
#: this discipline from ``gate_schema`` ("silently dropping a misspelled
#: threshold disables it") and ``[env]`` keys fail loudly at env construction;
#: the ``[jax]`` table had neither, and the configs' own history shows the trap
#: is real ("species default was leaking into stage 1 via key mismatch bug").
#: ``obs_rms_decay_on_resume`` is consumed by :func:`run_curriculum` itself
#: (cross-stage init), not forwarded to ``train_jax``.
KNOWN_JAX_KEYS = frozenset(_JAX_KEY_MAP) | frozenset(_JAX_ENV_OVERRIDE_KEYS) | frozenset({"obs_rms_decay_on_resume"})


def validate_jax_kwargs(jax_kwargs: dict[str, Any], *, source: str) -> None:
    """Reject unknown ``[jax]`` keys, fail-closed.

    Args:
        jax_kwargs: The stage config's ``jax_kwargs`` table.
        source: Human-readable origin for the error message, e.g.
            ``"trex stage 2 [jax]"``.

    Raises:
        ValueError: If the table carries a key no consumer reads.  Silently
            dropping a misspelled or unwired key disables it — the failure
            mode this module shipped twice (``ramp_attr``,
            ``obs_rms_decay_on_resume``).
    """
    unknown = sorted(set(jax_kwargs) - KNOWN_JAX_KEYS)
    if unknown:
        raise ValueError(
            f"{source} declares keys nothing consumes: {unknown}. "
            "A [jax] key that no training path reads is silently inert — fix the "
            "spelling, or add it to _JAX_KEY_MAP / KNOWN_JAX_KEYS in jax_curriculum.py "
            "alongside the code that consumes it."
        )
    policy_kwargs = jax_kwargs.get("policy_kwargs") or {}
    unknown_policy = sorted(set(policy_kwargs) - KNOWN_JAX_POLICY_KWARGS)
    if unknown_policy:
        raise ValueError(
            f"{source} [jax.policy_kwargs] declares keys the JAX network factory does not read: "
            f"{unknown_policy}; only {sorted(KNOWN_JAX_POLICY_KWARGS)} is wired (make_actor_critic hidden_dims)."
        )


def network_hidden_dims(jax_kwargs: Mapping[str, Any], default: tuple[int, ...] = (512, 256)) -> tuple[int, ...]:
    """The actor-critic backbone widths a stage's ``[jax.policy_kwargs]`` asks for.

    Every path that builds the network — training, and every load/eval path
    that rebuilds it to apply saved params — must call this with the SAME
    stage's ``jax_kwargs``: a mismatch there is a shape error at parameter
    load, or worse, a silently different network than the checkpoint was
    trained with.

    Args:
        jax_kwargs: The stage config's ``jax_kwargs`` table.
        default: Widths used when the table declares no ``net_arch``.
    """
    net_arch = (jax_kwargs.get("policy_kwargs") or {}).get("net_arch")
    if net_arch is None:
        return tuple(default)
    dims = tuple(int(width) for width in net_arch)
    if not dims or any(width <= 0 for width in dims):
        raise ValueError(f"[jax.policy_kwargs] net_arch must be a non-empty list of positive widths, got {net_arch!r}")
    return dims


def episode_return_for_gate(eval_metrics: dict[str, float], *, threshold: float) -> float:
    """The EPISODE-level mean return the TOML reward thresholds are stated in.

    ``min_avg_reward`` is an episode-level threshold shared with the SB3
    TOMLs — trex stage 1 sets 1950.0.  The MJX trainer emits **both** a
    per-step ``mean_reward`` (~3.3 for a standing T-Rex) and an episode-level
    ``mean_episode_return``; only the second is comparable.

    Three call sites used to substitute the per-step value when the episode
    return was absent, with three different behaviours: the
    ``reward_and_length`` branch warned and defaulted to ``0.0``, the stance
    branch fell back **silently** and defaulted to ``-inf``, and
    ``run_curriculum``'s log line did its own third version.  Substituting is
    always wrong — it compares numbers three orders of magnitude apart — and
    it merely happens to give the right verdict sometimes.  Against trex
    stage 1's rail it gives 3.3 < 1950, so the stage never advances and the
    only clue is a rail failure that reads like a bad policy.

    Absent-and-needed is therefore fatal, matching how this module already
    treats a declared ``min_avg_episode_length`` with no ``mean_episode_length``
    to check it against.

    Args:
        eval_metrics: The trainer's evaluation metrics.
        threshold: The reward criterion this value will be compared against.
            A non-finite threshold means no reward criterion is configured —
            the stance gate's rail is optional — so nothing is compared and
            the absence does not matter.

    Raises:
        GateSchemaError: If a finite reward threshold is configured but the
            metrics carry no ``mean_episode_return``.
    """
    episode_return = eval_metrics.get("mean_episode_return")
    if episode_return is not None:
        return float(episode_return)

    if not math.isfinite(threshold):
        # No reward criterion to check, so there is nothing to be wrong about.
        return -math.inf

    per_step = eval_metrics.get("mean_reward")
    measured = (
        f" The metrics do carry a per-step mean_reward of {float(per_step):.4g}, which is NOT comparable: "
        f"substituting it would compare it against {threshold:g}."
        if per_step is not None
        else ""
    )
    raise GateSchemaError(
        "stage config sets min_avg_reward but eval_metrics carries no "
        f"mean_episode_return, so the reward criterion cannot be checked.{measured} "
        "Emit mean_episode_return from the trainer instead."
    )


#: Calibrations already warned about, keyed by (config name, bar) rather than
#: by the stage reference: the gate is checked many times per stage
#: (pre-flight, every evaluation) and not every caller knows the stage —
#: ``load_stage_config`` emits no ``"stage"`` key, so :func:`check_stage_gate`
#: sees ``"?"`` — but the bar the warning describes does not change between
#: checks.  (The fall penalties are deliberately not part of the key:
#: :func:`run_curriculum` applies the ``[jax]`` override into ``env_kwargs``
#: in place before training, so they read differently after the stage.)
_sb3_calibrated_bar_warned: set[tuple[Any, ...]] = set()


def jax_gate_thresholds(stage: int | str, stage_config: dict[str, Any]) -> dict[str, Any]:
    """The ``[curriculum]`` thresholds as the JAX path must read them.

    Applies the stage's ``[curriculum.jax]`` override table (additive: absent,
    the shared thresholds are returned unchanged) and, when a reward-denominated
    threshold is left at its shared value, warns ONCE that the bar is
    SB3-calibrated.  ``min_avg_reward`` is compared against raw episode returns
    on both backends, but the backends do not pay the same return for the same
    behaviour: the MJX kernel height-gates the alive bonus by the
    ``healthy_z_range`` fraction whenever ``support_conditioned_alive_fraction``
    is 0 (the stage-2/3 configs; ~0.27x of ``alive_bonus`` for a standing
    trex — a deliberate legacy), and ``[jax] fall_penalty`` overrides
    ``[env] fall_penalty`` (-10 vs -150 on trex locomotion).  The other
    overridable thresholds (length, velocity, success) are not
    reward-denominated, so they raise no warning.

    Args:
        stage: Stage reference, named in the warning.
        stage_config: The loaded stage config (``curriculum_kwargs`` already
            validated by :func:`validate_gate_config`; ``env_kwargs`` /
            ``jax_kwargs`` are read only to name the fall penalties).
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    if has_backend_overrides(curriculum, "jax"):
        return apply_backend_overrides(curriculum, "jax")
    shared = apply_backend_overrides(curriculum, "jax")
    min_avg_reward = finite_gate_metric(shared.get("min_avg_reward"))
    if min_avg_reward is None:
        return shared
    env_fall = stage_config.get("env_kwargs", {}).get("fall_penalty")
    jax_fall = stage_config.get("jax_kwargs", {}).get("fall_penalty")
    key = (stage_config.get("name"), min_avg_reward)
    if key not in _sb3_calibrated_bar_warned:
        _sb3_calibrated_bar_warned.add(key)
        fall_note = (
            f"[jax] fall_penalty={jax_fall} overrides [env] fall_penalty={env_fall}"
            if jax_fall is not None
            else f"[env] fall_penalty={env_fall} applies unchanged (no [jax] override)"
        )
        _logger.warning(
            "stage %s: [curriculum] min_avg_reward=%s is compared against raw MJX episode returns but "
            "was calibrated on the SB3 backend: the MJX kernel height-gates the alive bonus by the "
            "healthy_z_range fraction whenever support_conditioned_alive_fraction is 0 (~0.27x of "
            "alive_bonus for a standing trex; deliberate legacy), and %s. The same number is therefore "
            "a different bar on this backend. Declare [curriculum.jax] (%s) to state a JAX-calibrated bar.",
            stage,
            min_avg_reward,
            fall_note,
            ", ".join(sorted(BACKEND_OVERRIDABLE_KEYS)),
        )
    return shared


def check_stage_gate(
    eval_metrics: dict[str, float],
    stage_config: dict[str, Any],
) -> bool:
    """Check if curriculum gate thresholds are met.

    Args:
        eval_metrics: Evaluation metrics dict (keys like ``"mean_reward"``).
        stage_config: Stage configuration dict as returned by
            :func:`~environments.shared.config.load_stage_config` (the TOML
            ``[curriculum]`` section lives under ``"curriculum_kwargs"``).

    Returns:
        ``True`` if the gate is passed and training should advance.

    Raises:
        GateSchemaError: If the stage's gate declaration is missing, unknown,
            or malformed — this used to log a warning and return ``True``, so a
            stage with no reward threshold advanced unconditionally, the same
            fail-open behaviour the SB3 path had, reached by a different route.
            Also raised for a schema-valid kind no arm below evaluates
            (recovery_quality/v1 today): the reward arm used to be the
            fall-through for every non-stance kind, so such a stage either
            crashed on a missing threshold key after its whole budget or
            advanced on its optional reward rail alone.  P5 wired the frozen
            resolver into the path that CAN reach its evidence
            (:func:`~environments.shared.reporting.gates.evaluate_stage_gate`,
            given the stage directory and the pushed panel) and deliberately
            left this one refusing — see :func:`_require_evaluable_gate_kind`.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    stage = stage_config.get("stage", "?")
    gate_kind = validate_gate_config(stage, curriculum, advancement_enabled=True)
    _require_evaluable_gate_kind(stage, gate_kind)
    curriculum = jax_gate_thresholds(stage, stage_config)

    if gate_kind == STANCE_GATE_KIND:
        return _check_stance_gate(eval_metrics, curriculum)

    # Only reward_and_length/v1 reaches here — _require_evaluable_gate_kind
    # refused every other non-stance kind. It requires min_avg_reward, so a
    # validated config of this kind always carries it; a KeyError here would
    # mean the schema and this branch fell out of sync.
    min_reward = float(curriculum["min_avg_reward"])
    episode_return = episode_return_for_gate(eval_metrics, threshold=min_reward)
    if not bool(episode_return >= min_reward):
        return False

    # The length half of reward_and_length/v1.  This used to be ignored here
    # while the SB3 CurriculumManager enforced it, so a stage carrying
    # min_avg_episode_length was gated on reward alone under JAX -- the exact
    # "one backend silently ignores a gate the other enforces" divergence the
    # gate schema exists to prevent.  It matters more since stage 1 began
    # encoding its full-horizon floor in this field (PLANT_VALIDATION §12).
    min_length = curriculum.get("min_avg_episode_length")
    if min_length is None:
        return True
    episode_length = eval_metrics.get("mean_episode_length")
    if episode_length is None:
        raise GateSchemaError(
            "stage config sets min_avg_episode_length but eval_metrics carries "
            "no mean_episode_length, so the length half of the gate cannot be "
            "checked. Passing on reward alone would silently half-enforce the "
            "gate; emit mean_episode_length from the trainer instead."
        )
    return bool(episode_length >= min_length)


#: Metrics the MJX trainer must emit for ``stance_quality/v1``.  Named here so
#: the error message can say exactly what is missing.
_STANCE_METRIC_KEYS = (
    "n_eval_episodes",
    "full_horizon_fraction",
    "n_duty_episodes",
    "mean_unsupported_duty",
    "unsupported_duty_ucb",
)


def _check_stance_gate(eval_metrics: dict[str, float], curriculum: dict[str, Any]) -> bool:
    """Evaluate ``stance_quality/v1`` on the JAX path.

    Runs the *same* :func:`~environments.shared.curriculum.stance_gate.evaluate_stance_gate`
    the SB3 ``CurriculumManager`` runs, on a panel the MJX trainer must supply
    already reduced -- so the two backends cannot disagree about the criteria.

    Raises:
        GateSchemaError: If the trainer emitted none of the stance metrics.
            Deliberately loud rather than a warning-and-pass: a stage that
            declares this kind and is then gated on nothing is precisely the
            "one backend silently ignores a gate the other enforces"
            divergence the versioned schema exists to prevent.
    """
    missing = [key for key in _STANCE_METRIC_KEYS if eval_metrics.get(key) is None]
    if missing:
        raise GateSchemaError(
            f"stage declares gate_kind {STANCE_GATE_KIND!r} but eval_metrics is missing "
            f"{missing}, so stance quality cannot be checked. Passing on the metrics that "
            "are present would half-enforce the gate. The MJX evaluator must reduce its "
            "rollout into a StancePanel with stance_gate.summarize_stance_panel() -- "
            "reconstructing episode boundaries from cumsum(lengths) over the per-step "
            "diag_* arrays -- and emit these keys. Note that reconstruction is valid for "
            "BIPEDS only: see the diag_r_foot/diag_l_foot interleaving defect in "
            "KNOWN_ISSUES, which must be fixed before this gate can be used for "
            "brachiosaurus or dibothrosuchus."
        )

    # The rail is optional for this kind, so resolve it against the declared
    # threshold: absent-and-unused is fine, absent-and-needed is fatal. This
    # used to fall back to the per-step mean_reward silently — 3.3 against
    # trex stage 1's 1950 rail, which never advances and reads like a bad
    # policy rather than a missing metric.
    min_avg_reward = float(curriculum.get("min_avg_reward", -math.inf))
    panel = StancePanel(
        n_episodes=int(eval_metrics["n_eval_episodes"]),
        full_horizon_fraction=float(eval_metrics["full_horizon_fraction"]),
        mean_reward=episode_return_for_gate(eval_metrics, threshold=min_avg_reward),
        n_duty_episodes=int(eval_metrics["n_duty_episodes"]),
        mean_unsupported_duty=float(eval_metrics["mean_unsupported_duty"]),
        unsupported_duty_ucb=float(eval_metrics["unsupported_duty_ucb"]),
    )
    thresholds = StanceGateThresholds(
        min_full_horizon_fraction=float(curriculum["min_full_horizon_fraction"]),
        max_unsupported_duty=float(curriculum["max_unsupported_duty"]),
        max_unsupported_duty_ucb=float(curriculum["max_unsupported_duty_ucb"]),
        settle_steps=int(curriculum.get("settle_steps", 0)),
        min_eval_episodes=int(curriculum.get("min_eval_episodes", 40)),
        min_avg_reward=min_avg_reward,
        required_consecutive=int(curriculum.get("required_consecutive", 3)),
    )
    passed, failures = evaluate_stance_gate(panel, thresholds)
    if not passed:
        _logger.info("Stance gate not met: %s", "; ".join(failures))
    return passed


def run_curriculum(
    species: str,
    train_fn: Callable,
    stages: tuple[int, ...] = (1, 2, 3),
    **train_kwargs: Any,
) -> dict[int, Any]:
    """Run full curriculum: train each stage, evaluate gate, advance.

    Args:
        species: Species name (``"trex"``, ``"velociraptor"``, ``"brachiosaurus"``).
        train_fn: Training function with signature
            ``train_fn(species, stage, **kwargs) -> (params, eval_metrics, obs_stats)``
            (a legacy 2-tuple without *obs_stats* is also accepted).
        stages: Tuple of stage references (legacy numbers or semantic ids)
            to train through, in order.
        **train_kwargs: Extra keyword arguments forwarded to ``train_fn``.

    Returns:
        Dict mapping stage number to final ``(params, eval_metrics)``.

    Raises:
        GateSchemaError: If a stage whose gate will be checked declares a
            malformed gate or a kind :func:`check_stage_gate` cannot evaluate.
            Raised before any training compute is spent — the gate check runs
            only after a stage's full budget, which is the most expensive
            possible time to learn its verdict was never computable.
    """
    # Only gated stages are pre-checked: the final stage's gate is never
    # evaluated here, and single-stage pilots legitimately run configs that
    # would not validate under advancement.
    for stage in stages[:-1]:
        stage_config = load_stage_config(species, stage)
        curriculum = stage_config.get("curriculum_kwargs", {})
        _require_evaluable_gate_kind(stage, validate_gate_config(stage, curriculum, advancement_enabled=True))
        # Warn about an SB3-calibrated reward bar here, before the stage's
        # budget is spent, rather than only at the gate check after it.
        jax_gate_thresholds(stage, stage_config)

    results: dict[int, Any] = {}

    params = None
    obs_stats = None
    for index, stage in enumerate(stages):
        stage_config = load_stage_config(species, stage)
        jax_kwargs = stage_config.get("jax_kwargs", {})
        env_kwargs = stage_config.get("env_kwargs", {})
        validate_jax_kwargs(jax_kwargs, source=f"{species} stage {stage} [jax]")

        # Pass previous stage's params AND observation-normalization stats to
        # the next stage — carrying only the weights would feed the policy
        # freshly re-scaled inputs it was never trained on (the SB3 path
        # carries obs_rms across stages for the same reason).
        if params is not None:
            train_kwargs["init_params"] = params
        if obs_stats is not None:
            # Decay the carried normalization count so the entered stage's
            # shifted obs distribution (near-zero velocity in balance →
            # sustained velocity in locomotion) re-anchors the stats within a
            # few updates — with the prior stage's count of millions,
            # update_running_stats is nearly a no-op exactly when the
            # distribution moves.  Same default (0.01) and per-stage override
            # (obs_rms_decay_on_resume, 1.0 disables) as the notebook's
            # resume cell, which honored this key while both library paths
            # silently dropped it.
            decay = float(jax_kwargs.get("obs_rms_decay_on_resume", 0.01))
            if decay != 1.0:
                from .jax_normalization import decay_running_stats

                obs_stats = decay_running_stats(obs_stats, decay_factor=decay)
                _logger.info("Stage %s init: obs normalization count decayed by %.4g", stage, decay)
            train_kwargs["init_obs_stats"] = obs_stats

        # Merge TOML [jax] and [env] sections into train_kwargs so that
        # stage-specific hyperparameters and reward weights reach train_jax.
        stage_train_kwargs = dict(train_kwargs)
        for toml_key, param_name in _JAX_KEY_MAP.items():
            if toml_key in jax_kwargs:
                stage_train_kwargs[param_name] = jax_kwargs[toml_key]

        # Always pass env_kwargs so reward weights reach the MJX env
        stage_train_kwargs["env_kwargs"] = env_kwargs

        # Override fall_penalty / reset noise from [jax] section if specified.
        # Use direct assignment — setdefault is a no-op when [env] already
        # defines the key, which silently ignores the JAX-specific override.
        if "fall_penalty" in jax_kwargs:
            env_kwargs["fall_penalty"] = jax_kwargs["fall_penalty"]
        for noise_key in ("reset_noise_scale", "init_qpos_noise", "init_yaw_noise"):
            if noise_key in jax_kwargs:
                env_kwargs[noise_key] = jax_kwargs[noise_key]

        result = train_fn(species=species, stage=stage, **stage_train_kwargs)
        if len(result) == 3:
            params, eval_metrics, obs_stats = result
        else:  # legacy 2-tuple train_fn
            params, eval_metrics = result
            obs_stats = None
        results[stage] = (params, eval_metrics)

        # Check gate (skip for last stage)
        if stage != stages[-1]:
            if not check_stage_gate(eval_metrics, stage_config):
                episode_return = eval_metrics.get("mean_episode_return")
                _logger.warning(
                    "Stage %s gate NOT passed (episode return=%s). Stopping early.",
                    stage,
                    # Never the per-step mean_reward: labelling it "episode
                    # return" in the one message a stopped run leaves behind
                    # is how a unit mismatch stays invisible.
                    f"{float(episode_return):.1f}" if episode_return is not None else "not reported",
                )
                break
            _logger.info("Stage %s gate passed. Advancing to stage %s.", stage, stages[index + 1])

    return results
