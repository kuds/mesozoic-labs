"""Versioned, fail-closed schema for curriculum advancement gates.

The gate plumbing used to fail *open* in three separate ways, so a stage
could advance on evidence nobody had checked:

1. :func:`~environments.shared.curriculum.manager.thresholds_from_configs`
   copied six known keys out of ``[curriculum]`` and silently dropped
   everything else.  A stage config carrying only new-style gate fields
   therefore produced *no* thresholds at all, and SB3 fell back to
   :class:`~environments.shared.curriculum.manager.StageThreshold`'s
   permissive defaults (``min_avg_reward = -inf``, length and success floors
   ``0``) — which advance on any evaluation whatsoever.
2. :func:`~environments.shared.jax_curriculum.check_stage_gate` logged a
   warning and returned ``True`` when ``min_avg_reward`` was absent.
3. Neither backend rejected an unrecognised key, so a typo in a threshold
   name disabled that threshold instead of failing.

Together those mean *removing* a gate key is indistinguishable from having
no gate, and the permissive reading always wins.  That is the wrong default
for a mechanism whose entire job is to decide whether a policy is good
enough to build the next stage on.

This module makes the gate explicit and versioned instead.  Every stage
config declares ``gate_schema_version`` and ``gate_kind``; unknown keys,
unknown kinds and unsupported versions are **fatal** whenever advancement is
enabled.  Running without a gate is still possible, but only by declaring
``gate_kind = "none/v1"``, which is recorded in the config and refuses to
advance rather than passing by default.

Adding a new gate kind means adding an entry to :data:`GATE_KINDS` and
teaching both backends to evaluate it — the schema deliberately will not let
one backend understand a gate the other silently ignores.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .stance_gate import STANCE_GATE_KIND

#: Bumped when the meaning of an existing key changes.  Adding a new gate
#: kind does not require a bump; changing how an existing one is evaluated
#: does.
GATE_SCHEMA_VERSION = 1

#: Threshold fields each gate kind consumes.  A stage config may carry only
#: the threshold keys belonging to the kind it declares, so a field left over
#: from a different gate is an error rather than dead config.
GATE_KINDS: dict[str, frozenset[str]] = {
    # The historical gate: a conjunction of per-evaluation means, repeated
    # ``required_consecutive`` times.  Named so a future capability gate can
    # coexist with it rather than silently replace it.
    "reward_and_length/v1": frozenset(
        {
            "min_avg_reward",
            "min_avg_episode_length",
            "min_avg_forward_vel",
            "min_success_rate",
            "min_eval_episodes",
            "required_consecutive",
        }
    ),
    # Stage 1a's stance-quality gate.  Judges physical stance -- did the body
    # reach the horizon, and did it keep both feet loaded once settled --
    # rather than return, because on stage 1 the zero-action statue is the
    # global optimum and no reward threshold separates it from a competent
    # policy (docs/STAGE1_SPLIT_PLAN.md 2.3.1).  ``min_avg_reward`` is carried
    # here too, but as a RAIL set well below the statue: its only job is
    # rejecting a policy that threw away most of the available return, and it
    # is deliberately not the gate.  See
    # :mod:`environments.shared.curriculum.stance_gate` for the statistic.
    STANCE_GATE_KIND: frozenset(
        {
            "min_full_horizon_fraction",
            "max_unsupported_duty",
            "max_unsupported_duty_ucb",
            "settle_steps",
            "min_avg_reward",
            "min_eval_episodes",
            "required_consecutive",
        }
    ),
    # An explicit, recorded non-advancing mode for pilots and diagnostics.
    # Declaring it is the ONLY supported way to run a stage with no gate, and
    # it refuses to advance rather than passing by default.
    "none/v1": frozenset(),
}

#: Threshold fields a gate kind cannot function without.  Declaring a kind and
#: omitting its required fields used to fail OPEN on the SB3 path: the schema
#: passed, ``thresholds_from_configs`` produced no fields, and
#: ``CurriculumManager`` fell back to ``StageThreshold``'s permissive defaults
#: (``min_avg_reward = -inf``) — advancing on any evaluation — while the JAX
#: path raised.  Requiring the core field here closes that hole and keeps the
#: two backends agreeing, which is the schema's whole job.
_REQUIRED_THRESHOLD_KEYS: dict[str, frozenset[str]] = {
    "reward_and_length/v1": frozenset({"min_avg_reward"}),
    # All three stance criteria are required.  The UCB in particular is the
    # one that certifies -- omitting it would leave the gate resting on raw
    # panel fractions, which is the low-power reading this kind exists to
    # replace.  ``min_avg_reward`` is NOT required: it is a rail, and a config
    # may legitimately decline to set one.
    STANCE_GATE_KIND: frozenset(
        {
            "min_full_horizon_fraction",
            "max_unsupported_duty",
            "max_unsupported_duty_ucb",
        }
    ),
    "none/v1": frozenset(),
}

# A gate kind with no required-keys declaration would fail with a bare
# KeyError mid-validation; make the omission unmissable at import instead.
# Deliberately a raise, not an assert: `python -O` strips asserts, and a
# fail-closed guarantee that disappears under an optimisation flag is the
# same class of defect as the gates this module exists to harden.
if set(_REQUIRED_THRESHOLD_KEYS) != set(GATE_KINDS):
    raise RuntimeError(
        "every gate kind must declare its required threshold fields; "
        f"missing from _REQUIRED_THRESHOLD_KEYS: {sorted(set(GATE_KINDS) - set(_REQUIRED_THRESHOLD_KEYS))}"
    )

#: Keys that configure the stage's schedule rather than its gate.
_SCHEDULE_KEYS = frozenset(
    {
        "timesteps",
        "ramp_timesteps",
        "ramp_start_value",
        "warmup_timesteps",
        "warmup_clip_range",
        "warmup_ent_coef",
    }
)

#: Keys that configure collapse/early-stop detection rather than the gate.
_COLLAPSE_KEYS = frozenset(
    {
        "collapse_min_evals",
        "collapse_patience",
        "collapse_drop_fraction",
        "collapse_peak_floor",
        "collapse_peak_floor_fraction",
        "collapse_peak_floor_reference",
        "collapse_peak_warmup_timesteps",
        "collapse_smoothing_window",
    }
)

#: Keys that configure diagnostics and reporting rather than the gate. Each
#: is already read with a default somewhere in the trainer, which means each
#: was *intended* to be settable from a TOML -- but the fail-closed check
#: below rejects any key it does not know, so setting one was fatal. Same
#: trap ``max_checkpoints`` hit; registering them is what makes them reachable.
_DIAGNOSTIC_KEYS = frozenset(
    {
        "diagnostics_plateau_window",
        "diagnostics_plateau_min_relative_variation",
        "supplementary_episodes",
        "stance_report_episodes",
        "baseline_warn_after_budget_fraction",
        "stance_probe_filter_hz",
        "stance_probe_hold_constant",
        "stance_probe_release_ablation",
        "stance_probe_impulse_speeds",
    }
)

#: Keys that configure artifact retention rather than the gate.  Without an
#: entry here the fail-closed unknown-key check below would reject any TOML
#: that set one, which would make the setting unreachable from config —
#: the only place it is meant to be set.
_RETENTION_KEYS = frozenset({"max_checkpoints"})

#: Keys that record measurement provenance rather than configure anything.
#: ``statue_constants_physics_revision`` pins the plant revision the stage's
#: statue-derived constants (``min_avg_reward``,
#: ``collapse_peak_floor_reference``) were measured on;
#: test_statue_constant_freshness.py cross-checks it against the plant
#: manifest so a plant bump that forgets the re-measurement fails CI.  No
#: trainer reads it.
_PROVENANCE_KEYS = frozenset({"statue_constants_physics_revision"})

#: The schema's own declaration keys.
_SCHEMA_KEYS = frozenset({"gate_schema_version", "gate_kind"})

#: Every threshold key any gate kind knows about, used to tell "belongs to a
#: different gate kind" apart from "not a gate key at all".
_ALL_THRESHOLD_KEYS = frozenset().union(*GATE_KINDS.values())


def finite_gate_metric(value: Any) -> float | None:
    """The metric as a float, or ``None`` when it was not measured.

    Lives here, at the bottom of the gate stack, because both backends need
    the same answer and both got it wrong in the same way.  A threshold
    comparison against an unmeasured metric is the archetypal fail-open:
    ``nan < threshold`` is ``False``, so an unfiltered NaN *clears* every
    floor beneath it, and a gate that could not measure anything reports a
    pass.  Measured on both paths before this guard existed --
    ``min_avg_reward = 1950`` against a NaN reward returned ``(True, [])``
    from ``reporting.gates`` and from ``jax_eval.check_stage_gate`` alike.

    ``None`` and ``""`` are the two "not measured" sentinels the trainers
    actually write, and non-finite values join them, so every caller can
    treat one return of ``None`` as "this criterion is unproven" and fail.

    The empty-string test is ``isinstance`` rather than ``value == ""``
    because the latter is an elementwise comparison against a numpy array,
    and ``if array_of_bools`` then raises "truth value ... is ambiguous".
    A gate helper that can raise on its input is a gate that can take a
    stage's artifacts down with it, so the comparison is one that cannot.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class GateSchemaError(ValueError):
    """A stage config's advancement gate is missing, unknown, or malformed."""


def _describe(stage: int | str) -> str:
    return f"stage {stage}"


def validate_gate_config(
    stage: int | str,
    curriculum_kwargs: Mapping[str, Any],
    *,
    advancement_enabled: bool = True,
) -> str:
    """Validate one stage's ``[curriculum]`` block and return its gate kind.

    Args:
        stage: Stage identifier, used only in error messages.
        curriculum_kwargs: The stage config's ``curriculum_kwargs`` mapping.
        advancement_enabled: Whether this run may advance between stages.
            When ``False`` a missing gate declaration is tolerated, because a
            single-stage pilot has nothing to advance to; every other defect
            is still fatal, since a malformed gate is a bug either way.

    Returns:
        The declared gate kind, or ``"none/v1"`` when advancement is disabled
        and no kind was declared.

    Raises:
        GateSchemaError: If the block carries an unknown key, declares an
            unknown gate kind or schema version, carries a threshold field
            that its declared kind does not consume, or omits the gate
            declaration entirely while advancement is enabled.
    """
    known = (
        _SCHEMA_KEYS
        | _SCHEDULE_KEYS
        | _COLLAPSE_KEYS
        | _RETENTION_KEYS
        | _DIAGNOSTIC_KEYS
        | _PROVENANCE_KEYS
        | _ALL_THRESHOLD_KEYS
    )
    unknown = sorted(set(curriculum_kwargs) - known)
    if unknown:
        raise GateSchemaError(
            f"{_describe(stage)}: unrecognised [curriculum] key(s) {unknown}. "
            "Unknown keys are rejected rather than ignored, because silently "
            "dropping a misspelled threshold disables it. Add the key to "
            "environments/shared/curriculum/gate_schema.py if it is real."
        )

    declared_kind = curriculum_kwargs.get("gate_kind")
    declared_version = curriculum_kwargs.get("gate_schema_version")

    if declared_kind is None:
        if advancement_enabled:
            raise GateSchemaError(
                f"{_describe(stage)}: no gate_kind declared, but advancement is "
                "enabled. A stage with no gate must say so explicitly with "
                'gate_kind = "none/v1", which refuses to advance; it must not '
                "advance by default. See docs/STAGE1_SPLIT_PLAN.md §5.2."
            )
        return "none/v1"

    if declared_kind not in GATE_KINDS:
        raise GateSchemaError(
            f"{_describe(stage)}: unknown gate_kind {declared_kind!r}. "
            f"Known kinds: {sorted(GATE_KINDS)}. An unknown kind is fatal so a "
            "backend cannot silently ignore a gate the other backend enforces."
        )

    if declared_version is None:
        raise GateSchemaError(
            f"{_describe(stage)}: gate_kind is declared but gate_schema_version "
            "is missing; both are required so a config cannot be read under a "
            "schema it was not written for."
        )
    if declared_version != GATE_SCHEMA_VERSION:
        raise GateSchemaError(
            f"{_describe(stage)}: gate_schema_version {declared_version!r} is not "
            f"supported (this build understands {GATE_SCHEMA_VERSION})."
        )

    allowed = GATE_KINDS[declared_kind]
    misplaced = sorted((set(curriculum_kwargs) & _ALL_THRESHOLD_KEYS) - allowed)
    if misplaced:
        raise GateSchemaError(
            f"{_describe(stage)}: gate_kind {declared_kind!r} does not consume "
            f"threshold field(s) {misplaced}. Leaving them in the config implies "
            "a gate that is not actually enforced."
        )

    missing = sorted(_REQUIRED_THRESHOLD_KEYS[declared_kind] - set(curriculum_kwargs))
    if missing:
        raise GateSchemaError(
            f"{_describe(stage)}: gate_kind {declared_kind!r} is missing required "
            f"threshold field(s) {missing}. Declaring a gate without them fails "
            "open on the SB3 path (StageThreshold defaults to min_avg_reward = "
            '-inf) while the JAX path rejects it; declare gate_kind = "none/v1" '
            "for a non-advancing pilot instead."
        )

    if advancement_enabled and declared_kind == "none/v1":
        raise GateSchemaError(
            f'{_describe(stage)}: gate_kind "none/v1" declares a non-advancing '
            "pilot, so it cannot be used in a run that advances between stages."
        )

    return str(declared_kind)


def validate_gate_configs(
    configs: Mapping[int, Mapping[str, Any]],
    *,
    advancement_enabled: bool = True,
) -> dict[int, str]:
    """Validate every stage config, returning each stage's gate kind."""
    return {
        stage: validate_gate_config(
            stage,
            cfg.get("curriculum_kwargs", {}),
            advancement_enabled=advancement_enabled,
        )
        for stage, cfg in configs.items()
    }
