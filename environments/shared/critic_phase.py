"""The critic's episode phase: ``[ppo] critic_phase = true`` (off by default; no committed stage config sets it).

Why.  On the T. rex stance (``TyrannosaurusRexAnatomicalEnv``) one reward term of the committed stance is paid only
after the settle: ``reward_stance_width`` under ``stance_width_reference = "settled"``, while
``_step_count > stance_width_settle_steps`` (200).  The lateral-lean term has the same gate in code but weight 0 in
the committed stance; the never-merged r10 run branch on which this option was screened also paid stagger and
CoP-band terms on that gate (four in all).  The 63-value observation carries no clock, so on that branch the critic
read a slowly creeping input (the tail_3 pitch angle, observation 6) as one, and the policy gradient paid the actor
for raising the tail.  A CPU screen of 2026-10-31 gave the critic alone the episode phase and the tail climb
stopped.  A three-seed GPU round on the run branch's recipe (2026-11-01/02) read "consistent with the phase critic
removing the habit, not confirmatory" (its pre-registered test did not separate the arms).  Whether the committed
recipe forms the habit is not measured.  This module makes the phase a training option: the critic, and only the
critic, also reads three features of the env's own per-episode step counter ``k`` (``BaseDinoEnv._step_count`` when
the observation was made):

    ramp      min(k, S) / S
    settled   1 if k >= S else 0
    discount  gamma ** max(0, S - k)

``S`` is the env's ``stance_width_settle_steps`` (read from the env instances, never restated) and ``gamma`` the
stage's PPO discount: the screen's definition (S 200, gamma 0.98 on the stance), the same float arithmetic.  No
off-by-one: ``BaseDinoEnv.step`` increments ``_step_count`` before it builds the observation and the reward, so the
observation returned with ``_step_count == k`` is followed by the reward of step ``k + 1``, which the gates pay iff
``k + 1 > S``, i.e. ``k >= S`` -- ``settled``; the first gated reward is ``S - k`` steps after the next one, so its
weight in ``V(o_k)`` is ``gamma ** (S - k)`` -- ``discount``.  A reset observation has ``k = 0``.

How (the SB3 classes are in :mod:`environments.shared.critic_phase_ppo`).  The phase is a side input of the
critic, separate from the actor's observation: the policy interface -- the model's ``observation_space``, the
VecNormalize sidecar, every evaluation, probe, K3 panel, judge and deploy path -- stays the env's 63-wide
observation, and nothing reuses the three command slots (60-62), which the walking stages need live.  Only the
TRAINING ``VecNormalize`` returns ``[normalized observation, phase]`` (66 values) to PPO; PPO's rollout buffer is
3 columns wider; the actor's first layer reads columns 0-62 alone and the critic's first layer all 66.  The phase
is appended after normalisation and never normalised: it is bounded in [0, 1] by construction, and running
statistics would first see ``settled`` constant 0 (variance ~1e-8 until an episode passes step 200, which would
then normalise to the clip) and would keep moving the critic's input.  The screen froze those slots at mean
0 / variance 1, the same input up to the factor 1 / sqrt(1 + 1e-8).

This module is numpy only (the no-SB3 CI legs import it).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The ``[ppo]`` key (a boolean; absent or false: off).  Popped before the PPO constructor, recorded in
#: ``stage_config.json`` with the rest of the table, so ``hyperparameters_sha256`` covers it.
CRITIC_PHASE_KEY = "critic_phase"
#: The features, in column order after the actor's observation.
PHASE_FEATURES: tuple[str, ...] = ("ramp", "settled", "discount")
PHASE_WIDTH = len(PHASE_FEATURES)
#: The env attribute that holds the reward gates' settle step (``TyrannosaurusRexEnv``).
SETTLE_ATTRIBUTE = "stance_width_settle_steps"
#: The SB3 checkpoint attribute (JSON) that records the phase a checkpoint's critic reads.
CHECKPOINT_ATTRIBUTE = "mesozoic_critic_phase"
CHECKPOINT_SCHEMA = "mesozoic.critic-phase/v1"
#: The policy keyword arguments that carry the spec into the checkpoint's ``policy_kwargs``.
POLICY_KWARGS = ("critic_phase_settle_steps", "critic_phase_gamma")


class CriticPhaseError(ValueError):
    """The critic-phase option cannot be applied as asked (fail closed: nothing has trained)."""


def _step_counter(k: Any) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or int(k) < 0:
        raise CriticPhaseError(f"the step counter {k!r} is not a non-negative integer")
    return int(k)


def phase_features(k: int, settle_steps: int, gamma: float) -> tuple[float, float, float]:
    """``(ramp, settled, discount)`` at step counter *k*: the screen's definition, the same arithmetic."""
    k = _step_counter(k)
    return (
        min(k, settle_steps) / settle_steps,
        1.0 if k >= settle_steps else 0.0,
        gamma ** max(0, settle_steps - k),
    )


@dataclass(frozen=True)
class CriticPhaseSpec:
    """The phase a critic reads: the env's settle step ``settle_steps`` and the discount ``gamma``."""

    settle_steps: int
    gamma: float

    def __post_init__(self) -> None:
        settle = self.settle_steps
        if isinstance(settle, (bool, np.bool_)) or not isinstance(settle, (int, np.integer)) or int(settle) < 1:
            raise CriticPhaseError(f"the settle step must be a whole number of steps, at least 1, got {settle!r}")
        gamma = self.gamma
        if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, (int, float, np.floating)):
            raise CriticPhaseError(f"the phase discount must be a number, got {gamma!r}")
        if not (math.isfinite(float(gamma)) and 0.0 < float(gamma) <= 1.0):
            raise CriticPhaseError(f"the phase discount must lie in (0, 1], got {gamma!r}")
        object.__setattr__(self, "settle_steps", int(settle))
        object.__setattr__(self, "gamma", float(gamma))

    def features(self, k: int) -> tuple[float, float, float]:
        return phase_features(k, self.settle_steps, self.gamma)

    def matrix(self, ks: Any) -> np.ndarray:
        """``(n, 3)`` float64: :meth:`features` of each counter in *ks*."""
        flat = np.asarray(ks).reshape(-1)
        return np.array([self.features(k) for k in flat.tolist()], dtype=np.float64).reshape(len(flat), PHASE_WIDTH)

    def record(self) -> dict[str, Any]:
        """The checkpoint attribute :data:`CHECKPOINT_ATTRIBUTE` (JSON)."""
        return {
            "schema": CHECKPOINT_SCHEMA,
            "features": list(PHASE_FEATURES),
            "settle_steps": self.settle_steps,
            "gamma": self.gamma,
            "width": PHASE_WIDTH,
            "step_counter": "the env's _step_count when the observation was made (0 at reset)",
            "input": "the critic's only: appended after the normalized observation, never normalized",
        }

    def policy_kwargs(self) -> dict[str, Any]:
        return {"critic_phase_settle_steps": self.settle_steps, "critic_phase_gamma": self.gamma}

    @classmethod
    def from_record(cls, record: Any) -> CriticPhaseSpec:
        if not isinstance(record, Mapping) or record.get("schema") != CHECKPOINT_SCHEMA:
            raise CriticPhaseError(f"not a {CHECKPOINT_SCHEMA} record: {record!r}")
        if list(record.get("features", ())) != list(PHASE_FEATURES) or record.get("width") != PHASE_WIDTH:
            raise CriticPhaseError(f"the record's features {record.get('features')!r} are not {PHASE_FEATURES}")
        if "settle_steps" not in record or "gamma" not in record:
            raise CriticPhaseError(f"the record names no settle step or discount: {record!r}")
        return cls(settle_steps=record["settle_steps"], gamma=record["gamma"])


def read_critic_phase(ppo_kwargs: Mapping[str, Any] | None) -> bool:
    """The ``[ppo] critic_phase`` switch of a stage's PPO table (validated; absent: ``False``)."""
    if not ppo_kwargs or CRITIC_PHASE_KEY not in ppo_kwargs:
        return False
    value = ppo_kwargs[CRITIC_PHASE_KEY]
    if not isinstance(value, bool):
        raise CriticPhaseError(f"[ppo] {CRITIC_PHASE_KEY} must be true or false, got {value!r}")
    return value


def pop_critic_phase(ppo_kwargs: dict[str, Any]) -> bool:
    """Remove ``critic_phase`` from *ppo_kwargs* (in place; it is not a PPO constructor argument); the switch."""
    switch = read_critic_phase(ppo_kwargs)
    ppo_kwargs.pop(CRITIC_PHASE_KEY, None)
    return switch
