"""Probe the policy at every PPO update on a fixed bank of states: report-only.

TensorBoard's ``train/std`` is the mean exploration noise over the
actuators; the r9 arm L runs hid a 9-13x spread between them behind it.
Nothing recorded what the policy *does* at a known posture as training
goes on, nor what its critic thinks of a lean.  :class:`PolicyProbeCallback`
records, at the end of every rollout (the policy that collected it, before
``train()`` updates it), for a fixed bank of states built once per session:

* the per-actuator exploration std, ``exp(log_std)``;
* VecNormalize's per-dimension statistics, the entropy coefficient and the
  learning rate the coming ``train()`` applies;
* the critic's value and the deterministic mean action at every bank state,
  under the current normalization and under the one frozen at the session's
  start (so a value change can be split into the policy's and the
  normalizer's share).  On a stage that starts with a fresh VecNormalize the
  frozen statistics are the initial reset observations' alone, a
  near-degenerate reference (the docs say how to read it);
* per actuator, over the rollout buffer that update trains on: the share
  of the sampled actions above the action space's high bound and below its
  low one (the ones ``env.step`` clipped; the buffer keeps them, and their
  log-probabilities, unclipped), the share at ``|clip(a)| >=
  SATURATION_ABS`` (the gate's threshold), and, from the policy's mean
  action at every buffer observation, the share of means beyond each bound
  and the means' average, median and 95th percentile
  (:func:`rollout_action_shares`).  Deterministic play acts on the mean
  clipped, training on samples around it; the two can sit on opposite
  sides of a bound.

The bank (:data:`BANK_SPEC`) is ten states of the stage's own env, on two
reset seeds: the settled statue (step 199) and the statue at step 599,
mirror-symmetric ankle leans of +0.03 and -0.03 (action units, ramped over
steps 200-300 and held) at step 599, and a -0.05 lean at step 400.  It is
built on a separate env instance that nothing else steps, with its own
horizon long enough for every row (:func:`bank_horizon`) and any scheduled
pushes off (:func:`bank_env_kwargs`), under
:class:`GlobalRngGuard`, which snapshots and restores the global generators
(the build draws none; the guard is a backstop and warns if one moved).

Per update the probe only reads parameters and the rollout buffer and runs
``torch.no_grad()`` forward passes of the constant bank and of the buffer's
observations (float32, in chunks of :data:`ROLLOUT_CHUNK_ROWS`): no
sampling, no gradient, no parameter, optimizer, buffer or ``training`` flag
touched, and it never calls ``predict()`` (which sets the training mode).
The normalization is VecNormalize's own formula in numpy.  Without a
rollout buffer (SAC) there is no bank and no row.

``<stage_dir>/policy_probe.npz`` is rewritten atomically on the checkpoint
cadence and at training end; a same-stage resume keeps the prior session's
rows at or before the resumed step when its bank equals this session's.
TensorBoard gets ``policy_probe/*`` (never the console) and
``progress.json`` the latest row (:meth:`PolicyProbeCallback.latest`).
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .curriculum import sb3_compat
from .curriculum.sb3_compat import BaseCallback
from .model_id import ModelId

logger = logging.getLogger(__name__)

#: The fixed state bank: ``(label, reset seed, mirror-even ankle offset in action units, kept step)``.  The offset
#: ``x`` is applied as ``left_sign * x`` and ``right_sign * x`` to every sagittal ankle pair, ramped linearly over
#: steps 200-300 and then held; the kept step is the step whose post-step observation is kept (0-based).
BANK_SPEC: tuple[tuple[str, int, float, int], ...] = (
    ("statue_settled", 3042, 0.0, 199),
    ("statue", 3042, 0.0, 599),
    ("ankle_plus_0.03", 3042, +0.03, 599),
    ("ankle_minus_0.03", 3042, -0.03, 599),
    ("ankle_minus_0.05_at400", 3042, -0.05, 400),
    ("statue_settled", 3043, 0.0, 199),
    ("statue", 3043, 0.0, 599),
    ("ankle_plus_0.03", 3043, +0.03, 599),
    ("ankle_minus_0.03", 3043, -0.03, 599),
    ("ankle_minus_0.05_at400", 3043, -0.05, 400),
)
#: The ramp of the ankle offset: 0 up to this step, 1 from :data:`RAMP_END` on.
RAMP_START = 200
RAMP_END = 300
#: The file the callback writes into the stage directory.
POLICY_PROBE_FILENAME = "policy_probe.npz"
#: The bank rows the value differences and the gain compare (``value_<x>_minus_statue``: ``(tag, label)``).
_STATUE = "statue"
_LEANS: tuple[tuple[str, str], ...] = (
    ("value_ankle_plus_minus_statue", "ankle_plus_0.03"),
    ("value_ankle_minus_minus_statue", "ankle_minus_0.03"),
    ("value_ankle_minus_0.05_minus_statue", "ankle_minus_0.05_at400"),
)
_PLUS, _MINUS = "ankle_plus_0.03", "ankle_minus_0.03"
#: The per-update arrays of the file, aligned with ``update_timesteps``.
_UPDATE_KEYS: tuple[str, ...] = (
    "update_timesteps",
    "std_per_actuator",
    "obs_rms_mean",
    "obs_rms_var",
    "obs_rms_count",
    "ret_rms_var",
    "ent_coef",
    "learning_rate",
    "bank_value_now",
    "bank_value_frozen",
    "bank_action_now",
    "bank_action_frozen",
    "frozen_reference_timesteps",
)
#: The per-update arrays read from the rollout buffer (:func:`rollout_action_shares`), aligned with
#: ``update_timesteps``; a prior session's file written before them reads NaN (and 0 rows) on its rows.
_ROLLOUT_KEYS: tuple[str, ...] = (
    "rollout_rows",
    "rollout_action_clip_high_share",
    "rollout_action_clip_low_share",
    "rollout_action_saturated_share",
    "rollout_mu_beyond_high_share",
    "rollout_mu_beyond_low_share",
    "rollout_mu_mean",
    "rollout_mu_q50",
    "rollout_mu_q95",
)
#: The buffer's observations go through the policy this many rows at a time (the T. rex stance's 8,192 in one).
ROLLOUT_CHUNK_ROWS = 16_384
#: The bank's description in the file: a prior session's rows are kept only when it is equal.
_BANK_KEYS: tuple[str, ...] = ("bank_labels", "bank_seeds", "bank_steps", "bank_ankle_offset", "bank_raw_obs")

NAN = float("nan")


class StateBank(NamedTuple):
    """The fixed states the policy is probed at (:func:`build_state_bank`)."""

    labels: tuple[str, ...]
    seeds: np.ndarray  # (B,) int64
    steps: np.ndarray  # (B,) int64, the kept step
    ankle_offset: np.ndarray  # (B,) float64, the mirror-even ankle offset in action units
    raw_obs: np.ndarray  # (B, obs_dim) float64, the env's own observation (before VecNormalize)
    com_fore_aft_m: np.ndarray  # (B,) the CoM ahead of the mean foot reference site along the heading
    ankle_actuators: np.ndarray  # (P, 2) int64, each ankle pair's (left, right) actuator
    ankle_signs: np.ndarray  # (P, 2) float64, each pair's leftward-axis signs
    horizon: int  # the bank env's max_episode_steps
    build_seconds: float  # the build's wall time (descriptive only)


def bank_horizon(stage_horizon: int, spec: Sequence[tuple[str, int, float, int]] = BANK_SPEC) -> int:
    """The bank env's horizon: the stage's, or long enough that no kept step is the truncation step."""
    return max(int(stage_horizon), max(int(step) for *_rest, step in spec) + 2)


#: The env keyword that switches a stage's scheduled pushes on (the recovery stages'); 0 is off.
PUSH_MULTIPLE_KWARG = "perturbation_capture_velocity_multiple"


def bank_env_kwargs(env_kwargs: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    """The bank env's keywords: the stage's own, at *horizon*, with any scheduled pushes off.

    A recovery stage pushes the animal on a schedule, and a push topples
    the statue and the leans before their kept steps: on the T. rex
    recovery stage 6 of the 10 rows were lost, every statue row among them.
    The bank is the unpushed statue and held leans on every stage.  With
    the multiple at 0 the env runs no push code and its reset draws exactly
    what a push-free reset of the same seed draws (``base_env``); the
    observation does not depend on it.  The key is set only where the stage
    declares it, so an env class without it is never handed it.
    """
    kwargs = {**dict(env_kwargs), "max_episode_steps": int(horizon)}
    if PUSH_MULTIPLE_KWARG in kwargs:
        kwargs[PUSH_MULTIPLE_KWARG] = 0.0
    return kwargs


def ankle_ramp(step: int) -> float:
    """The fraction of the ankle offset applied at step *step*: 0 before 200, linear to 1 at 300, then held."""
    return min(max((step - RAMP_START) / float(RAMP_END - RAMP_START), 0.0), 1.0)


def _com_fore_aft(data: Any, root: int, sites: np.ndarray | None) -> float:
    """``f . (subtree_com[root] - mean foot reference site)``, horizontal, ``f`` the heading's forward vector."""
    if sites is None or not len(sites):
        return NAN
    from .gait.balance_telemetry import heading_frames

    forward, _pitch = heading_frames(np.asarray(data.xquat[root], dtype=np.float64)[None])
    offset = np.asarray(data.subtree_com[root], dtype=np.float64) - np.asarray(data.site_xpos[sites]).mean(axis=0)
    return float(forward[0] @ offset[:2])


def build_state_bank(
    env_factory: Callable[[int], Any],
    model_id: ModelId,
    spec: Sequence[tuple[str, int, float, int]] = BANK_SPEC,
    *,
    stage_horizon: int,
) -> StateBank:
    """Roll the bank's states on ``env_factory(bank_horizon(stage_horizon))``, a fresh env nothing else steps.

    One ``reset(seed=...)`` per (seed, offset) group, zero action elsewhere,
    and the offset on every sagittal ankle pair (``sagittal_leg_dc_pairs``
    whose key ends in ``_ankle_deg``) as ``sign * x``: mirror-symmetric on
    every model, whatever its axes.  The rows of one group share the episode
    (the same reset and actions give the same states).  A row whose episode
    ends at or before its kept step is dropped, with one warning naming every
    dropped row (the scalars that compare with it then read NaN).  A model
    with no ankle pair keeps its statue rows only.  The env is closed
    afterwards.
    """
    from .eval_diagnostics import SagittalLegPair, sagittal_leg_dc_pairs
    from .gait.morphology import Morphology

    started = time.perf_counter()  # descriptive only: recorded, never used to decide anything
    horizon = bank_horizon(stage_horizon, spec)
    env = env_factory(horizon)
    try:
        unwrapped = getattr(env, "unwrapped", env)
        model = unwrapped.model
        root: int = 0
        sites: np.ndarray | None = None
        pairs: tuple[SagittalLegPair, ...] = ()
        try:
            morphology = Morphology.from_env(unwrapped, model_id)
            root = int(morphology.root_body)
            sites = np.array([foot.primary_site for foot in morphology.feet], dtype=np.int64)
            pairs = tuple(
                pair for pair in sagittal_leg_dc_pairs(unwrapped, morphology) if pair.key.endswith("_ankle_deg")
            )
        except Exception:  # noqa: BLE001 - a report must not sink a run
            logger.warning("Policy probe: could not describe %s's feet; the bank keeps its statue rows.", model_id)
        space = env.action_space
        low = np.asarray(space.low, dtype=np.float64)
        high = np.asarray(space.high, dtype=np.float64)
        groups: dict[tuple[int, float], list[int]] = {}
        for index, (_label, seed, offset, _step) in enumerate(spec):
            if offset != 0.0 and not pairs:
                logger.info("Policy probe: %s has no sagittal ankle pair; bank row %r left out.", model_id, _label)
                continue
            groups.setdefault((int(seed), float(offset)), []).append(index)
        kept: dict[int, tuple[np.ndarray, float]] = {}
        dropped: list[str] = []
        for (seed, offset), members in groups.items():
            wanted: dict[int, list[int]] = {int(spec[index][3]): [] for index in members}
            for index in members:
                wanted[int(spec[index][3])].append(index)
            env.reset(seed=seed)
            for step in range(max(wanted) + 1):
                action = np.zeros(int(model.nu), dtype=np.float64)
                lean = offset * ankle_ramp(step)
                for pair in pairs:
                    action[pair.left] = pair.left_sign * lean
                    action[pair.right] = pair.right_sign * lean
                obs, _reward, terminated, truncated, _info = env.step(np.clip(action, low, high).astype(space.dtype))
                if terminated or truncated:
                    dropped += [
                        f"{spec[index][0]!r} (seed {seed}, ended at step {step})"
                        for kept_step, rows in wanted.items()
                        if kept_step >= step
                        for index in rows
                    ]
                    break
                for index in wanted.get(step, ()):
                    kept[index] = (
                        np.asarray(obs, dtype=np.float64).reshape(-1).copy(),
                        _com_fore_aft(unwrapped.data, root, sites),
                    )
        if dropped:
            logger.warning(
                "Policy probe: %d of %d bank rows dropped, their episodes ending at or before the kept step: %s. "
                "The probe's scalars that compare with a dropped row read NaN.",
                len(dropped),
                len(spec),
                "; ".join(dropped),
            )
        order = [index for index in range(len(spec)) if index in kept]
        if not order:
            raise RuntimeError("no bank state could be built")
        return StateBank(
            labels=tuple(str(spec[index][0]) for index in order),
            seeds=np.array([int(spec[index][1]) for index in order], dtype=np.int64),
            steps=np.array([int(spec[index][3]) for index in order], dtype=np.int64),
            ankle_offset=np.array([float(spec[index][2]) for index in order], dtype=np.float64),
            raw_obs=np.stack([kept[index][0] for index in order]),
            com_fore_aft_m=np.array([kept[index][1] for index in order], dtype=np.float64),
            ankle_actuators=np.array([(pair.left, pair.right) for pair in pairs], dtype=np.int64).reshape(-1, 2),
            ankle_signs=np.array([(pair.left_sign, pair.right_sign) for pair in pairs], dtype=np.float64).reshape(
                -1, 2
            ),
            horizon=horizon,
            build_seconds=time.perf_counter() - started,
        )
    finally:
        env.close()


def _same_numpy_state(first: Any, second: Any) -> bool:
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and tuple(first[2:]) == tuple(second[2:])


class GlobalRngGuard:
    """Snapshot the global generators on entry and restore them bit for bit on exit; ``changed`` says if any moved.

    ``numpy.random``'s legacy global state, ``random``'s, torch's CPU
    generator and, once CUDA is initialised, every CUDA device's.
    """

    def __init__(self) -> None:
        self.changed = False

    def __enter__(self) -> GlobalRngGuard:
        self._numpy = np.random.get_state()
        self._python = random.getstate()
        self._torch: Any = None
        self._cuda: Any = None
        try:
            import torch
        except ImportError:
            return self
        self._torch = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            self._cuda = torch.cuda.get_rng_state_all()
        return self

    def __exit__(self, *exc: object) -> None:
        changed = not _same_numpy_state(np.random.get_state(), self._numpy) or random.getstate() != self._python
        np.random.set_state(self._numpy)
        random.setstate(self._python)
        if self._torch is not None:
            import torch

            changed = changed or not torch.equal(torch.get_rng_state(), self._torch)
            torch.set_rng_state(self._torch)
            if self._cuda is not None:
                now = torch.cuda.get_rng_state_all()
                changed = changed or any(not torch.equal(a, b) for a, b in zip(now, self._cuda))
                torch.cuda.set_rng_state_all(self._cuda)
        self.changed = changed


def _mirror_even(actions: np.ndarray, pairs: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """(B, nu) -> (B,): ``(sL a_L + sR a_R) / 2`` averaged over the ankle pairs; NaN without one."""
    if not len(pairs):
        return np.full(actions.shape[0], np.nan)
    values = (signs[:, 0] * actions[:, pairs[:, 0]] + signs[:, 1] * actions[:, pairs[:, 1]]) / 2.0
    result: np.ndarray = values.mean(axis=1)
    return result


def _nanmean(values: Sequence[float] | np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else NAN


def rollout_action_shares(actions: np.ndarray, means: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    """The ``rollout_*`` arrays of one row from (N, nu) sampled actions and the policy's means at the same steps.

    Per actuator: the share of the samples above *high* and below *low*
    (``env.step`` clips them, as SB3 does, with ``np.clip``), the share at
    ``|clip(a)| >= SATURATION_ABS``, the share of the means above *high* and
    below *low*, and the means' average, median and 95th percentile; NaN
    with no row.  ``rollout_rows`` is N.
    """
    from .gait.constants import SATURATION_ABS

    actions = np.asarray(actions)
    means = np.asarray(means, dtype=np.float64)
    rows, nu = actions.shape
    if not rows:
        empty = np.full(nu, np.nan, dtype=np.float32)
        return {"rollout_rows": np.int64(0), **{key: empty.copy() for key in _ROLLOUT_KEYS[1:]}}
    clipped = np.clip(actions, low, high)

    def share(mask: np.ndarray) -> np.ndarray:
        return np.asarray(mask.mean(axis=0), dtype=np.float32)

    q50, q95 = np.quantile(means, (0.5, 0.95), axis=0)
    return {
        "rollout_rows": np.int64(rows),
        "rollout_action_clip_high_share": share(actions > high),
        "rollout_action_clip_low_share": share(actions < low),
        "rollout_action_saturated_share": share(np.abs(clipped) >= SATURATION_ABS),
        "rollout_mu_beyond_high_share": share(means > high),
        "rollout_mu_beyond_low_share": share(means < low),
        "rollout_mu_mean": np.asarray(means.mean(axis=0), dtype=np.float32),
        "rollout_mu_q50": np.asarray(q50, dtype=np.float32),
        "rollout_mu_q95": np.asarray(q95, dtype=np.float32),
    }


def _max_share(row: Mapping[str, Any], *keys: str) -> float:
    """The largest per-actuator sum of *keys*' shares in *row*; NaN when a key is absent or nothing is finite."""
    if any(key not in row for key in keys):
        return NAN
    total = sum(np.asarray(row[key], dtype=np.float64) for key in keys)
    finite = np.asarray(total)[np.isfinite(total)]
    return float(finite.max()) if finite.size else NAN


def probe_scalars(bank: StateBank, row: Mapping[str, Any]) -> dict[str, float]:
    """The TensorBoard ``policy_probe/*`` scalars of one row (NaN where the bank lacks a row they compare)."""
    std = np.asarray(row["std_per_actuator"], dtype=np.float64)
    finite_std = std[np.isfinite(std)]
    var = np.asarray(row["obs_rms_var"], dtype=np.float64)
    finite_var = var[np.isfinite(var)]
    now = np.asarray(row["bank_value_now"], dtype=np.float64)
    frozen = np.asarray(row["bank_value_frozen"], dtype=np.float64)
    ankle = _mirror_even(np.asarray(row["bank_action_now"], dtype=np.float64), bank.ankle_actuators, bank.ankle_signs)
    where = {(label, int(seed)): index for index, (label, seed) in enumerate(zip(bank.labels, bank.seeds))}
    seeds = sorted({int(seed) for seed in bank.seeds})
    statue = [index for index, label in enumerate(bank.labels) if label == _STATUE]
    out = {
        "std_min": float(finite_std.min()) if finite_std.size else NAN,
        "std_max": float(finite_std.max()) if finite_std.size else NAN,
        "std_max_over_min": float(finite_std.max() / finite_std.min())
        if finite_std.size and finite_std.min() > 0
        else NAN,
        "ent_coef": float(row["ent_coef"]),
        "obs_var_min": float(finite_var.min()) if finite_var.size else NAN,
        "obs_var_max": float(finite_var.max()) if finite_var.size else NAN,
        "value_statue": _nanmean(now[statue]) if statue else NAN,
    }
    for tag, label in _LEANS:
        out[tag] = _nanmean(
            [
                now[where[(label, seed)]] - now[where[(_STATUE, seed)]]
                for seed in seeds
                if (label, seed) in where and (_STATUE, seed) in where
            ]
        )
    out["value_normalization_shift_statue"] = _nanmean(now[statue] - frozen[statue]) if statue else NAN
    out["statue_ankle_action_mean"] = _nanmean(ankle[statue]) if statue else NAN
    gains = []
    for seed in seeds:
        if (_PLUS, seed) in where and (_MINUS, seed) in where:
            plus, minus = where[(_PLUS, seed)], where[(_MINUS, seed)]
            lever = bank.com_fore_aft_m[plus] - bank.com_fore_aft_m[minus]
            if math.isfinite(lever) and lever != 0.0:
                gains.append((ankle[plus] - ankle[minus]) / lever)
    out["ankle_gain_per_m"] = _nanmean(gains)
    # The rollout's worst actuator, both bounds together.
    out["rollout_action_clip_share_max"] = _max_share(
        row, "rollout_action_clip_high_share", "rollout_action_clip_low_share"
    )
    out["rollout_action_saturated_share_max"] = _max_share(row, "rollout_action_saturated_share")
    out["rollout_mu_beyond_share_max"] = _max_share(row, "rollout_mu_beyond_high_share", "rollout_mu_beyond_low_share")
    return out


class PolicyProbeCallback(BaseCallback):  # type: ignore[misc]
    """Write ``policy_probe.npz``: one row per PPO update on a fixed state bank (see the module docstring).

    Args:
        model_id: The model trained (its feet and ankle pairs, for the bank).
        env_factory: ``env_factory(horizon)`` builds a bare env of the stage
            (the stage's ``env_kwargs`` with ``max_episode_steps = horizon``),
            never a Monitor and never ``make_env`` (which seeds the global
            generators).
        stage_horizon: The stage's ``max_episode_steps``.
        probe_dir: The stage directory, where the file goes (``None``: kept in memory).
        flush_every_calls: The ``CheckpointCallback``'s ``save_freq`` (already
            divided by ``n_envs``): the file is rewritten on the calls a
            checkpoint is saved on, and at training end.
        spec: The bank (:data:`BANK_SPEC`).
    """

    def __init__(
        self,
        *,
        model_id: ModelId,
        env_factory: Callable[[int], Any],
        stage_horizon: int,
        probe_dir: str | Path | None,
        flush_every_calls: int,
        spec: Sequence[tuple[str, int, float, int]] = BANK_SPEC,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PolicyProbeCallback.")
        super().__init__(verbose)
        if int(flush_every_calls) < 1:
            raise ValueError("flush_every_calls must be at least 1")
        self.model_id = model_id
        self.env_factory = env_factory
        self.stage_horizon = int(stage_horizon)
        self.path = None if probe_dir is None else Path(probe_dir) / POLICY_PROBE_FILENAME
        self.flush_every_calls = int(flush_every_calls)
        self.spec = tuple(spec)
        self.bank: StateBank | None = None
        self._rows: list[dict[str, Any]] = []
        self._prior_rows: list[dict[str, Any]] = []
        self._prior_bank: dict[str, np.ndarray] | None = None
        self._frozen: tuple[np.ndarray, np.ndarray, int] | None = None
        self._latest: dict[str, float] = {}
        self._dirty = False
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str, *, exc_info: bool = False) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, exc_info=exc_info)

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Every row the file holds, the prior session's kept ones first."""
        return [*self._prior_rows, *self._rows]

    # ------------------------------------------------------------------ the bank
    def _on_training_start(self) -> None:
        if getattr(self.model, "rollout_buffer", None) is None:
            logger.info(
                "Policy probe: %s has no rollout buffer (not PPO); no bank, no rows.", type(self.model).__name__
            )
            return
        try:
            with GlobalRngGuard() as guard:
                bank = build_state_bank(self.env_factory, self.model_id, self.spec, stage_horizon=self.stage_horizon)
            if guard.changed:
                self._warn_once(
                    "rng",
                    "Policy probe: building the state bank moved a global random generator; it was restored.",
                )
            width = int(np.prod(self.model.observation_space.shape or ()))
            if bank.raw_obs.shape[1] != width:
                raise ValueError(
                    f"the bank's observations have {bank.raw_obs.shape[1]} dimensions, the policy's {width}"
                )
        except Exception:  # noqa: BLE001 - a report must not sink a run
            self._warn_once(
                "bank", "Policy probe: could not build the state bank; no rows this session.", exc_info=True
            )
            return
        self.bank = bank
        logger.info(
            "Policy probe: a bank of %d states (%s) at horizon %d, built in %.1f s.",
            len(bank.labels),
            ", ".join(f"{label}/{seed}" for label, seed in zip(bank.labels, bank.seeds)),
            bank.horizon,
            bank.build_seconds,
        )
        self._adopt_prior_rows()
        vec_normalize = self.model.get_vec_normalize_env()
        stats = self._obs_stats(vec_normalize)
        if stats is not None:
            mean, var, _epsilon, _clip = stats
            self._frozen = (np.array(mean, dtype=np.float64), np.array(var, dtype=np.float64), int(self.num_timesteps))

    def _adopt_prior_rows(self) -> None:
        """The deferred check of :meth:`seed_prior_record`: keep the held rows when the prior bank is this one."""
        prior, self._prior_bank = self._prior_bank, None
        if prior is None or self.bank is None:
            return
        bank = self.bank
        same = (
            [str(label) for label in prior["bank_labels"]] == list(bank.labels)
            and np.array_equal(prior["bank_seeds"], bank.seeds)
            and np.array_equal(prior["bank_steps"], bank.steps)
            and np.array_equal(prior["bank_ankle_offset"], bank.ankle_offset)
            and np.array_equal(prior["bank_raw_obs"], bank.raw_obs)
        )
        if same and self._prior_rows:
            first = self._prior_rows[0]
            nu = int(np.prod(self.model.action_space.shape or ()))
            same = np.asarray(first["std_per_actuator"]).shape == (nu,) and np.asarray(
                first["bank_value_now"]
            ).shape == (len(bank.labels),)
        if not same:
            if self._prior_rows:
                self._warn_once(
                    "prior",
                    f"Policy probe: the prior session's state bank differs from this one's; its "
                    f"{len(self._prior_rows)} rows are left out of {POLICY_PROBE_FILENAME}.",
                )
            self._prior_rows = []

    # ------------------------------------------------------------------ one row per update
    @staticmethod
    def _obs_stats(vec_normalize: Any) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """``(mean, var, epsilon, clip_obs)`` of a VecNormalize that normalizes a flat observation, else ``None``."""
        if vec_normalize is None or not getattr(vec_normalize, "norm_obs", False):
            return None
        rms = getattr(vec_normalize, "obs_rms", None)
        if rms is None or isinstance(rms, dict):
            return None
        return rms.mean, rms.var, float(vec_normalize.epsilon), float(vec_normalize.clip_obs)

    @staticmethod
    def _normalized(raw: np.ndarray, mean: np.ndarray, var: np.ndarray, epsilon: float, clip: float) -> np.ndarray:
        """VecNormalize's own formula (``_normalize_obs``), in numpy, on the raw bank."""
        normalized: np.ndarray = np.clip((raw - mean) / np.sqrt(var + epsilon), -clip, clip)
        return normalized

    def _rollout_means(self, policy: Any, nu: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The rollout buffer's (N, nu) sampled actions and the policy's means at its observations, or ``None``.

        Called inside :meth:`_row`'s ``no_grad`` and its restore of
        ``action_dist``.  ``None`` (the row reads NaN) for a buffer of dict
        observations or an action space that is not a ``Box``.
        """
        import torch
        from gymnasium import spaces

        model: Any = self.model  # PPO's rollout buffer is not on the base class's type
        buffer = model.rollout_buffer
        observations = buffer.observations
        if isinstance(observations, dict) or not isinstance(model.action_space, spaces.Box):
            self._warn_once(
                "rollout",
                "Policy probe: the rollout buffer holds dict observations or the action space is not a Box; the "
                "rollout_* arrays read NaN.",
            )
            return None
        steps = buffer.buffer_size if buffer.full else buffer.pos
        obs = observations[:steps].reshape((-1, *buffer.obs_shape))
        actions = buffer.actions[:steps].reshape(-1, nu)
        means = np.empty(actions.shape, dtype=np.float32)
        for start in range(0, len(obs), ROLLOUT_CHUNK_ROWS):
            tensor = torch.as_tensor(obs[start : start + ROLLOUT_CHUNK_ROWS], dtype=torch.float32, device=policy.device)
            means[start : start + len(tensor)] = policy.get_distribution(tensor).mode().cpu().numpy().reshape(-1, nu)
        return actions, means

    def _learning_rate(self) -> float:
        """The rate the coming ``train()`` applies: SB3's own formula (``_update_learning_rate``)."""
        model = self.model
        total = float(getattr(model, "_total_timesteps", 0) or 0)
        remaining = 1.0 - float(model.num_timesteps) / total if total > 0 else 1.0
        return float(model.lr_schedule(remaining))

    def _row(self) -> dict[str, Any]:
        import torch
        from stable_baselines3.common.distributions import DiagGaussianDistribution

        assert self.bank is not None
        model = self.model
        policy: Any = model.policy  # an ActorCriticPolicy (PPO); typed loosely, as its attributes are dynamic
        bank = self.bank
        nu = int(np.prod(model.action_space.shape or ()))
        obs_dim = bank.raw_obs.shape[1]
        vec_normalize = model.get_vec_normalize_env()
        stats = self._obs_stats(vec_normalize)
        passes: dict[str, np.ndarray | None] = {"now": bank.raw_obs, "frozen": None}
        if stats is not None:
            mean, var, epsilon, clip = stats
            passes["now"] = self._normalized(bank.raw_obs, mean, var, epsilon, clip)
            if self._frozen is not None:
                passes["frozen"] = self._normalized(bank.raw_obs, self._frozen[0], self._frozen[1], epsilon, clip)
        diagonal = isinstance(policy.action_dist, DiagGaussianDistribution)
        if not diagonal:
            self._warn_once(
                "distribution",
                f"Policy probe: the action distribution is {type(policy.action_dist).__name__}, not a diagonal "
                "Gaussian; std_per_actuator reads NaN.",
            )
        values: dict[str, np.ndarray] = {}
        actions: dict[str, np.ndarray] = {}
        rollout: tuple[np.ndarray, np.ndarray] | None = None
        # get_distribution() stores the distribution it builds on the policy's action_dist; the one the
        # rollout left there is put back, so the probe leaves the policy object as it found it.
        stored = dict(vars(policy.action_dist))
        try:
            with torch.no_grad():
                std = torch.exp(policy.log_std).cpu().numpy() if diagonal else np.full(nu, np.nan)
                for name, observations in passes.items():
                    if observations is None:
                        values[name] = np.full(len(bank.labels), np.nan)
                        actions[name] = np.full((len(bank.labels), nu), np.nan)
                        continue
                    tensor = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
                    values[name] = policy.predict_values(tensor).flatten().cpu().numpy()
                    actions[name] = policy.get_distribution(tensor).mode().cpu().numpy().reshape(len(bank.labels), -1)
                try:
                    rollout = self._rollout_means(policy, nu)
                except Exception:  # noqa: BLE001 - the buffer's pass costs the rollout_* arrays only, not the row
                    self._warn_once(
                        "rollout_pass",
                        "Policy probe: could not read the rollout buffer at this update; its rollout_* arrays read NaN.",
                        exc_info=True,
                    )
        finally:
            vars(policy.action_dist).clear()
            vars(policy.action_dist).update(stored)
        ret_rms = getattr(vec_normalize, "ret_rms", None) if vec_normalize is not None else None
        count = float(getattr(getattr(vec_normalize, "obs_rms", None), "count", NAN)) if stats is not None else NAN
        ent_coef = getattr(model, "ent_coef", NAN)
        if rollout is not None:
            space: Any = model.action_space  # a Box: _rollout_means returns None otherwise
            shares = rollout_action_shares(*rollout, space.low, space.high)
        else:  # no buffer row to read: NaN
            shares = rollout_action_shares(np.zeros((0, nu)), np.zeros((0, nu)), -np.ones(nu), np.ones(nu))
        return {
            "update_timesteps": np.int64(self.num_timesteps),
            "std_per_actuator": np.asarray(std, dtype=np.float32).reshape(nu),
            "obs_rms_mean": np.asarray(stats[0] if stats else np.full(obs_dim, np.nan), dtype=np.float32),
            "obs_rms_var": np.asarray(stats[1] if stats else np.full(obs_dim, np.nan), dtype=np.float32),
            "obs_rms_count": np.float64(count),
            "ret_rms_var": np.float32(float(np.asarray(ret_rms.var)) if ret_rms is not None else np.nan),
            "ent_coef": np.float32(ent_coef if isinstance(ent_coef, (int, float)) else np.nan),
            "learning_rate": np.float32(self._learning_rate()),
            "bank_value_now": np.asarray(values["now"], dtype=np.float32),
            "bank_value_frozen": np.asarray(values["frozen"], dtype=np.float32),
            "bank_action_now": np.asarray(actions["now"], dtype=np.float32),
            "bank_action_frozen": np.asarray(actions["frozen"], dtype=np.float32),
            "frozen_reference_timesteps": np.int64(self._frozen[2] if self._frozen is not None else -1),
            **shares,
        }

    def _on_rollout_end(self) -> None:
        if self.bank is None:
            return
        try:
            row = self._row()
            scalars = probe_scalars(self.bank, row)
        except Exception:  # noqa: BLE001 - a report must not sink a run
            self._warn_once("row", "Policy probe: could not probe the policy at this update.", exc_info=True)
            return
        self._rows.append(row)
        self._dirty = True
        self._latest = {"update_timesteps": int(row["update_timesteps"]), **scalars}
        for key, value in scalars.items():
            if math.isfinite(value):
                # TensorBoard only: SB3's logger prints every recorded key at each dump.
                self.logger.record(f"policy_probe/{key}", value, exclude=("stdout", "log"))

    def _on_step(self) -> bool:
        if self.n_calls % self.flush_every_calls == 0:
            self.write()
        return True

    def _on_training_end(self) -> None:
        self.write()

    # ------------------------------------------------------------------ the file
    def arrays(self) -> dict[str, np.ndarray]:
        """The file's arrays: the per-update ones, then the bank's description and the frozen statistics."""
        assert self.bank is not None
        bank = self.bank
        rows = self.rows
        out: dict[str, np.ndarray] = {
            key: np.stack([np.asarray(row[key]) for row in rows]) for key in (*_UPDATE_KEYS, *_ROLLOUT_KEYS)
        }
        frozen_mean = self._frozen[0] if self._frozen is not None else np.full(bank.raw_obs.shape[1], np.nan)
        frozen_var = self._frozen[1] if self._frozen is not None else np.full(bank.raw_obs.shape[1], np.nan)
        described: dict[str, np.ndarray] = {
            "bank_raw_obs": bank.raw_obs,
            "bank_labels": np.array(bank.labels, dtype=str),
            "bank_seeds": bank.seeds,
            "bank_steps": bank.steps,
            "bank_ankle_offset": bank.ankle_offset,
            "bank_com_fore_aft_m": bank.com_fore_aft_m,
            "bank_horizon": np.asarray(bank.horizon, dtype=np.int64),
            "bank_build_seconds": np.asarray(bank.build_seconds, dtype=np.float64),
            "ankle_actuators": bank.ankle_actuators,
            "ankle_signs": bank.ankle_signs,
            "frozen_obs_mean": np.asarray(frozen_mean, dtype=np.float64),
            "frozen_obs_var": np.asarray(frozen_var, dtype=np.float64),
        }
        out.update(described)
        return out

    def write(self) -> bool:
        """Rewrite the file atomically when a row was added since the last write; ``True`` when it wrote."""
        if self.path is None or self.bank is None or not self._dirty or not self.rows:
            return False
        try:
            from .file_io import atomic_savez

            atomic_savez(self.path, **self.arrays())
        except Exception:  # noqa: BLE001 - a report must not sink a run
            self._warn_once("write", f"Could not write {self.path}; the run continues without it.", exc_info=True)
            return False
        self._dirty = False
        return True

    def seed_prior_record(self, directory: str | Path, *, max_timesteps: int | None = None) -> int:
        """Hold the prior session's rows at or before *max_timesteps*, and its bank; compare nothing yet.

        ``seed_resume_eval_state`` calls this before ``learn()``, when this
        session has no bank: :meth:`_on_training_start` keeps the held rows
        once it has built one equal to the prior one.  Returns the rows held.
        """
        path = Path(directory) / POLICY_PROBE_FILENAME
        if not path.exists():
            logger.info("Resume: no prior %s; it starts with this session's updates.", path)
            return 0
        try:
            with np.load(path, allow_pickle=False) as data:
                steps = np.asarray(data["update_timesteps"], dtype=np.int64)
                keep = np.ones(steps.shape, dtype=bool) if max_timesteps is None else steps <= int(max_timesteps)
                updates = {key: np.asarray(data[key])[keep] for key in _UPDATE_KEYS}
                count, nu = int(keep.sum()), int(np.asarray(data["std_per_actuator"]).shape[-1])
                for key in _ROLLOUT_KEYS:  # absent from a file written before them: NaN, and 0 rows
                    if key in data.files:
                        updates[key] = np.asarray(data[key])[keep]
                    elif key == "rollout_rows":
                        updates[key] = np.zeros(count, dtype=np.int64)
                    else:
                        updates[key] = np.full((count, nu), np.nan, dtype=np.float32)
                bank = {key: np.asarray(data[key]) for key in _BANK_KEYS}
        except Exception:  # noqa: BLE001 - a report must not sink a run
            logger.warning("Resume: could not read %s; it starts with this session's updates.", path, exc_info=True)
            return 0
        self._prior_rows = [{key: values[index] for key, values in updates.items()} for index in range(int(keep.sum()))]
        self._prior_bank = bank
        return len(self._prior_rows)

    def latest(self) -> dict[str, float]:
        """``progress.json``'s ``policy``: the latest row's ``policy_probe/*`` scalars and its ``update_timesteps``."""
        if not self._latest and self.bank is not None and self._prior_rows:
            row = self._prior_rows[-1]
            return {"update_timesteps": int(row["update_timesteps"]), **probe_scalars(self.bank, row)}
        return dict(self._latest)


def build_policy_probe_callback(
    *,
    model_id: ModelId | None,
    env_factory: Callable[[int], Any] | None,
    stage_horizon: int,
    log_dir: str | Path | None,
    flush_every_calls: int,
    verbose: int = 0,
) -> PolicyProbeCallback | None:
    """The probe for a stage, or ``None`` without a model, an env factory or a stage directory."""
    if model_id is None or env_factory is None or log_dir is None:
        return None
    return PolicyProbeCallback(
        model_id=model_id,
        env_factory=env_factory,
        stage_horizon=stage_horizon,
        probe_dir=log_dir,
        flush_every_calls=flush_every_calls,
        verbose=verbose,
    )
