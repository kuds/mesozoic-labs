"""Centre-of-mass stance regression tests: the MuJoCo side, the bars and the base class each species extends.

``static_balance_helpers`` checks the home keyframe (the CoM over its support, not too far back) and that the
neutral action survives.  A stance can pass all of that and still be a poor one: a steady lean to one side, a
slow drift of the CoM toward a support edge, the body turning or twisting on its feet, the CoM sagging, a sway
or a rock.  The tests here track the whole-body CoM through a full standing episode and judge it with
``stance_track_helpers``' reduction:

* :class:`EnvStanceSampler` reads ``env.data`` after the reset and after each ``env.step``: the root subtree's
  CoM, the root quaternion, each foot's reference site and heading, the per-foot floor normal loads and the
  loaded floor-contact points of the leg geoms (the support polygon).  No ``mj_*`` call it makes writes the
  env's data (``mj_contactForce`` only reads it), so the episode is the one the env runs untracked.  An
  evaluation episode the gait recorder captured reads the same way through
  ``stance_track_helpers.track_from_episode_trace`` (the base class checks the two captures agree).
* :func:`run_tracked_episode` runs one episode under the neutral action, or under ``action_fn(obs, step)``,
  ``pre_step(env, step)`` and ``post_step(env, step, action)``, the hooks the scripted poor stances use.  A
  checkpoint's rollout goes through the same hook (offline; no policy is committed in the tests)::

      model, normalizer, _ = load_sb3_checkpoint(zip_path, vecnorm_path, env_factory)
      act = lambda obs, step: model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]
      episodes = [run_tracked_episode(env, model_id, seed=seed, action_fn=act) for seed in seeds]
      numbers = [stance_track_quality(e.track, window_start_s=settle_s) for e in episodes]
      lean_sign_summary(numbers)  # a lean of one sign on every seed is the bias per-episode bars hide

  A policy is not mirror-exact, so compare its numbers with the statue's at the same spawns, not with the
  symmetry bars.
* The scripted poor stances the bars must catch, each standing the full horizon (so the survival tests pass
  them): :func:`hip_roll_lean_action`, every hip-roll servo's control shifted by the same angle about the root's
  forward axis (a steady sideways lean); :func:`root_yaw_torque`, a constant torque about world z on the root (a
  slow pivot on the feet); :func:`fore_aft_rock`, a sinusoidal fore-aft force on the root (a persistent rock).
* :class:`Bar` and the two tiers of bars, each a width of :func:`resolution_floor` (0.5 mm, 0.5 mm/s,
  0.05 deg, 0.005 of the load) or more.  On the noise-free statue, which is exactly mirror-symmetric, a key that
  symmetry fixes (:data:`SYMMETRIC_VALUES`) must hold that value to within twice its floor, and every other key
  the statue's own pinned value to within twice its floor: the statue is deterministic, so anything more is a
  change.  On a spawn at reset noise 0.02 every key must stay within the species' spread width of the same
  value: twice the worst deviation of thirty calibration spawns (seeds 0-29), never under twice the floor,
  checked on fifty held-out spawns (seeds 100-149) and run on two of those.
  :mod:`.com_stance_calibration` re-measures all of it.
* :class:`ComStanceQualityBase`: a provenance check, six checks on the noise-free statue (support at every
  step, lateral lean and drift, fore-aft posture, height, pivot, sway), one on each of two noisy spawns, three
  that prove the bars separate a poor stance from a good one (the scripted lean, pivot and rock each stand the
  full horizon and exceed their bars at least :attr:`ComStanceQualityBase.detection_ratio` times) and one that
  the sampler reads what the evaluation recorder records.

The window starts at each species' settle time (``window_start_s``): the first whole second from which the
statue's and every calibration spawn's horizontal CoM speed stays under 1 mm/s in every second to the end.
A species whose settle leaves under 5 s of its env's horizon (the T. rex and the Brachiosaurus) runs twice that
horizon (``horizon_steps``), so its window is the stance it settles into, not the settling.  The window is this
measured settle, not a stance gate's ``settle_steps`` (1 s on the Velociraptor, 4 s on both Compsognathus
models): those are training choices, and a window that opens while the statue still rings reads the ringing.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mujoco
import numpy as np
import pytest

from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.shared.model_id import ModelId
from environments.shared.species_registry import env_class

from .stance_track_helpers import (
    FAMILIES,
    KINEMATIC_KEYS,
    LOAD_KEYS,
    SUPPORT_SERIES,
    StanceTrack,
    convex_hull,
    first_violation,
    nearest_edge_side,
    stance_track_quality,
    stance_track_series,
    track_from_episode_trace,
    window_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Each key's bar kind: ``abs`` |x - ref| <= w with ref the symmetric 0; ``half`` |x - 0.5| <= w; ``dev``
#: |x - ref| <= w; ``upper`` x <= ref + w (more than the statue is worse); ``lower`` x >= ref - w (less is worse).
BAR_KINDS: dict[str, str] = {
    "com_margin_min_m": "lower",
    "xcom_margin_min_m": "lower",
    "com_margin_lateral_min_m": "lower",
    "xcom_margin_lateral_min_m": "lower",
    "com_margin_fore_aft_min_m": "lower",
    "xcom_margin_fore_aft_min_m": "lower",
    "com_lateral_mean_m": "abs",
    "root_roll_mean_deg": "abs",
    "left_load_share_mean": "half",
    "lightest_foot_share_min": "lower",
    "com_lateral_drift_m": "abs",
    "root_roll_drift_deg": "abs",
    "com_fore_aft_mean_m": "dev",
    "com_fore_aft_drift_m": "dev",
    "root_pitch_mean_deg": "dev",
    "front_load_share_mean": "dev",
    "com_height_mean_m": "dev",
    "com_height_drift_m": "dev",
    "root_turn_deg": "abs",
    "root_turn_end_deg": "abs",
    "root_turn_from_spawn_deg": "abs",
    "root_minus_feet_yaw_from_spawn_deg": "abs",
    "foot_turn_max_deg": "upper",
    "foot_turn_from_spawn_max_deg": "upper",
    "foot_splay_change_max_deg": "upper",
    "foot_travel_max_m": "upper",
    "foot_travel_from_spawn_max_m": "upper",
    "com_speed_rms_mps": "upper",
    "com_speed_end_rms_mps": "upper",
    "com_lateral_sd_m": "upper",
    "com_fore_aft_sd_m": "upper",
    "com_height_sd_m": "upper",
    "root_roll_sd_deg": "upper",
    "root_pitch_sd_deg": "upper",
    "root_yaw_sd_deg": "upper",
}
#: The value mirror symmetry fixes on the noise-free statue.  The reset is exactly symmetric, and the statue
#: reads 1.6e-13 or less on these keys on five models and at most 8e-6 (deg, its twist) on the Compsognathus robot.
#: ``lightest_foot_share_min`` is fixed (each foot carries half) on two feet only.
SYMMETRIC_VALUES: dict[str, float] = {
    "com_lateral_mean_m": 0.0,
    "root_roll_mean_deg": 0.0,
    "left_load_share_mean": 0.5,
    "lightest_foot_share_min": 0.5,
    "com_lateral_drift_m": 0.0,
    "root_roll_drift_deg": 0.0,
    "root_turn_deg": 0.0,
    "root_turn_end_deg": 0.0,
    "root_turn_from_spawn_deg": 0.0,
    "root_minus_feet_yaw_from_spawn_deg": 0.0,
    "com_lateral_sd_m": 0.0,
    "root_roll_sd_deg": 0.0,
    "root_yaw_sd_deg": 0.0,
}
#: The keys of the scripted lean that must clear their bar: where the CoM stands, how the load splits, the roll.
LEAN_DETECTION_KEYS: tuple[str, ...] = (
    "com_lateral_mean_m",
    "left_load_share_mean",
    "lightest_foot_share_min",
    "root_roll_mean_deg",
)
#: The key of the scripted pivot that must clear its bar.
PIVOT_DETECTION_KEYS: tuple[str, ...] = ("root_turn_deg",)
#: The keys of the scripted fore-aft rock that must clear their bar.
ROCK_DETECTION_KEYS: tuple[str, ...] = ("com_fore_aft_sd_m", "com_speed_rms_mps", "root_pitch_sd_deg")
#: The episode the sampler-against-recorder check runs (control steps from the reset).
CROSS_CHECK_STEPS = 200
_SYMMETRY_BASIS = "mirror symmetry fixes it on the noise-free statue"
_STATUE_BASIS = "the noise-free statue's pinned value"
_SPREAD_BASIS = "twice the worst of thirty spawns at reset noise 0.02"


def resolution_floor(key: str) -> float:
    """The smallest change of *key* the bars treat as real: 0.5 mm, 0.5 mm/s, 0.05 deg or 0.005 of the load.

    Every bar's width is at least twice this.  A drift is the change across the window, so its floor is a
    position's, whatever the window's length.
    """
    if key.endswith("_mps"):
        return 0.0005
    if key.endswith("_m"):
        return 0.0005
    if key.endswith("_deg"):
        return 0.05
    return 0.005


# ── the sampler ──────────────────────────────────────────────────────────────


class EnvStanceSampler:
    """Collects one :class:`StanceTrack` from a live env, one sample per call to :meth:`sample`.

    A foot's heading is the heading of the axis of its reference site's body that lies closest to the root's
    forward axis at the first sample.  A floor contact on any leg geom (``Morphology.geom_foot >= 0``) adds its
    normal force to its foot's load, and counts as a support point when that force exceeds 0.01 N or 1e-3 of
    the body weight, whichever is larger.  The floor is the highest of the env's static floor planes, read from
    the model (``geom_pos``); a heightfield floor is refused.
    """

    def __init__(self, env: Any, morphology: Morphology) -> None:
        unwrapped = getattr(env, "unwrapped", env)
        model = unwrapped.model
        self.model = model
        self.root = int(morphology.root_body)
        self.sites = np.array([int(foot.primary_site) for foot in morphology.feet])
        self.site_body = np.array([int(model.site_bodyid[site]) for site in self.sites])
        self.labels = tuple(foot.label for foot in morphology.feet)
        self.geom_foot = np.asarray(morphology.geom_foot)
        self.is_floor = np.asarray(morphology.is_floor)
        self.is_animal = np.asarray(morphology.is_animal)
        self.gravity = abs(float(model.opt.gravity[2]))
        self.load_threshold_n = max(0.01, 1e-3 * float(model.body_subtreemass[self.root]) * self.gravity)
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        if not morphology.floor_geoms or any(int(model.geom_type[g]) != plane for g in morphology.floor_geoms):
            raise ValueError("the sampler reads a flat floor: every static floor geom must be a plane")
        self.floor_z = float(max(model.geom_pos[geom, 2] for geom in morphology.floor_geoms))
        self._axes: list[tuple[int, float]] | None = None
        self._force = np.zeros(6)
        self._rows: dict[str, list[Any]] = {
            key: [] for key in ("t", "com", "quat", "feet", "heading", "load", "support")
        }

    def sample(self, env: Any) -> None:
        """Append the env's current state (call after ``reset`` and after each ``step``)."""
        data = getattr(env, "unwrapped", env).data
        rows = self._rows
        rows["t"].append(float(data.time))
        rows["com"].append(data.subtree_com[self.root].copy())
        rows["quat"].append(data.xquat[self.root].copy())
        rows["feet"].append(data.site_xpos[self.sites].copy())
        frames = data.xmat[self.site_body].reshape(-1, 3, 3)
        if self._axes is None:
            root_x = data.xmat[self.root].reshape(3, 3)[:2, 0]
            root_x = root_x / max(float(np.linalg.norm(root_x)), 1e-12)
            self._axes = []
            for frame in frames:
                dots = frame[:2, :].T @ root_x
                axis = int(np.argmax(np.abs(dots)))
                self._axes.append((axis, math.copysign(1.0, float(dots[axis]))))
        rows["heading"].append(
            [
                math.atan2(sign * frame[1, axis], sign * frame[0, axis])
                for (axis, sign), frame in zip(self._axes, frames)
            ]
        )
        load = np.zeros(len(self.labels))
        points = []
        ncon = int(data.ncon)
        if ncon:
            geoms = data.contact.geom[:ncon]
            floor1, floor2 = self.is_floor[geoms[:, 0]], self.is_floor[geoms[:, 1]]
            other = np.where(floor1, geoms[:, 1], geoms[:, 0])
            selected = np.flatnonzero((floor1 != floor2) & self.is_animal[other] & (self.geom_foot[other] >= 0))
            for index in selected:
                mujoco.mj_contactForce(self.model, data, int(index), self._force)
                normal = float(self._force[0])
                load[self.geom_foot[other[index]]] += normal
                if normal > self.load_threshold_n:
                    points.append(data.contact.pos[index, :2].copy())
        rows["load"].append(load)
        rows["support"].append(np.asarray(points, dtype=np.float64).reshape(-1, 2))

    def track(self) -> StanceTrack:
        """The samples so far as a :class:`StanceTrack` (the CoM velocity left to the reduction's differences)."""
        rows = self._rows
        return StanceTrack(
            t=np.asarray(rows["t"]),
            com=np.asarray(rows["com"]),
            root_quat=np.asarray(rows["quat"]),
            foot_pos=np.asarray(rows["feet"]),
            labels=self.labels,
            support=tuple(rows["support"]),
            foot_load=np.asarray(rows["load"]),
            foot_heading=np.asarray(rows["heading"]),
            floor_z=self.floor_z,
            gravity=self.gravity,
        )


@dataclass(frozen=True)
class TrackedEpisode:
    """One tracked episode: the track, how far it ran and how it ended."""

    track: StanceTrack
    steps: int
    horizon: int
    terminated: bool
    truncated: bool
    termination_reason: str | None
    dt: float

    @property
    def stood(self) -> bool:
        """It ran its whole horizon without terminating."""
        return not self.terminated and self.steps == self.horizon


def run_tracked_episode(
    env: Any,
    model_id: ModelId,
    *,
    seed: int,
    action_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    pre_step: Callable[[Any, int], None] | None = None,
    post_step: Callable[[Any, int, np.ndarray], None] | None = None,
    steps: int | None = None,
    morphology: Morphology | None = None,
) -> TrackedEpisode:
    """``env.reset(seed=seed)``, then up to *steps* (default the env's horizon) steps, sampled after each.

    The action is the neutral one (zeros) unless ``action_fn(obs, step)`` gives it; ``pre_step(env, step)``
    runs before each step (step 1 is the first) and ``post_step(env, step, action)`` after it.  The episode
    stops at a termination or truncation.
    """
    unwrapped = getattr(env, "unwrapped", env)
    sampler = EnvStanceSampler(env, morphology or Morphology.from_env(env, model_id))
    horizon = int(steps if steps is not None else unwrapped.max_episode_steps)
    neutral = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    sampler.sample(env)
    terminated = truncated = False
    reason: str | None = None
    step = 0
    for step in range(1, horizon + 1):
        if pre_step is not None:
            pre_step(env, step)
        action = neutral if action_fn is None else action_fn(obs, step)
        obs, _, terminated, truncated, info = env.step(action)
        sampler.sample(env)
        if post_step is not None:
            post_step(env, step, action)
        if terminated:
            reason = str(info.get("termination_reason", "unknown"))
        if terminated or truncated:
            break
    return TrackedEpisode(
        track=sampler.track(),
        steps=step,
        horizon=horizon,
        terminated=bool(terminated),
        truncated=bool(truncated),
        termination_reason=reason,
        dt=float(unwrapped.dt),
    )


# ── the scripted poor stances ────────────────────────────────────────────────


def _home_frame(env: Any, morphology: Morphology) -> tuple[mujoco.MjData, np.ndarray]:
    """Forward kinematics of the reset keyframe on a scratch MjData, and the root's horizontal forward axis there."""
    unwrapped = getattr(env, "unwrapped", env)
    model = unwrapped.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, int(getattr(unwrapped, "_reset_keyframe_id", 0)))
    mujoco.mj_kinematics(model, data)
    forward = data.xmat[morphology.root_body].reshape(3, 3)[:, 0].copy()
    forward[2] = 0.0
    return data, forward / np.linalg.norm(forward)


def body_weight_n(env: Any, morphology: Morphology) -> float:
    """The animal's weight (the free root's subtree mass times |g|), N."""
    model = getattr(env, "unwrapped", env).model
    return float(model.body_subtreemass[morphology.root_body]) * abs(float(model.opt.gravity[2]))


def hip_roll_servos(env: Any) -> list[int]:
    """The actuators that drive a joint whose name contains ``hip_roll``."""
    model = getattr(env, "unwrapped", env).model
    joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    return [
        actuator
        for actuator in range(model.nu)
        if int(model.actuator_trntype[actuator]) == joint
        and "hip_roll" in model.joint(int(model.actuator_trnid[actuator, 0])).name
    ]


def action_for_deltas(env: Any, deltas: Mapping[int, float]) -> np.ndarray:
    """The normalised action whose control differs from the neutral action's by *deltas* (actuator -> rad).

    Found by bisection on ``env._scale_action``, one actuator at a time, and checked to 1e-6 rad.
    """
    unwrapped = getattr(env, "unwrapped", env)
    nu = unwrapped.model.nu
    home = unwrapped._scale_action(np.zeros(nu))
    action = np.zeros(nu)
    for actuator, delta in deltas.items():
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            probe = np.zeros(nu)
            probe[actuator] = mid
            if unwrapped._scale_action(probe)[actuator] - home[actuator] < delta:
                lo = mid
            else:
                hi = mid
        action[actuator] = 0.5 * (lo + hi)
    reached = unwrapped._scale_action(action) - home
    for actuator, delta in deltas.items():
        if abs(reached[actuator] - delta) > 1e-6:
            name = unwrapped.model.actuator(actuator).name
            raise ValueError(
                f"actuator {name}: wanted a {delta} rad shift, the action mapping reaches {reached[actuator]}"
            )
    return action


def hip_roll_lean_action(env: Any, morphology: Morphology, degrees: float) -> np.ndarray:
    """Every hip-roll servo's control shifted by *degrees* about the root's forward axis, so both legs roll alike.

    Each servo's sign is its joint axis against the root's forward axis at the reset keyframe.  Held from the
    first step, it leans the stance sideways and the legs hold it there.
    """
    model = getattr(env, "unwrapped", env).model
    servos = hip_roll_servos(env)
    if not servos:
        raise ValueError("the model has no hip-roll servo to lean with")
    data, forward = _home_frame(env, morphology)
    deltas = {
        actuator: math.copysign(1.0, float(data.xaxis[int(model.actuator_trnid[actuator, 0])] @ forward))
        * math.radians(degrees)
        for actuator in servos
    }
    return action_for_deltas(env, deltas)


def root_yaw_torque(env: Any, morphology: Morphology, k: float) -> Callable[[Any, int], None]:
    """A ``pre_step`` that applies ``k * m * g * w`` N.m about world z to the root before every step.

    ``m g`` is the animal's weight and ``w`` the mean distance of the foot reference sites from their centroid
    at the reset keyframe: a torque scaled by the weight and the foot spread, not by the floor's friction or the
    real contact geometry (the floors' friction differs between models, 0.8 to 1.2), so *k* is a comparable
    twisting load across models, not the same fraction of the moment the feet can resist.
    """
    data, _ = _home_frame(env, morphology)
    sites = np.array([data.site_xpos[foot.primary_site][:2] for foot in morphology.feet])
    spread = float(np.mean(np.linalg.norm(sites - sites.mean(axis=0), axis=1)))
    torque = k * body_weight_n(env, morphology) * spread
    root = int(morphology.root_body)

    def apply(env_: Any, step: int) -> None:
        getattr(env_, "unwrapped", env_).data.xfrc_applied[root, 5] = torque

    return apply


def fore_aft_rock(env: Any, morphology: Morphology, force_bw: float, hz: float) -> Callable[[Any, int], None]:
    """A ``pre_step`` that pushes the root fore and aft: ``force_bw * m g * sin(2 pi hz t)`` along its forward axis.

    The axis is the root's horizontal forward axis at the reset keyframe and ``t`` the time of the step's
    start, so the push rocks the CoM fore and aft for the whole episode: a stance that never settles.
    """
    _, forward = _home_frame(env, morphology)
    amplitude = force_bw * body_weight_n(env, morphology)
    root = int(morphology.root_body)
    dt = float(getattr(env, "unwrapped", env).dt)

    def apply(env_: Any, step: int) -> None:
        push = amplitude * math.sin(2.0 * math.pi * hz * (step - 1) * dt)
        getattr(env_, "unwrapped", env_).data.xfrc_applied[root, :3] = push * forward

    return apply


# ── the bars ─────────────────────────────────────────────────────────────────


def _display(key: str) -> tuple[float, str, int]:
    """How a key's value is shown: its scale, its unit and its decimals."""
    if key.endswith("_mps"):
        return 1e3, " mm/s", 3
    if key.endswith("_m"):
        return 1e3, " mm", 3
    if key.endswith("_deg"):
        return 1.0, " deg", 4
    return 1.0, "", 4


def shown(key: str, value: float, decimals: int | None = None) -> str:
    """*value* of *key* in display units (mm, mm/s, deg or a load fraction)."""
    scale, unit, default = _display(key)
    places = default if decimals is None else decimals
    return f"{value * scale:.{places}f}{unit}" if math.isfinite(value) else f"{value}"


@dataclass(frozen=True)
class Bar:
    """One key's bar: its kind (:data:`BAR_KINDS`), its width and its reference (the symmetric or pinned value)."""

    key: str
    kind: str
    width: float
    reference: float = 0.0
    basis: str = ""

    def badness(self, value: float) -> float:
        """How far *value* is from good, in the key's units: the bar holds while it is at most the width."""
        if self.kind in ("abs", "dev"):
            return abs(value - self.reference)
        if self.kind == "half":
            return abs(value - 0.5)
        if self.kind == "upper":
            return value - self.reference
        if self.kind == "lower":
            return self.reference - value
        raise ValueError(f"unknown bar kind {self.kind!r}")

    def ratio(self, value: float) -> float:
        """``badness / width``: over 1 fails; NaN for a value that was not measured."""
        return self.badness(value) / self.width if math.isfinite(value) else math.nan

    def holds(self, value: float) -> bool:
        return math.isfinite(value) and self.badness(value) <= self.width

    def bounds(self) -> tuple[float, float]:
        """The interval of values the bar holds (an infinite end where it is one-sided)."""
        if self.kind in ("abs", "dev"):
            return self.reference - self.width, self.reference + self.width
        if self.kind == "half":
            return 0.5 - self.width, 0.5 + self.width
        if self.kind == "upper":
            return -math.inf, self.reference + self.width
        return self.reference - self.width, math.inf

    def describe(self, decimals: int | None = None) -> str:
        key, width = self.key, self.width
        lo, hi = self.bounds()
        if self.kind == "abs" and self.reference == 0.0:
            text = f"|x| <= {shown(key, width, decimals)}"
        elif self.kind in ("abs", "dev"):
            text = f"within {shown(key, width, decimals)} of {shown(key, self.reference, decimals)}"
        elif self.kind == "half":
            text = f"|x - 0.5| <= {shown(key, width, decimals)}"
        elif self.kind == "upper":
            text = f"<= {shown(key, hi, decimals)} ({shown(key, self.reference, decimals)} + {shown(key, width, decimals)})"
        else:
            text = f">= {shown(key, lo, decimals)} ({shown(key, self.reference, decimals)} - {shown(key, width, decimals)})"
        return f"{text}; {self.basis}" if self.basis else text


def _decimals_apart(key: str, value: float, bar: Bar) -> int:
    """Enough decimals that a value the bar breaks never prints equal to the bound it breaks."""
    _, _, decimals = _display(key)
    lo, hi = bar.bounds()
    bound = hi if value > hi else lo
    while decimals < 9 and math.isfinite(bound) and shown(key, value, decimals) == shown(key, bound, decimals):
        decimals += 1
    return decimals


@dataclass(frozen=True)
class Violation:
    """A key whose value its bar does not hold (NaN: the track did not measure it)."""

    bar: Bar
    value: float

    def describe(self) -> str:
        key = self.bar.key
        if not math.isfinite(self.value):
            return f"{key} = {self.value} (bar {self.bar.describe()}, not measured)"
        decimals = _decimals_apart(key, self.value, self.bar)
        ratio = self.bar.ratio(self.value)
        return f"{key} = {shown(key, self.value, decimals)} (bar {self.bar.describe(decimals)}; {ratio:.2f}x its width)"


def check(metrics: Mapping[str, float], bars: Iterable[Bar]) -> list[Violation]:
    """Every bar *metrics* breaks, a missing key among them."""
    out = []
    for bar in bars:
        value = float(metrics.get(bar.key, math.nan))
        if not bar.holds(value):
            out.append(Violation(bar, value))
    return out


def format_violations(title: str, violations: list[Violation]) -> str:
    return title + ":\n" + "\n".join(f"  {violation.describe()}" for violation in violations)


# ── provenance ───────────────────────────────────────────────────────────────


def manifest_physics_revision(model_id: ModelId) -> int:
    """The model's physics revision in the committed plant manifest."""
    manifest = json.loads((REPO_ROOT / "configs" / "plant_manifest.generated.json").read_text())
    return int(manifest["plants"][model_id.species][model_id.embodiment]["physics"]["revision"])


def canonical_mujoco_version() -> str:
    """The MuJoCo version every committed fingerprint and pin is read with (``configs/plant_versions.toml``)."""
    with open(REPO_ROOT / "configs" / "plant_versions.toml", "rb") as handle:
        return str(tomllib.load(handle)["canonical_mujoco_version"])


# ── the base class ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Statue:
    episode: TrackedEpisode
    series: dict[str, np.ndarray]
    metrics: dict[str, float]


#: Each test class's noise-free statue, rolled and reduced once for its six statue tests and kept for the
#: session (under a MB each); pytest 9.1 deprecates a class-scoped fixture written as a method.
_STATUES: dict[type, _Statue] = {}


def _held(action: np.ndarray) -> Callable[[np.ndarray, int], np.ndarray]:
    """An ``action_fn`` that holds *action* at every step."""

    def action_fn(obs: np.ndarray, step: int) -> np.ndarray:
        return action

    return action_fn


class ComStanceQualityBase:
    """Tracks the CoM through full standing episodes and fails on a stance that stands but is a poor one.

    Subclasses provide:
        - ``model_id``, ``species_label`` and ``env_kwargs`` (the static-balance tests' constructor, reset
          noise 0)
        - ``physics_revision``: the plant revision the pins were read on (checked against the manifest)
        - ``horizon_steps``: the episode's length (None: the env's default) and ``window_start_s``: the settle
          time, where the window starts
        - ``reference``: the noise-free statue's value of every key mirror symmetry does not fix
        - ``spread_width``: each key's width on a noisy spawn
        - ``noisy_seeds``: the two held-out seeds of the noisy-spawn test
        - ``quadruped``: four feet (``lightest_foot_share_min`` is a symmetry key on two only)
        - ``rock_force_bw`` and ``rock_hz``: the scripted rock's push
        - ``imu_lean_keys``: on a robot, the lean keys its hardware reads without foot force sensing; the
          scripted lean must clear its bar on one of them too

    :mod:`.com_stance_calibration` prints every pinned number from a fresh measurement.  One episode per class,
    the noise-free statue, serves the six statue tests; each noisy spawn and each scripted stance is one more.
    """

    model_id: ClassVar[ModelId]
    env_kwargs: ClassVar[dict[str, Any]] = {"reset_noise_scale": 0.0}
    species_label: ClassVar[str] = ""
    physics_revision: ClassVar[int] = 0
    #: The MuJoCo version the pins were read with: ``configs/plant_versions.toml``'s canonical one.
    pinned_mujoco_version: ClassVar[str] = "3.10.0"
    horizon_steps: ClassVar[int | None] = None
    window_start_s: ClassVar[float] = 2.0
    #: From here on (past the reset's settle onto the floor) the CoM and the XcoM must stay inside the support.
    support_from_s: ClassVar[float] = 1.0
    reference: ClassVar[dict[str, float]] = {}
    spread_width: ClassVar[dict[str, float]] = {}
    noisy_seeds: ClassVar[tuple[int, ...]] = ()
    quadruped: ClassVar[bool] = False
    noisy_reset_noise_scale: ClassVar[float] = 0.02
    lean_degrees: ClassVar[float] = 1.0
    yaw_torque_k: ClassVar[float] = 0.05
    rock_force_bw: ClassVar[float] = 0.005
    rock_hz: ClassVar[float] = 0.75
    imu_lean_keys: ClassVar[tuple[str, ...]] = ()
    #: How many times its bar a scripted poor stance must reach: room on the poor side, so a later change that
    #: moves the bars or the plant until the two sit close fails here instead of passing at 1.1x.
    detection_ratio: ClassVar[float] = 2.0

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        if "noisy_seed" in metafunc.fixturenames:
            metafunc.parametrize("noisy_seed", self.noisy_seeds, ids=[f"seed{seed}" for seed in self.noisy_seeds])

    # ── episodes ──

    def make_env(self, **overrides: Any) -> Any:
        kwargs = dict(self.env_kwargs)
        if self.horizon_steps is not None:
            kwargs["max_episode_steps"] = self.horizon_steps
        return env_class(self.model_id)(**{**kwargs, **overrides})

    def run_episode(
        self,
        *,
        seed: int,
        reset_noise_scale: float | None = None,
        action: Callable[[Any, Morphology], np.ndarray] | None = None,
        pre_step: Callable[[Any, Morphology], Callable[[Any, int], None]] | None = None,
    ) -> TrackedEpisode:
        """One full-horizon episode, under a constant *action* and a *pre_step* each built from the live env."""
        overrides = {} if reset_noise_scale is None else {"reset_noise_scale": reset_noise_scale}
        env = self.make_env(**overrides)
        try:
            morphology = Morphology.from_env(env, self.model_id)
            action_fn = _held(action(env, morphology)) if action is not None else None
            hook = pre_step(env, morphology) if pre_step is not None else None
            return run_tracked_episode(
                env, self.model_id, seed=seed, action_fn=action_fn, pre_step=hook, morphology=morphology
            )
        finally:
            env.close()

    def reduce(self, episode: TrackedEpisode) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """The episode's per-sample series and its window numbers."""
        series = stance_track_series(episode.track)
        return series, stance_track_quality(episode.track, window_start_s=self.window_start_s, series=series)

    @pytest.fixture
    def statue(self) -> _Statue:
        """The noise-free statue: the reset keyframe, the neutral action, the full horizon (run once per class)."""
        cls = type(self)
        if cls not in _STATUES:
            episode = self.run_episode(seed=0)
            # A statue that falls before its window has nothing to reduce: each test fails on assert_stood.
            series, metrics = self.reduce(episode) if episode.stood else ({}, {})
            _STATUES[cls] = _Statue(episode, series, metrics)
        return _STATUES[cls]

    # ── bars ──

    def symmetric(self, key: str) -> bool:
        return key in SYMMETRIC_VALUES and not (key == "lightest_foot_share_min" and self.quadruped)

    def bar_reference(self, key: str) -> float:
        """The value both tiers hold *key* to: the symmetric value, or the statue's pinned one."""
        return SYMMETRIC_VALUES[key] if self.symmetric(key) else float(self.reference[key])

    def statue_bar(self, key: str) -> Bar:
        """The noise-free statue's bar: twice the floor about the symmetric value or the statue's pinned value."""
        basis = _SYMMETRY_BASIS if self.symmetric(key) else _STATUE_BASIS
        return Bar(key, BAR_KINDS[key], 2.0 * resolution_floor(key), self.bar_reference(key), basis)

    def spread_bar(self, key: str) -> Bar:
        """A noisy spawn's bar: the species' spread width about the same value."""
        return Bar(key, BAR_KINDS[key], self.spread_width[key], self.bar_reference(key), _SPREAD_BASIS)

    def keys(self, family: str | None = None) -> list[str]:
        """The keys of *family* (every family when None) this model has (``front_load_share_mean`` on four feet)."""
        families = list(FAMILIES) if family is None else [family]
        return [key for name in families for key in FAMILIES[name] if key != "front_load_share_mean" or self.quadruped]

    def provenance(self) -> str:
        """A note for a failure message when the running MuJoCo is not the one the pins were read with."""
        running = mujoco.__version__
        if running == self.pinned_mujoco_version:
            return ""
        return f" [pins read on mujoco {self.pinned_mujoco_version}; running {running}]"

    # ── checks ──

    def assert_stood(self, episode: TrackedEpisode, what: str) -> None:
        assert episode.stood, (
            f"{self.species_label} {what} terminated at step {episode.steps}/{episode.horizon} "
            f"(t = {episode.steps * episode.dt:.2f} s): {episode.termination_reason}. A stance that falls has no "
            "stance quality to measure, and the full-horizon survival tests fail it too."
        )

    def assert_family_holds(self, statue: _Statue, family: str, what: str) -> None:
        """The noise-free statue stood, and every key of *family* holds its statue bar (*what* names the fault)."""
        self.assert_stood(statue.episode, "noise-free statue")
        violations = check(statue.metrics, [self.statue_bar(key) for key in self.keys(family)])
        assert not violations, format_violations(
            f"{self.species_label} {what} ({family} keys, window {self.window_start_s:g} s on){self.provenance()}",
            violations,
        )

    def _edge(self, track: StanceTrack, series: Mapping[str, np.ndarray], index: int, xcom: bool) -> str:
        points = track.support[index] if track.support is not None else np.zeros((0, 2))
        hull = convex_hull(points) if len(points) else np.zeros((0, 2))
        point = series["xcom"][index] if xcom else np.asarray(track.com)[index, :2]
        return nearest_edge_side(hull, point, series["forward"][index])

    def support_failures(
        self, episode: TrackedEpisode, series: Mapping[str, np.ndarray], bars: Mapping[str, Bar]
    ) -> list[str]:
        """Per step: where the CoM or the XcoM first leaves the support polygon (from ``support_from_s``), and
        where each support margin first breaks its bar in the window, naming the nearest edge's side."""
        t = series["t"]
        track = episode.track
        out = []
        for name, label in (("com_margin", "CoM"), ("xcom_margin", "XcoM")):
            first = first_violation(t, series[name], below=0.0, window_start_s=self.support_from_s)
            if first is not None:
                out.append(
                    f"the {label} leaves its support polygon at step {first} (t = {t[first]:.2f} s; checked from "
                    f"{self.support_from_s:g} s): margin {shown('com_margin_min_m', float(series[name][first]))}, nearest edge "
                    f"{self._edge(track, series, first, name == 'xcom_margin')}"
                )
        window = window_mask(t, self.window_start_s)
        samples = np.flatnonzero(window)
        for key, name in SUPPORT_SERIES.items():
            bar = bars[key]
            margin = series[name]
            first = first_violation(t, margin, below=bar.reference - bar.width, window_start_s=self.window_start_s)
            if first is None:
                continue
            worst = int(samples[np.argmin(np.where(np.isfinite(margin[samples]), margin[samples], -np.inf))])
            value = float(margin[first])
            decimals = _decimals_apart(key, value, bar) if math.isfinite(value) else None
            out.append(
                f"{key}: the margin first breaks its bar ({bar.describe(decimals)}) at step {first} "
                f"(t = {t[first]:.2f} s), at {shown(key, value, decimals)}, nearest edge "
                f"{self._edge(track, series, first, key.startswith('xcom'))}; its window minimum is "
                f"{shown(key, float(margin[worst]), decimals)} at step {worst}; "
                f"{int((window & ~np.isfinite(margin)).sum())} window samples have no support polygon "
                f"(window {t[window][0]:.2f}-{t[-1]:.2f} s, {int(window.sum())} samples)"
            )
        return out

    def detection_ratios(self, metrics: Mapping[str, float], keys: Iterable[str]) -> list[tuple[float, str, float]]:
        """(badness / statue bar, key, value) for each key, largest first."""
        out = []
        for key in keys:
            value = float(metrics.get(key, math.nan))
            ratio = self.statue_bar(key).ratio(value)
            out.append((ratio if math.isfinite(ratio) else -math.inf, key, value))
        return sorted(out, reverse=True)

    def assert_detected(self, metrics: Mapping[str, float], keys: Iterable[str], what: str) -> None:
        ratios = self.detection_ratios(metrics, keys)
        ratio, key, value = ratios[0]
        listed = ", ".join(f"{k} {shown(k, v)} ({r:.2f}x)" for r, k, v in ratios)
        assert ratio >= self.detection_ratio, (
            f"{self.species_label} {what}: its largest ratio is {ratio:.2f}x ({key} = {shown(key, value)} against "
            f"{self.statue_bar(key).describe()}), under the {self.detection_ratio:g}x the bars must clear to show "
            f"they separate a poor stance from a good one. All: {listed}{self.provenance()}"
        )

    # ── provenance ──

    def test_pins_were_read_on_this_plant(self):
        """The pinned numbers name the plant revision and the MuJoCo they were read on, and both are current."""
        current = manifest_physics_revision(self.model_id)
        assert self.physics_revision == current, (
            f"{self.species_label}: the CoM stance pins were read on physics revision {self.physics_revision}, "
            f"but the manifest is at revision {current}. Re-measure them on the current plant "
            f"(python -m environments.shared.tests.com_stance_calibration {type(self).__module__}:"
            f"{type(self).__name__}), review the printed differences, and update the pins and physics_revision "
            "in the same commit."
        )
        canonical = canonical_mujoco_version()
        assert self.pinned_mujoco_version == canonical, (
            f"{self.species_label}: the pins were read on mujoco {self.pinned_mujoco_version}, but the canonical "
            f"version is {canonical}: re-measure them with com_stance_calibration."
        )
        missing = [key for key in self.keys() if not self.symmetric(key) and key not in self.reference]
        unwidthed = [key for key in self.keys() if key not in self.spread_width]
        assert not missing and not unwidthed, (missing, unwidthed)

    # ── the noise-free statue ──

    def test_statue_com_stays_inside_its_support_at_every_step(self, statue):
        """The CoM and the XcoM stay inside the loaded support polygon at every step from ``support_from_s``, and
        keep their margins (to the nearest edge, sideways and fore-aft) at every window step."""
        self.assert_stood(statue.episode, "noise-free statue")
        failures = self.support_failures(
            statue.episode, statue.series, {key: self.statue_bar(key) for key in self.keys("support")}
        )
        assert not failures, f"{self.species_label} noise-free statue{self.provenance()}: " + "; ".join(failures)

    def test_statue_does_not_lean_or_drift_sideways(self, statue):
        """Lateral CoM offset, roll and the left/right load split stay mirror-symmetric, with no sideways drift."""
        self.assert_family_holds(statue, "lean", "noise-free statue leans or drifts sideways")

    def test_statue_holds_its_fore_aft_posture(self, statue):
        """The fore-aft CoM offset, its drift and the root pitch (and the front/rear split) hold the statue's own."""
        self.assert_family_holds(statue, "sagittal", "noise-free statue moved off its fore-aft posture")

    def test_statue_com_height_holds(self, statue):
        """The CoM stands at the statue's own height and neither sags nor rises across the window."""
        self.assert_family_holds(statue, "height", "noise-free statue's CoM height moved")

    def test_statue_does_not_pivot(self, statue):
        """The root does not turn (in the window, at its end or from the spawn) nor twist over its feet, and the
        feet do not turn, splay or slide more than the statue's own."""
        self.assert_family_holds(statue, "pivot", "noise-free statue pivots")

    def test_statue_sway_stays_small(self, statue):
        """The CoM's speed (over the window and its end) and the spread of the CoM and the root's angles about
        their trends stay at the statue's own."""
        self.assert_family_holds(statue, "sway", "noise-free statue sways or rocks")

    # ── noisy spawns ──

    def test_noisy_spawn_stays_within_the_calibrated_spread(self, noisy_seed):
        """A held-out spawn at reset noise 0.02 stays inside every spread bar, the support at every step: no
        regression in how small spawn errors settle.  (It does not say the current plant's spread is good:
        the species files list what it allows.)"""
        self.check_noisy_spawn(noisy_seed)

    def check_noisy_spawn(self, seed: int) -> None:
        """The noisy-spawn check on *seed*: it stands, and every key holds its spread bar."""
        noisy_seed = seed
        what = f"spawn at reset noise {self.noisy_reset_noise_scale} (seed {noisy_seed})"
        episode = self.run_episode(seed=noisy_seed, reset_noise_scale=self.noisy_reset_noise_scale)
        self.assert_stood(episode, what)
        series, metrics = self.reduce(episode)
        # The support keys are checked per step (a window sample with no support polygon breaks them too).
        problems = self.support_failures(episode, series, {key: self.spread_bar(key) for key in self.keys("support")})
        keys = [key for family in FAMILIES if family != "support" for key in self.keys(family)]
        problems += [violation.describe() for violation in check(metrics, [self.spread_bar(key) for key in keys])]
        assert not problems, f"{self.species_label} {what}{self.provenance()}:\n" + "\n".join(
            f"  {problem}" for problem in problems
        )

    # ── the bars catch a poor stance that survives ──

    def test_bars_catch_a_scripted_lateral_lean(self):
        """A hip-roll lean held from the first step stands the full horizon and breaks a lean bar twice over (on a
        robot, also one its hardware reads without foot force sensing)."""
        degrees = self.lean_degrees
        episode = self.run_episode(
            seed=0, action=lambda env, morphology: hip_roll_lean_action(env, morphology, degrees)
        )
        what = f"scripted {degrees:g} deg hip-roll lean"
        self.assert_stood(episode, what)
        _, metrics = self.reduce(episode)
        self.assert_detected(metrics, [key for key in LEAN_DETECTION_KEYS if key in self.keys("lean")], what)
        if self.imu_lean_keys:
            self.assert_detected(metrics, self.imu_lean_keys, f"{what}, read without foot force sensing")

    def test_bars_catch_a_scripted_pivot(self):
        """A constant yaw torque on the root stands the full horizon and turns the root twice its bar."""
        k = self.yaw_torque_k
        episode = self.run_episode(seed=0, pre_step=lambda env, morphology: root_yaw_torque(env, morphology, k))
        what = f"scripted yaw torque of {k:g} m g w"
        self.assert_stood(episode, what)
        self.assert_detected(self.reduce(episode)[1], PIVOT_DETECTION_KEYS, what)

    def test_bars_catch_a_scripted_fore_aft_rock(self):
        """A sinusoidal fore-aft push on the root stands the full horizon and breaks a sway bar twice over."""
        force, hz = self.rock_force_bw, self.rock_hz
        episode = self.run_episode(seed=0, pre_step=lambda env, morphology: fore_aft_rock(env, morphology, force, hz))
        what = f"scripted fore-aft rock of {force:g} m g at {hz:g} Hz"
        self.assert_stood(episode, what)
        self.assert_detected(self.reduce(episode)[1], ROCK_DETECTION_KEYS, what)

    # ── the capture ──

    def test_the_sampler_reads_what_the_eval_recorder_records(self):
        """The sampler and the evaluation pipeline's ``SubstepContactRecorder(balance=True)`` capture the same CoM,
        root attitude and foot points, so a recorded evaluation episode reads the same through
        ``track_from_episode_trace``; each per-foot load the sampler reads lies within the recorder's substeps'."""
        env = self.make_env()
        try:
            morphology = Morphology.from_env(env, self.model_id)
            recorder = SubstepContactRecorder(env, morphology, balance=True).attach()
            try:
                episode = run_tracked_episode(
                    env,
                    self.model_id,
                    seed=0,
                    steps=CROSS_CHECK_STEPS,
                    morphology=morphology,
                    post_step=lambda env_, step, action: recorder.end_step(action),
                )
                trace = recorder.end_episode(terminated=episode.terminated, truncated=True)
            finally:
                recorder.detach()
        finally:
            env.close()
        sampled = episode.track
        recorded = track_from_episode_trace(trace, gravity=sampled.gravity)
        np.testing.assert_allclose(recorded.t, sampled.t - sampled.t[0], rtol=0.0, atol=1e-9)
        for name in ("com", "root_quat", "foot_pos"):
            np.testing.assert_allclose(
                getattr(recorded, name), getattr(sampled, name), rtol=0.0, atol=1e-12, err_msg=name
            )
        loads = np.asarray(sampled.foot_load)[1:]
        assert np.all(loads >= np.asarray(trace.leg_floor_min) - 1e-6), "a sampled load under the recorder's least"
        assert np.all(loads <= np.asarray(trace.leg_floor_max) + 1e-6), "a sampled load over the recorder's most"
        # The second half of what ran (the comparison is of the captures, not of a settled stance).
        window = 0.5 * float(sampled.t[-1] - sampled.t[0])
        mine = stance_track_quality(sampled, window_start_s=window)
        theirs = stance_track_quality(recorded, window_start_s=window)
        for key in KINEMATIC_KEYS:
            assert theirs[key] == pytest.approx(mine[key], rel=1e-9, abs=1e-12), key
        for key in LOAD_KEYS:
            assert theirs[key] == pytest.approx(mine[key], abs=0.01), key
