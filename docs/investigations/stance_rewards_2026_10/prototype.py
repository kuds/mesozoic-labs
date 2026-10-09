"""Research-only stance rewards; no production env, gate, or interface changes.

The streaming probe matches the floor-truth recorder's force-weighted CoP
over a control step. It records only that step and never writes MjData.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from environments.shared.gait.constants import CONTACT_THRESHOLD_N, SPAWN_GRACE_S
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder


@dataclass
class BudgetSchedule:
    """Pickled by module reference; preserve production LR versus elapsed steps."""

    schedule: object
    pilot_steps: int
    production_steps: int

    def __call__(self, progress_remaining):
        elapsed = (1 - progress_remaining) * self.pilot_steps
        return self.schedule(max(0.0, 1.0 - elapsed / self.production_steps))


@dataclass(frozen=True)
class Variant:
    name: str
    species: str
    cop_weight: float = 0.0
    cop_safe_fraction: float = 0.65
    support_fraction: float | None = None
    airborne_weight: float = 0.0
    impact_weight: float = 0.0
    impact_threshold_bw: float = 2.0
    coverage_weight: float = 0.0

    def __post_init__(self):
        for key in ("cop_weight", "airborne_weight", "impact_weight", "coverage_weight", "impact_threshold_bw"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not 0 <= self.cop_safe_fraction < 1:
            raise ValueError("cop_safe_fraction must be in [0, 1)")
        if self.support_fraction is not None and not 0 <= self.support_fraction <= 1:
            raise ValueError("support_fraction must be in [0, 1]")

    def as_dict(self):
        return asdict(self)


VARIANTS = {
    ("trex", "control"): Variant("control", "trex"),
    ("trex", "cop_margin"): Variant("cop_margin", "trex", cop_weight=0.10),
    ("velociraptor", "control"): Variant("control", "velociraptor"),
    ("velociraptor", "support_fraction"): Variant("support_fraction", "velociraptor", support_fraction=0.5),
    # Separate bundle: preserve the existing alive fraction and price flight,
    # hard landings, and incomplete registered support. The support-only arm
    # isolates the alive-fraction change; the bundle cannot attribute its
    # outcome to one of its three terms without a subsequent ablation.
    ("velociraptor", "contact_quality"): Variant(
        "contact_quality", "velociraptor", airborne_weight=0.5, impact_weight=0.25, coverage_weight=0.15
    ),
}


def cop_margin_cost(cop: np.ndarray, loaded: np.ndarray, safe_fraction: float = 0.65) -> float:
    """Worst foot's squared edge deficit, bounded [0, 1]. Missing support costs 1.

    This is a dense per-step surrogate for the gate's worst-foot temporal
    mean, not the identical episode reduction. No left/right averaging can
    hide an edge-loaded foot. A safe region permits ordinary pressure motion.
    """
    coordinate = np.where(loaded & np.isfinite(cop), np.abs(cop), 1.0)
    deficit = np.clip((coordinate - safe_fraction) / (1.0 - safe_fraction), 0.0, 1.0)
    return float(np.max(deficit**2))


class StepFloorProbe:
    """Constant-memory floor measurements, chained after any previous hook."""

    def __init__(self, env: Any, species: str):
        self.env = env.unwrapped
        self.morphology = m = Morphology.from_env(env, species)
        # Reuse the recorder's already-verified normal-force decoder. No
        # buffers are populated and the full recorder is never attached here.
        self.decoder = SubstepContactRecorder(env, m)
        self.decoder._configure_decode(self.env.model)
        self.soles = np.array([f.sole_geom for f in m.feet] + [-2])
        self._had_instance_hook = "_substep_probe_hook" in vars(self.env)
        self._previous_hook = self.env._substep_probe_hook
        self._bound_hook = self._hook
        self.env._substep_probe_hook = self._bound_hook
        self.begin_step()

    def begin_step(self):
        m = self.morphology
        self.count = 0
        self.cop_numerator = np.zeros(m.n_feet)
        self.cop_denominator = np.zeros(m.n_feet)
        self.sole_loaded = np.zeros(m.n_feet, dtype=bool)
        self.leg_forces = np.zeros((m.frame_skip, m.n_feet))
        self.support_loaded = np.zeros((m.frame_skip, len(m.support_geom_ids)))
        self.total_force = np.zeros(m.frame_skip)

    def _hook(self):
        if self._previous_hook is not None:
            self._previous_hook()
        if self.count >= self.morphology.frame_skip:
            raise RuntimeError("begin_step was not called before advancing physics")
        data, model, m = self.env.data, self.env.model, self.morphology
        row = self.count
        self.count += 1
        if data.ncon == 0:
            return
        contact = data.contact
        normal = self.decoder._normal_forces(contact, data.efc_force)
        geoms = contact.geom
        floor1, floor2 = m.is_floor[geoms[:, 0]], m.is_floor[geoms[:, 1]]
        other = np.where(floor1, geoms[:, 1], geoms[:, 0])
        on_floor = (floor1 ^ floor2) & m.is_animal[other]
        foot = m.geom_foot[other]
        leg_floor = on_floor & (foot >= 0)
        self.total_force[row] = float(normal[on_floor].sum())
        self.leg_forces[row] = np.bincount(foot[leg_floor], weights=normal[leg_floor], minlength=m.n_feet)
        support = m.geom_support[other]
        registered = leg_floor & (support >= 0)
        support_force = np.bincount(support[registered], weights=normal[registered], minlength=len(m.support_geom_ids))
        self.support_loaded[row] = support_force > CONTACT_THRESHOLD_N
        on_sole = leg_floor & (other == self.soles[foot])
        for index in np.unique(foot[on_sole]).tolist():
            spec = m.feet[index]
            chosen = np.flatnonzero(on_sole & (foot == index))
            rotation = data.geom_xmat[spec.sole_geom].reshape(3, 3)
            local = (contact.pos[chosen] - data.geom_xpos[spec.sole_geom]) @ rotation
            coordinate = (
                spec.sole_forward_sign
                * local[:, spec.sole_forward_axis]
                / model.geom_size[spec.sole_geom, spec.sole_forward_axis]
            )
            weight = normal[chosen]
            self.cop_numerator[index] += float(np.sum(weight * coordinate))
            self.cop_denominator[index] += float(weight.sum())
            self.sole_loaded[index] |= bool(np.any(weight > CONTACT_THRESHOLD_N))

    def end_step(self):
        m = self.morphology
        if self.count != m.frame_skip:
            raise RuntimeError(f"floor probe saw {self.count}/{m.frame_skip} substeps; hook was displaced")
        cop = np.full(m.n_feet, np.nan)
        np.divide(self.cop_numerator, self.cop_denominator, out=cop, where=self.cop_denominator > 1e-9)
        coverage = [self.support_loaded[:, np.asarray(m.support_geom_foot) == i].mean() for i in range(m.n_feet)]
        return {
            "cop": cop,
            "sole_loaded": self.sole_loaded.copy(),
            "airborne_fraction": float(np.all(self.leg_forces <= CONTACT_THRESHOLD_N, axis=1).mean()),
            "peak_floor_force_bw": float(self.total_force.max() / m.body_weight_n),
            "support_coverage": float(min(coverage)),
        }

    def detach(self):
        if self.env._substep_probe_hook is not self._bound_hook:
            raise RuntimeError("another observer owns the hook; detach it before the experiment wrapper")
        if self._had_instance_hook:
            self.env._substep_probe_hook = self._previous_hook
        else:
            del self.env.__dict__["_substep_probe_hook"]


def reward_deltas(variant: Variant, floor: dict, info: dict, env: Any, step_index: int, settle_steps: int):
    grace_steps = int(np.ceil(SPAWN_GRACE_S / env.dt - 1e-9))
    after_grace = step_index >= grace_steps
    after_settle = step_index >= settle_steps
    support_delta = 0.0
    if variant.support_fraction is not None:
        support_delta = (
            -float(info["raw_alive"])
            * (variant.support_fraction - env.support_conditioned_alive_fraction)
            * (1.0 - float(info["bilateral_support_quality"]))
        )
    return {
        "reward_exp_cop_margin": -variant.cop_weight
        * cop_margin_cost(floor["cop"], floor["sole_loaded"], variant.cop_safe_fraction)
        if variant.cop_weight and after_settle
        else 0.0,
        "reward_exp_support_delta": support_delta,
        "reward_exp_airborne": -variant.airborne_weight * floor["airborne_fraction"] if after_grace else 0.0,
        "reward_exp_floor_impact": -variant.impact_weight
        * max(0.0, floor["peak_floor_force_bw"] - variant.impact_threshold_bw)
        if after_grace
        else 0.0,
        "reward_exp_support_coverage": -variant.coverage_weight * (1.0 - floor["support_coverage"])
        if after_grace
        else 0.0,
    }


class ExperimentalReward(gym.Wrapper):
    """Same plant/observations/actions; altered reward only, explicitly Stance."""

    def __init__(self, env, variant: Variant, *, settle_steps: int):
        super().__init__(env)
        self.variant = variant
        self.settle_steps = settle_steps
        self.probe = StepFloorProbe(env, variant.species)
        if variant.cop_weight and any(
            f.sole_type != int(mujoco.mjtGeom.mjGEOM_BOX) for f in self.probe.morphology.feet
        ):
            self.probe.detach()
            raise ValueError("CoP margin arm requires box soles on every foot")
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.probe.begin_step()
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        floor = self.probe.end_step()
        deltas = reward_deltas(self.variant, floor, info, self.unwrapped, self.steps, self.settle_steps)
        reward = float(base_reward) + sum(deltas.values())
        info = dict(info, **deltas)
        info["experiment_base_reward"] = float(base_reward)
        info["experiment_reward"] = reward
        info["experiment_cop"] = floor["cop"].copy()
        info["experiment_sole_loaded"] = floor["sole_loaded"].copy()
        info["experiment_airborne_fraction"] = floor["airborne_fraction"]
        info["experiment_peak_floor_force_bw"] = floor["peak_floor_force_bw"]
        info["experiment_support_coverage"] = floor["support_coverage"]
        self.steps += 1
        return obs, reward, terminated, truncated, info

    def close(self):
        self.probe.detach()
        self.env.close()
