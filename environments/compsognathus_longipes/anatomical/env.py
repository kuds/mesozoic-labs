"""The anatomical Compsognathus longipes (decision D-D33).

The species base
:class:`~environments.compsognathus_longipes.base_env.CompsognathusLongipesEnv`
with the soft-cubic leg residual (policy interface r3, D-D26), the
heading-free observation (``bipedal-target-heading-free/v1``, policy
interface r4, D-D31) and the stance-quality reward terms.  The robot keeps
the species base's linear residual and world-frame observation.
``environments/__init__.py`` registers this class with Gymnasium from the
species manifest.
"""

from __future__ import annotations

from typing import cast

import mujoco
import numpy as np

from environments.compsognathus_longipes.base_env import CompsognathusLongipesEnv
from environments.shared import reward_functions
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.heading_observation import heading_free_root_segments
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT


class CompsognathusLongipesAnatomicalEnv(CompsognathusLongipesEnv):
    """The anatomical model as the species trains it: soft-cubic leg residuals.

    Every leg servo saturates at forcerange / kp = 0.05 rad of position error,
    yet home-keyframe-residual/v1 spreads +-1 over the whole 0.35-1.0 rad
    half-ranges, so 0.007-0.015 of action is already the full knee or ankle
    holding torque and 86-93% of the action range is saturated torque.  On
    the physics r1 plant the zero-action statue survived white action noise
    only to sigma 0.01: at PPO's initial sigma 0.135 (log_std_init -2.0) it
    fell within about ten steps, PPO never sampled the quiet stance, and both
    certified r1 stances were noise-robust tiptoe marches.  The ten leg
    residuals (hip pitch, hip roll, knee, ankle and toe on both legs, found by
    actuator name) are therefore shaped before the unchanged home-keyframe span:
    ``b = s*a + (1 - s)*a**3`` with ``s = residual_linear_slope = 0.1``.
    ``b(0) = 0`` keeps action zero at the gravity-preloaded home ctrl,
    ``b(+-1) = +-1`` keeps both ctrlrange ends for recovery steps and the walk,
    and ``b' = s + 3 (1 - s) a**2 > 0`` keeps the map monotone with a tenth of
    the linear slope at home.  A pure cubic (``s = 0``) leaves a dead zone for
    small corrections; a linear residual scaled by 0.1 reaches only 2-6 deg
    past home.  Neck, jaw and tail keep the linear residual.

    Zero-mean white action noise on the physics r2 statue (stance [env],
    seeds 3042+i, paired noise streams): sigma 0.02 20/20 full episodes
    (the r1 plant on the linear map 5/20), 0.10 20/20, 0.135 39/40 (151/160
    over four seed blocks; r1 0/20, median 12 steps), 0.20 0/20.  The cliff
    between the recipe's initial sigma and 0.20 is a constraint on
    log_std_init and on any entropy schedule.

    The robot keeps CompsognathusLongipesEnv's linear residual and its digests: this
    is policy interface r3 of the anatomical species only (D-D26).  Policy
    interface r4 (D-D31) adds the anatomical-only heading-free ``_get_obs``
    below; the robot keeps the inherited world-frame observation.

    The constructor adds the stance-quality reward terms on top of the
    inherited reward, every one inert at its default (bilateral support and
    support-conditioned alive bonus, sole flatness and stance width gated on
    each foot's load, leg home pose, support-geom coverage, floor impact,
    airborne substeps, action saturation and jerk); the stance task's weights
    and their measurements are in ``configs/compsognathus_longipes/anatomical/stance.toml``.  Their
    ids are resolved in ``__init__``, not in ``_cache_ids``: this SB3-only
    species fingerprints ``_cache_ids`` as part of its policy interface, and a
    reward target is not part of what a checkpoint observes.  The shaped
    residual mask is built there too, so the interface keeps hashing the
    inherited ``CompsognathusLongipesEnv._cache_ids`` (the home ctrl and the contact
    threshold it sets), and the mask enters the digest as the action mapping's
    ``residual_shaping``.
    """

    embodiment = "anatomical"
    MODEL_PATH = ModelId("compsognathus_longipes", "anatomical").asset_path(ENVIRONMENTS_ROOT)
    nominal_pelvis_height = 0.244391440454
    #: The floor-truth foot registry entry (``gait/morphology.py``): the geoms
    #: a flat, loaded foot stands on, the sole and the foot's reference site,
    #: as name templates.  It equals the generic answer: plantar pad plus
    #: d2-d4.  The metatarsus is a TERMINATING geom (``_body_ground_geoms``),
    #: not support.
    FOOT_SUPPORT = {
        "support": ("{s}_plantar_pad", "{s}_toe_d2_geom", "{s}_toe_d3_geom", "{s}_toe_d4_geom"),
        "sole": "{s}_plantar_pad",
        "foot_site": "{s}_foot_touch_volume",
    }
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    REWARD_PROBE = (0.01, "body", "pelvis")
    action_mapping = "home-keyframe-residual-softcubic/v1"
    #: Slope of the shaped leg residual at home, ``b'(0)``; 1.0 is the linear map.
    residual_linear_slope = 0.1
    #: Joints whose actuators ``{r,l}_<joint>_act`` get the shaped residual.
    shaped_residual_joints = ("hip_pitch", "hip_roll", "knee", "ankle", "toe")
    #: Joints the leg_home_pose term holds at the home keyframe, both legs.
    leg_home_pose_joints = ("hip_pitch", "hip_roll", "knee", "ankle", "toe")
    #: Each foot's support geoms: FOOT_SUPPORT's support set without the
    #: ``{s}_`` foot prefix, resolved per foot by this module, which does not
    #: import the gait package.
    support_geoms = ("plantar_pad", "toe_d2_geom", "toe_d3_geom", "toe_d4_geom")
    _SUPPORT_FORCE_AGGREGATIONS = ("min", "mean")
    _STANCE_WIDTH_REFERENCES = ("spawn", "settled")
    #: A floor contact carries load above this normal force, and a foot is
    #: airborne at or below it: the floor-truth library's CONTACT_THRESHOLD_N.
    _CONTACT_THRESHOLD_N = 0.1
    #: A support geom is loaded on a step when it carries load on at least
    #: this share of the step's substeps (SUPPORT_GEOM_DOWN_FRACTION).
    _SUPPORT_GEOM_DOWN_FRACTION = 0.5
    #: The floor-impact term charges a step's peak at most this far above its
    #: threshold, in body weights: a crushing contact at a fall reads tens of
    #: body weights, and the fall is the fall penalty's to price.
    _FLOOR_IMPACT_EXCESS_CAP_BW = 1.0

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 1000,
        forward_vel_weight: float = 0.0,
        forward_vel_max: float = 0.25,
        alive_bonus: float = 1.0,
        energy_penalty_weight: float = 0.02,
        fall_penalty: float = -25.0,
        posture_weight: float = 1.0,
        height_weight: float = 1.0,
        home_pose_weight: float = 0.15,
        smoothness_weight: float = 0.02,
        spin_penalty_weight: float = 0.02,
        drift_penalty_weight: float = 0.5,
        lateral_penalty_weight: float = 0.1,
        heading_weight: float = 0.0,
        gait_symmetry_weight: float = 0.0,
        target_approach_weight: float = 0.0,
        target_reach_bonus: float = 0.0,
        # Stance-quality terms.  Every default is inert: zero weights (and
        # alive fraction) reproduce CompsognathusLongipesEnv's reward bit for bit.
        # The weights, tolerances and saturation force are named as on
        # TyrannosaurusRexEnv, and so, since D-D27, are the stance-width reference, the
        # per-foot load gate (foot_terms_min_support_force) and the
        # floor-impact and airborne terms, in this species' own forms (the
        # gate reads support_force_aggregation's load; the impact is capped);
        # support_force_aggregation and the support-geom coverage are this
        # species' own (D-D26).  Tolerances default to the stance task's
        # calibrated values.
        bilateral_support_weight: float = 0.0,
        foot_contact_saturation_force: float = 4.4,
        support_force_aggregation: str = "min",
        support_conditioned_alive_fraction: float = 0.0,
        foot_flatness_weight: float = 0.0,
        foot_flatness_tolerance_deg: float = 2.0,
        stance_width_weight: float = 0.0,
        stance_width_tolerance_m: float = 0.03,
        stance_width_reference: str = "settled",
        stance_width_settle_steps: int = 200,
        foot_terms_min_support_force: float = 0.0,
        leg_home_pose_weight: float = 0.0,
        leg_home_pose_tolerance: float = 0.15,
        support_geom_coverage_weight: float = 0.0,
        floor_impact_weight: float = 0.0,
        floor_impact_threshold_bw: float = 2.5,
        airborne_substep_weight: float = 0.0,
        action_saturation_weight: float = 0.0,
        action_saturation_threshold: float = 0.9,
        action_jerk_weight: float = 0.0,
        target_radius: float = 0.08,
        target_max_speed: float = 0.10,
        prey_distance_range: tuple[float, float] = (0.6, 1.0),
        prey_lateral_range: tuple[float, float] = (-0.2, 0.2),
        healthy_z_range: tuple[float, float] | None = None,
        max_tilt_angle: float = 0.7,
        reset_noise_scale: float = 0.01,
        reset_height_noise_scale: float = 0.0,
        perturbation_capture_velocity_multiple: float = 0.0,
        perturbation_interval: float = 2.0,
        perturbation_jitter: float = 0.5,
        perturbation_duration: float = 0.20,
        perturbation_direction: str = "uniform_horizontal",
        command_mode: str = "none",
        command_config: DirectionCommandConfig | None = None,
    ):
        if support_force_aggregation not in self._SUPPORT_FORCE_AGGREGATIONS:
            raise ValueError(f"support_force_aggregation must be one of {self._SUPPORT_FORCE_AGGREGATIONS}")
        if stance_width_reference not in self._STANCE_WIDTH_REFERENCES:
            raise ValueError(f"stance_width_reference must be one of {self._STANCE_WIDTH_REFERENCES}")
        # No upper bound against max_episode_steps: a horizon shortened below
        # the settle (a smoke run, a short report panel) simply never pays it.
        if (
            isinstance(stance_width_settle_steps, bool)
            or not isinstance(stance_width_settle_steps, int)
            or stance_width_settle_steps < 1
        ):
            raise ValueError("stance_width_settle_steps must be a positive integer")
        for name, value in (
            ("bilateral_support_weight", bilateral_support_weight),
            ("foot_flatness_weight", foot_flatness_weight),
            ("stance_width_weight", stance_width_weight),
            ("foot_terms_min_support_force", foot_terms_min_support_force),
            ("leg_home_pose_weight", leg_home_pose_weight),
            ("support_geom_coverage_weight", support_geom_coverage_weight),
            ("floor_impact_weight", floor_impact_weight),
            ("airborne_substep_weight", airborne_substep_weight),
            ("action_saturation_weight", action_saturation_weight),
            ("action_jerk_weight", action_jerk_weight),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("foot_contact_saturation_force", foot_contact_saturation_force),
            ("foot_flatness_tolerance_deg", foot_flatness_tolerance_deg),
            ("stance_width_tolerance_m", stance_width_tolerance_m),
            ("leg_home_pose_tolerance", leg_home_pose_tolerance),
            ("floor_impact_threshold_bw", floor_impact_threshold_bw),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= support_conditioned_alive_fraction <= 1.0:
            raise ValueError("support_conditioned_alive_fraction must be in [0, 1]")
        if not 0.0 <= action_saturation_threshold < 1.0:
            raise ValueError("action_saturation_threshold must be in [0, 1)")
        self.bilateral_support_weight = bilateral_support_weight
        self.foot_contact_saturation_force = foot_contact_saturation_force
        self.support_force_aggregation = support_force_aggregation
        self.support_conditioned_alive_fraction = support_conditioned_alive_fraction
        self.foot_flatness_weight = foot_flatness_weight
        self.foot_flatness_tolerance_deg = foot_flatness_tolerance_deg
        self.stance_width_weight = stance_width_weight
        self.stance_width_tolerance_m = stance_width_tolerance_m
        self.stance_width_reference = stance_width_reference
        self.stance_width_settle_steps = stance_width_settle_steps
        self.foot_terms_min_support_force = foot_terms_min_support_force
        self.leg_home_pose_weight = leg_home_pose_weight
        self.leg_home_pose_tolerance = leg_home_pose_tolerance
        self.support_geom_coverage_weight = support_geom_coverage_weight
        self.floor_impact_weight = floor_impact_weight
        self.floor_impact_threshold_bw = floor_impact_threshold_bw
        self.airborne_substep_weight = airborne_substep_weight
        self.action_saturation_weight = action_saturation_weight
        self.action_saturation_threshold = action_saturation_threshold
        self.action_jerk_weight = action_jerk_weight
        # Per-step state of the coverage accumulator (_accumulate_substep):
        # each substep's support-geom floor contacts, tagged with the step
        # count they belong to.
        self._support_geom_records: list[tuple[np.ndarray, ...]] = []
        self._support_geom_counts_step = -1

        super().__init__(
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            forward_vel_weight=forward_vel_weight,
            forward_vel_max=forward_vel_max,
            alive_bonus=alive_bonus,
            energy_penalty_weight=energy_penalty_weight,
            fall_penalty=fall_penalty,
            posture_weight=posture_weight,
            height_weight=height_weight,
            home_pose_weight=home_pose_weight,
            smoothness_weight=smoothness_weight,
            spin_penalty_weight=spin_penalty_weight,
            drift_penalty_weight=drift_penalty_weight,
            lateral_penalty_weight=lateral_penalty_weight,
            heading_weight=heading_weight,
            gait_symmetry_weight=gait_symmetry_weight,
            target_approach_weight=target_approach_weight,
            target_reach_bonus=target_reach_bonus,
            target_radius=target_radius,
            target_max_speed=target_max_speed,
            prey_distance_range=prey_distance_range,
            prey_lateral_range=prey_lateral_range,
            healthy_z_range=healthy_z_range,
            max_tilt_angle=max_tilt_angle,
            reset_noise_scale=reset_noise_scale,
            reset_height_noise_scale=reset_height_noise_scale,
            perturbation_capture_velocity_multiple=perturbation_capture_velocity_multiple,
            perturbation_interval=perturbation_interval,
            perturbation_jitter=perturbation_jitter,
            perturbation_duration=perturbation_duration,
            perturbation_direction=perturbation_direction,
            command_mode=command_mode,
            command_config=command_config,
        )

        # The shaped-residual mask, by actuator name.  Built here, not in an
        # override of _cache_ids: this SB3-only species' policy interface
        # hashes the _cache_ids it is handed, and an override would put its own
        # few lines there in place of CompsognathusLongipesEnv._cache_ids, so an edit
        # of the parent's (the home ctrl, the contact threshold) would no
        # longer move this interface.  The mask itself is in the digest, as
        # the action mapping's residual_shaping.  The behavior env's model swap
        # re-runs _cache_ids only; its scenes share these actuator names.
        names = [self.model.actuator(index).name for index in range(self.model.nu)]
        shaped = {side + "_" + joint + "_act" for side in ("r", "l") for joint in self.shaped_residual_joints}
        if not shaped.issubset(names):
            raise ValueError("soft-cubic leg residual: no actuator named " + ", ".join(sorted(shaped - set(names))))
        self._shaped_residual_mask = np.array([name in shaped for name in names])

        # The reward-term ids are resolved here rather than in _cache_ids,
        # whose tokens are part of this SB3-only species' policy-interface
        # digest: a reward target is not part of what a checkpoint observes.
        model = self.model
        if model.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
            raise ValueError("support-geom coverage decodes elliptic contact forces; the model's cone changed")
        sides = ("r", "l")
        self._sole_geom_ids = np.array([model.geom(side + "_plantar_pad").id for side in sides])
        self._stance_site_ids = np.array([model.site(side + "_foot_touch_volume").id for side in sides])
        leg_joints = [side + "_" + joint for side in sides for joint in self.leg_home_pose_joints]
        self._leg_home_qpos_indices = np.array([int(model.jnt_qposadr[model.joint(name).id]) for name in leg_joints])
        self._leg_home_qpos = self._home_qpos[self._leg_home_qpos_indices].copy()
        # Support-geom slots: right foot first, in support_geoms order; -1 off the support set.
        support_ids = [model.geom(side + "_" + name).id for side in sides for name in self.support_geoms]
        self._support_geom_ids = np.array(support_ids)
        # A contact's slot by its geom pair: the support geom's when the other
        # geom is the floor, -1 for every other pair (one lookup per substep).
        slot = np.full(model.ngeom, -1, dtype=np.int64)
        slot[support_ids] = np.arange(len(support_ids))
        self._floor_contact_slot = np.full((model.ngeom, model.ngeom), -1, dtype=np.int64)
        self._floor_contact_slot[self.floor_geom_id, :] = slot
        self._floor_contact_slot[:, self.floor_geom_id] = slot
        self._support_geom_is_digit = np.array([name.startswith("toe_") for _ in sides for name in self.support_geoms])
        # Which way along each digit capsule's local z axis its tip lies (+1 or
        # -1; 0 for the pad), read off the model, not off a frame convention:
        # the tip is the end away from the MTP joint, so it lies on the side
        # of the capsule's centre that points away from the joint.  (A fromto
        # capsule's z axis runs from its "to" end back to its "from" end: here
        # from the tip to the joint, so every digit reads -1.)
        tipward = np.zeros(len(support_ids))
        axis = np.zeros(3)
        for index, geom in enumerate(support_ids):
            if self._support_geom_is_digit[index]:
                joint = int(model.body_jntadr[model.geom_bodyid[geom]])
                mujoco.mju_rotVecQuat(axis, np.array([0.0, 0.0, 1.0]), model.geom_quat[geom])
                tipward[index] = np.sign(float((model.geom_pos[geom] - model.jnt_pos[joint]) @ axis))
        if not np.all(tipward[self._support_geom_is_digit]):
            raise ValueError("support-geom coverage: a digit capsule is centred on its MTP joint")
        self._support_geom_tipward = tipward
        self._body_weight_n = self.body_mass * abs(float(model.opt.gravity[2]))
        # Scratch data for the spawn width (_spawn_target); until the first
        # reset the width target is the home keyframe's own.
        self._width_data = mujoco.MjData(model)
        self._width_data.qpos[:] = self._home_qpos
        mujoco.mj_kinematics(model, self._width_data)
        self._spawn_stance_width = self._stance_width_target = self._stance_width(self._width_data)

    def _get_obs(self) -> np.ndarray:
        """The bipedal-target-heading-free/v1 observation (55 values; policy interface r4, D-D31).

        CompsognathusLongipesEnv's layout with its three world-frame blocks replaced:
        projected gravity for the pelvis quaternion, the root's world linear
        velocity in the gravity-aligned heading frame, and the target
        direction in pelvis coordinates, all from the free root's own
        quaternion qpos[3:7] (environments/shared/heading_observation.py).
        The gyro and accelerometer stay in their own site frame.  Overridden
        here, not in CompsognathusLongipesEnv, so the robot keeps its observation and
        its digests.
        """
        relative = self.data.mocap_pos[self._target_mocap_id] - self.data.xpos[self.pelvis_id]
        distance = np.linalg.norm(relative)
        sensors = self.data.sensordata
        gravity, heading_velocity, target_direction_body = heading_free_root_segments(
            self.data.qpos, self.data.qvel[:3], relative / (distance + 1e-8)
        )
        return np.asarray(
            np.concatenate(
                [
                    self.data.qpos[7:],
                    self.data.qvel[6:],
                    gravity,
                    sensors[self._sensor_gyro_start : self._sensor_gyro_start + 3],
                    heading_velocity,
                    sensors[self._sensor_accel_start : self._sensor_accel_start + 3],
                    self._foot_contact_forces(),
                    target_direction_body,
                    [distance],
                    self._command,  # Body-relative command (v_x, v_y, yaw_rate), pre-scaled; zeros unless command_mode != "none"
                ]
            ),
            dtype=np.float32,
        )

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        residual = np.clip(action, -1.0, 1.0)
        slope = self.residual_linear_slope
        residual = np.where(self._shaped_residual_mask, slope * residual + (1.0 - slope) * residual**3, residual)
        minimum, maximum = self.model.actuator_ctrlrange.T
        return cast(
            np.ndarray,
            self._home_ctrl
            + np.where(residual < 0, residual * (self._home_ctrl - minimum), residual * (maximum - self._home_ctrl)),
        )

    def _spawn_target(self) -> None:
        super()._spawn_target()
        # The spawn's own stance width: reset has applied the joint jitter
        # and leaves the root's planar pose alone, so forward kinematics of
        # this qpos on a scratch MjData gives the width the animal is born
        # with (self.data is not forwarded yet, and is left untouched).
        self._width_data.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, self._width_data)
        self._spawn_stance_width = self._stance_width(self._width_data)
        # Under "settled" the spawn width only stands in (unpaid) until the
        # settle ends and the animal's own width replaces it.
        self._stance_width_target = self._spawn_stance_width
        self._support_geom_counts_step = -1

    def _stance_width(self, data: mujoco.MjData) -> float:
        """Planar (x-y) distance between the two foot sites (the gate's stance width), free of heading."""
        right, left = data.site_xpos[self._stance_site_ids, :2]
        return float(np.linalg.norm(right - left))

    def _sole_tilts_deg(self) -> np.ndarray:
        """Each plantar pad's tilt from level, in degrees (right, left): the box's local z against vertical."""
        cos_tilt = np.clip(self.data.geom_xmat[self._sole_geom_ids, 8], -1.0, 1.0)
        return np.asarray(np.degrees(np.arccos(cos_tilt)))

    def _support_geom_contacts(self) -> tuple[np.ndarray, ...]:
        """The current ``data``'s floor contacts on the support geoms, copied: what the coverage reads.

        ``(slot, normal force, position)`` per contact, and the support geoms'
        centres and frames.  Only these rows are copied (``data.contact``
        fields are views the next ``mj_step`` overwrites); the classification
        runs once per control step over all of a step's substeps
        (:meth:`_support_geoms_loaded_on`).  The force is decoded as
        mj_contactForce does for an elliptic cone, ``efc_force[efc_address]``,
        0 for an excluded contact.
        """
        data = self.data
        contact = data.contact
        pairs = contact.geom
        slots = self._floor_contact_slot[pairs[:, 0], pairs[:, 1]]
        rows = np.flatnonzero(slots >= 0)
        address = contact.efc_address[rows]
        return (
            slots[rows],
            data.efc_force[address] * (address >= 0),
            contact.pos[rows],
            data.geom_xpos[self._support_geom_ids],
            data.geom_xmat[self._support_geom_ids],
        )

    def _support_geoms_loaded_on(self, records: list[tuple[np.ndarray, ...]]) -> np.ndarray:
        """Which support geoms carry load on the floor in each record, ``(len(records), slots)`` bool.

        As the floor-truth recorder reads a geom: the normal forces of its
        floor contacts, summed, above the contact threshold.  Each digit
        capsule starts at the MTP joint, under the pad, so at the statue its
        joint-end contact carries load whether or not the digit reaches the
        floor; a digit therefore sums only its contacts on the DISTAL half
        (beyond the capsule's centre, away from the joint), so a foot rocked
        back onto its pad with the digit tips up does not read as digit
        support, and a digit loaded at its tip always does.  The gate counts
        a digit through either end.
        """
        n_slots = len(self._support_geom_ids)
        counts = [len(item[0]) for item in records]
        if not sum(counts):
            return np.zeros((len(records), n_slots), dtype=bool)
        record = np.repeat(np.arange(len(records)), counts)
        slots = np.concatenate([item[0] for item in records])
        normal = np.concatenate([item[1] for item in records])
        pos = np.concatenate([item[2] for item in records])
        centre = np.stack([item[3] for item in records])[record, slots]
        # Each contact's capsule axis, pointing to the tip (zero on the pads).
        tipward = np.stack([item[4] for item in records])[record, slots][:, 2::3]
        axial = np.einsum("ij,ij->i", pos - centre, tipward) * self._support_geom_tipward[slots]
        counted = ~self._support_geom_is_digit[slots] | (axial > 0.0)
        force = np.bincount(record * n_slots + slots, weights=normal * counted, minlength=len(records) * n_slots)
        return np.asarray(force.reshape(len(records), n_slots) > self._CONTACT_THRESHOLD_N)

    def _support_geoms_loaded(self) -> np.ndarray:
        """Which support geoms carry load on the floor in the current ``data``, one bool per slot."""
        return np.asarray(self._support_geoms_loaded_on([self._support_geom_contacts()])[0])

    def _accumulate_substep(self, substep: int) -> None:
        # Each substep's support-geom floor contacts, for the coverage term:
        # the gate's per-step definition (a geom loaded on at least half the
        # substeps) rather than the boundary sample, which a control-locked
        # rock could time.  Only copies here; skipped when the term is off.
        if self.support_geom_coverage_weight <= 0.0:
            return
        if substep == 0:
            self._support_geom_records = []
            self._support_geom_counts_step = self._step_count + 1
        self._support_geom_records.append(self._support_geom_contacts())

    def _support_geom_counts(self) -> np.ndarray:
        """Substeps of the current control step on which each support geom carried load."""
        return np.asarray(self._support_geoms_loaded_on(self._support_geom_records).sum(axis=0))

    def _support_geom_coverage(self) -> np.ndarray:
        """Each foot's share of its support geoms loaded on the current step, (right, left).

        From the step's substeps when they belong to this step, and from the
        current contacts otherwise (a state scored outside step()).
        """
        fresh = self._support_geom_counts_step == self._step_count == self._substep_contact_step
        if fresh:
            loaded = self._support_geom_counts() >= self._SUPPORT_GEOM_DOWN_FRACTION * self.frame_skip
        else:
            loaded = self._support_geoms_loaded()
        return np.asarray(loaded.reshape(2, -1).mean(axis=1))

    def _get_reward_info(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        # Jerk BEFORE the legacy reward: its smoothness term rotates the
        # action history, which would leave the jerk reading this step's own
        # action as its first lag.
        reward_action_jerk, action_jerk = self._reward_action_jerk(action)
        reward, info = super()._get_reward_info(action)

        # Per-foot load.  The legacy alive, posture and height terms above
        # keep their support gate on the per-substep MIN (the sum of the two
        # feet's MIN above 4% of body weight): a step on which both feet
        # unload on some substep, together or in turn, reads as unsupported,
        # which is what prices a hop.  A one-foot unload leaves the other
        # foot's MIN (about 4.9 N at the statue) above that bar, so a
        # one-foot chatter keeps the legacy terms whole and only the
        # bilateral and coverage terms below price it (a right foot lifted
        # for one substep 130-190 times an episode keeps 0.972 of the
        # statue).  support_force_aggregation feeds ONLY the new terms below
        # (the bilateral term, the alive conditioning and the flatness/width
        # load gate); "mean" reads the step's substep MEAN, so a foot that
        # unloads on one substep of ten keeps nine tenths of its load
        # instead of reading zero.
        block = self._substep_foot_force_block()
        right_min, left_min = self._aggregated_foot_contact_forces()
        if self.support_force_aggregation == "mean":
            support_forces = block.mean(axis=0)
        else:
            support_forces = np.asarray((right_min, left_min))
        _, support_quality = reward_functions.reward_bilateral_support(
            support_forces, self.foot_contact_saturation_force, 1.0
        )
        bilateral_support_quality = float(support_quality)
        reward_bilateral_support = self.bilateral_support_weight * bilateral_support_quality

        legacy_alive = info["reward_alive"]
        alive_fraction = self.support_conditioned_alive_fraction
        alive_gate = (1.0 - alive_fraction) + alive_fraction * bilateral_support_quality
        reward_alive = legacy_alive * alive_gate

        # A foot earns its flatness share, and the pair the width term, only
        # while it carries foot_terms_min_support_force on the same aggregate:
        # an airborne or grazing foot held level must not earn them.
        if self.foot_terms_min_support_force > 0.0:
            foot_supported = support_forces >= self.foot_terms_min_support_force
        else:
            foot_supported = np.ones(2, dtype=bool)

        sole_tilts = self._sole_tilts_deg()
        foot_flatness = np.exp(-np.square(sole_tilts / self.foot_flatness_tolerance_deg))
        foot_flatness_quality = float(np.mean(np.where(foot_supported, foot_flatness, 0.0)))
        reward_foot_flatness = self.foot_flatness_weight * foot_flatness_quality

        stance_width = self._stance_width(self.data)
        if self.stance_width_reference == "settled" and self._step_count == self.stance_width_settle_steps:
            self._stance_width_target = stance_width
        width_paid = self.stance_width_reference == "spawn" or self._step_count > self.stance_width_settle_steps
        _, stance_width_error, width_quality = reward_functions.reward_stance_width(
            np.asarray(stance_width), self._stance_width_target, self.stance_width_tolerance_m, 1.0
        )
        stance_width_quality = float(width_quality) * float(foot_supported.all() and width_paid)
        reward_stance_width = self.stance_width_weight * stance_width_quality

        _, leg_rms_error, leg_quality = reward_functions.reward_soft_home_pose(
            self.data.qpos[self._leg_home_qpos_indices], self._leg_home_qpos, self.leg_home_pose_tolerance, 1.0
        )
        reward_leg_home_pose = self.leg_home_pose_weight * float(leg_quality)

        if self.support_geom_coverage_weight > 0.0:
            coverage = self._support_geom_coverage()
            support_geom_coverage = float(np.mean(coverage))
            info["r_support_geom_coverage"] = float(coverage[0])
            info["l_support_geom_coverage"] = float(coverage[1])
            info["support_geom_coverage"] = support_geom_coverage
            reward_support_geom_coverage = -self.support_geom_coverage_weight * (1.0 - support_geom_coverage)
        else:
            reward_support_geom_coverage = 0.0

        # Floor impact and airborne substeps, on every substep of every step
        # from the first (the settle included): the summed foot force is the
        # animal's floor force on this plant (the metatarsus and every body
        # geom terminate on contact).
        peak_foot_force_bw = float(block.sum(axis=1).max()) / self._body_weight_n
        airborne_substeps = int(np.count_nonzero(np.all(block <= self._CONTACT_THRESHOLD_N, axis=1)))
        impact_excess = min(
            max(0.0, peak_foot_force_bw - self.floor_impact_threshold_bw), self._FLOOR_IMPACT_EXCESS_CAP_BW
        )
        reward_floor_impact = -self.floor_impact_weight * impact_excess
        reward_airborne_substeps = -self.airborne_substep_weight * airborne_substeps / self.frame_skip

        reward_action_saturation, action_saturation = reward_functions.reward_action_saturation(
            action, self.action_saturation_weight, self.action_saturation_threshold
        )

        reward += (
            (reward_alive - legacy_alive)
            + reward_bilateral_support
            + reward_foot_flatness
            + reward_stance_width
            + reward_leg_home_pose
            + reward_support_geom_coverage
            + reward_floor_impact
            + reward_airborne_substeps
            + float(reward_action_saturation)
            + reward_action_jerk
        )
        info.update(
            {
                "raw_alive": legacy_alive,
                "alive_gate": alive_gate,
                "reward_alive": reward_alive,
                "bilateral_support_quality": bilateral_support_quality,
                "reward_bilateral_support": reward_bilateral_support,
                "r_sole_tilt_deg": float(sole_tilts[0]),
                "l_sole_tilt_deg": float(sole_tilts[1]),
                "foot_flatness_quality": foot_flatness_quality,
                "reward_foot_flatness": reward_foot_flatness,
                "stance_width": stance_width,
                "stance_width_target": self._stance_width_target,
                "stance_width_error": float(stance_width_error),
                "stance_width_quality": stance_width_quality,
                "reward_stance_width": reward_stance_width,
                "leg_home_pose_error": float(leg_rms_error),
                "leg_home_pose_quality": float(leg_quality),
                "reward_leg_home_pose": reward_leg_home_pose,
                "reward_support_geom_coverage": reward_support_geom_coverage,
                "peak_foot_force_bw": peak_foot_force_bw,
                "airborne_substeps": airborne_substeps,
                "reward_floor_impact": reward_floor_impact,
                "reward_airborne_substeps": reward_airborne_substeps,
                "action_saturation": float(action_saturation),
                "reward_action_saturation": float(reward_action_saturation),
                "action_jerk": action_jerk,
                "reward_action_jerk": reward_action_jerk,
                "reward_total": reward,
            }
        )
        return reward, info
