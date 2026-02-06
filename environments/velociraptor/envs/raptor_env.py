"""
Velociraptor Gymnasium Environment

A bipedal dinosaur locomotion environment with predatory strike behavior.

Observation space:
    - Joint positions (qpos) excluding root freejoint
    - Joint velocities (qvel) excluding root freejoint
    - Pelvis orientation (quaternion)
    - Pelvis angular velocity
    - Pelvis linear velocity
    - Foot contact states
    - Prey relative position
    - Prey distance

Action space:
    - Continuous control for all actuators [-1, 1] normalized

Reward components:
    - Forward velocity
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Tail stability
    - Strike bonus (when claw contacts prey)
"""

import gymnasium as gym
import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


class RaptorEnv(gym.Env):
    """Velociraptor locomotion and strike environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        # Reward weights (tune these!)
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        tail_stability_weight: float = 0.05,
        strike_bonus: float = 500.0,
        strike_approach_weight: float = 0.5,
        # Environment settings
        prey_distance_range: Tuple[float, float] = (3.0, 8.0),
        prey_lateral_range: Tuple[float, float] = (-2.0, 2.0),
        healthy_z_range: Tuple[float, float] = (0.25, 1.0),
    ):
        super().__init__()

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent / "assets" / "raptor.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Reward weights
        self.forward_vel_weight = forward_vel_weight
        self.alive_bonus = alive_bonus
        self.energy_penalty_weight = energy_penalty_weight
        self.fall_penalty = fall_penalty
        self.tail_stability_weight = tail_stability_weight
        self.strike_bonus = strike_bonus
        self.strike_approach_weight = strike_approach_weight

        # Environment settings
        self.prey_distance_range = prey_distance_range
        self.prey_lateral_range = prey_lateral_range
        self.healthy_z_range = healthy_z_range

        # Cache body/geom/site IDs for faster lookup
        self._cache_ids()

        # Define action space (normalized to [-1, 1])
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )

        # Define observation space
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )

        # Rendering
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""
        # Body IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        # Geom IDs for contact detection
        self.prey_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
        self.r_claw_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_claw_geom")
        self.l_claw_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_claw_geom")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.r_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot")
        self.l_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot")
        self.tail_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")

        # Mocap body for prey
        self.prey_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prey")

        # Sensor indices (order matches MJCF definition)
        self._sensor_gyro_start = 0
        self._sensor_accel_start = 3
        self._sensor_quat_start = 6
        self._sensor_r_foot = 10
        self._sensor_l_foot = 11

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (exclude root freejoint: first 7 values are pos + quat)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values are lin + ang vel)
        qvel = self.data.qvel[6:].copy()

        # Pelvis state from sensors
        pelvis_gyro = self.data.sensordata[self._sensor_gyro_start:self._sensor_gyro_start + 3].copy()
        pelvis_accel = self.data.sensordata[self._sensor_accel_start:self._sensor_accel_start + 3].copy()
        pelvis_quat = self.data.sensordata[self._sensor_quat_start:self._sensor_quat_start + 4].copy()

        # Pelvis linear velocity (from root freejoint)
        pelvis_linvel = self.data.qvel[0:3].copy()

        # Foot contact (from touch sensors)
        foot_contact = np.array([
            self.data.sensordata[self._sensor_r_foot],
            self.data.sensordata[self._sensor_l_foot],
        ])

        # Prey info (relative to pelvis)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]  # First (and only) mocap body
        prey_rel = prey_pos - pelvis_pos
        prey_distance = np.linalg.norm(prey_rel)

        # Normalize prey direction
        prey_direction = prey_rel / (prey_distance + 1e-8)

        obs = np.concatenate([
            qpos,                   # Joint positions
            qvel,                   # Joint velocities
            pelvis_quat,            # Orientation (quaternion)
            pelvis_gyro,            # Angular velocity
            pelvis_linvel,          # Linear velocity
            pelvis_accel,           # Accelerometer
            foot_contact,           # Foot contacts
            prey_direction,         # Direction to prey (unit vector)
            [prey_distance],        # Distance to prey (scalar)
        ]).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        # 1. Forward velocity reward
        forward_vel = self.data.qvel[0]  # x-velocity of root
        info["forward_vel"] = forward_vel
        reward_forward = self.forward_vel_weight * forward_vel
        info["reward_forward"] = reward_forward

        # 2. Alive bonus
        reward_alive = self.alive_bonus
        info["reward_alive"] = reward_alive

        # 3. Energy penalty (encourage efficiency)
        energy = np.sum(np.square(action))
        reward_energy = -self.energy_penalty_weight * energy
        info["reward_energy"] = reward_energy

        # 4. Tail stability (penalize high angular velocity at tail tip)
        tail_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_SITE, self.tail_tip_site_id,
            tail_vel, 0
        )
        tail_tip_angvel = tail_vel[3:6]  # Angular velocity (last 3 elements)
        tail_instability = np.linalg.norm(tail_tip_angvel)
        reward_tail = -self.tail_stability_weight * tail_instability
        info["tail_instability"] = tail_instability
        info["reward_tail"] = reward_tail

        # 5. Strike bonus (check claw-prey contact)
        strike_reward = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Check if either claw touched prey
            claw_geoms = {self.r_claw_geom_id, self.l_claw_geom_id}
            if (geom1 in claw_geoms and geom2 == self.prey_geom_id) or \
               (geom2 in claw_geoms and geom1 == self.prey_geom_id):
                strike_reward = self.strike_bonus
                info["strike_success"] = 1.0
                break
        else:
            info["strike_success"] = 0.0

        reward_strike = strike_reward
        info["reward_strike"] = reward_strike

        # 6. Approach shaping (smooth gradient toward prey for claw strike)
        pelvis_pos = self.data.xpos[self.pelvis_id]
        prey_pos = self.data.mocap_pos[0]
        prey_distance = np.linalg.norm(prey_pos - pelvis_pos)
        reward_approach = -self.strike_approach_weight * prey_distance
        info["prey_distance"] = prey_distance
        info["reward_approach"] = reward_approach

        # Total reward
        total_reward = (
            reward_forward + reward_alive + reward_energy
            + reward_tail + reward_strike + reward_approach
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        # Get pelvis height
        pelvis_z = self.data.xpos[self.pelvis_id, 2]
        info["pelvis_height"] = pelvis_z

        # Termination: pelvis too low (fallen)
        if pelvis_z < self.healthy_z_range[0]:
            info["termination_reason"] = "fallen"
            return True, info

        # Termination: pelvis too high (shouldn't happen, but safety check)
        if pelvis_z > self.healthy_z_range[1]:
            info["termination_reason"] = "too_high"
            return True, info

        # Check for torso-ground contact (fallen over)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Torso touching floor = fallen
            if (geom1 == self.torso_geom_id and geom2 == self.floor_geom_id) or \
               (geom2 == self.torso_geom_id and geom1 == self.floor_geom_id):
                info["termination_reason"] = "torso_contact"
                return True, info

        return False, info

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)."""
        return self._step_count >= self.max_episode_steps

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Scale action from [-1, 1] to actuator control ranges
        ctrl = self._scale_action(action)
        self.data.ctrl[:] = ctrl

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, reward_info = self._get_reward_info(action)

        # Check termination
        terminated, term_info = self._is_terminated()
        if terminated:
            reward += self.fall_penalty
            reward_info["reward_total"] = reward

        truncated = self._is_truncated()

        # Combine info
        info = {**reward_info, **term_info}
        info["step"] = self._step_count

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [-1, 1] to actuator control range."""
        ctrl_range = self.model.actuator_ctrlrange
        ctrl_min = ctrl_range[:, 0]
        ctrl_max = ctrl_range[:, 1]

        # Linear interpolation from [-1, 1] to [min, max]
        scaled = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        return scaled

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbation to initial pose
        if self.np_random is not None:
            noise_scale = 0.01
            self.data.qpos[7:] += self.np_random.uniform(
                -noise_scale, noise_scale, size=self.data.qpos[7:].shape
            )
            self.data.qvel[:] += self.np_random.uniform(
                -noise_scale, noise_scale, size=self.data.qvel.shape
            )

        # Randomize prey position
        self._spawn_prey()

        # Forward pass to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        obs = self._get_obs()
        info = {"step": 0}

        return obs, info

    def _spawn_prey(self):
        """Spawn prey at random location ahead of raptor."""
        if self.np_random is not None:
            distance = self.np_random.uniform(*self.prey_distance_range)
            lateral = self.np_random.uniform(*self.prey_lateral_range)
        else:
            distance = np.mean(self.prey_distance_range)
            lateral = 0.0

        # Prey position (relative to origin, raptor starts at origin)
        prey_pos = np.array([distance, lateral, 0.3])
        self.data.mocap_pos[0] = prey_pos

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.distance = 3.0
                self._viewer.cam.azimuth = 135
                self._viewer.cam.elevation = -20
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register with Gymnasium
gym.register(
    id="Raptor-v0",
    entry_point="envs.raptor_env:RaptorEnv",
    max_episode_steps=1000,
)
