"""
Brachiosaurus Gymnasium Environment

A quadrupedal sauropod locomotion environment with food-reaching behavior.

Observation space:
    - Joint positions (qpos) excluding root freejoint
    - Joint velocities (qvel) excluding root freejoint
    - Torso orientation (quaternion)
    - Torso angular velocity
    - Torso linear velocity
    - Foot contact states (4 feet)
    - Food relative position
    - Food distance

Action space:
    - Continuous control for all actuators [-1, 1] normalized
    - 22 actuators: 6 neck + 16 legs (4 per leg)

Reward components:
    - Forward velocity toward food
    - Alive bonus
    - Fall penalty
    - Energy penalty
    - Gait stability (encourage coordinated quadrupedal gait)
    - Food reach bonus (when head gets close to food)
"""

import gymnasium as gym
import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


class BrachioEnv(gym.Env):
    """Brachiosaurus quadrupedal locomotion and food-reaching environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        # Reward weights
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        gait_stability_weight: float = 0.05,
        food_reach_bonus: float = 500.0,
        food_reach_threshold: float = 0.5,
        # Environment settings
        food_distance_range: Tuple[float, float] = (3.0, 8.0),
        food_lateral_range: Tuple[float, float] = (-2.0, 2.0),
        food_height_range: Tuple[float, float] = (2.0, 4.0),
        healthy_z_range: Tuple[float, float] = (1.0, 3.5),
    ):
        super().__init__()

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent / "assets" / "brachiosaurus.xml"
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
        self.gait_stability_weight = gait_stability_weight
        self.food_reach_bonus = food_reach_bonus
        self.food_reach_threshold = food_reach_threshold

        # Environment settings
        self.food_distance_range = food_distance_range
        self.food_lateral_range = food_lateral_range
        self.food_height_range = food_height_range
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
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # Geom IDs for contact detection
        self.food_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "food_geom"
        )
        self.torso_main_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_main"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        # Site IDs for sensors
        self.imu_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self.head_tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "head_tip"
        )
        self.tail_tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tail_tip"
        )

        # Foot site IDs
        self.foot_site_ids = {
            "fr": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot_contact"
            ),
            "fl": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot_contact"
            ),
            "rr": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "rr_foot_contact"
            ),
            "rl": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "rl_foot_contact"
            ),
        }

        # Sensor indices (order matches MJCF definition)
        self._sensor_gyro_start = 0
        self._sensor_accel_start = 3
        self._sensor_quat_start = 6
        self._sensor_fr_foot = 10
        self._sensor_fl_foot = 11
        self._sensor_rr_foot = 12
        self._sensor_rl_foot = 13

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (exclude root freejoint: first 7 values)
        qpos = self.data.qpos[7:].copy()

        # Joint velocities (exclude root freejoint: first 6 values)
        qvel = self.data.qvel[6:].copy()

        # Torso state from sensors
        torso_gyro = self.data.sensordata[
            self._sensor_gyro_start:self._sensor_gyro_start + 3
        ].copy()
        torso_accel = self.data.sensordata[
            self._sensor_accel_start:self._sensor_accel_start + 3
        ].copy()
        torso_quat = self.data.sensordata[
            self._sensor_quat_start:self._sensor_quat_start + 4
        ].copy()

        # Torso linear velocity (from root freejoint)
        torso_linvel = self.data.qvel[0:3].copy()

        # Foot contacts (from touch sensors)
        foot_contact = np.array([
            self.data.sensordata[self._sensor_fr_foot],
            self.data.sensordata[self._sensor_fl_foot],
            self.data.sensordata[self._sensor_rr_foot],
            self.data.sensordata[self._sensor_rl_foot],
        ])

        # Food info (relative to torso)
        torso_pos = self.data.xpos[self.torso_id]
        food_pos = self.data.mocap_pos[0]
        food_rel = food_pos - torso_pos
        food_distance = np.linalg.norm(food_rel)

        # Normalize food direction
        food_direction = food_rel / (food_distance + 1e-8)

        obs = np.concatenate([
            qpos,                   # Joint positions
            qvel,                   # Joint velocities
            torso_quat,             # Orientation (quaternion)
            torso_gyro,             # Angular velocity
            torso_linvel,           # Linear velocity
            torso_accel,            # Accelerometer
            foot_contact,           # Foot contacts (4)
            food_direction,         # Direction to food (unit vector)
            [food_distance],        # Distance to food (scalar)
        ]).astype(np.float32)

        return obs

    def _get_reward_info(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown for logging."""
        info = {}

        # 1. Forward velocity reward (toward food)
        torso_pos = self.data.xpos[self.torso_id]
        food_pos = self.data.mocap_pos[0]
        food_dir_2d = food_pos[:2] - torso_pos[:2]
        food_dist_2d = np.linalg.norm(food_dir_2d)
        if food_dist_2d > 1e-6:
            food_dir_2d = food_dir_2d / food_dist_2d

        # Project velocity onto food direction
        vel_2d = self.data.qvel[0:2]
        forward_vel = np.dot(vel_2d, food_dir_2d)
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

        # 4. Gait stability (penalize high angular velocity of torso)
        torso_angvel = self.data.qvel[3:6]
        gait_instability = np.linalg.norm(torso_angvel)
        reward_gait = -self.gait_stability_weight * gait_instability
        info["gait_instability"] = gait_instability
        info["reward_gait"] = reward_gait

        # 5. Food reach bonus (head tip close to food)
        head_tip_pos = self.data.site_xpos[self.head_tip_site_id]
        head_food_dist = np.linalg.norm(head_tip_pos - food_pos)
        info["head_food_distance"] = head_food_dist

        food_reward = 0.0
        if head_food_dist < self.food_reach_threshold:
            food_reward = self.food_reach_bonus
            info["food_reached"] = 1.0
        else:
            info["food_reached"] = 0.0

        reward_food = food_reward
        info["reward_food"] = reward_food

        # Total reward
        total_reward = (
            reward_forward + reward_alive + reward_energy
            + reward_gait + reward_food
        )
        info["reward_total"] = total_reward

        return total_reward, info

    def _is_terminated(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        # Get torso height
        torso_z = self.data.xpos[self.torso_id, 2]
        info["torso_height"] = torso_z

        # Termination: torso too low (fallen)
        if torso_z < self.healthy_z_range[0]:
            info["termination_reason"] = "fallen"
            return True, info

        # Termination: torso too high (shouldn't happen, safety check)
        if torso_z > self.healthy_z_range[1]:
            info["termination_reason"] = "too_high"
            return True, info

        # Check for torso-ground contact (fallen over)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if (geom1 == self.torso_main_geom_id and geom2 == self.floor_geom_id) or \
               (geom2 == self.torso_main_geom_id and geom1 == self.floor_geom_id):
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

        # Randomize food position
        self._spawn_food()

        # Forward pass to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        obs = self._get_obs()
        info = {"step": 0}

        return obs, info

    def _spawn_food(self):
        """Spawn food at random location (elevated, since Brachiosaurus browses high)."""
        if self.np_random is not None:
            distance = self.np_random.uniform(*self.food_distance_range)
            lateral = self.np_random.uniform(*self.food_lateral_range)
            height = self.np_random.uniform(*self.food_height_range)
        else:
            distance = np.mean(self.food_distance_range)
            lateral = 0.0
            height = np.mean(self.food_height_range)

        food_pos = np.array([distance, lateral, height])
        self.data.mocap_pos[0] = food_pos

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
                self._viewer.cam.distance = 8.0
                self._viewer.cam.azimuth = 135
                self._viewer.cam.elevation = -20
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=480, width=640
                )
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
    id="Brachio-v0",
    entry_point="envs.brachio_env:BrachioEnv",
    max_episode_steps=1000,
)
