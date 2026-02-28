"""
Base Gymnasium environment for MuJoCo dinosaur simulations.

Provides the common lifecycle (init, step, reset, render, close) shared
across all dinosaur species. Subclasses override species-specific methods:
  - _cache_ids()
  - _get_obs()
  - _get_reward_info()
  - _is_terminated()
  - _spawn_target()
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np


class BaseDinoEnv(gym.Env, ABC):
    """Abstract base class for dinosaur locomotion environments."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # Subclasses override for camera positioning
    _camera_distance: float = 3.0
    _camera_azimuth: float = 135
    _camera_elevation: float = -20
    _camera_track_body: Optional[str] = None  # Body name to track, or None for fixed

    def __init__(
        self,
        model_path: str,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        forward_vel_weight: float = 1.0,
        alive_bonus: float = 0.1,
        energy_penalty_weight: float = 0.001,
        fall_penalty: float = -100.0,
        healthy_z_range: Tuple[float, float] = (0.25, 1.0),
        max_tilt_angle: float = 1.047,
    ):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Common reward weights
        self.forward_vel_weight = forward_vel_weight
        self.alive_bonus = alive_bonus
        self.energy_penalty_weight = energy_penalty_weight
        self.fall_penalty = fall_penalty

        # Environment settings
        self.healthy_z_range = healthy_z_range
        self.max_tilt_angle = max_tilt_angle

        # Cache body/geom/site IDs (species-specific)
        self._cache_ids()

        # Define action space (normalized to [-1, 1])
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Define observation space from initial obs
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        # Rendering
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None
        self._camera = None

    @staticmethod
    def _quat_to_tilt(quat: np.ndarray) -> float:
        """Compute tilt angle (radians) between body up-axis and world up.

        Args:
            quat: MuJoCo quaternion (w, x, y, z).

        Returns:
            Angle in radians between the body's Z-axis and world Z-axis.
            0 means perfectly upright, pi/2 means horizontal.
        """
        w, x, y, z = quat
        # Body Z-axis (up) rotated into world frame
        body_up_z = 1.0 - 2.0 * (x * x + y * y)
        return float(np.arccos(np.clip(body_up_z, -1.0, 1.0)))

    # ------------------------------------------------------------------
    # Abstract methods: subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, geoms, and sites."""

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector."""

    @abstractmethod
    def _get_reward_info(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute reward and breakdown dict for logging."""

    @abstractmethod
    def _is_terminated(self) -> Tuple[bool, Dict[str, Any]]:
        """Check species-specific termination conditions."""

    @abstractmethod
    def _spawn_target(self):
        """Randomize the target (prey/food) position on reset."""

    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------

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
            if not term_info.get("success"):
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
        return np.asarray(scaled)

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)."""
        return self._step_count >= self.max_episode_steps

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset MuJoCo state using keyframe if available, otherwise default
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbation to initial pose
        if self.np_random is not None:
            noise_scale = 0.01
            self.data.qpos[7:] += self.np_random.uniform(-noise_scale, noise_scale, size=self.data.qpos[7:].shape)
            self.data.qvel[:] += self.np_random.uniform(-noise_scale, noise_scale, size=self.data.qvel.shape)

        # Randomize target position (species-specific)
        self._spawn_target()

        # Forward pass to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        obs = self._get_obs()
        info = {"step": 0}

        return obs, info

    def _make_camera(self) -> mujoco.MjvCamera:
        """Create a configured MjvCamera for rendering."""
        camera = mujoco.MjvCamera()
        if self._camera_track_body is not None:
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._camera_track_body)
        camera.distance = self._camera_distance
        camera.azimuth = self._camera_azimuth
        camera.elevation = self._camera_elevation
        return camera

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                cam = self._viewer.cam
                if self._camera_track_body is not None:
                    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._camera_track_body)
                cam.distance = self._camera_distance
                cam.azimuth = self._camera_azimuth
                cam.elevation = self._camera_elevation
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
                self._camera = self._make_camera()
            self._renderer.update_scene(self.data, self._camera)
            return self._renderer.render()

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
