"""Pixel observation wrapper for the BaseDinoEnv-derived environments.

The dinosaur envs all return proprioceptive state vectors. JEPA pretraining
and downstream pixel-conditioned policies need RGB renders as well. This
wrapper renders the MuJoCo scene at a fixed (small) resolution on every
``step``/``reset`` and either:

- replaces the observation with the rendered frame (``mode="pixels"``),
  used for JEPA pretraining data collection;
- returns a ``Dict({"state": <orig>, "pixels": <rgb>})`` (``mode="dict"``,
  the default), used by downstream RL with a JEPA feature extractor that
  may want both modalities.

Notes
-----
- One ``mujoco.Renderer`` per wrapper instance. ``SubprocVecEnv`` workers
  each get their own GL context, which is the desired behaviour.
- The wrapper does **not** call the env's ``render()``; it owns its own
  renderer/camera so that human-mode visualization and pixel observation
  do not interfere.
- Frames are returned as ``uint8`` HxWx3 by default. The JEPA feature
  extractor expects this and rescales to ``float32 [0, 1]``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - mujoco is required at runtime
    mujoco = None  # type: ignore[assignment]


class PixelObservationWrapper(gym.ObservationWrapper):
    """Render the underlying MuJoCo scene to RGB pixels each step.

    Parameters
    ----------
    env:
        A ``BaseDinoEnv``-derived environment (or anything that exposes
        ``self.model`` and ``self.data`` MuJoCo handles plus the camera
        attributes ``_camera_track_body``, ``_camera_distance``,
        ``_camera_azimuth``, ``_camera_elevation``).
    width, height:
        Target render resolution. 84x84 matches the canonical pixel-RL
        setup; 64x64 is also reasonable for tighter VRAM budgets.
    mode:
        ``"pixels"`` - replace the observation with the RGB frame.
        ``"dict"``   - return ``{"state": <orig>, "pixels": <rgb>}``.
    grayscale:
        If True, return a single-channel grayscale frame (HxWx1) instead
        of RGB. Reduces VRAM in JEPA pretraining at the cost of color
        cues (the prey is red by default; greyscaling weakens the prey
        signal).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        width: int = 84,
        height: int = 84,
        mode: str = "dict",
        grayscale: bool = False,
        camera_name: str | None = None,
    ) -> None:
        super().__init__(env)
        if mujoco is None:
            raise RuntimeError("PixelObservationWrapper requires the `mujoco` package.")
        if mode not in {"pixels", "dict"}:
            raise ValueError(f"mode must be 'pixels' or 'dict', got {mode!r}")

        self._width = int(width)
        self._height = int(height)
        self._mode = mode
        self._grayscale = bool(grayscale)
        self._camera_name = camera_name

        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None

        channels = 1 if self._grayscale else 3
        pixel_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, channels),
            dtype=np.uint8,
        )

        if self._mode == "pixels":
            self.observation_space = pixel_space
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": env.observation_space,
                    "pixels": pixel_space,
                }
            )

    # -- core ObservationWrapper API -------------------------------------

    def observation(self, observation: np.ndarray) -> Any:
        pixels = self._render_pixels()
        if self._mode == "pixels":
            return pixels
        return {"state": observation, "pixels": pixels}

    # -- rendering helpers -----------------------------------------------

    def _ensure_renderer(self) -> None:
        if self._renderer is not None:
            return
        base = self.env.unwrapped
        self._renderer = mujoco.Renderer(base.model, height=self._height, width=self._width)
        self._camera = self._build_camera(base)

    def _build_camera(self, base: gym.Env) -> "mujoco.MjvCamera":
        camera = mujoco.MjvCamera()
        if self._camera_name is not None:
            cam_id = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
            if cam_id >= 0:
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = cam_id
                return camera
        track_body = getattr(base, "_camera_track_body", None)
        if track_body is not None:
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_BODY, track_body)
        camera.distance = float(getattr(base, "_camera_distance", 3.0))
        camera.azimuth = float(getattr(base, "_camera_azimuth", 135.0))
        camera.elevation = float(getattr(base, "_camera_elevation", -20.0))
        return camera

    def _render_pixels(self) -> np.ndarray:
        self._ensure_renderer()
        base = self.env.unwrapped
        assert self._renderer is not None and self._camera is not None
        self._renderer.update_scene(base.data, self._camera)
        frame = self._renderer.render()  # uint8 HxWx3
        if self._grayscale:
            # Rec. 601 luma; matches what most pixel-RL setups use.
            gray = (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(np.uint8)
            return gray[..., None]
        return frame

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            finally:
                self._renderer = None
                self._camera = None
        super().close()
