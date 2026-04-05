"""Velociraptor mjlab species configuration.

Registers the velociraptor with the mjlab adapter so that
``make_mjlab_env("velociraptor", stage=1)`` works once the optional
``mjlab`` dependency is installed.

Pilot target: reproduce Stage 1 balance (ref: SB3 PPO 2:57:25, best reward
1964.43) on a single NVIDIA GPU via MuJoCo-Warp batch simulation, then
measure wall-clock speedup and envs/sec vs the existing ``MJXDinoEnv``.
"""

from __future__ import annotations

from pathlib import Path

from environments.shared.mjlab_env import MJLabSpeciesConfig, register_species_mjlab

_ASSETS = Path(__file__).parent / "assets"


def _build_observation_manager() -> object:
    """Compose the 67-dim velociraptor observation via mjlab obs terms.

    Intentionally deferred: mirrors the observation layout already documented
    in ``envs/raptor_env.py`` (joint pos/vel, pelvis quat/gyro/linvel/accel,
    foot contacts, prey direction + distance). Implemented in the pilot PR.
    """
    # from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
    # from mjlab.envs.mdp import observations as mdp_obs
    # return ObservationGroupCfg(
    #     joint_pos=ObservationTermCfg(func=mdp_obs.joint_pos_rel),
    #     joint_vel=ObservationTermCfg(func=mdp_obs.joint_vel_rel),
    #     base_quat=ObservationTermCfg(func=mdp_obs.base_quat),
    #     base_ang_vel=ObservationTermCfg(func=mdp_obs.base_ang_vel),
    #     base_lin_vel=ObservationTermCfg(func=mdp_obs.base_lin_vel),
    #     ...
    # )
    raise NotImplementedError("Pilot PR: wire mjlab obs terms for velociraptor.")


def _build_reward_manager(weights: dict[str, float]) -> object:
    """Compose the staged reward via mjlab reward terms.

    Reuses the same pure reward functions in ``environments/shared/
    reward_functions.py`` via small ``ManagerTermCfg`` wrappers, so the
    velociraptor TOML stage configs remain the single source of truth.
    """
    raise NotImplementedError("Pilot PR: wrap shared reward functions as mjlab terms.")


def _build_termination_manager() -> object:
    """Fall/time-out/success terminations (mirrors MJXDinoEnv logic)."""
    raise NotImplementedError("Pilot PR: define mjlab termination terms.")


def _build_event_manager() -> object:
    """Domain randomization via mjlab events (Phase 2 roadmap item).

    Event managers are mjlab's idiomatic DR mechanism — friction, damping,
    gravity, actuator gain, external pushes, reset noise.
    """
    raise NotImplementedError("Pilot PR: add DR event terms using randomization ranges.")


register_species_mjlab(
    MJLabSpeciesConfig(
        species="velociraptor",
        mjcf_path=_ASSETS / "raptor.xml",
        frame_skip=5,
        episode_length_s=10.0,  # 1000 steps * 0.01s timestep * frame_skip=5 / 5
        num_envs_default=4096,
        obs_dim=67,
        action_dim=22,
        observation_manager_factory=_build_observation_manager,
        reward_manager_factory=_build_reward_manager,
        termination_manager_factory=_build_termination_manager,
        event_manager_factory=_build_event_manager,
    )
)
