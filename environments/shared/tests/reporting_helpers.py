"""Shared fixtures for the reporting package test modules.

``test_reporting_*.py`` each cover one submodule of
``environments.shared.reporting``; these two builders are the inputs several
of them need in common.
"""

from __future__ import annotations

from environments.shared.plant_contract import PlantIdentity


def plant_identity():
    return PlantIdentity(
        species="velociraptor",
        model_path="environments/velociraptor/assets/raptor.xml",
        physics_revision=1,
        policy_interface_revision=1,
        visual_revision=1,
        source_closure_sha256="sha256:" + "1" * 64,
        policy_interface_sha256="sha256:" + "2" * 64,
        physics_sha256="sha256:" + "3" * 64,
        visual_sha256="sha256:" + "4" * 64,
        nq=31,
        nv=30,
        nu=22,
        observation_dim=67,
        action_dim=22,
    )


def make_stage_result(stage=1, **overrides):
    """Build a minimal stage result dict for testing."""
    result = {
        "stage": stage,
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "timesteps": 100_000 * stage,
        "duration_seconds": 300.0 * stage,
        "mean_reward": 50.0 + stage * 10,
        "std_reward": 5.0,
        "mean_episode_length": 200.0,
        "std_episode_length": 20.0,
        "mean_forward_vel": 0.5 * stage,
        "std_forward_vel": 0.1,
        "mean_success_rate": 0.0,
        "model_path": f"/tmp/stage{stage}/best_model",
        "vecnorm_path": f"/tmp/stage{stage}/vecnorm.pkl",
        "sim_dt": 0.01,
        "gate_passed": True,
        "publication_gate_passed": True,
        "gate_failures": [],
    }
    result.update(overrides)
    return result
