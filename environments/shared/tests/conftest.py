"""Directory-wide pytest fixtures for the shared test suite."""

from __future__ import annotations

import pytest

from environments.shared.plant_contract import (
    GENERATED_MANIFEST_PATH,
    PlantVersion,
    fingerprint_model_layers,
)
from environments.shared.result_bundle import provenance as result_bundle_provenance
from environments.velociraptor.envs.raptor_env import RaptorEnv

from .result_bundle_helpers import _COMMIT


@pytest.fixture
def stable_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove host Git/package state from result-bundle assertions."""
    monkeypatch.setattr(
        result_bundle_provenance,
        "_repository_state",
        lambda _root: {
            "repository_url": "https://github.com/kuds/mesozoic-labs.git",
            "repository_commit": _COMMIT,
            "repository_dirty": False,
            "repository_patch_sha256": None,
        },
    )
    monkeypatch.setattr(
        result_bundle_provenance,
        "_dependency_versions",
        lambda: {
            "mesozoic_labs": "0.3.3.dev0",
            "mujoco": "3.10.0",
            "mujoco_mjx": "3.10.0",
            "gymnasium": "1.2.0",
            "numpy": "2.3.0",
            "stable_baselines3": "2.7.0",
            "jax": "0.11.0",
            "flax": "0.12.7",
            "optax": "0.2.8",
        },
    )


@pytest.fixture(scope="module")
def raptor_layers():
    """Return original model source, interface shell, version, and hashes."""
    env = RaptorEnv(reset_noise_scale=0.0)
    try:
        source = (GENERATED_MANIFEST_PATH.parent.parent / "environments/velociraptor/assets/raptor.xml").read_text()
        interface = env
        version = PlantVersion(
            species="velociraptor",
            physics_revision=1,
            policy_interface_revision=1,
            visual_revision=1,
            observation_schema="bipedal-target/v1",
        )
        layers = fingerprint_model_layers(env.model, interface, version)
        yield source, interface, version, layers
    finally:
        env.close()
