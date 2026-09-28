"""Pins the frozen MJX interface core (D-D17; docs/CLEANUP_PLAN_2026_09.md §4.1-§4.3).

Nothing trains on MJX, but trex, velociraptor, brachiosaurus and dibothrosuchus
still declare ``jax-mjx``: their policy-interface digests hash the tokens of the
functions below (``action_filter``'s through trex's low-pass filter), and the
plant contract's MJX probe reads the ``mjx_config`` registrations.
``plant_contract --check`` reports an edit as a stale manifest; these pins name
what moved. Delete this file with the frozen core, once the last of the four
species has declared itself SB3-only inside a ``policy_interface_revision`` bump.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from environments.shared.plant_contract.digests import _callable_semantics
from environments.shared.plant_contract.versions import _species_entries

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: ``_callable_semantics(...)["tokens_sha256"]``, unchanged since f850815.
FROZEN_TOKEN_DIGESTS = {
    "mjx_env.build_mjx_observation": "d1a8ac56e6f533670690897f47756ebf07cc2e499d6f56da89a5bea7d2446ac4",
    "jax_setup.make_obs_fn": "66d4e404736f5c23159eec010e125a3a9a5c0981fae300897040c16b6fa923c5",
    "mjx_utils.scale_action_around_nominal_jax": "f01be749e8a78ac9c9266d462f31511a06e9f0c0fef2b7d7292c4b8c431b2eae",
    "mjx_utils.reset_mujoco_data_to_home": "f780f506759e615e1a0e9292deeae31fa8113a6feef5442993bd26bfbec28eb0",
    "obs_functions._array_mod": "cbcfa3fe9292bf98e8d8c70d88d4f1ef03e49e11cfcbd50d0d6f87e6967d48f4",
    "obs_functions.build_bipedal_obs": "8b46a7e02e14059c38d723ff46f60eae31b633bb3b61bb2455caf862dc6f75e4",
    "obs_functions.build_quadruped_obs": "7b4e74c4c97a78b143f80902d6e4eb542ecc8a3830579182c1921ad5065dbded",
    "action_filter.low_pass_alpha": "f32bb3f1cda4b52493b8bbd4c61fff9c86231f705ddcf9d462900f97af7f0908",
    "action_filter.apply_low_pass": "8c619d7021723c3f74bfc3d5ccb45c27738c931d176c487a3a76a18b1affaa19",
}

#: The keys ``policy_layer._jax_policy_interface_payload`` reads, per species.
_PROBED = {"action_mapping", "frame_skip", "body_ids", "sensor_foot_indices"}
_PROBED |= {"sensor_gyro_start", "sensor_accel_start", "sensor_quat_start"}
REGISTRATION_KEYS = {
    "trex": _PROBED | {"sensor_foot_aux_indices", "action_filter_cutoff_hz"},
    "brachiosaurus": _PROBED | {"sensor_foot_aux_indices"},
    "velociraptor": _PROBED,
    "dibothrosuchus": _PROBED,
}


@pytest.mark.parametrize("target", list(FROZEN_TOKEN_DIGESTS))
def test_frozen_function_keeps_its_hashed_tokens(target: str) -> None:
    module, name = target.split(".")
    function = getattr(importlib.import_module(f"environments.shared.{module}"), name)
    expected = {"qualname": name, "tokens_sha256": f"sha256:{FROZEN_TOKEN_DIGESTS[target]}"}
    assert _callable_semantics(function) == expected, (
        f"environments/shared/{module}.py: {name} was edited, reformatted or moved. It is frozen by D-D17: "
        "its tokens are part of four species' policy-interface digests. Revert the change."
    )


@pytest.mark.parametrize("species", list(REGISTRATION_KEYS))
def test_frozen_mjx_registration_keeps_exactly_the_probed_keys(species: str) -> None:
    registration = importlib.import_module(f"environments.{species}.mjx_config")
    registry = importlib.import_module("environments.shared.mjx_env")._SPECIES_CONFIGS
    if species not in registry:
        importlib.reload(registration)
    assert set(registry[species]) == REGISTRATION_KEYS[species], (
        f"environments/{species}/mjx_config.py registers {sorted(registry[species])}; D-D17 freezes it "
        f"to the keys the plant contract's MJX probe reads: {sorted(REGISTRATION_KEYS[species])}"
    )


def test_frozen_registrations_are_exactly_the_dual_backend_species() -> None:
    default = ("stable-baselines3", "jax-mjx")
    dual = {name for name, entry in _species_entries().items() if "jax-mjx" in entry.get("training_backends", default)}
    assert dual == set(REGISTRATION_KEYS), (
        f"species declaring jax-mjx: {sorted(dual)}; the D-D17 frozen core registers {sorted(REGISTRATION_KEYS)}. "
        "A species leaves it only inside a policy_interface_revision bump (CLEANUP_PLAN_2026_09.md §4.1)."
    )


def test_ruff_leaves_the_frozen_modules_alone() -> None:
    frozen = {f"environments/shared/{module}.py" for module in ("mjx_env", "jax_setup", "mjx_utils", "obs_functions")}
    frozen |= {f"environments/{species}/mjx_config.py" for species in REGISTRATION_KEYS}
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        ruff = tomllib.load(handle)["tool"]["ruff"]
    assert frozen <= set(ruff.get("extend-exclude", ())) and ruff.get("force-exclude") is True, (
        "pyproject.toml's [tool.ruff] must extend-exclude (with force-exclude) the D-D17 frozen modules, "
        f"{sorted(frozen)}: a formatter or autofix change to them moves four species' policy-interface digests"
    )
