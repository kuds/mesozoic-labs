"""Phase C interface pins (BEHAVIOR_RECIPES_PLAN §4.6, WS-C1): the MJX half.

Runs under the JAX CPU job through the ``test_mjx_*.py`` glob (amendment
A2); skipped wherever JAX / MJX is not installed.  Never imports
stable_baselines3.  Pins:

* the jax-mjx backend refuses every live ``command_mode`` — in
  ``MJXDinoEnv.__init__``, in ``canonicalize_env_kwargs`` (before any
  unknown-key warning) and therefore on ``jax_setup.setup_species``'s CPU
  evaluation path — invariant 9, fail-closed half (amendment A7);
* ``command_mode = "none"`` widens the observation by the 3 zero command
  dims and is bit-identical to an env built without the key;
  ``EnvState.command`` rides the pytree through jit/vmap;
* ``validate_mjx_environment_plant`` checks the MJX observation width
  against the plant identity (decision D-C16);
* ``MJXDinoEnv.command_manifest()`` is ``None`` (amendments A1(iv), A9).

The SB3 / backend-neutral half lives in ``test_phase_c_interface.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

import environments.trex.mjx_config  # noqa: E402,F401  (registers the species)
from environments.shared.command_frame import COMMAND_MODES, COMMAND_WIDTH  # noqa: E402
from environments.shared.mjx_env import (  # noqa: E402
    _PLANT_INTERFACE_CONFIG_FIELDS,
    _SB3_ONLY_ENV_KEYS,
    MJXDinoEnv,
    canonicalize_env_kwargs,
)
from environments.shared.plant_contract import (  # noqa: E402
    PlantCompatibilityError,
    current_plant_identity,
    validate_mjx_environment_plant,
)

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
_MJX_REFUSAL = "not implemented on the jax-mjx backend"


def _stance_toml_with(tmp_path: Path, command_mode: str) -> Path:
    """A copy of trex stance.toml whose ``[env]`` table declares *command_mode*."""
    source = (_CONFIGS_DIR / "trex" / "stance.toml").read_text(encoding="utf-8")
    assert "\n[env]\n" in source
    target = tmp_path / f"stance_{command_mode}.toml"
    target.write_text(source.replace("\n[env]\n", f'\n[env]\ncommand_mode = "{command_mode}"\n', 1), encoding="utf-8")
    return target


def test_mjx_refuses_command_mode_other_than_none(monkeypatch, tmp_path, caplog):
    with pytest.raises(ValueError, match=_MJX_REFUSAL):
        MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs={"command_mode": "heading"})
    with pytest.raises(ValueError, match=_MJX_REFUSAL):
        MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs={"command_mode": "heading_and_speed"})

    # canonicalize_env_kwargs refuses BEFORE any unknown-key warning.
    with caplog.at_level(logging.WARNING, logger="environments.shared.mjx_env"):
        with pytest.raises(ValueError, match=_MJX_REFUSAL):
            canonicalize_env_kwargs({"command_mode": "heading", "definitely_unknown_key": 1.0})
    assert not caplog.records
    with pytest.raises(ValueError, match=re.escape(f"command_mode='bogus' is not one of {COMMAND_MODES}")):
        canonicalize_env_kwargs({"command_mode": "bogus"})
    assert canonicalize_env_kwargs({"command_mode": "none"}) == {"command_mode": "none"}

    # The CPU-evaluation path (jax_setup.setup_species) canonicalises the
    # stage TOML without constructing an MJXDinoEnv and must refuse too.
    from environments.shared import config as config_module
    from environments.shared.jax_setup import setup_species

    original = config_module.load_stage_config
    live_toml = _stance_toml_with(tmp_path, "heading")
    monkeypatch.setattr(
        config_module,
        "load_stage_config",
        lambda species, stage, config_path=None: original(species, stage, config_path=str(live_toml)),
    )
    with pytest.raises(ValueError, match=_MJX_REFUSAL):
        setup_species("trex", stage=1)

    # Positive control: the same TOML with "none" evaluates at the Phase C width.
    inert_toml = _stance_toml_with(tmp_path, "none")
    monkeypatch.setattr(
        config_module,
        "load_stage_config",
        lambda species, stage, config_path=None: original(species, stage, config_path=str(inert_toml)),
    )
    ctx = setup_species("trex", stage=1)
    assert ctx.obs_dim == current_plant_identity("trex", verify_generated=False).observation_dim == 64


def test_mjx_command_mode_none_widens_obs_and_is_bit_identical_to_no_kwarg():
    import jax.numpy as jnp

    explicit = MJXDinoEnv("trex", stage=1, num_envs=2, env_kwargs={"command_mode": "none"})
    plain = MJXDinoEnv("trex", stage=1, num_envs=2)
    assert explicit.config.command_mode == plain.config.command_mode == "none"

    rng = jax.random.PRNGKey(3)
    states_explicit = explicit.reset(rng)
    states_plain = plain.reset(rng)
    assert states_explicit.obs.shape == (2, 64)
    assert states_explicit.obs.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(states_explicit.obs[:, -COMMAND_WIDTH:]), np.zeros((2, COMMAND_WIDTH)))
    np.testing.assert_array_equal(np.asarray(states_explicit.obs), np.asarray(states_plain.obs))

    # EnvState.command survives the vmapped reset and the jitted step.
    assert states_explicit.command.shape == (2, COMMAND_WIDTH)
    assert states_explicit.command.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(states_explicit.command), np.zeros((2, COMMAND_WIDTH)))
    leaves = jax.tree_util.tree_leaves(states_explicit)
    assert any(leaf is states_explicit.command for leaf in leaves)

    actions = jnp.zeros((2, explicit.action_dim), dtype=jnp.float32)
    for step in range(5):
        step_rng = jax.random.PRNGKey(100 + step)
        states_explicit, _, _, _ = explicit.step(states_explicit, actions, step_rng)
        states_plain, _, _, _ = plain.step(states_plain, actions, step_rng)
    np.testing.assert_array_equal(np.asarray(states_explicit.data.qpos), np.asarray(states_plain.data.qpos))
    np.testing.assert_array_equal(np.asarray(states_explicit.obs), np.asarray(states_plain.obs))
    assert states_explicit.obs.shape == (2, 64)
    np.testing.assert_array_equal(np.asarray(states_explicit.obs[:, -COMMAND_WIDTH:]), np.zeros((2, COMMAND_WIDTH)))
    assert states_explicit.command.shape == (2, COMMAND_WIDTH)
    np.testing.assert_array_equal(np.asarray(states_explicit.command), np.zeros((2, COMMAND_WIDTH)))


def test_validate_mjx_environment_plant_checks_observation_width():
    env = MJXDinoEnv("trex", stage=1, num_envs=1)
    identity = current_plant_identity("trex", verify_generated=False)
    assert identity.observation_dim == 64
    validate_mjx_environment_plant(env, identity)

    stale = replace(identity, observation_dim=identity.observation_dim - COMMAND_WIDTH)
    with pytest.raises(
        PlantCompatibilityError, match=r"MJX observation width 64 does not match the plant identity \(61\)"
    ):
        validate_mjx_environment_plant(env, stale)


def test_mjx_command_manifest_is_none_and_the_command_kwargs_are_task_level():
    env = MJXDinoEnv("trex", stage=1, num_envs=1)
    assert env.command_manifest() is None
    assert env.config.command_mode == "none"
    assert tuple(env.config.command_speed_range) == (0.0, 0.0)
    assert tuple(env.config.command_lateral_range) == (0.0, 0.0)
    assert env.config.command_yaw_rate_max == 0.0
    assert env.config.command_switch_interval == 0.0
    assert env.config.command_switch_jitter == 0.0
    command_fields = {
        "command_mode",
        "command_speed_range",
        "command_lateral_range",
        "command_yaw_rate_max",
        "command_switch_interval",
        "command_switch_jitter",
    }
    # Task-level like perturbation_*: never part of the versioned plant
    # interface, and never silently ignored as an SB3-only key.
    assert not command_fields & _PLANT_INTERFACE_CONFIG_FIELDS
    assert not command_fields & _SB3_ONLY_ENV_KEYS

    # A stage may pass the inert defaults explicitly without tripping the
    # versioned-interface guard.
    explicit = MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs={"command_mode": "none", "command_yaw_rate_max": 0.0})
    assert explicit.command_manifest() is None
