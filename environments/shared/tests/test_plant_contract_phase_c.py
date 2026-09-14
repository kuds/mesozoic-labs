"""Phase C plant-contract pin: SB3/MJX observation parity on a NON-ZERO command.

Lives beside ``test_plant_contract_layers.py`` (the ``test_plant_contract_*``
glob of the plant-contract CI job, amendment A2) and follows its pattern:
``_policy_interface_payload(..., require_backend_parity=True)`` resolves the
MJX observation builder with NumPy, so no JAX is needed and the job's
``.[test]`` install is enough.  The shared matrix job runs it too.

Decision D-C4 / invariant 9, parity half: both probes inject
``COMMAND_PROBE_VECTOR`` so a dropped or mis-ordered command slot diverges
instead of comparing two identical zero vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.shared import mjx_env
from environments.shared.command_frame import COMMAND_PROBE_VECTOR, COMMAND_WIDTH
from environments.shared.plant_contract import PlantContractError, load_plant_versions
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

DUAL_BACKEND_SPECIES = (
    pytest.param(RaptorEnv, "velociraptor", id="velociraptor"),
    pytest.param(TRexEnv, "trex", id="trex"),
    pytest.param(BrachioEnv, "brachiosaurus", id="brachiosaurus"),
    pytest.param(DibothrosuchusEnv, "dibothrosuchus", id="dibothrosuchus"),
)


@pytest.mark.parametrize(("env_class", "species"), DUAL_BACKEND_SPECIES)
def test_plant_contract_probe_exercises_a_non_zero_command_on_both_backends(env_class, species, monkeypatch):
    expected_tail = np.round(np.asarray(COMMAND_PROBE_VECTOR, dtype=np.float64), 6)
    assert np.all(expected_tail != 0.0)
    env = env_class(reset_noise_scale=0.0)
    try:
        version = load_plant_versions()[1][species]
        payload = _policy_interface_payload(env.model, env, version, require_backend_parity=True)

        sb3_values = np.asarray(payload["observation_probe"]["values"])
        mjx_values = np.asarray(payload["jax_interface"]["observation_probe"]["values"])
        np.testing.assert_array_equal(sb3_values[-COMMAND_WIDTH:], expected_tail)
        np.testing.assert_array_equal(mjx_values[-COMMAND_WIDTH:], expected_tail)
        assert payload["backend_observation_equal"] is True
        np.testing.assert_array_equal(sb3_values, mjx_values)
        # The probe never leaks into the live env: the command is restored.
        np.testing.assert_array_equal(env._command, np.zeros(COMMAND_WIDTH, dtype=np.float32))

        # Non-vacuous: an MJX builder that drops the command must diverge.
        original = mjx_env.build_mjx_observation

        def dropping_command(data, target_pos, config, command=None):
            return original(data, target_pos, config, command=None)

        monkeypatch.setattr(mjx_env, "build_mjx_observation", dropping_command)
        with pytest.raises(PlantContractError, match="observation probes diverge"):
            _policy_interface_payload(env.model, env, version, require_backend_parity=True)
        relaxed = _policy_interface_payload(env.model, env, version, require_backend_parity=False)
        assert relaxed["backend_observation_equal"] is False
    finally:
        env.close()
