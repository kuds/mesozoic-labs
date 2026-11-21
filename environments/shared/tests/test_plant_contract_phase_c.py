"""Phase C plant-contract pin: the observation probe runs on a NON-ZERO command.

Lives beside ``test_plant_contract_layers.py`` (the ``test_plant_contract_*``
files, amendment A2) and follows its pattern: ``_policy_interface_payload``
runs the SB3 observation on deterministic probe data, so the ``.[test]``
install is enough.  The shared legs of the test matrix run it, with the rest of
the ``test_plant_contract_*`` files (the plant-contract CI job runs no pytest
since CU-14a).

Decision D-C4 / invariant 9: the probe injects ``COMMAND_PROBE_VECTOR`` so a
dropped or mis-ordered command slot hashes differently from a zero one.  The
SB3/MJX parity half of this pin left with the frozen MJX interface core
(configs/plant_versions.toml note 23).
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv
from environments.compsognathus_longipes.anatomical.env import CompsognathusLongipesAnatomicalEnv
from environments.dibothrosuchus_elaphros.anatomical.env import DibothrosuchusElaphrosAnatomicalEnv
from environments.shared.command_frame import COMMAND_PROBE_VECTOR, COMMAND_WIDTH
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import load_plant_versions
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")


@pytest.mark.parametrize(
    ("env_class", "model_id"),
    (
        pytest.param(
            VelociraptorMongoliensisAnatomicalEnv, VELOCIRAPTOR_MONGOLIENSIS, id="velociraptor_mongoliensis-anatomical"
        ),
        pytest.param(TyrannosaurusRexAnatomicalEnv, TYRANNOSAURUS_REX, id="tyrannosaurus_rex-anatomical"),
        pytest.param(
            CompsognathusLongipesAnatomicalEnv, COMPSOGNATHUS_LONGIPES, id="compsognathus_longipes-anatomical"
        ),
        pytest.param(
            BrachiosaurusAltithoraxAnatomicalEnv, BRACHIOSAURUS_ALTITHORAX, id="brachiosaurus_altithorax-anatomical"
        ),
        pytest.param(
            DibothrosuchusElaphrosAnatomicalEnv, DIBOTHROSUCHUS_ELAPHROS, id="dibothrosuchus_elaphros-anatomical"
        ),
    ),
)
def test_sb3_only_probe_still_exercises_a_non_zero_command(env_class, model_id):
    """The SB3 half of the pin survives each species' exit from the frozen MJX core.

    Since the velociraptor's policy-interface revision 11 (plant_versions note
    14), the T-Rex's revision 14 (note 18, the heading-free observation) and
    the brachiosaurus's and dibothrosuchus's revisions 11 and 10 (note 23, the
    core removed) there is no MJX probe to compare against, and the anatomical
    compsognathus never had one, but the SB3 probe must still inject the
    non-zero command, or a dropped command slot would hash like a zero one.
    """
    expected_tail = np.round(np.asarray(COMMAND_PROBE_VECTOR, dtype=np.float64), 6)
    env = env_class(reset_noise_scale=0.0)
    try:
        version = load_plant_versions()[1][model_id]
        payload = _policy_interface_payload(env.model, env, version)
        sb3_values = np.asarray(payload["observation_probe"]["values"])
        np.testing.assert_array_equal(sb3_values[-COMMAND_WIDTH:], expected_tail)
        assert payload["jax_interface"] == {"supported": False, "backend": "jax-mjx"}
        assert payload["backend_observation_equal"] is None
        np.testing.assert_array_equal(env._command, np.zeros(COMMAND_WIDTH, dtype=np.float32))
    finally:
        env.close()
