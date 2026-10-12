#!/usr/bin/env python3
"""
Test actuators by applying sinusoidal control signals.
Useful for verifying joint ranges and actuator gains.

Usage (from the repository root):
    python -m environments.tyrannosaurus_rex.scripts.test_actuators --embodiment anatomical
"""

import argparse

from environments.shared.harnesses.actuators import ActuatorTestConfig, run_actuator_test
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.shared.species_names import embodiments_of

SPECIES = "tyrannosaurus_rex"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embodiment", required=True, choices=embodiments_of(SPECIES), help="The model to drive")
    model_id = ModelId(SPECIES, parser.parse_args().embodiment)
    run_actuator_test(
        ActuatorTestConfig(
            model_path=model_id.asset_path(ENVIRONMENTS_ROOT),
            camera_distance=3.0,
        )
    )
