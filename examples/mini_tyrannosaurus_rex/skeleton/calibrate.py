"""Bind measured M1 PD-null evidence to the shared recovery-profile contract.

Pass --measure to repeat the shared calibration experiment first. The default
only packages existing measurements, refusing changed source files or settings.
It never restamps stale measurements or edits the task to fit a selected force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.shared.recovery_calibration import CAPABILITY_KEYS, load_recovery_calibration
from environments.shared.task_fingerprint import stage_task_fingerprint
from examples.mini_tyrannosaurus_rex.skeleton.build import ROOT

MODEL_ID = ModelId("tyrannosaurus_rex", "robot")
MEASUREMENT = ROOT / "hardware/mini_tyrannosaurus_rex/m01_skeleton/recovery_measurement.json"
PROFILE = ROOT / "configs/tyrannosaurus_rex/robot/recovery_calibration.json"


def package_profile() -> None:
    measurement = json.loads(MEASUREMENT.read_text())
    if not MODEL_ID.matches(measurement):
        raise ValueError("Measurements name a different model")
    for path, expected in measurement["provenance"]["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Calibration source changed: {path}; repeat --measure")
    config = load_stage_config(MODEL_ID, "recovery")
    kwargs = config["env_kwargs"]
    stance = measurement["stance_env_kwargs"]
    measured_kwargs = {
        **stance,
        "perturbation_capture_velocity_multiple": measurement["selected_capture_velocity_multiple"],
        "perturbation_interval": 2.0,
        "perturbation_jitter": 0.25,
        "perturbation_duration": 0.2,
        "perturbation_direction": "uniform_horizontal",
    }
    if json.dumps(measured_kwargs, sort_keys=True) != json.dumps(kwargs, sort_keys=True):
        raise ValueError("Recovery config does not match the measured task and selected force")
    quiet = measurement["quiet_calibration"]
    plant = current_plant_identity(MODEL_ID).to_dict()
    profile = {
        "schema": "mesozoic.recovery-calibration/v2",
        **MODEL_ID.record(),
        "calibration_id": "tyrannosaurus_rex-robot-m1-hw-r01-recovery-v1",
        "status": "Simulation PD-null task calibration; learned-policy and hardware qualification outstanding",
        "control_dt_s": 0.02,
        "height_reference_m": quiet["height_reference_m"],
        "safe_set": quiet["safe_set"],
        "recovery_env_kwargs": kwargs,
        "capability_spec": {key: config["curriculum_kwargs"][key] for key in sorted(CAPABILITY_KEYS)},
        "required_paired_nulls": ["zero_action", "brace"],
        "plant_identity": plant,
        "task_sha256": stage_task_fingerprint(MODEL_ID, "recovery")["task_sha256"],
        "derived_push_manifest": measurement["heldout_diagnostic_panels"][0]["perturbation"],
        "provenance": {
            "measurement_artifact": MEASUREMENT.relative_to(ROOT).as_posix(),
            "measurement_sha256": hashlib.sha256(MEASUREMENT.read_bytes()).hexdigest(),
            "quiet_seed_range": [1042, 1049],
            "selection_seed_range": [2042, 2073],
            "diagnostic_seed_range": [7042, 7081],
            "stress_seed_range": [8042, 8081],
            "reserved_publication_seed_range": [3042, 3081],
            "method": "Shared calibrate_recovery.py; 8 quiet, 32 selection, 40 held-out episodes; unchanged engineering floors",
            "qualification_targets": "Experimental; not measured policy achievement",
        },
    }
    PROFILE.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    load_recovery_calibration(MODEL_ID)
    print(PROFILE.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    if args.measure:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "environments.shared.scripts.calibrate_recovery",
                "--species",
                MODEL_ID.species,
                "--embodiment",
                MODEL_ID.embodiment,
                "--output",
                str(MEASUREMENT),
            ],
            cwd=ROOT,
            check=True,
        )
    package_profile()


if __name__ == "__main__":
    main()
