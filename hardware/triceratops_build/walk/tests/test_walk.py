"""Tests for the model-based trot's sweep driver. Run by hand:
    python -m pytest hardware/triceratops_build/walk/tests
"""

import gzip
import json
import sys
from pathlib import Path

WALK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WALK))

import sweep_walk60 as SW  # noqa: E402


def test_driver_reproduces_a_stored_terrain_episode():
    """The driver's own path (controller built on the base env, pointed at a terrain env) reproduces the stored 3 cm
    step episode, seed 200000, exactly (the review found a controller compiled from the terrain env's round-tripped
    model drifts away chaotically)."""
    res = json.loads(gzip.decompress((WALK / "results" / "sweep_walk60" / "sweep_walk60.json.gz").read_bytes()))
    key = next(k for k, v in res["points"].items() if v["point"]["terrain"] == ["step", 0.03])
    stored = next(e for e in res["points"][key]["episodes"] if e["seed"] == 200000)
    SW._init_worker(SW.env_config())
    _, rec = SW._task((key, SW.S.make_point(terrain=("step", 0.03)), 200000))
    assert rec["fell"] == stored["fell"]
    assert rec["forward_speed_mps"] == stored["forward_speed_mps"]
    assert rec["sideways_drift_m"] == stored["sideways_drift_m"]
