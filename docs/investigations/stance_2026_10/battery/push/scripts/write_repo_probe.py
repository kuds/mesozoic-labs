"""Write main's stance_gate_probe_impulse.{txt,json} per checkpoint from the Part A sweeps (policy vs statue).

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints; report only; certifies nothing.
"""
import pickle, sys
from pathlib import Path
import environments
assert "main_ro" in environments.__file__, environments.__file__
from environments.shared.reporting import stage_artifacts
from environments.shared.reporting.stance_report import write_impulse_probe
base = Path(sys.argv[1])
statue = pickle.load(open(base / "sweep_statue.pkl", "rb"))
for role in ("s60_6M", "s52_6M", "s42_6M"):
    policy = pickle.load(open(base / f"sweep_{role}.pkl", "rb"))
    written = write_impulse_probe(base / role, policy, statue, probe_episodes=stage_artifacts._IMPULSE_EPISODES)
    print(role, {k: str(v) for k, v in written.items()})
