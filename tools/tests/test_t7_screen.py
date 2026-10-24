"""T7a, screen form (build tier; no optimizer step): a BOUND zip made from the real s42 start loads with stock PPO in
the measuring copy.

run_tailscreen.build(state="s42", rep_seed=101, out_dir=None) loads the start without learning; the bound is enabled
at b = 0.60, coef 2.7; the model and its VecNormalize are saved to SCREEN/tests_out/t7/.  To pass: (1) the zip
carries the mesozoic_tail3_bound record and no _tail3_bound key; (2) its policy parameters are bit-equal to the start
zip's (no step was taken); (3) in a subprocess on the measuring copy, episode_trace_report.py --seed 3042
--episodes 1 succeeds with no warning on stderr.  Reported only (never pass/fail): whether the trace's tail_3 window
command equals the start panel's seed-3042 row (DRAFT/trace_s42, 6 decimals).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RT = _load("run_tailscreen")
C = RT.C
SCREEN = Path(C.SCREEN)
OUT = SCREEN / "tests_out" / "t7"
MEASURE = SCREEN / "measure_copy"
ZIP = OUT / "stage1_t7.zip"
PKL = OUT / "stage1_t7_vecnormalize.pkl"
COEF, BOUND = 2.7, 0.60


def _policy_tensors(path: Path) -> dict:
    import torch

    with zipfile.ZipFile(path) as zf:
        state = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


@pytest.fixture(scope="module")
def t7_zip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    built = RT.build("s42", 101, None)
    from environments.shared import mirror_kl_ppo as MK

    model, vecnorm = built["model"], built["vecnorm"]
    assert model.num_timesteps == C.START_STEPS
    n_updates = int(model._n_updates)
    config = MK.Tail3UpperBoundConfig(COEF, BOUND)
    MK.enable_tail3_upper_bound(model, config)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    model.save(str(ZIP))
    vecnorm.save(str(PKL))
    assert int(model._n_updates) == n_updates and torch.get_num_threads() == 1
    model.env.close()
    return {"config": config, "record": config.checkpoint_record()}


def test_t7a_screen_zip_carries_the_record_and_the_start_parameters(t7_zip):
    data = json.loads(zipfile.ZipFile(ZIP).read("data"))
    assert data["mesozoic_tail3_bound"] == t7_zip["record"]
    assert data["mesozoic_tail3_bound"] == {"coef": 2.7, "bound": 0.6, "index": 14, "side": "upper",
                                            "loss": "coef * mean_rows relu(mu[:, index] - bound)^2"}
    assert "_tail3_bound" not in data and "_mirror" not in data
    start = json.loads(zipfile.ZipFile(C.STATES["s42"]["zip"]).read("data"))
    assert "mesozoic_tail3_bound" not in start
    assert set(data) - set(start) == {"mesozoic_tail3_bound"}
    import numpy as np

    saved, original = _policy_tensors(ZIP), _policy_tensors(Path(C.STATES["s42"]["zip"]))
    assert saved.keys() == original.keys()
    assert all(np.array_equal(saved[k], original[k]) and saved[k].dtype == original[k].dtype for k in saved)


def _window_cmd(csv_path: Path, settle: int) -> float:
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    values = [float(r["tail_3_pitch_act_command"]) for r in rows[settle:]]
    return sum(values) / len(values)


def test_t7a_screen_zip_runs_in_the_measuring_copy(t7_zip):
    script = MEASURE / "environments" / "shared" / "scripts" / "episode_trace_report.py"
    if not script.is_file() or not (MEASURE / "TREE_DIGEST.txt").is_file():
        pytest.skip("the measuring copy is not built yet")
    for rel, digest in C.OVERLAYS.items():
        assert hashlib.sha256((MEASURE / rel).read_bytes()).hexdigest() == digest, rel
    trace_dir = OUT / "trace"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    done = subprocess.run(
        [sys.executable, "-I", "-B", str(script), "--species", "tyrannosaurus_rex", "--embodiment", "anatomical",
         "--model", str(ZIP), "--vecnorm", str(PKL), "--seed", "3042", "--episodes", "1", "--out-dir", str(trace_dir)],
        capture_output=True, text=True, env=env, cwd=str(OUT), timeout=1200,
    )
    (OUT / "trace_stdout.txt").write_text(done.stdout)
    (OUT / "trace_stderr.txt").write_text(done.stderr)
    assert done.returncode == 0, done.stderr[-4000:]
    assert "warn" not in done.stderr.lower(), done.stderr[-4000:]
    trace = json.loads((trace_dir / "episode_trace.json").read_text())
    settle = int(trace["settle_steps"])
    got = _window_cmd(trace_dir / "episode_trace_seed3042.csv", settle)
    reference = _window_cmd(Path(C.DRAFT) / "trace_s42" / "episode_trace_seed3042.csv", settle)
    start_pkl = Path(C.STATES["s42"]["pkl"])
    report = {
        "report_only": True, "t7_tail3_window_cmd_seed3042": round(got, 6),
        "start_panel_tail3_window_cmd_seed3042": round(reference, 6), "equal_6dp": round(got, 6) == round(reference, 6),
        "t7_pkl_sha256": hashlib.sha256(PKL.read_bytes()).hexdigest(),
        "start_pkl_sha256": hashlib.sha256(start_pkl.read_bytes()).hexdigest(),
        "note": "a difference can come only through the VecNormalize statistics that were saved",
    }
    (OUT / "t7_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
