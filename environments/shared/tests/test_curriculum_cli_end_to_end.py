"""End-to-end test of the command-line curriculum path.

Cleanup PR-A2 (docs/CLEANUP_PLAN_2026_09.md §4.5, decision D-D17) removes GCS
artifact upload from ``cli.py``, ``config.py``, ``train_base.train_curriculum``
and ``reporting/csv_output.py``.  Every other curriculum test replaces the
trainer (``test_train_base.py::TestTrainCurriculumWalksTheManifest`` mocks the
model, environments, callbacks, config writer, CSV writer and verdict writer;
``test_cli.py`` mocks ``train_curriculum`` itself), so nothing ran the path from
argv to artifacts.  These tests do: the real entry point,
``environments/velociraptor/scripts/train_sb3.py curriculum``, runs in a
subprocess with tiny budgets, and ``--override`` opens every gate so that each
stage advances on its first evaluation.  They check the run's on-disk contract,
not learning.

Velociraptor rather than Compsognathus: all three of its advancing stages gate
on ``reward_and_length/v1``, which ``--override`` can open at a 32-step
horizon.  Compsognathus's stance gate (``stance_quality/v1``) needs an
evaluation episode that reaches the horizon ``CurriculumManager`` re-reads from
the stage TOML (1,000 steps, not an overridden ``env.max_episode_steps``), so
its stance evaluation alone costs 40 thousand-step episodes (about 90 s
measured, against about 16 s for the whole velociraptor ladder; docs/KNOWN_ISSUES.md,
Training / RL, records the horizon defect).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from environments.shared.cli import _apply_overrides  # noqa: E402
from environments.shared.config import load_all_stages  # noqa: E402
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view  # noqa: E402
from environments.shared.reporting import CSV_METRIC_COLUMNS  # noqa: E402
from environments.shared.result_bundle import sha256_file  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname, stage_label  # noqa: E402
from environments.shared.train_base import CURRICULUM_MANAGER_JUDGED_BY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECIES = "velociraptor"
SCRIPT = REPO_ROOT / "environments" / SPECIES / "scripts" / "train_sb3.py"
LABEL = "cli-e2e"

# The run's shape: one 64-step stage (two 32-step PPO rollouts), one evaluation
# at step 64 over 32-step episodes, one supplementary episode, and a single
# passing evaluation to advance.
RUN_SHAPE = (
    "curriculum.timesteps=64",
    "curriculum.required_consecutive=1",
    "curriculum.supplementary_episodes=1",
    "env.max_episode_steps=32",
    "ppo.n_steps=32",
    "ppo.batch_size=32",
    "ppo.n_epochs=1",
)
# Every reward_and_length/v1 criterion opened: 0.0 disables the forward-velocity
# and success-rate criteria, and no panel reward is below the rail.
OPEN_GATES = (
    "curriculum.min_avg_reward=-1000000000",
    "curriculum.min_avg_episode_length=0",
    "curriculum.min_avg_forward_vel=0",
    "curriculum.min_success_rate=0",
)
# The CLI flags, all real arguments of the `curriculum` subparser.
FLAGS = ("--n-envs", "1", "--seed", "0", "--eval-freq", "64", "--save-freq", "64", "--verbose", "0")

# ``-X importtime`` writes one stderr line for each module an ``import``
# statement (or ``__import__``) first loads; ``importlib.import_module`` is not
# logged. A GCS client module on it means the curriculum path reached
# cloud-upload code.
_GOOGLE_CLOUD_IMPORT = re.compile(r"^import time:.*\|\s+google\.cloud(\.|\s*$)")


def _run_cli(output_dir: Path, *extra: str, overrides: tuple[str, ...] = RUN_SHAPE + OPEN_GATES):
    """Run ``train_sb3.py curriculum`` as a user would, and return the finished process."""
    env = dict(os.environ)
    # One thread per process: the SB3 job runs one pytest process on a
    # four-vCPU runner, and torch's default would oversubscribe it.
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [
        sys.executable,
        "-X",
        "importtime",
        str(SCRIPT),
        "curriculum",
        *FLAGS,
        "--label",
        LABEL,
        "--output-dir",
        str(output_dir),
        *extra,
        "--override",
        *overrides,
    ]
    return subprocess.run(
        command, capture_output=True, text=True, cwd=output_dir.parent, env=env, timeout=900, check=False
    )


def _assert_succeeded(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"exit {result.returncode}\n--- stdout\n{result.stdout[-4000:]}\n--- stderr\n{result.stderr[-8000:]}"
    )


def _assert_no_cloud_imports(result: subprocess.CompletedProcess) -> None:
    lines = result.stderr.splitlines()
    # The instrumentation works (otherwise the negative check below is vacuous).
    assert any(line.startswith("import time:") and "environments.shared.train_base" in line for line in lines)
    assert not [line for line in lines if _GOOGLE_CLOUD_IMPORT.search(line)]


def _expected_configs(*overrides: str) -> dict:
    """The stage configs the run judged under: the TOMLs with the same overrides applied."""
    configs = load_all_stages(SPECIES)
    _apply_overrides(configs, list(overrides), SPECIES)
    return configs


def _read(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "curriculum_results.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _header(run_dir: Path) -> list[str]:
    with (run_dir / "curriculum_results.csv").open(newline="") as handle:
        return next(csv.reader(handle))


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """One full CLI curriculum run (stance -> locomotion -> behavior), shared by the tests below."""
    run_dir = tmp_path_factory.mktemp("cli_curriculum") / "ladder"
    result = _run_cli(run_dir)
    return run_dir, result


def test_the_cli_curriculum_trains_every_advancing_stage_and_records_it(ladder):
    run_dir, result = ladder
    _assert_succeeded(result)
    _assert_no_cloud_imports(result)
    assert "Curriculum training complete!" in result.stderr

    manifest = load_stage_manifest(SPECIES)
    chain = [entry for entry in manifest.stages if entry in manifest.advancing_stages]
    assert [entry.reference for entry in chain] == [1, 2, 3]
    configs = _expected_configs(*RUN_SHAPE, *OPEN_GATES)

    # The run directory: one stage directory per advancing node, the run's
    # plant identity and the stage CSV; nothing reused, nothing else written.
    assert {path.name for path in run_dir.iterdir()} == {stage_dirname(SPECIES, entry.reference) for entry in chain} | {
        "plant_identity.json",
        "curriculum_results.csv",
    }
    assert (run_dir / "plant_identity.json").is_file()
    assert not (run_dir / "ancestors").exists()
    # The notebook writes collected_results.csv; this path never has
    # (docs/KNOWN_ISSUES.md, Post-training artifacts item 2: CLI curriculum parity).
    assert not (run_dir / "collected_results.csv").exists()

    parent_dir = parent_verdict = None
    for entry in chain:
        stage = entry.reference
        stage_dir = run_dir / stage_dirname(SPECIES, stage)
        label = stage_label(stage)
        models = stage_dir / "models"

        # Per-stage records.
        for name in ("stage_config.json", "plant_identity.json", "task_fingerprint.json", "evaluations.npz"):
            assert (stage_dir / name).is_file(), f"{stage_dir.name}/{name}"
        # metrics.json is the CLI `train` subcommand's panel record
        # (train(report_metrics=True)); train_curriculum never writes it
        # (docs/KNOWN_ISSUES.md, Post-training artifacts item 2: CLI curriculum parity).
        assert not (stage_dir / "metrics.json").exists()

        # The final pair and the evaluation-selected pairs.
        for stem in (f"{label}_final", "best_model"):
            assert (models / f"{stem}.zip").is_file(), f"{stage_dir.name}/models/{stem}.zip"
            assert (models / f"{stem}_vecnorm.pkl").is_file(), f"{stage_dir.name}/models/{stem}_vecnorm.pkl"

        # The in-training verdict (D-A5), hash-bound to the handoff pair and to
        # the gate this run judged under (D-A22).
        verdict = _read(stage_dir / "gate_verdict.json")
        assert verdict["passed"] is True and verdict["failures"] == []
        assert verdict["species"] == SPECIES
        assert verdict["stage"] == stage and verdict["stage_id"] == entry.id
        assert verdict["judged_by"] == CURRICULUM_MANAGER_JUDGED_BY
        assert verdict["gate_kind"] == "reward_and_length/v1"
        cur_kwargs = configs[stage]["curriculum_kwargs"]
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(cur_kwargs))
        assert verdict["gate"]["thresholds"]["min_avg_reward"] == -1_000_000_000
        checkpoint = stage_dir / verdict["checkpoint"]
        normalization = stage_dir / verdict["normalization"]
        assert checkpoint.parent == models and checkpoint.name.endswith(".zip")
        assert verdict["checkpoint_sha256"] == sha256_file(checkpoint)
        assert verdict["normalization_sha256"] == sha256_file(normalization)
        assert verdict["task_sha256"] == _read(stage_dir / "task_fingerprint.json")["task_sha256"]

        # The stage config: the recipe, the run block and the lineage.
        recorded = _read(stage_dir / "stage_config.json")
        assert recorded["species"] == SPECIES and recorded["stage"] == stage
        assert recorded["algorithm"] == "PPO"
        run = recorded["run"]
        assert (run["seed"], run["n_envs"], run["timesteps"]) == (0, 1, 64)
        assert run["label"] == LABEL
        assert run["hyperparameters_sha256"].startswith("sha256:")
        assert "parent_run_id" not in run  # nothing was reused from another run
        if parent_dir is None:
            # The root loads nothing and records no lineage.
            assert "load_path" not in run and "load_mode" not in run
        else:
            # The handoff: the child entered on exactly the checkpoint its
            # parent's verdict certified (warm_start_from edge), across a
            # stage boundary.
            assert entry.warm_start_from == chain[chain.index(entry) - 1].id
            assert Path(run["load_path"]) == parent_dir / parent_verdict["checkpoint"]
            assert run["load_mode"] == "initialize_next_stage"
            assert run["parent_checkpoint_sha256"] == parent_verdict["checkpoint_sha256"]
            assert run["parent_task_sha256"] == parent_verdict["task_sha256"]
        parent_dir, parent_verdict = stage_dir, verdict

    # curriculum_results.csv: one row per trained stage, through the shared
    # writer (reporting.csv_output.write_results_csv in append mode), so the
    # header is the sorted hyperparameter columns then CSV_METRIC_COLUMNS.
    rows = _rows(run_dir)
    assert [row["stage"] for row in rows] == ["1", "2", "3"]
    assert {row["stage_passed"] for row in rows} == {"True"}
    assert {(row["species"], row["algorithm"], row["run_dir"], row["seed"], row["n_envs"]) for row in rows} == {
        (SPECIES, "PPO", run_dir.name, "0", "1")
    }
    assert {row["plant_species"] for row in rows} == {SPECIES}
    assert {row["reward_threshold"] for row in rows} == {"-1000000000"}
    header = _header(run_dir)
    assert header[-len(CSV_METRIC_COLUMNS) :] == CSV_METRIC_COLUMNS
    assert header[: -len(CSV_METRIC_COLUMNS)] == sorted(header[: -len(CSV_METRIC_COLUMNS)])


def test_a_cli_run_serves_as_the_next_runs_trunk(ladder):
    """The verdicts the CLI writes are what --trunk-from reuses (D-A5, §4.2): the
    certified ancestors are recorded under ancestors/, never trained, and only
    the target is trained here."""
    trunk, trunk_result = ladder
    _assert_succeeded(trunk_result)
    run_dir = trunk.parent / "child"
    result = _run_cli(run_dir, "--trunk-from", str(trunk))
    _assert_succeeded(result)
    _assert_no_cloud_imports(result)

    target_dir = stage_dirname(SPECIES, 3)
    assert {path.name for path in run_dir.iterdir()} == {
        "ancestors",
        target_dir,
        "plant_identity.json",
        "curriculum_results.csv",
    }
    for stage_id in ("stance", "locomotion"):
        assert (run_dir / "ancestors" / stage_id / "ancestor.json").is_file()
        assert (run_dir / "ancestors" / stage_id / "gate_verdict.json").is_file()
        # A record, never the checkpoint.
        assert not list((run_dir / "ancestors" / stage_id).glob("*.zip"))

    parent_verdict = _read(trunk / stage_dirname(SPECIES, 2) / "gate_verdict.json")
    run = _read(run_dir / target_dir / "stage_config.json")["run"]
    assert Path(run["load_path"]) == trunk / stage_dirname(SPECIES, 2) / parent_verdict["checkpoint"]
    assert run["parent_checkpoint_sha256"] == parent_verdict["checkpoint_sha256"]
    assert run["parent_run_id"] == trunk.name
    assert _read(run_dir / target_dir / "gate_verdict.json")["passed"] is True

    # A reused node writes no curriculum_results.csv row.
    assert [row["stage"] for row in _rows(run_dir)] == ["3"]


def test_a_failed_gate_stops_the_cli_curriculum_after_recording_it(tmp_path):
    run_dir = tmp_path / "stopped"
    closed_stance = RUN_SHAPE + OPEN_GATES + ("stance.curriculum.min_avg_reward=1000000000",)
    result = _run_cli(run_dir, overrides=closed_stance)
    # The run itself succeeds: the failure is recorded, not raised.
    _assert_succeeded(result)
    _assert_no_cloud_imports(result)
    assert "Stopping curriculum" in result.stderr

    stance_dir = stage_dirname(SPECIES, 1)
    assert {path.name for path in run_dir.iterdir()} == {stance_dir, "plant_identity.json", "curriculum_results.csv"}
    verdict = _read(run_dir / stance_dir / "gate_verdict.json")
    assert verdict["passed"] is False and verdict["failures"]
    assert verdict["gate"]["thresholds"]["min_avg_reward"] == 1_000_000_000
    # The pair is still saved and hash-bound, so the node can be re-judged.
    assert verdict["checkpoint_sha256"] == sha256_file(run_dir / stance_dir / verdict["checkpoint"])
    assert [(row["stage"], row["stage_passed"]) for row in _rows(run_dir)] == [("1", "False")]
