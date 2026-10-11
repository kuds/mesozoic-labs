"""Cross-producer presentation must preserve run and checkpoint identities."""

import copy
import csv
import json
from unittest.mock import MagicMock

import pytest

from environments.shared.config import load_stage_config
from environments.shared.reporting import save_results_csv, save_results_json
from environments.shared.reporting.text_summaries import write_stage_summary, write_training_summary
from environments.shared.species_names import species_display_names
from environments.shared.stage_manifest import stage_dirname
from environments.shared.wandb_integration import init_wandb

from .reporting_helpers import make_stage_result


@pytest.mark.parametrize("species,label", species_display_names().items())
@pytest.mark.parametrize("recorded_name", ["balance", "stance"])
def test_stance_artifacts_use_full_names_without_changing_machine_identity(
    tmp_path, monkeypatch, species, label, recorded_name
):
    config = load_stage_config(species, 1)
    assert config["name"] == "stance"
    stage_dir = tmp_path / stage_dirname(species, 1)
    assert stage_dir.name == "01_stance"
    stage_dir.mkdir()
    result = make_stage_result(1, name=recorded_name, gate_passed=False, publication_gate_passed=False)
    before = copy.deepcopy(result)

    stage_summary = write_stage_summary(stage_dir, result, species, "PPO").read_text()
    run_summary = write_training_summary(tmp_path, [result], species, "PPO", 42, 4).read_text()
    assert f"Species:        {label}" in stage_summary and f"Species:        {label}" in run_summary
    assert "1 (Stance)" in stage_summary and "Stage 1: Stance" in run_summary
    assert "Verdict:      FAIL" in stage_summary

    summary = json.loads(save_results_json([result], species, "PPO", 42, tmp_path).read_text())
    csv_path = save_results_csv([result], {1: config}, species, "PPO", 42, tmp_path)
    with csv_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert summary["species"] == row["species"] == species
    assert summary["stages"]["1"]["name"] == recorded_name
    assert row["stage"] == "1"
    assert result == before

    # The displayed title changes on resume; the existing run must not fork.
    wandb = MagicMock()
    wandb.init.return_value.id = "existing-run"
    monkeypatch.setattr("environments.shared.wandb_integration.wandb", wandb)
    (stage_dir / "wandb_run_id.txt").write_text("existing-run")
    init_wandb(species, 1, {**config, "name": recorded_name}, run_dir=str(stage_dir))
    kwargs = wandb.init.call_args.kwargs
    assert kwargs["name"] == f"{label} / Stance"
    assert kwargs["id"] == "existing-run" and kwargs["resume"] == "allow"
    assert kwargs["tags"] == [species, "stage1"]
    assert kwargs["config"]["species"] == species and kwargs["config"]["stage"] == 1
    assert kwargs["config"]["species_display_name"] == label
    assert kwargs["config"]["stage_display_name"] == "Stance"
