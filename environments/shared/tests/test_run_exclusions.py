"""Tests for environments.shared.run_exclusions: the r10 T. rex stance screen runs never count (the maintainer's
stance-landing decision of 2026-11-06).

The nine screen runs trained the landed stance recipe on branch commits, so their task, gate, plant and recipe digests
equal a new certification seed's: every rule replicate discovery and reuse apply would accept them.  These tests pin
the one rule that keeps them out, at every call site (replicate discovery for a sibling and for the run itself, reuse
rule 9 directly and through a followed record, trunk selection, the older-interface widening hint, the widening tool,
the result-bundle audit and the species catalog), on each of its four rules (the run's provenance commit, the stage
directory's recorded commit, the verdict's handoff checkpoint, a load of a screen's checkpoint the stage records), and
that the same files at another commit, or without the load, still count, so the rule alone excludes them.  The judged-run directories are ``ancestors_helpers.build_trunk_run``'s.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest

from environments.shared import run_exclusions
from environments.shared.ancestors import AncestorReuseError, find_certified_ancestor, record_ancestor, select_trunk
from environments.shared.config import hyperparameters_sha256
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
from environments.shared.model_id import ModelId
from environments.shared.replication import discover_replicates, discover_replicates_for_run
from environments.shared.result_bundle import sha256_file
from environments.shared.run_exclusions import (
    EXCLUDED_COMMITS,
    EXCLUDED_HANDOFF_SHA256,
    EXCLUDED_RUN_IDS,
    EXCLUSION_DECISION,
    LINEAGE_DIGEST_KEYS,
    LINEAGE_PATH_KEYS,
    LINEAGE_RUN_ID_KEYS,
    excluded_checkpoint_reason,
    excluded_commit_reason,
    excluded_handoff_reason,
    excluded_lineage_reason,
    excluded_path_reason,
    excluded_run_id_reason,
    excluded_run_reason,
)
from environments.shared.stage_manifest import load_stage_manifest

from .ancestors_helpers import (
    LOCOMOTION_CURRICULUM,
    OTHER_TASK,
    STANCE_CURRICULUM,
    STANCE_TASK,
    build_trunk_run,
    trunk_plant,
)
from .reporting_helpers import make_plant_identity

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
STANCE_ENTRY = load_stage_manifest(TYRANNOSAURUS_REX).by_id("stance")
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The phase-critic and switch-off screens' commit, the stance6 screens' commit, and main's commit at the landing's base.
SCREEN_COMMIT = "1dfe2327f773007fa7fd68460dc0c53740936ff9"
STANCE6_COMMIT = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"
MAIN_COMMIT = "77c63ade7bcdc4bbfeb84dcf76f6bdf6d0363b02"
SCREEN_RUNS = {f"stance7_r10LTSB{arm}_s{seed}" for arm in ("pc", "off") for seed in (42, 60, 80)} | {
    f"stance6_r10LTSB_s{seed}" for seed in (42, 60, 80)
}
#: stance7_r10LTSBpc_s42's handoff (robust_best_model.zip) and its stage1_final.zip.
PC_S42_HANDOFF = "sha256:77959e30f4f20e1a3bb184d210b4430a14b6df1a046cdbc2e5b279487fbfc299"
PC_S42_FINAL = "sha256:1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6"

RECIPE = {"learning_rate": 3e-4, "n_steps": 2048}
RECIPE_DIGEST = hyperparameters_sha256({"ppo_kwargs": RECIPE, "curriculum_kwargs": STANCE_CURRICULUM}, "PPO")
STANCE_GATE = gate_config_sha256(gate_config_view(STANCE_CURRICULUM))


@pytest.fixture
def log_base(tmp_path: Path) -> Path:
    base = tmp_path / "logs" / "tyrannosaurus_rex" / "anatomical" / "ppo"
    base.mkdir(parents=True)
    return base


def _run(
    log_base: Path,
    name: str,
    *,
    seed: int,
    commit: "str | None" = None,
    stage_commit: "str | None" = None,
    lineage: "dict[str, Any] | None" = None,
):
    """A judged same-recipe run; *commit* writes its provenance.json, *stage_commit* its stage_config git_commit,
    *lineage* the load keys of its stage_config run block."""
    run = log_base / name
    run.mkdir(parents=True)
    stage_dir = build_trunk_run(
        run, seed=seed, algorithm="PPO", hyperparameters=RECIPE, record_recipe_digest=True, lineage=lineage
    )
    if commit is not None:
        _write(run / "provenance.json", {"run_id": name, "repository_commit": commit})
    if stage_commit is not None:
        _set_stage_commit(stage_dir, stage_commit)
    return run


def _write(path: Path, record: Any) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _set_stage_commit(stage_dir: Path, commit: str) -> None:
    record = json.loads((stage_dir / "stage_config.json").read_text(encoding="utf-8"))
    record["git_commit"] = commit
    _write(stage_dir / "stage_config.json", record)


def _exclude(monkeypatch: pytest.MonkeyPatch, digest: str) -> str:
    excluded = {**run_exclusions.EXCLUDED_HANDOFF_SHA256, digest: "stand-in"}
    monkeypatch.setattr(run_exclusions, "EXCLUDED_HANDOFF_SHA256", excluded)
    return digest


def _exclude_checkpoint(monkeypatch: pytest.MonkeyPatch, digest: str) -> str:
    excluded = {**run_exclusions.EXCLUDED_CHECKPOINT_SHA256, digest: "stand-in/stage1_final.zip"}
    monkeypatch.setattr(run_exclusions, "EXCLUDED_CHECKPOINT_SHA256", excluded)
    return digest


def _exclude_handoff(monkeypatch: pytest.MonkeyPatch, stage_dir: Path) -> str:
    """Make *stage_dir*'s verdict judge a screen's handoff: the fixtures' checkpoints are byte-identical across runs,
    so the verdict is re-stamped with a digest of its own, which the excluded digests then list."""
    digest = "sha256:" + hashlib.sha256(str(stage_dir).encode("utf-8")).hexdigest()
    verdict = json.loads((stage_dir / "gate_verdict.json").read_text(encoding="utf-8"))
    verdict["checkpoint_sha256"] = digest
    _write(stage_dir / "gate_verdict.json", verdict)
    return _exclude(monkeypatch, digest)


def _discover(run: Path, **overrides: Any) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = dict(
        model_id=TYRANNOSAURUS_REX,
        entry=STANCE_ENTRY,
        task_sha256=STANCE_TASK,
        plant_identity=trunk_plant(),
        gate_sha256=STANCE_GATE,
        hyperparameters_sha256=RECIPE_DIGEST,
        training_seed=44,
    )
    kwargs.update(overrides)
    return discover_replicates(run, **kwargs)


def _find(run_dir: Path, **overrides: Any):
    kwargs: dict[str, Any] = dict(
        model_id=TYRANNOSAURUS_REX,
        entry=STANCE_ENTRY,
        current_task_sha256=STANCE_TASK,
        plant_identity=trunk_plant(),
        current_gate_config=STANCE_CURRICULUM,
    )
    kwargs.update(overrides)
    return find_certified_ancestor(run_dir, **kwargs)


# ------------------------------------------------------------------------------------------------- the lists


def test_the_lists_name_the_two_branch_commits_and_the_nine_screen_handoffs():
    assert set(EXCLUDED_COMMITS) == {SCREEN_COMMIT, STANCE6_COMMIT}
    assert set(EXCLUDED_HANDOFF_SHA256.values()) == SCREEN_RUNS
    for digest in EXCLUDED_HANDOFF_SHA256:
        assert digest.startswith("sha256:") and len(digest) == 71 and digest == digest.lower()
    for run in SCREEN_RUNS:
        commit = STANCE6_COMMIT if run.startswith("stance6") else SCREEN_COMMIT
        assert run.removesuffix("_s42").removesuffix("_s60").removesuffix("_s80") in EXCLUDED_COMMITS[commit]


def test_the_stance6_handoffs_are_the_committed_judge_files_checkpoints():
    """The stance6 screens' handoff digests are the ones their committed offline judge files hash."""
    folder = REPO_ROOT / "docs" / "investigations" / "stance_2026_10" / "stance6_results" / "complete"
    for seed in (42, 60, 80):
        judged = json.loads((folder / f"complete_stance6_r10LTSB_s{seed}.json").read_text(encoding="utf-8"))
        digest = judged["completeness"]["certificate"]["binding"]["checkpoint_sha256"]
        assert EXCLUDED_HANDOFF_SHA256["sha256:" + digest] == f"stance6_r10LTSB_s{seed}"


def test_the_reasons_name_the_runs_and_the_decision_and_read_both_digest_forms():
    reason = excluded_commit_reason(f"  {SCREEN_COMMIT.upper()} ")
    assert reason is not None and "stance7_r10LTSBpc_s42" in reason and EXCLUSION_DECISION in reason
    assert excluded_commit_reason(MAIN_COMMIT) is None
    assert excluded_commit_reason(None) is None and excluded_commit_reason(12) is None
    digest = "sha256:77959e30f4f20e1a3bb184d210b4430a14b6df1a046cdbc2e5b279487fbfc299"
    for form in (digest, digest.removeprefix("sha256:"), digest.upper().replace("SHA256:", "sha256:")):
        handoff = excluded_handoff_reason(form)
        assert handoff is not None and "stance7_r10LTSBpc_s42" in handoff
    assert excluded_handoff_reason("sha256:" + "0" * 64) is None and excluded_handoff_reason("") is None


# ------------------------------------------------------------------------------------------------- the three keys


def test_any_of_the_three_records_excludes_a_run(log_base, monkeypatch):
    by_provenance = _run(log_base, "by_provenance", seed=42, commit=SCREEN_COMMIT)
    by_stage_record = _run(log_base, "by_stage_record", seed=60, stage_commit=STANCE6_COMMIT)
    by_handoff = _run(log_base, "by_handoff", seed=80)
    _exclude_handoff(monkeypatch, by_handoff / "01_stance")
    control = _run(log_base, "control", seed=43, commit=MAIN_COMMIT, stage_commit=MAIN_COMMIT)

    assert "provenance.json" in (excluded_run_reason(by_provenance) or "")
    assert "01_stance/stage_config.json" in (excluded_run_reason(by_stage_record) or "")
    assert "01_stance/gate_verdict.json" in (excluded_run_reason(by_handoff) or "")
    assert excluded_run_reason(by_handoff, by_handoff / "01_stance") is not None
    assert excluded_run_reason(control) is None


def test_a_screen_run_resumed_on_main_stays_excluded(log_base, monkeypatch):
    """A same-stage resume rewrites stage_config.json with main's commit; provenance keeps the screen's, and a
    provenance-less copy still carries the screen's handoff in its verdict."""
    resumed = _run(log_base, "resumed", seed=42, commit=SCREEN_COMMIT, stage_commit=MAIN_COMMIT)
    assert excluded_run_reason(resumed) is not None
    copied = _run(log_base, "copied", seed=60, stage_commit=MAIN_COMMIT)
    _exclude_handoff(monkeypatch, copied / "01_stance")
    assert excluded_run_reason(copied) is not None


def test_unreadable_records_decide_nothing(log_base):
    run = _run(log_base, "malformed", seed=42)
    (run / "provenance.json").write_text("{not json", encoding="utf-8")
    (run / "01_stance" / "stage_config.json").write_text("[1, 2]\n", encoding="utf-8")
    assert excluded_run_reason(run) is None
    assert excluded_run_reason(log_base / "absent") is None


# ------------------------------------------------------------------------------------------------- replicate discovery


def test_a_screen_sibling_never_counts_and_the_same_files_at_main_do(log_base, monkeypatch, caplog):
    """Cases A and B of the plan review: a new seed beside the screens (by provenance, by a provenance-less copy's
    stage record, by the handoff) counts only the other new seed; with main's commit the same siblings count."""
    this_run = _run(log_base, "cert_s44", seed=44, commit=MAIN_COMMIT)
    _run(log_base, "stance7_r10LTSBpc_s42", seed=42, commit=SCREEN_COMMIT, stage_commit=SCREEN_COMMIT)
    _run(log_base, "copy_without_provenance", seed=60, stage_commit=SCREEN_COMMIT)
    by_handoff = _run(log_base, "stance6_r10LTSB_s80", seed=80, commit=MAIN_COMMIT)
    _exclude_handoff(monkeypatch, by_handoff / "01_stance")
    _run(log_base, "cert_s43", seed=43, commit=MAIN_COMMIT)

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        assert _discover(this_run) == [{"run_id": "cert_s43", "training_seed": 43}]
    skipped = "\n".join(record.getMessage() for record in caplog.records if "skipping" in record.getMessage())
    for name in ("stance7_r10LTSBpc_s42", "copy_without_provenance", "stance6_r10LTSB_s80"):
        assert f"skipping {name}: " in skipped and EXCLUSION_DECISION in skipped

    monkeypatch.setattr(run_exclusions, "EXCLUDED_COMMITS", {})
    monkeypatch.setattr(run_exclusions, "EXCLUDED_HANDOFF_SHA256", {})
    # Sorted by run id: cert_s43, copy_without_provenance, stance6_r10LTSB_s80, stance7_r10LTSBpc_s42.
    assert [entry["training_seed"] for entry in _discover(this_run)] == [43, 60, 80, 42]


def test_a_screen_run_counts_no_replicate_of_its_own(log_base, caplog):
    """Case A': a screen published or re-published on main beside the new seeds counts none of them, so its record
    stays at itself alone (provisional); a new seed beside it still counts the other new seed (case C)."""
    screen = _run(log_base, "stance7_r10LTSBpc_s42", seed=42, commit=SCREEN_COMMIT, stage_commit=SCREEN_COMMIT)
    _run(log_base, "cert_s44", seed=44, commit=MAIN_COMMIT)
    _run(log_base, "cert_s43", seed=43, commit=MAIN_COMMIT)

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        assert _discover(screen, training_seed=42) == []
        assert discover_replicates_for_run(screen, model_id=TYRANNOSAURUS_REX, plant_identity=trunk_plant()) == {}
    assert sum("the run itself never counts" in record.getMessage() for record in caplog.records) == 2
    assert discover_replicates_for_run(
        log_base / "cert_s44", model_id=TYRANNOSAURUS_REX, plant_identity=trunk_plant()
    ) == {"1": [{"run_id": "cert_s43", "training_seed": 43}]}


def test_a_copied_screen_stage_directory_never_counts(log_base):
    """Case B: a screen's stage directory copied into a run folder with no provenance.json (named by its folder)."""
    screen = _run(log_base, "stance7_r10LTSBoff_s60", seed=60, commit=SCREEN_COMMIT, stage_commit=SCREEN_COMMIT)
    copy = log_base / "my_copy"
    copy.mkdir()
    shutil.copytree(screen / "01_stance", copy / "01_stance")
    this_run = _run(log_base, "cert_s44", seed=44, commit=MAIN_COMMIT)
    assert _discover(this_run) == []


# ------------------------------------------------------------------------------------------------- reuse


def test_reuse_rule_9_refuses_a_screen_run_and_accepts_the_same_run_at_main(tmp_path):
    run = tmp_path / "stance7_r10LTSBpc_s80"
    run.mkdir()
    stage_dir = build_trunk_run(run)
    _write(run / "provenance.json", {"run_id": run.name, "repository_commit": SCREEN_COMMIT})
    with pytest.raises(AncestorReuseError, match="reuse rule 9") as refusal:
        _find(run)
    assert EXCLUSION_DECISION in str(refusal.value)

    _write(run / "provenance.json", {"run_id": run.name, "repository_commit": MAIN_COMMIT})
    assert _find(run).stage_dir == stage_dir
    _set_stage_commit(stage_dir, STANCE6_COMMIT)
    with pytest.raises(AncestorReuseError, match="reuse rule 9"):
        _find(run)


def test_reuse_rule_9_is_applied_at_a_followed_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    stage_dir = build_trunk_run(source)
    middle = tmp_path / "middle"
    middle.mkdir()
    record_ancestor(middle, _find(source))
    assert _find(middle, follow_records=True).source_run_dir == source

    _set_stage_commit(stage_dir, SCREEN_COMMIT)
    with pytest.raises(AncestorReuseError, match="followed the ancestor record") as refusal:
        _find(middle, follow_records=True)
    assert "reuse rule 9" in str(refusal.value)


def _chain_and_configs():
    manifest = load_stage_manifest(TYRANNOSAURUS_REX)
    chain = tuple(manifest.by_id(stage_id) for stage_id in ("stance", "locomotion", "behavior"))
    configs = {
        1: {"env_kwargs": {}, "curriculum_kwargs": STANCE_CURRICULUM},
        2: {"env_kwargs": {}, "curriculum_kwargs": LOCOMOTION_CURRICULUM},
        3: {"env_kwargs": {}, "curriculum_kwargs": {"gate_kind": "task_success/v1", "timesteps": 10}},
    }
    return chain, configs


@pytest.fixture
def fixture_tasks(monkeypatch):
    """Derive the stance node's task digest as the trunk fixtures were judged under."""
    from environments.shared import task_fingerprint as task_fingerprint_module

    monkeypatch.setattr(
        task_fingerprint_module, "derive_stage_task_fingerprint", lambda **kwargs: {"task_sha256": STANCE_TASK}
    )


def test_trunk_selection_passes_over_a_screen_run(tmp_path, fixture_tasks):
    """A screen's name sorts above timestamp run ids, so without the rule TRUNK_FROM = "auto" would pick it."""
    screen = tmp_path / "stance7_r10LTSBpc_s60"
    build_trunk_run(screen)
    _write(screen / "provenance.json", {"run_id": screen.name, "repository_commit": SCREEN_COMMIT})
    certified = tmp_path / "20261110_120000"
    build_trunk_run(certified)
    chain, configs = _chain_and_configs()

    selection = select_trunk(
        tmp_path, model_id=TYRANNOSAURUS_REX, chain=chain, stage_configs=configs, plant_identity=trunk_plant()
    )

    assert selection.run_dir == certified
    (refused,) = [candidate for candidate in selection.candidates if candidate.run_dir == screen]
    assert refused.refusal is not None and "reuse rule 9" in refused.refusal[1]


def test_the_older_interface_hint_never_offers_a_screen_run(tmp_path, fixture_tasks):
    from environments.shared.command_frame import COMMAND_WIDTH

    current = trunk_plant()
    older_plant = make_plant_identity(
        species="tyrannosaurus_rex",
        embodiment="anatomical",
        model_path="environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml",
        policy_interface_revision=current.policy_interface_revision - 1,
        observation_dim=current.observation_dim - COMMAND_WIDTH,
    )
    for name in ("20260815_000000", "20260816_000000"):
        build_trunk_run(tmp_path / name, plant=older_plant, task_sha256=OTHER_TASK)
    _set_stage_commit(tmp_path / "20260816_000000" / "01_stance", SCREEN_COMMIT)
    chain, configs = _chain_and_configs()

    selection = select_trunk(
        tmp_path, model_id=TYRANNOSAURUS_REX, chain=chain, stage_configs=configs, plant_identity=current
    )

    assert [parent.run_dir.name for parent in selection.older_interface] == ["20260815_000000"]


# ------------------------------------------------------------------------------------------------- widening


def test_widening_refuses_a_screen_parent_stage_directory(tmp_path):
    pytest.importorskip("stable_baselines3")
    from environments.shared.scripts.widen_checkpoint import WidenError, _resolve_parent

    run = tmp_path / "stance7_r10LTSBpc_s42"
    stage_dir = build_trunk_run(run)
    arguments: dict[str, Any] = dict(
        model_zip=None,
        vecnorm_pkl=None,
        algorithm=None,
        seed=None,
        n_envs=None,
        timesteps=None,
        parent_run_id=None,
    )
    # Not excluded: refused later, for the run block the fixture leaves short.
    with pytest.raises(WidenError, match="n_envs") as other:
        _resolve_parent(parent_stage_dir=stage_dir, **arguments)
    assert "never widened" not in str(other.value)
    _write(run / "provenance.json", {"run_id": run.name, "repository_commit": SCREEN_COMMIT})
    with pytest.raises(WidenError, match="never widened"):
        _resolve_parent(parent_stage_dir=stage_dir, **arguments)


def test_widening_refuses_a_screen_handoff_given_as_an_explicit_pair(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    from environments.shared.scripts.widen_checkpoint import WidenError, _resolve_parent

    stage_dir = build_trunk_run(tmp_path / "run")
    _exclude(monkeypatch, sha256_file(stage_dir / "models" / "robust_best_model.zip"))
    with pytest.raises(WidenError, match="never widened"):
        _resolve_parent(
            parent_stage_dir=None,
            model_zip=stage_dir / "models" / "robust_best_model.zip",
            vecnorm_pkl=stage_dir / "models" / "robust_best_model_vecnorm.pkl",
            algorithm="ppo",
            seed=1,
            n_envs=4,
            timesteps=1000,
            parent_run_id=None,
        )


# ------------------------------------------------------------------------------------------------- results


def test_the_bundle_audit_and_the_catalog_refuse_a_screen_result():
    from environments.shared.result_bundle.audit import _excluded_run_errors
    from environments.shared.species_catalog import CatalogError, _refuse_excluded_run

    handoff = "sha256:77959e30f4f20e1a3bb184d210b4430a14b6df1a046cdbc2e5b279487fbfc299"
    screen = {"repository_commit": SCREEN_COMMIT, "deliverables": {"1": {"model_hash": handoff}}}
    errors = _excluded_run_errors(screen, where="provenance.json")
    assert len(errors) == 2 and all("never a result" in error for error in errors)
    assert _excluded_run_errors({"repository_commit": MAIN_COMMIT, "deliverables": {}}, where="x") == []
    assert _excluded_run_errors({"repository_commit": MAIN_COMMIT, "deliverables": screen["deliverables"]}, where="x")

    with pytest.raises(CatalogError, match="never a committed result"):
        _refuse_excluded_run({"repository_commit": SCREEN_COMMIT}, relative_path="results/x/summary.json")
    with pytest.raises(CatalogError, match="never a committed result"):
        _refuse_excluded_run({"repository_commit": MAIN_COMMIT, "model_hash": handoff}, relative_path="r")
    _refuse_excluded_run({"repository_commit": MAIN_COMMIT, "model_hash": "sha256:" + "0" * 64}, relative_path="r")


# ------------------------------------------------------------------------------------------------- rule (d): a load


def test_rule_d_reads_every_digest_path_and_run_id_key_the_stage_record_has():
    """The lineage keys save_stage_config records for a --load, an edge resume and a widened root: every digest,
    path and run-id key among them is one rule (d) reads, so a renamed key fails here."""
    from environments.shared.config import LOAD_LINEAGE_KEYS, RESUME_LINEAGE_KEYS, WIDEN_LINEAGE_KEYS

    recorded = {*LOAD_LINEAGE_KEYS, *RESUME_LINEAGE_KEYS, *WIDEN_LINEAGE_KEYS}
    assert {key for key in recorded if key.endswith("checkpoint_sha256")} == set(LINEAGE_DIGEST_KEYS)
    assert {key for key in recorded if key.endswith("path")} == set(LINEAGE_PATH_KEYS)
    assert {key for key in recorded if key.endswith("run_id")} == set(LINEAGE_RUN_ID_KEYS)


def test_rule_d_reads_a_screen_checkpoint_a_screen_folder_and_a_screen_run_id():
    assert EXCLUDED_RUN_IDS == SCREEN_RUNS
    final = excluded_checkpoint_reason(PC_S42_FINAL)
    assert final is not None and "stance7_r10LTSBpc_s42/stage1_final.zip" in final and EXCLUSION_DECISION in final
    assert excluded_checkpoint_reason(PC_S42_FINAL.removeprefix("sha256:").upper()) == final
    assert excluded_checkpoint_reason(PC_S42_HANDOFF) == excluded_handoff_reason(PC_S42_HANDOFF)
    assert excluded_checkpoint_reason("sha256:" + "0" * 64) is None and excluded_checkpoint_reason(None) is None
    for inside in (
        "/content/drive/MyDrive/runs/stance7_r10LTSBpc_s42/01_stance/models/stage1_final.zip",
        "stance7_r10LTSBoff_s80/01_stance/models/stage1_4194304_steps",
        "C:\\runs\\stance6_r10LTSB_s60\\01_stance\\models\\robust_best_model.zip",
    ):
        assert excluded_path_reason(inside) is not None, inside
    for outside in ("/runs/stance7_r10LTSBpc_s42_copy/01_stance/models/stage1_final.zip", "/runs/cert_s43/x.zip", 3):
        assert excluded_path_reason(outside) is None, outside
    assert excluded_run_id_reason(" stance7_r10LTSBpc_s60 ") is not None and excluded_run_id_reason("cert_s43") is None
    reason = excluded_lineage_reason({"load_path": None, "parent_run_id": "stance6_r10LTSB_s42"})
    assert reason is not None and reason.startswith("run.parent_run_id: ")
    assert excluded_lineage_reason({"seed": 1, "hyperparameters_sha256": RECIPE_DIGEST}) is None
    assert excluded_lineage_reason(None) is None and excluded_lineage_reason([PC_S42_FINAL]) is None


def test_rule_d_reads_what_save_stage_config_records_for_a_load(tmp_path, monkeypatch):
    """The trainer's own writer: a --load from a screen's run folder, and from a copy of a screen's checkpoint kept
    elsewhere (its digest listed), are excluded; a load from a new seed's folder is not."""
    from environments.shared.config import save_stage_config

    config = {"env_kwargs": {}, "curriculum_kwargs": STANCE_CURRICULUM, "ppo_kwargs": RECIPE}

    def loaded_from(source: Path, name: str, task: str) -> Path:
        # The fixture's checkpoints differ only by the task digest they carry, so each source gets its own.
        handoff = build_trunk_run(source, task_sha256=task) / "models" / "robust_best_model.zip"
        stage_dir: Path = tmp_path / name / "01_stance"
        save_stage_config(
            stage_dir,
            1,
            config,
            "PPO",
            extra={"seed": 45},
            model_id=TYRANNOSAURUS_REX,
            load_path=str(handoff),
            load_mode="resume_same_stage",
        )
        return stage_dir

    from_folder = loaded_from(tmp_path / "drive" / "stance7_r10LTSBpc_s42", "loaded_from_folder", STANCE_TASK)
    copy = tmp_path / "parents" / "copy"
    from_copy = loaded_from(copy, "loaded_from_copy", OTHER_TASK)
    control = loaded_from(tmp_path / "drive" / "cert_s43", "loaded_from_a_new_seed", "sha256:" + "c" * 64)
    _exclude_checkpoint(monkeypatch, sha256_file(copy / "01_stance" / "models" / "robust_best_model.zip"))

    assert "stage_config.json run.load_path: " in (excluded_run_reason(from_folder.parent) or "")
    assert "stage_config.json run.parent_checkpoint_sha256: " in (excluded_run_reason(from_copy.parent) or "")
    assert excluded_run_reason(control.parent) is None


@pytest.mark.parametrize(
    "lineage",
    [
        pytest.param(
            {
                "load_path": "/content/drive/runs/stance7_r10LTSBpc_s42/01_stance/models/stage1_final.zip",
                "load_mode": "resume_same_stage",
                "parent_checkpoint_sha256": PC_S42_FINAL,
            },
            id="a --load of a screen's final from its run folder",
        ),
        pytest.param(
            {
                "load_path": "/content/parents/stance_final.zip",
                "load_mode": "resume_same_stage",
                "parent_checkpoint_sha256": PC_S42_FINAL,
            },
            id="a --load of a copy of a screen's final kept elsewhere",
        ),
        pytest.param(
            {
                "load_path": "/content/drive/runs/stance7_r10LTSBoff_s60/01_stance/models/stage1_99_steps.zip",
                "load_mode": "resume_same_stage",
                "parent_checkpoint_sha256": "sha256:" + "1" * 64,
            },
            id="a --load of an unlisted checkpoint from a screen's run folder",
        ),
        pytest.param(
            {
                "resume_load_path": "/content/drive/runs/cert_s45/01_stance/models/stage1_8388608_steps.zip",
                "resume_checkpoint_sha256": "sha256:" + "2" * 64,
                "parent_run_id": "stance7_r10LTSBpc_s80",
            },
            id="an edge resume of a stage whose parent was a screen run",
        ),
        pytest.param(
            {
                "widened_from_path": "/content/parents/robust_best_model.zip",
                "widened_from_checkpoint_sha256": PC_S42_HANDOFF,
                "widened_from_run_id": "parent",
            },
            id="a widened copy of a screen's handoff",
        ),
    ],
)
def test_a_new_run_that_loaded_a_screen_checkpoint_never_counts(log_base, monkeypatch, lineage):
    """Cases B5 and B6 of the equivalence review: a fresh run at main's commit, its own id and seed, whose stance
    was loaded from a screen's weights.  Rules (a) to (c) see nothing of the screen; rule (d) alone keeps it out of
    discovery (both ways) and reuse, and with the lists emptied the same run counts."""
    this_run = _run(log_base, "cert_s44", seed=44, commit=MAIN_COMMIT)
    loaded = _run(log_base, "cert_s45", seed=45, commit=MAIN_COMMIT, stage_commit=MAIN_COMMIT, lineage=lineage)
    _run(log_base, "cert_s43", seed=43, commit=MAIN_COMMIT)

    assert "01_stance/stage_config.json run." in (excluded_run_reason(loaded) or "")
    assert _discover(this_run) == [{"run_id": "cert_s43", "training_seed": 43}]
    assert _discover(loaded, training_seed=45) == []
    with pytest.raises(AncestorReuseError, match="reuse rule 9"):
        _find(loaded)

    for name in ("EXCLUDED_HANDOFF_SHA256", "EXCLUDED_CHECKPOINT_SHA256"):
        monkeypatch.setattr(run_exclusions, name, {})
    monkeypatch.setattr(run_exclusions, "EXCLUDED_RUN_IDS", frozenset())
    assert [entry["training_seed"] for entry in _discover(this_run)] == [43, 45]


def test_a_copied_screen_stage_resumed_and_rejudged_on_main_never_counts(log_base):
    """The equivalence review's narrower gap: a screen's stage directory copied without provenance.json, resumed on
    main from its copied final (main's commit in stage_config.json; a root's resume records the load under the
    load keys) and re-judged (the fixture's verdict hashes a handoff no list names).  Only the resume's recorded
    checkpoint names the screen."""
    this_run = _run(log_base, "cert_s44", seed=44, commit=MAIN_COMMIT)
    resume = {
        "load_path": "/content/drive/runs/my_copy/01_stance/models/stage1_final.zip",
        "load_mode": "resume_same_stage",
        "parent_checkpoint_sha256": PC_S42_FINAL,
    }
    copy = _run(log_base, "my_copy", seed=42, stage_commit=MAIN_COMMIT, lineage=resume)

    assert "run.parent_checkpoint_sha256: " in (excluded_run_reason(copy) or "")
    assert _discover(this_run) == []


def test_the_bundle_audit_applies_rules_b_and_d_to_every_stage_config_it_hashes():
    from environments.shared.run_exclusions import excluded_stage_config_reason

    assert excluded_stage_config_reason({"git_commit": STANCE6_COMMIT, "run": {}}) is not None
    loaded = {"git_commit": MAIN_COMMIT, "run": {"seed": 45, "parent_checkpoint_sha256": PC_S42_FINAL}}
    assert "run.parent_checkpoint_sha256: " in (excluded_stage_config_reason(loaded) or "")
    assert excluded_stage_config_reason({"git_commit": MAIN_COMMIT, "run": {"seed": 44}}) is None
    assert excluded_stage_config_reason(None) is None


def test_the_results_layer_refuses_a_result_built_on_a_screen_ancestor():
    """An ``ancestors`` record (a node reused from another run) naming a screen's handoff or a screen's run id makes
    the result never a result, in the audit and in the catalog; a deliverable that is a screen's final does too."""
    from environments.shared.result_bundle.audit import _excluded_run_errors
    from environments.shared.run_exclusions import excluded_provenance_reasons
    from environments.shared.species_catalog import CatalogError, _refuse_excluded_run

    unlisted = "sha256:" + "0" * 64
    by_handoff = {"run_id": "20261110_120000", "model_hash": PC_S42_HANDOFF}
    by_run_id = {"run_id": "stance7_r10LTSBpc_s60", "model_hash": unlisted}
    for ancestor in (by_handoff, by_run_id):
        provenance = {"repository_commit": MAIN_COMMIT, "deliverables": {}, "ancestors": {"1": ancestor}}
        errors = _excluded_run_errors(provenance, where="provenance.json")
        assert len(errors) == 1 and "never a result: ancestors.1: " in errors[0] and EXCLUSION_DECISION in errors[0]
        with pytest.raises(CatalogError, match="never a committed result: ancestors.1: "):
            _refuse_excluded_run(provenance, relative_path="results/x/summary.json")

    clean = {
        "repository_commit": MAIN_COMMIT,
        "model_hash": unlisted,
        "deliverables": {"2": {"model_hash": unlisted}},
        "ancestors": {"1": {"run_id": "cert_s44", "model_hash": unlisted}},
    }
    assert excluded_provenance_reasons(clean) == [] and _excluded_run_errors(clean, where="x") == []
    _refuse_excluded_run(clean, relative_path="r")
    final = {**clean, "deliverables": {"1": {"model_hash": PC_S42_FINAL}}}
    assert [reason.split(": ")[0] for reason in excluded_provenance_reasons(final)] == ["deliverables.1"]
    assert excluded_provenance_reasons(None) == [] and excluded_provenance_reasons({"ancestors": [1]}) == []
