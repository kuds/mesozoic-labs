"""The SB3 notebook's publish cell (section 6, ``PUBLISH_RUN``; the maintainer's decision of 2026-11-26).

The chain loop writes a run's training summary and result bundle only when it trains or judges a node: a node it
reuses from the run's own records does ``continue`` without saving. So a session that reuses every node writes no
bundle, and two runs took a hand-typed ``save_run_bundle`` call: a run cut off after its last verdict but before its
bundle (T. rex run ``20261121_175147``, whose runtime ended during its stance probes), and a ``partial`` run that
should count a replicate which certified later (decision D-B10). The opt-in cell after the chain loop writes both
files with the chain loop's own two calls, from what the run holds: a node judged in this run re-enters from its
``gate_verdict.json``, as the chain loop's REUSE branch re-enters one, and an ancestor reused from another run is
read from its ``ancestors/`` record by the bundle writer. Before anything is written it refuses a ``complete`` run
and a chain node that is neither.

The walk follows the chain loop: root first, an ancestor's ``ancestors/`` record standing for it, and stopping after
the first FAILED verdict, where the chain loop halted after publishing it. Before anything is written the cell also
refuses a BEHAVIOR whose target differs from the one the run's ``provenance.json`` records, and a judged node the
bundle would leave out (a review of 2026-11-26 showed a shallower BEHAVIOR sealing a ``partial`` walk run
``complete`` under a stance target, dropping its failed locomotion).

A judged node whose verdict passed re-enters only while its stage hands on the pair the verdict hashes, as the chain
loop's reuse of a node of this run checks it (``find_certified_ancestor``, rule 5, through
``select_handoff_checkpoint``). On the command-driven walk (decision D-D61) that pair is the one the stage's handoff
choice (``models/handoff_choice.json``) names, and the walk's verdict is the ``command_tracking/v1`` judge's, from
the panel report rolled on that pair after training.

The cell runs here as the other notebook tests run theirs: its source, in a namespace holding the helpers cell's
real ``chain_results``, ``write_training_summary`` and ``save_run_bundle``, over runs the result-bundle helpers build
in the chain loop's stage layout (``stage_dirname``), for every model of the catalog and in the screen and quick-test
trees. Each node holds an SB3 pair layout (``<name>.zip`` beside ``<name>_vecnorm.pkl``). The walk node (locomotion)
holds the records a command-driven walk leaves: its handoff choice, ``command_tracking_report.json`` and the panel CSV,
and the ``gate_verdict.json`` the shared judge (``reporting.stage_artifacts._apply_stage_gate``) writes from them, so
the bundle writer re-derives its pass from the panel. Its gate and command are the command-tracking wiring tests'
design walk (a 4-cruise, 2-stand panel), at each model's own horizon and control step. Stance and recovery keep the
helpers' stand-ins (a reward gate, and recovery's none/v1). ``SPECIES_CFG`` is the model's real registry entry and
``STAGE_CONFIGS`` carries each stage's real ``env_kwargs``, so the bare env the cell builds for a node's control step
is the model's own, the walk's with its live command. The one other stand-in is the library writer's evaluation
protocol: the helpers' evidence is three episodes on seed 101, not the notebook's thirty on its seed roles, so
``_lib_save_result_bundle`` forwards the notebook's call with the fixture's protocol (``save_run_bundle``'s own
arguments are pinned in ``test_sb3_notebook_pins.py``).
"""

from __future__ import annotations

import ast
import json
import shutil
from functools import cache
from pathlib import Path
from typing import Any, Callable

import pytest

from environments.shared.command_tracking_panel import plan_command_panel
from environments.shared.config import env_kwargs_record, load_all_stages
from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum import select_handoff_checkpoint
from environments.shared.curriculum.checkpoints import HANDOFF_CHOICE_SCHEMA, write_handoff_choice
from environments.shared.curriculum.command_tracking_evidence import COMMAND_TRACKING_PANEL_CSV
from environments.shared.curriculum.command_tracking_gate import (
    COMMAND_TRACKING_GATE_KIND,
    CRUISE_FAMILY,
    thresholds_from_curriculum,
)
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
from environments.shared.model_id import ModelId
from environments.shared.replication import discover_replicates_for_run
from environments.shared.reporting import save_result_bundle
from environments.shared.reporting import write_training_summary as library_training_summary
from environments.shared.reporting.command_tracking_report import (
    HANDOFF_SELECTION_SEED_START,
    assemble_command_tracking_report,
    write_command_tracking_report,
)
from environments.shared.reporting.stage_artifacts import _apply_stage_gate
from environments.shared.result_bundle import (
    ResultBundleError,
    read_bundle_status,
    sha256_file,
    validate_result_bundle,
    write_gate_verdict,
)
from environments.shared.species_names import model_ids
from environments.shared.species_registry import get_species_config
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname

from .notebook_cells import exec_top_level_def
from .reporting_helpers import make_plant_identity
from .result_bundle_helpers import (
    _ANCESTOR_TASK_SHA256,
    _TRUNK_RUN_ID,
    _complete_bundle_inputs,
    _legacy_stage_dirname,
    _write_ancestor_record,
)
from .test_command_tracking_wiring import MEASUREMENT, PLAN, WALK_CONFIG, WALK_CURRICULUM, _panel
from .test_sb3_notebook_pins import (
    CHAIN_CELL_MARKER,
    INFRA_CELL_MARKER,
    MANUAL_CELL_MARKER,
    _all_cell_sources,
    _call,
    _calls,
    _cell,
    _top_level_assigns,
)

#: Host Git and package state stay out of the bundles (conftest.stable_provenance): a dirty checkout is refused.
pytestmark = pytest.mark.usefixtures("stable_provenance")

PUBLISH_CELL_MARKER = "# ===== PUBLISH THIS RUN"
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
#: Every model the notebook's SPECIES and EMBODIMENT menus select.
MODELS = model_ids()
#: The models whose manifest holds the recovery node (BEHAVIOR = "recover").
RECOVERY_MODELS = tuple(model for model in MODELS if "recovery" in {e.id for e in load_stage_manifest(model).stages})
#: The evaluation protocol of the helpers' evidence (result_bundle_helpers._complete_bundle_inputs).
FIXTURE_PROTOCOL = {"evaluation_episodes": 3, "evaluation_seeds": [101, 102, 103], "seed_roles": None}
#: The names the publish cell must never call: it trains, judges, probes, records and releases nothing.
NEVER_CALLED = (
    "train_stage",
    "evaluate_stage_checkpoints",
    "generate_stage_artifacts",
    "find_certified_ancestor",
    "record_ancestor",
    "freeze_recovery_gate",
    "roll_policy_panel",
    "write_recovery_evidence",
    "write_gate_verdict",
    "halt",
    "disconnect_runtime",
)


def _publish_cell(*, publish: bool = True) -> str:
    src = _cell(PUBLISH_CELL_MARKER)
    return src.replace("PUBLISH_RUN = False", "PUBLISH_RUN = True", 1) if publish else src


def _run_dir(tmp_path: Path, model_id: ModelId, tree: str = "ppo", run_id: str = "20261121_175147") -> Path:
    run_dir = model_id.path(tmp_path / "logs") / tree / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def _plant(model_id: ModelId) -> Any:
    """The plant identity the helpers' stage results and stage configs record (result_bundle_helpers._plant_identity)."""
    return make_plant_identity(**model_id.record(), model_path=model_id.asset_path(Path("environments")).as_posix())


def _judge(
    run_dir: Path, model_id: ModelId, result: dict[str, Any], config: dict[str, Any], dirname: Callable[[Any], str]
) -> Path:
    """Write the node's ``gate_verdict.json`` as ``generate_stage_artifacts`` does: hash-bound to its selected pair,
    with the results it was judged on as ``stage_result``."""
    entry = load_stage_manifest(model_id).resolve(result["stage"])
    stage_dir = run_dir / dirname(entry.reference)
    return write_gate_verdict(
        stage_dir,
        model_id=model_id,
        stage=entry.reference,
        stage_id=entry.id,
        gate_kind="reward_and_length/v1",
        gate_schema_version=1,
        passed=bool(result["publication_gate_passed"]),
        failures=[] if result["publication_gate_passed"] else ["min_avg_reward: evidence=0.5 threshold=1.0"],
        task_sha256=_ANCESTOR_TASK_SHA256,
        judged_by="reporting.stage_artifacts.generate_stage_artifacts",
        checkpoint=Path(f"{result['model_path']}.zip"),
        normalization=Path(result["vecnorm_path"]),
        gate_config=gate_config_view(config["curriculum_kwargs"]),
        stage_result=result,
    )


def _sb3_pairs(stage_dir: Path, result: dict[str, Any]) -> None:
    """Give the node the pair layout ``evaluate_stage_checkpoints`` leaves: each checkpoint a ``.zip`` beside its
    ``_vecnorm.pkl`` (the helpers write ``.pkl`` stand-ins; their evidence binds the bytes, which a rename keeps), and
    ``model_path`` the selected pair's stem, as that function records it."""
    models = stage_dir / "models"
    for path in models.glob("*.pkl"):
        if not path.name.endswith("_vecnorm.pkl"):
            path.rename(path.with_suffix(".zip"))
    result["model_path"] = str(models / "best_model")


#: The task the walk's stage records (its own fingerprint: the panel, the handoff choice and the verdict are bound to
#: it).
WALK_TASK_SHA256 = "sha256:" + "5a" * 32


def _walk_env(model_id: ModelId) -> dict[str, Any]:
    """The walk's ``[env]`` block as ``save_stage_config`` records it under ``reward_weights``: the model's own horizon
    and frame skip, and the design walk's live command (``WALK_CONFIG``)."""
    real = _real_stage_configs(model_id)[2]["env_kwargs"]
    return {
        "max_episode_steps": int(real["max_episode_steps"]),
        "frame_skip": int(real.get("frame_skip", 5)),
        "command_mode": "heading_and_speed",
        "command_config": env_kwargs_record({"command_config": WALK_CONFIG})["command_config"],
    }


def _choose_walk_handoff(models: Path, selected: str = "best_model") -> None:
    """Record *selected* as the walk's handoff choice among every pair of the stage, under its gate and task, on the
    selection panel (as ``command_tracking_report.choose_command_handoff`` records it at the end of training)."""
    ranked = sorted(path.name[: -len("_vecnorm.pkl")] for path in models.glob("*_vecnorm.pkl"))
    selection = plan_command_panel(
        thresholds_from_curriculum(WALK_CURRICULUM), WALK_CONFIG, seed_start=HANDOFF_SELECTION_SEED_START
    )
    write_handoff_choice(
        models,
        {
            "schema": HANDOFF_CHOICE_SCHEMA,
            "gate_kind": COMMAND_TRACKING_GATE_KIND,
            "gate_sha256": gate_config_sha256(gate_config_view(WALK_CURRICULUM)),
            "selection_seed_start": HANDOFF_SELECTION_SEED_START,
            "plan": selection.as_dict(),
            "task_sha256": WALK_TASK_SHA256,
            "selected": selected,
            **_handoff_digests(models, selected),
            "candidates": [{"name": name, **_handoff_digests(models, name)} for name in ranked],
        },
    )


def _judge_walk(
    stage_dir: Path, model_id: ModelId, result: dict[str, Any], config: dict[str, Any], *, passed: bool
) -> None:
    """Judge the walk as training judges it (decision D-D61): its stage records the live command and its task, its
    handoff choice names the pair, the command panel is rolled on that pair (``command_tracking_report.json`` and the
    panel CSV; one cruise episode never settles when *passed* is False), and the shared judge writes the verdict from
    the report, recording the command-tracking numbers onto the stage result."""
    env = _walk_env(model_id)
    record_path = stage_dir / "stage_config.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.update(
        curriculum_kwargs=dict(WALK_CURRICULUM),
        reward_weights=env,
        task_fingerprint={"schema": "mesozoic.task-fingerprint/v2", "task_sha256": WALK_TASK_SHA256},
    )
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config["curriculum_kwargs"] = dict(WALK_CURRICULUM)
    models = stage_dir / "models"
    _choose_walk_handoff(models)
    horizon, control_dt = env["max_episode_steps"], 0.002 * env["frame_skip"]
    overrides: dict[str, dict[str, Any]] = {
        f"seed_{seed}": {"length": horizon, "horizon": horizon, "dt": control_dt} for seed in PLAN.seeds
    }
    if not passed:
        overrides[f"seed_{PLAN.seeds[PLAN.families.index(CRUISE_FAMILY)]}"]["first_event_settled"] = False
    report = assemble_command_tracking_report(
        model_id,
        2,
        curriculum=WALK_CURRICULUM,
        command_config=WALK_CONFIG,
        plan=PLAN,
        episodes=_panel(PLAN, **overrides),
        measurement=MEASUREMENT,
        horizon=horizon,
        control_dt=control_dt,
        policy="best_model.zip — VecNormalize stats applied (clip 10)",
        normalization={"mode": "vecnormalize-frozen", "clip_obs": 10.0, "epsilon": 1e-8},
        handoff={"checkpoint": "best_model.zip", **_handoff_digests(models, "best_model")},
        task_sha256=WALK_TASK_SHA256,
        plant_identity=_plant(model_id).to_dict(),
        checkpoint_plant_validated=True,
    )
    write_command_tracking_report(stage_dir, report)
    _apply_stage_gate(
        stage=2,
        stage_config=config,
        stage_results=result,
        stance_report=None,
        stage_dir=stage_dir,
        model_id=model_id,
        command_tracking_report=report,
    )
    assert result["publication_gate_passed"] is passed, result["gate_failures"]
    assert json.loads((stage_dir / "gate_verdict.json").read_text(encoding="utf-8"))["passed"] is passed


def _handoff_digests(models: Path, name: str) -> dict[str, str]:
    return {
        "checkpoint_sha256": sha256_file(models / f"{name}.zip"),
        "normalization_sha256": sha256_file(models / f"{name}_vecnorm.pkl"),
    }


def _record_seed(stage_dir: Path, seed: int) -> None:
    """Give the node's ``stage_config.json`` the run block ``train_base.train`` records (its training seed)."""
    path = stage_dir / "stage_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["run"] = {**config.get("run", {}), "seed": seed, "n_envs": 4}
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _judged_run(
    run_dir: Path,
    model_id: ModelId,
    stage_refs: "tuple[int | str, ...]",
    *,
    failed: "tuple[int | str, ...]" = (),
    seed: int = 42,
    dirname: "Callable[[Any], str] | None" = None,
) -> "dict[int | str, dict[str, Any]]":
    """A run the chain loop trained and judged node by node in its stage layout, cut off before its bundle: each
    node holds its stage config, selected and final pairs, evaluation evidence and ``gate_verdict.json`` (the walk's,
    locomotion's, judged from its command panel by ``_judge_walk``), and the run holds no ``summary.json``,
    ``collected_results.csv`` or ``artifact_manifest.json``. Returns its configs. *dirname* names a stage directory
    (default: the chain loop's ``stage_dirname``)."""
    layout = dirname if dirname is not None else (lambda stage: stage_dirname(model_id, stage))
    results, configs = _complete_bundle_inputs(
        run_dir, algorithm="PPO", dirname=layout, model_id=model_id, stage_refs=stage_refs
    )
    for result in results:
        stage_dir = run_dir / layout(result["stage"])
        _sb3_pairs(stage_dir, result)
        _record_seed(stage_dir, seed)
        if result["stage"] == 2:
            _judge_walk(stage_dir, model_id, result, configs[2], passed=2 not in failed)
            continue
        if result["stage"] in failed:
            result["gate_passed"] = result["publication_gate_passed"] = False
        _judge(run_dir, model_id, result, configs[result["stage"]], layout)
    return configs


def _trunked_run(run_dir: Path, model_id: ModelId) -> "dict[int | str, dict[str, Any]]":
    """A walk run that reused stance from the trunk run beside it (``ancestors/stance/``) and trained and judged
    locomotion here on its handoff, with the run-block lineage that reuse leaves (as
    ``result_bundle_helpers._reused_trunk_bundle_inputs`` writes it, in the chain loop's stage layout)."""
    configs = _judged_run(run_dir, model_id, (2,))
    trunk = run_dir.parent / "20261101_000000"
    _, checkpoint_sha256 = _write_ancestor_record(run_dir, 1, model_id=model_id, source_run_dir=trunk)
    config_path = run_dir / stage_dirname(model_id, 2) / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run"].update(
        load_path=str(trunk / "stage1" / "models" / "best_model"),
        load_mode="initialize_next_stage",
        parent_task_sha256=_ANCESTOR_TASK_SHA256,
        parent_checkpoint_sha256=checkpoint_sha256,
        parent_run_id=_TRUNK_RUN_ID,
    )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return configs


def _namespace(
    run_dir: Path,
    model_id: ModelId,
    behavior: str,
    stage_configs: "dict[int | str, dict[str, Any]]",
    *,
    quick_test: bool = False,
) -> dict[str, Any]:
    """What the publish cell reads after sections 1-6: the resolve cell's chain (``MANIFEST.resolve_behavior``,
    ``chain_for``), the storage cell's run and seeds, and the helpers cell's real wrappers over the library."""
    manifest = load_stage_manifest(model_id)
    target = manifest.resolve_behavior(behavior)
    calls: list[dict[str, Any]] = []
    # The writer reads every stage's config from its recorded stage_config.json and ignores STAGE_CONFIGS; the cell
    # reads only a stage's env_kwargs from it, for the bare env that answers the node's control step.
    real_configs = _real_stage_configs(model_id)
    session_configs = {
        ref: {**config, "env_kwargs": real_configs[ref]["env_kwargs"]} for ref, config in stage_configs.items()
    }

    def library_writer(stage_results_list, **kwargs):
        calls.append({"stage_results_list": stage_results_list, **kwargs})
        return save_result_bundle(stage_results_list, **{**kwargs, **FIXTURE_PROTOCOL, "backend_version": "2.9.0"})

    namespace: dict[str, Any] = {
        "Path": Path,
        "RUN_DIR": run_dir,
        "_ACTIVE_RUN_ID": run_dir.name,
        "MODEL_ID": model_id,
        "BEHAVIOR": behavior,
        "MANIFEST": manifest,
        "TARGET_NODE": target,
        "CHAIN": manifest.chain_for(target.id),
        "STAGE_CONFIGS": session_configs,
        "SPECIES_CFG": get_species_config(model_id),
        "NODE_RESULTS": {},
        "ALGORITHM": "ppo",
        "SEED": 42,
        "N_ENVS": 4,
        "QUICK_TEST": quick_test,
        "HARDWARE_LABEL": "local",
        "CHECKPOINT_SELECTION_SEED": 1042,
        "EVALUATION_SEED": 3042,
        "PUBLICATION_SEED_START": PUBLICATION_SEED_START,
        "PLANT_IDENTITY": _plant(model_id),
        "repo_root": None,
        "discover_replicates_for_run": discover_replicates_for_run,
        "_lib_write_training_summary": library_training_summary,
        "_lib_save_result_bundle": library_writer,
        "bundle_calls": calls,
    }
    infra = _cell(INFRA_CELL_MARKER)
    for name in ("chain_results", "write_training_summary", "save_run_bundle"):
        exec_top_level_def(infra, name, namespace)
    return namespace


@cache
def _real_stage_configs(model_id: ModelId) -> "dict[int | str, dict[str, Any]]":
    return load_all_stages(model_id)


@cache
def _control_dt(model_id: ModelId, stage: "int | str" = 1) -> float:
    """The node's control step, from a bare env of its committed task (0.02 s for the Compsognathus pair)."""
    env = get_species_config(model_id).env_class(**_real_stage_configs(model_id)[stage].get("env_kwargs", {}))
    try:
        return float(env.dt)
    finally:
        env.close()


def _publish(namespace: dict[str, Any], *, publish: bool = True) -> None:
    exec(compile(_publish_cell(publish=publish), "sb3_publish", "exec"), namespace)


def _chain_loop_write(namespace: dict[str, Any], *node_ids: str) -> None:
    """The chain loop's own write after its last judged node: the named nodes re-enter NODE_RESULTS from their
    verdicts, then its two calls run (``write_training_summary`` and ``save_run_bundle``)."""
    manifest, run_dir, model_id = namespace["MANIFEST"], namespace["RUN_DIR"], namespace["MODEL_ID"]
    for node_id in node_ids:
        stage_dir = run_dir / stage_dirname(model_id, manifest.resolve(node_id).reference)
        namespace["NODE_RESULTS"][node_id] = json.loads((stage_dir / "gate_verdict.json").read_text())["stage_result"]
    exec(
        "write_training_summary(RUN_DIR, chain_results())\nsave_run_bundle(chain_results(), model_id=MODEL_ID)",
        namespace,
    )


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _node_files(run_dir: Path) -> dict[str, bytes]:
    """Every file under the run's stage directories and ``ancestors/``: what the cell must never touch."""
    return {name: data for name, data in _files(run_dir).items() if "/" in name}


#: The identity arguments the storage cell passes ``initialize_result_bundle``, as ``save_run_bundle`` forwards them.
_IDENTITY_KEYS = ("model_id", "algorithm", "backend", "seed", "parallel_envs", "hardware", "run_id", "repository_root")
BUNDLE_FILES = ("summary.json", "collected_results.csv", "artifact_manifest.json", "training_summary.txt")


# ---------------------------------------------------------------------------------------------- the cell's shape
class TestTheCell:
    def test_it_sits_after_the_chain_loop_under_its_own_heading_and_is_off_by_default(self):
        sources = _all_cell_sources()
        index = next(i for i, src in enumerate(sources) if src.startswith(PUBLISH_CELL_MARKER))
        chain = next(i for i, src in enumerate(sources) if src.startswith(CHAIN_CELL_MARKER))
        manual = next(i for i, src in enumerate(sources) if src.startswith(MANUAL_CELL_MARKER))
        assert chain < index - 1 < index < manual, "after the chain loop, before the manual node"
        heading = sources[index - 1]
        assert heading.startswith("### Publish this run")
        for phrase in (
            "PUBLISH_RUN = True",
            "`complete`",
            "gate_verdict.json",
            "ancestors/<id>/ancestor.json",
            "`provenance.json` records another target deliverable",
            # A passed node whose stage hands on another pair than its verdict's (on the walk, its handoff choice's).
            "`models/handoff_choice.json`",
            "Skip the RESUME cell and the chain loop",
            # The publish path ends in section 9, whose disconnect flushes Drive's buffered writes of the bundle.
            "this cell, and last section 9's cell",
            "drive.flush_and_unmount()",
            # A complete run's late replicate is typed after the chain loop, before the auto-disconnect.
            "in a session run up to the chain loop rather than Run all",
            "`ALGORITHM`",
            "`N_ENVS`",
        ):
            assert phrase in heading, phrase
        assigns = _top_level_assigns(_cell(PUBLISH_CELL_MARKER))
        assert isinstance(assigns["PUBLISH_RUN"], ast.Constant) and assigns["PUBLISH_RUN"].value is False
        tree = ast.parse(_cell(PUBLISH_CELL_MARKER))
        guard = [node for node in tree.body if isinstance(node, ast.If)]
        assert len(guard) == 1 and ast.unparse(guard[0].test) == "PUBLISH_RUN"
        assert all(isinstance(node, (ast.Assign, ast.If)) for node in tree.body), "nothing runs outside the guard"

    def test_it_writes_with_the_chain_loops_two_calls_between_its_refusals_and_the_check(self):
        src = _cell(PUBLISH_CELL_MARKER)
        tree = ast.parse(src)
        refuse = _call(tree, "refuse_write_into_complete_run")
        assert [ast.unparse(arg) for arg in refuse.args] == ["RUN_DIR"]
        summary, bundle = _call(tree, "write_training_summary"), _call(tree, "save_run_bundle")
        assert ast.unparse(summary) == "write_training_summary(RUN_DIR, chain_results())"
        assert ast.unparse(bundle) == "save_run_bundle(chain_results(), model_id=MODEL_ID)"
        loop = _cell(CHAIN_CELL_MARKER)
        assert ast.unparse(summary) in loop and ast.unparse(bundle) in loop, "the chain loop's own two calls"
        check = _call(tree, "validate_result_bundle")
        raises = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]
        refusals = [node for node in raises if node.exc is not None]
        assert refuse.lineno < min(node.lineno for node in raises), "the complete-run refusal comes first"
        assert max(node.lineno for node in refusals) < summary.lineno < bundle.lineno < check.lineno
        # After the summary, only the re-raise of a refused bundle write, once the summary is put back.
        [reraise] = [node for node in raises if node.lineno > summary.lineno]
        assert reraise.exc is None and bundle.lineno < reraise.lineno < check.lineno
        for name in NEVER_CALLED:
            assert not _calls(tree, name), f"the publish cell calls {name}"


# ---------------------------------------------------------------------------------------------- executed
class TestOffByDefault:
    def test_the_committed_cell_writes_nothing(self, tmp_path, capsys):
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1,))
        before = _files(run_dir)
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs)
        _publish(namespace, publish=False)
        assert _files(run_dir) == before and namespace["bundle_calls"] == [] and namespace["NODE_RESULTS"] == {}
        assert capsys.readouterr().out == ""


class TestRefusals:
    def test_a_complete_run_is_refused_before_anything_is_written(self, tmp_path):
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1,))
        _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs))
        assert read_bundle_status(run_dir) == "complete"
        before = _files(run_dir)
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs)
        with pytest.raises(ResultBundleError, match=r"Run '20261121_175147' is already published") as refused:
            _publish(namespace)
        message = str(refused.value)
        # Nothing to publish: never the generic advice to train the target again in a fresh run.
        assert "has nothing to write and has written nothing. Set PUBLISH_RUN = False." in message
        assert "save_run_bundle(chain_results(), model_id=MODEL_ID) once the chain loop" in message
        assert "in a session run up to the chain loop rather than Run all" in message
        assert "fresh run" not in message and "Nothing has been trained" not in message
        assert _files(run_dir) == before and namespace["bundle_calls"] == [] and namespace["NODE_RESULTS"] == {}

    def test_an_unreadable_manifest_is_refused_with_its_own_reason(self, tmp_path):
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1,))
        (run_dir / "artifact_manifest.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ResultBundleError, match="cannot read artifact manifest"):
            _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs))

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_shallower_behavior_never_retargets_a_partial_run(self, tmp_path, model_id):
        """Walk, stance passed and locomotion failed, published partial under the locomotion target: BEHAVIOR =
        "stance" would seal it complete under a stance target without its failed locomotion, and immutable."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _judged_run(run_dir, model_id, (1, 2), failed=(2,))
        _publish(_namespace(run_dir, model_id, "walk", configs))
        assert read_bundle_status(run_dir) == "partial"
        before = _files(run_dir)
        namespace = _namespace(run_dir, model_id, "stance", configs)
        with pytest.raises(RuntimeError, match=r"records the target deliverable 'locomotion'") as refused:
            _publish(namespace)
        assert "BEHAVIOR='stance' resolves to 'stance'" in str(refused.value)
        assert "Set BEHAVIOR = 'locomotion'" in str(refused.value)
        assert _files(run_dir) == before and namespace["bundle_calls"] == [] and namespace["NODE_RESULTS"] == {}

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_run_stopped_above_its_target_is_never_sealed_under_a_shallower_one(self, tmp_path, model_id):
        """Walk stopped after stance (the chain loop's write left it partial, target locomotion): BEHAVIOR = "stance"
        would seal it complete, and the resolve cell would then refuse the walk session that trains locomotion."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _judged_run(run_dir, model_id, (1,))
        _chain_loop_write(_namespace(run_dir, model_id, "walk", configs), "stance")
        assert read_bundle_status(run_dir) == "partial"
        before = _files(run_dir)
        with pytest.raises(RuntimeError, match=r"records the target deliverable 'locomotion'"):
            _publish(_namespace(run_dir, model_id, "stance", configs))
        assert _files(run_dir) == before

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_judged_node_off_the_chain_is_never_left_out(self, tmp_path, model_id):
        """A walk run cut off before its bundle records no target, but its judged locomotion is not on the stance
        chain: publishing under "stance" would drop it."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _judged_run(run_dir, model_id, (1, 2))
        before = _files(run_dir)
        with pytest.raises(RuntimeError, match=r"'locomotion' is judged in run '20261121_175147'") as refused:
            _publish(_namespace(run_dir, model_id, "stance", configs))
        assert "it is not on the chain of BEHAVIOR='stance' (stance)" in str(refused.value)
        assert _files(run_dir) == before

    def test_a_judged_node_below_the_failed_one_is_never_left_out(self, tmp_path):
        """The manual cell judged the walk (locomotion) in a run whose stance failed: the walk of the chain stops at
        the failed stance, and the locomotion verdict below it would be dropped."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1, 2), failed=(1,))
        before = _files(run_dir)
        with pytest.raises(RuntimeError, match=r"'locomotion' is judged in run") as refused:
            _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs))
        assert "it lies below 'stance', whose verdict FAILED, where the chain loop stops" in str(refused.value)
        assert _files(run_dir) == before

    @pytest.mark.parametrize("node", ["stance", "locomotion"])
    def test_a_passed_node_whose_stage_hands_on_another_pair_is_refused_as_the_chain_loops_reuse_refuses_it(
        self, tmp_path, node
    ):
        """The chain loop's reuse of a node of this run takes the pair ``select_handoff_checkpoint`` picks now and
        refuses it unless it is the pair the verdict hashes (rule 5). On the walk that pair is the one its handoff
        choice names: a choice made again after the verdict, naming the final pair, moves it. On the stance a
        ``robust_best_model`` pair written after the verdict does (it comes first in the reward order)."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1, 2))
        entry = load_stage_manifest(VELOCIRAPTOR_MONGOLIENSIS).resolve(node)
        models = run_dir / stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, entry.reference) / "models"
        if node == "locomotion":
            _choose_walk_handoff(models, selected="stage2_final")
            moved_to = "'stage2_final'"
        else:
            for suffix in (".zip", "_vecnorm.pkl"):
                (models / f"robust_best_model{suffix}").write_bytes(b"written after the verdict" + suffix.encode())
            moved_to = "'robust_best_model'"
        assert select_handoff_checkpoint(models)[0] == moved_to.strip("'")
        before = _files(run_dir)
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs)
        with pytest.raises(RuntimeError, match=rf"{node!r} passed its gate in run '20261121_175147'") as refused:
            _publish(namespace)
        message = str(refused.value)
        assert "on the pair 'models/best_model.zip'" in message and f"hands on {moved_to} now" in message
        assert "models/handoff_choice.json" in message and "the chain loop's reuse refuses the node" in message
        assert "remove that directory's gate_verdict.json so the chain loop's JUDGE branch re-judges it" in message
        assert _files(run_dir) == before and namespace["bundle_calls"] == [] and namespace["NODE_RESULTS"] == {}

    def test_a_failed_walk_is_published_on_the_pair_it_was_judged_on_whatever_its_stage_hands_on_now(self, tmp_path):
        """A failed verdict is never re-entered by the chain loop's reuse (it halts on it): the cell publishes it as
        the chain loop's own write did, on the pair it was judged on."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1, 2), failed=(2,))
        locomotion = run_dir / stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, 2)
        _choose_walk_handoff(locomotion / "models", selected="stage2_final")
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs)
        _publish(namespace)
        assert read_bundle_status(run_dir) == "partial"
        [_, walk] = namespace["bundle_calls"][0]["stage_results_list"]
        assert walk["model_path"] == str(locomotion / "models" / "best_model")

    @pytest.mark.parametrize("held", ["nothing", "unjudged checkpoints", "an ancestor record of the target"])
    def test_an_unjudged_node_is_refused_by_name_before_anything_is_written(self, tmp_path, held):
        """Walk on the velociraptor: stance judged here, locomotion (the target) not judged. The chain loop judges a
        finished node and trains a missing one; an ancestors/ record never stands in for the target (D-A18)."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1,))
        locomotion = run_dir / stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, 2)
        if held == "unjudged checkpoints":
            (locomotion / "models").mkdir(parents=True)
            (locomotion / "models" / "stage2_final.zip").write_bytes(b"trained, never judged")
        elif held == "an ancestor record of the target":
            _write_ancestor_record(run_dir, 2, source_run_dir=run_dir.parent / "20261101_000000")
        before = _files(run_dir)
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs)
        with pytest.raises(RuntimeError, match=r"'locomotion' is not judged in run '20261121_175147'") as refused:
            _publish(namespace)
        message = str(refused.value)
        assert "refused before anything is written" in message and "Run the chain loop (section 6) first" in message
        assert "BEHAVIOR='walk' resolves to 'locomotion'" in message
        assert "re-run sections 2-3 and the helpers cell" in message
        assert "the target is never reused from another run (D-A18)" in message
        assert _files(run_dir) == before and namespace["bundle_calls"] == [] and namespace["NODE_RESULTS"] == {}

    def test_a_run_that_holds_no_node_is_refused_as_a_run_id_mistake(self, tmp_path):
        """RUN_ID = "" in a fresh runtime lets the storage cell open a new run, which holds only its provenance and the
        preflight records. Publishing it is a RUN_ID mistake, never a reason to run the chain loop, which would train
        the whole chain into that run."""
        run_dir = _run_dir(tmp_path, TYRANNOSAURUS_REX, run_id="20261127_003139")
        for name in ("provenance.json", "trunk_run.json", "zero_action_baseline.json"):
            (run_dir / name).write_text("{}", encoding="utf-8")
        before = _files(run_dir)
        namespace = _namespace(run_dir, TYRANNOSAURUS_REX, "stance", {})
        with pytest.raises(
            RuntimeError, match=r"run '20261127_003139' \(.*\) holds no node of tyrannosaurus_rex"
        ) as refused:
            _publish(namespace)
        message = str(refused.value)
        assert "neither a stage directory nor an ancestors/ record" in message
        assert "Set RUN_ID to the run to publish" in message and "the chain loop is not needed" in message
        assert 'RUN_ID = "" keeps the run this runtime opened last and, in a fresh runtime, opens a new one' in message
        assert "Run the chain loop" not in message
        assert _files(run_dir) == before and namespace["bundle_calls"] == [] and namespace["NODE_RESULTS"] == {}

    def test_an_ancestor_neither_judged_nor_recorded_is_refused(self, tmp_path):
        """Locomotion judged here, stance (an ancestor) nowhere: the chain loop would train it."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (2,))
        before = _files(run_dir)
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs)
        with pytest.raises(RuntimeError, match=r"'stance' is not judged in run .* nor recorded as a reused ancestor"):
            _publish(namespace)
        assert _files(run_dir) == before

    def test_a_verdict_without_stage_result_is_refused_as_the_chain_loop_refuses_it(self, tmp_path):
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1,))
        verdict_path = run_dir / stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, 1) / "gate_verdict.json"
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        verdict["stage_result"] = None
        verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        before = _files(run_dir)
        with pytest.raises(RuntimeError, match="records no stage_result to re-enter the bundle with") as refused:
            _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs))
        assert "run scripts/backfill_gate_verdict.py --force on that directory" in str(refused.value)
        assert _files(run_dir) == before

    def test_a_checkpoint_recorded_under_another_root_that_no_longer_matches_is_refused(self, tmp_path):
        """A run copied off its root whose selected checkpoint no longer has the digest its verdict was judged on."""
        judged = _run_dir(tmp_path / "drive", VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(judged, VELOCIRAPTOR_MONGOLIENSIS, (1,))
        run_dir = tmp_path / "local" / judged.relative_to(tmp_path / "drive")
        shutil.copytree(judged, run_dir)
        (run_dir / stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, 1) / "models" / "best_model.zip").write_bytes(b"other")
        before = _files(run_dir)
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs)
        with pytest.raises(RuntimeError, match=r"'stance' was judged with its model_path at .*drive.*, outside run"):
            _publish(namespace)
        assert _files(run_dir) == before and namespace["bundle_calls"] == []

    def test_a_refused_bundle_write_puts_the_summary_back(self, tmp_path):
        """The writer's own refusal comes after the summary is written (the chain loop's order): the summary and the
        session's NODE_RESULTS are put back, so a published partial run still verifies."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1, 2), failed=(2,))
        _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs))
        # A record of the target beside its verdict: the cell never reads it (D-A18), and the writer refuses a node
        # recorded both as trained here and as reused.
        _write_ancestor_record(run_dir, 2, source_run_dir=run_dir.parent / "20261101_000000")
        before = _files(run_dir)
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs)
        namespace["NODE_RESULTS"]["stance"] = {"left": "by the session"}
        with pytest.raises(ResultBundleError, match="recorded both as trained in this run and as reused ancestors"):
            _publish(namespace)
        assert _files(run_dir) == before, "training_summary.txt is put back"
        assert namespace["NODE_RESULTS"] == {"stance": {"left": "by the session"}}


class TestPublishing:
    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    @pytest.mark.parametrize("tree", ["ppo", "ppo_screen", "ppo_quick_test"])
    def test_a_run_cut_off_before_its_bundle_is_sealed_without_touching_a_stage_file(
        self, tmp_path, capsys, model_id, tree
    ):
        """The r11 T. rex stance case, for every model and every run tree: the stance verdict, summary, figures and
        replays are on disk and the stance probes are not (the runtime ended during them). The bundle appears,
        complete, and verifies; the stage directory is byte-for-byte what it was."""
        run_dir = _run_dir(tmp_path, model_id, tree)
        configs = _judged_run(run_dir, model_id, (1,))
        stage_dir = run_dir / stage_dirname(model_id, 1)
        for name in ("stage_summary.txt", "figures/training_curves.png", "replays/stage1_best.mp4"):
            (stage_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (stage_dir / name).write_bytes(b"written before the probes")
        assert not any((run_dir / name).exists() for name in BUNDLE_FILES)
        nodes = _node_files(run_dir)
        namespace = _namespace(run_dir, model_id, "stance", configs, quick_test=tree == "ppo_quick_test")
        _publish(namespace)
        assert all((run_dir / name).is_file() for name in BUNDLE_FILES)
        assert _node_files(run_dir) == nodes, "no stage file is written, rewritten or removed"
        report = validate_result_bundle(run_dir, require_complete=True)
        assert report["status"] == "canonical-valid" and report["deliverables"]["1"]["certified"] is True
        [call] = namespace["bundle_calls"]
        assert call["run_dir"] == run_dir and call["target_deliverable"] == "1"
        verdict = json.loads((stage_dir / "gate_verdict.json").read_text(encoding="utf-8"))
        assert call["stage_results_list"] == [{**verdict["stage_result"], "sim_dt": _control_dt(model_id)}], (
            "the node re-enters from its verdict, with its control step"
        )
        out = capsys.readouterr().out
        assert "Result bundle published and checked: complete (canonical-valid), target 'stance'" in out
        assert "  stance: certified, 1 run of 1 seed" in out
        # The summary's sim time is the episode length times the node's own control step (a verdict records none:
        # the cell takes it from a bare env of the task, 0.02 s for the Compsognathus pair, 0.01 s for the rest).
        summary = (run_dir / "training_summary.txt").read_text(encoding="utf-8")
        assert f"201.0 +/- 4.9 steps ({201.0 * _control_dt(model_id):.2f}s sim time)" in summary
        if model_id.species == "compsognathus_longipes":
            assert "(4.02s sim time)" in summary

    def test_a_run_in_the_legacy_stage_layout_is_sealed_as_the_reuse_rule_finds_it(self, tmp_path):
        """A run from before the 2026-08-20 layout (stage1/): the chain loop's reuse finds its verdict through
        stage_dir_candidates, and so does this cell."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1,), dirname=_legacy_stage_dirname)
        assert (run_dir / "stage1" / "gate_verdict.json").is_file()
        nodes = _node_files(run_dir)
        _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs))
        assert _node_files(run_dir) == nodes
        assert validate_result_bundle(run_dir, require_complete=True)["deliverables"]["1"]["certified"] is True

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_run_judged_under_another_root_is_sealed_from_its_verdicts_pair(self, tmp_path, model_id):
        """A run copied off Drive into a local tree, the original still in place: each verdict's stage_result names
        its checkpoint pair under the old root, and the bundle is sealed from the verdict's stage-directory-relative,
        hash-bound pair in this copy."""
        judged = _run_dir(tmp_path / "drive", model_id)
        configs = _judged_run(judged, model_id, (1,))
        run_dir = tmp_path / "local" / judged.relative_to(tmp_path / "drive")
        shutil.copytree(judged, run_dir)
        nodes = _node_files(run_dir)
        namespace = _namespace(run_dir, model_id, "stance", configs)
        _publish(namespace)
        assert _node_files(run_dir) == nodes
        report = validate_result_bundle(run_dir, require_complete=True)
        assert report["deliverables"]["1"]["certified"] is True
        [result] = namespace["bundle_calls"][0]["stage_results_list"]
        stage_dir = run_dir / stage_dirname(model_id, 1)
        assert result["model_path"] == str(stage_dir / "models" / "best_model.zip")
        assert result["vecnorm_path"] == str(stage_dir / "models" / "best_model_vecnorm.pkl")

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_run_that_reused_an_ancestor_from_a_trunk_run_is_sealed(self, tmp_path, capsys, model_id):
        """Walk, stance reused from the trunk (its ancestors/ record) and locomotion judged here: the record stands
        in for stance, which the bundle lists under its ancestors, and the run publishes locomotion."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _trunked_run(run_dir, model_id)
        nodes = _node_files(run_dir)
        namespace = _namespace(run_dir, model_id, "walk", configs)
        _publish(namespace)
        assert _node_files(run_dir) == nodes
        report = validate_result_bundle(run_dir, require_complete=True)
        assert set(report["deliverables"]) == {"2"} and report["deliverables"]["2"]["certified"] is True
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        assert set(provenance["ancestors"]) == {"1"} and provenance["ancestors"]["1"]["run_id"] == _TRUNK_RUN_ID
        assert [result["stage"] for result in namespace["bundle_calls"][0]["stage_results_list"]] == [2]
        out = capsys.readouterr().out
        assert "  locomotion: certified, 1 run of 1 seed" in out
        assert "  stance: reused from another run (ancestors/stance/), published by that run" in out

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    @pytest.mark.parametrize("own_stance", ["passed", "failed"])
    def test_an_ancestor_record_stands_for_the_node_beside_a_verdict_of_the_runs_own(
        self, tmp_path, model_id, own_stance
    ):
        """The run's own stance verdict no longer passed the reuse rule, so the chain loop reused stance from the
        trunk (ancestors/stance/) and trained locomotion on it, leaving the node out of NODE_RESULTS. The record
        stands for stance here too, as the bundle the chain loop wrote recorded it."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _trunked_run(run_dir, model_id)
        _judged_run(run_dir, model_id, (1,), failed=(1,) if own_stance == "failed" else ())
        nodes = _node_files(run_dir)
        namespace = _namespace(run_dir, model_id, "walk", configs)
        _publish(namespace)
        assert _node_files(run_dir) == nodes, "the run's own stance directory is left as it is"
        report = validate_result_bundle(run_dir, require_complete=True)
        assert set(report["deliverables"]) == {"2"} and report["deliverables"]["2"]["certified"] is True
        assert [result["stage"] for result in namespace["bundle_calls"][0]["stage_results_list"]] == [2]

    def test_stale_session_results_never_enter_the_bundle(self, tmp_path):
        """NODE_RESULTS left by another RUN_ID or BEHAVIOR in the kernel (the helpers cell not re-run): the bundle is
        rebuilt from the run's files alone."""
        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1,))
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "stance", configs)
        namespace["NODE_RESULTS"]["locomotion"] = {"stage": 2, "left": "by another run"}
        _publish(namespace)
        assert set(namespace["NODE_RESULTS"]) == {"stance"}
        assert [result["stage"] for result in namespace["bundle_calls"][0]["stage_results_list"]] == [1]
        assert set(validate_result_bundle(run_dir, require_complete=True)["deliverables"]) == {"1"}

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_walk_run_cut_off_before_its_bundle_is_sealed_from_its_command_tracking_verdict(
        self, tmp_path, capsys, model_id
    ):
        """The walk certifies on every model: stance and locomotion judged, the runtime lost after locomotion's verdict
        (its figures and replays are best effort). Locomotion re-enters from its command_tracking/v1 verdict, on the
        pair its handoff choice names, and the writer re-derives its pass from the panel CSV."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _judged_run(run_dir, model_id, (1, 2))
        locomotion = run_dir / stage_dirname(model_id, 2)
        assert {"gate_verdict.json", "command_tracking_report.json", COMMAND_TRACKING_PANEL_CSV} <= {
            path.name for path in locomotion.iterdir()
        }
        assert select_handoff_checkpoint(locomotion / "models")[0] == "best_model"
        nodes = _node_files(run_dir)
        namespace = _namespace(run_dir, model_id, "walk", configs)
        _publish(namespace)
        assert _node_files(run_dir) == nodes
        report = validate_result_bundle(run_dir, require_complete=True)
        assert report["status"] == "canonical-valid"
        assert report["deliverables"]["1"]["certified"] is True and report["deliverables"]["2"]["certified"] is True
        [_, walk] = namespace["bundle_calls"][0]["stage_results_list"]
        verdict = json.loads((locomotion / "gate_verdict.json").read_text(encoding="utf-8"))
        assert verdict["gate_kind"] == COMMAND_TRACKING_GATE_KIND and verdict["checkpoint"] == "models/best_model.zip"
        assert walk == {**verdict["stage_result"], "sim_dt": _control_dt(model_id, 2)}
        assert walk["command_profile"] == "walk" and walk["command_handoff_selected"] == "best_model"
        assert (walk["command_cruise_successes"], walk["command_stand_successes"]) == (4, 2)
        out = capsys.readouterr().out
        assert "Result bundle published and checked: complete (canonical-valid), target 'locomotion'" in out
        assert "  locomotion: certified, 1 run of 1 seed" in out

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_chain_stopped_at_its_failed_target_is_published_partial(self, tmp_path, capsys, model_id):
        """The usual partial run, on the model's deepest behavior (walk, since decision D-D59 removed the hunt):
        stance passed and locomotion failed its command panel. The walk of the chain stops after the failed node, as
        the chain loop halted there."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _judged_run(run_dir, model_id, (1, 2), failed=(2,))
        namespace = _namespace(run_dir, model_id, "walk", configs)
        _publish(namespace)
        report = validate_result_bundle(run_dir, require_complete=False, require_publishable=True)
        assert report["status"] == "canonical-partial" and read_bundle_status(run_dir) == "partial"
        assert report["deliverables"]["1"]["certified"] is True and report["deliverables"]["2"]["certified"] is False
        assert namespace["bundle_calls"][0]["target_deliverable"] == namespace["TARGET_NODE"].key
        out = capsys.readouterr().out
        assert f"partial (canonical-partial), target {namespace['TARGET_NODE'].id!r}" in out
        assert "  locomotion: NOT certified" in out

    @pytest.mark.parametrize("model_id", MODELS, ids=str)
    def test_a_failed_stance_cut_off_before_its_bundle_is_published_as_failed(self, tmp_path, capsys, model_id):
        """Walk, stance judged and FAILED, the runtime lost during its probes: the run gets the bundle the chain
        loop's write would have left, with nothing certified and no summary.json."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _judged_run(run_dir, model_id, (1,), failed=(1,))
        nodes = _node_files(run_dir)
        _publish(_namespace(run_dir, model_id, "walk", configs))
        assert _node_files(run_dir) == nodes
        assert read_bundle_status(run_dir) == "failed" and not (run_dir / "summary.json").exists()
        out = capsys.readouterr().out
        assert "Result bundle published and checked: failed (failed), target 'locomotion'" in out
        assert "  nothing certified (the bundle has no summary.json)" in out

    @pytest.mark.parametrize("model_id", RECOVERY_MODELS, ids=str)
    def test_a_failed_target_is_published_as_failed_and_the_run_stays_partial(self, tmp_path, capsys, model_id):
        """Recover on the T. rex and both Compsognathus models: stance passed, recovery judged and failed (the
        helpers' none/v1 recovery refuses to pass). Both re-enter from their verdicts and the bundle is partial."""
        run_dir = _run_dir(tmp_path, model_id)
        configs = _judged_run(run_dir, model_id, (1, "recovery"))
        _publish(_namespace(run_dir, model_id, "recover", configs))
        report = validate_result_bundle(run_dir, require_complete=False, require_publishable=True)
        assert report["status"] == "canonical-partial" and read_bundle_status(run_dir) == "partial"
        assert report["deliverables"]["1"]["certified"] is True
        assert report["deliverables"]["recovery"]["certified"] is False
        out = capsys.readouterr().out
        assert "partial (canonical-partial), target 'recovery'" in out and "  recovery: NOT certified" in out

    def test_a_partial_run_a_new_runtime_re_entered_verifies_again_once_published(self, tmp_path, monkeypatch):
        """docs/KNOWN_ISSUES.md: in a new runtime the storage cell appends its session to a partial run's
        provenance.json, which the manifest declares, and the resolve cell writes trunk_run.json, so the verify cell
        refuses the bundle. The publish cell, under the run's own BEHAVIOR, rebuilds the bundle over both files, and
        the bundle verifies again."""
        from environments.shared.result_bundle import initialize_result_bundle, record_trunk_run
        from environments.shared.result_bundle import provenance as provenance_module

        run_dir = _run_dir(tmp_path, VELOCIRAPTOR_MONGOLIENSIS)
        configs = _judged_run(run_dir, VELOCIRAPTOR_MONGOLIENSIS, (1, 2), failed=(2,))
        namespace = _namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs)
        _publish(namespace)
        validate_result_bundle(run_dir, require_complete=False, require_publishable=True)
        # The storage cell of a new runtime (another process: another session token) re-enters the run.
        monkeypatch.setattr(provenance_module, "_PROCESS_SESSION_TOKEN", "f" * 32)
        captured = {key: value for key, value in namespace["bundle_calls"][0].items() if key in _IDENTITY_KEYS}
        initialize_result_bundle(
            run_dir, **captured, **FIXTURE_PROTOCOL, plant_identity=_plant(VELOCIRAPTOR_MONGOLIENSIS).to_dict()
        )
        assert len(json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))["sessions"]) == 2
        with pytest.raises(ResultBundleError, match="provenance.json"):
            validate_result_bundle(run_dir, require_complete=False, require_publishable=True)
        _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs))
        report = validate_result_bundle(run_dir, require_complete=False, require_publishable=True)
        assert report["status"] == "canonical-partial"
        # The resolve cell of a session that resolved a trunk records it (the run holds no ancestors/ record).
        trunk = run_dir.parent / "20261101_000000"
        trunk.mkdir()
        record_trunk_run(run_dir, trunk_dir=trunk, log_dir=run_dir.parent)
        with pytest.raises(ResultBundleError, match="trunk_run.json"):
            validate_result_bundle(run_dir, require_complete=False, require_publishable=True)
        _publish(_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, "walk", configs))
        report = validate_result_bundle(run_dir, require_complete=False, require_publishable=True)
        assert report["status"] == "canonical-partial"
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        assert "trunk_run.json" in json.dumps(manifest)

    def test_a_partial_run_counts_a_replicate_that_certified_later(self, tmp_path, capsys):
        """Decision D-B10: the T. rex stance declares two certification seeds. Walk run A judged stance (seed 42) and
        failed locomotion, so its bundle is partial with stance at one run of two seeds. Run B then certifies the same
        stance recipe on seed 43; publishing A again rebuilds its partial bundle and counts B."""
        run_a = _run_dir(tmp_path, TYRANNOSAURUS_REX)
        configs = _judged_run(run_a, TYRANNOSAURUS_REX, (1, 2), failed=(2,))
        _declare_certification_seeds(run_a, TYRANNOSAURUS_REX, configs)
        _publish(_namespace(run_a, TYRANNOSAURUS_REX, "walk", configs))
        stance = validate_result_bundle(run_a, require_complete=False)["deliverables"]["1"]
        assert (stance["replication"]["count"], stance["provisional"]) == (1, True)
        assert "  stance: certified, 1 run of 2 seeds; provisional" in capsys.readouterr().out

        run_b = run_a.parent / "20261123_090000"
        run_b.mkdir()
        configs_b = _judged_run(run_b, TYRANNOSAURUS_REX, (1,), seed=43)
        _declare_certification_seeds(run_b, TYRANNOSAURUS_REX, configs_b)
        _publish(_namespace(run_a, TYRANNOSAURUS_REX, "walk", configs))
        report = validate_result_bundle(run_a, require_complete=False)
        stance = report["deliverables"]["1"]
        assert report["status"] == "canonical-partial"
        assert stance["replication"]["runs"] == [
            {"run_id": run_a.name, "training_seed": 42},
            {"run_id": run_b.name, "training_seed": 43},
        ]
        assert stance["provisional"] is False
        assert "  stance: certified, 2 runs of 2 seeds" in capsys.readouterr().out

    def test_a_walk_run_the_chain_loop_left_partial_counts_a_stance_replicate_that_certified_later(
        self, tmp_path, capsys
    ):
        """Decision D-B10 on the usual partial run: T. rex walk run A certified stance (seed 42) and its locomotion
        failed its command panel, and the chain loop's own write left the bundle partial. Run B certifies the same
        stance recipe on seed 43; publishing A counts it, with BEHAVIOR = "walk" as A's session had it."""
        run_a = _run_dir(tmp_path, TYRANNOSAURUS_REX)
        configs = _judged_run(run_a, TYRANNOSAURUS_REX, (1, 2), failed=(2,))
        _declare_certification_seeds(run_a, TYRANNOSAURUS_REX, configs)
        _chain_loop_write(_namespace(run_a, TYRANNOSAURUS_REX, "walk", configs), "stance", "locomotion")
        stance = validate_result_bundle(run_a, require_complete=False)["deliverables"]["1"]
        assert (stance["replication"]["count"], stance["provisional"]) == (1, True)

        run_b = run_a.parent / "20261123_090000"
        run_b.mkdir()
        configs_b = _judged_run(run_b, TYRANNOSAURUS_REX, (1,), seed=43)
        _declare_certification_seeds(run_b, TYRANNOSAURUS_REX, configs_b)
        capsys.readouterr()
        _publish(_namespace(run_a, TYRANNOSAURUS_REX, "walk", configs))
        report = validate_result_bundle(run_a, require_complete=False, require_publishable=True)
        assert report["status"] == "canonical-partial"
        assert report["deliverables"]["1"]["replication"]["count"] == 2
        assert report["deliverables"]["1"]["provisional"] is False
        out = capsys.readouterr().out
        assert "target 'locomotion'" in out and "  stance: certified, 2 runs of 2 seeds" in out
        assert "  locomotion: NOT certified" in out


def _declare_certification_seeds(
    run_dir: Path, model_id: ModelId, configs: "dict[int | str, dict[str, Any]]", seeds: int = 2
) -> None:
    """Declare the stance's ``certification_seeds`` (a publication key in no digest, decision D-B9) in the stage
    config the run recorded and in the session's STAGE_CONFIGS, as the T. rex stance TOML declares it."""
    configs[1]["curriculum_kwargs"]["certification_seeds"] = seeds
    path = run_dir / stage_dirname(model_id, 1) / "stage_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["curriculum_kwargs"]["certification_seeds"] = seeds
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
