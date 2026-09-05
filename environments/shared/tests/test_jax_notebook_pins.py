"""Cell-level pins for ``notebooks/jax_training.ipynb`` (gap review Phase J).

The JAX notebook is the MJX curriculum driver an operator actually runs, and
three of the 2026-08 pipeline gap review's findings lived in its cells rather
than in the library (``docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md``):

* **JX2** — a same-stage ``RESUME_FROM`` loaded params/obs_rms only, decayed
  the obs_rms count, and re-initialised the optimizer, so the "resumed" run
  silently was not a continuation: Adam moments zeroed, the linear LR
  schedule (its step count lives in opt_state) restarted from the top, and
  the normalization statistics re-fit within an update.
* **JX4** — the stage-2/3 auto-resume probed ``RUN_DIR / f"stage{N}"`` while
  every run since 2026-08-20 writes ``NN_id`` stage directories, so it
  raised FileNotFoundError for every current run and pushed operators onto
  the manual path that skips the publication-gate check.
* **JX7** (notebook half) — the evaluation video rolled the raw policy mean
  without the plant's command low-pass, so ``evaluation.mp4`` and its
  printed reward came from a different plant than training and gate eval.

These tests read the notebook JSON and assert on cell source text, the way
``test_recovery_gate_wiring.TestNotebookRecoveryFlowPin`` pins the SB3
notebook, so a refactor that quietly reintroduces one of the defects fails
here instead of in a Colab session weeks later.

Note on names: the notebook drives the functional ``jax_trainer.train()``
(not ``train_jax``/``JaxTrainer``), whose optimizer-state parameter is
``opt_state``; the restored state therefore travels resume cell ->
optimizer cell (``_resume_opt_state``) -> ``train(opt_state=...)``.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "jax_training.ipynb"

# Unique markers that identify the cells under test (not their positions —
# cell numbers move when a markdown cell is added).
RESUME_CELL_MARKER = "_auto_resume = CURRENT_STAGE > 1"
OPTIMIZER_CELL_MARKER = "optimizer = make_optimizer(ppo_config)"
TRAIN_CELL_MARKER = "result = train("
VIDEO_CELL_MARKER = "record_training_video("
IMPORT_CELL_MARKER = "from environments.shared.jax_checkpoint import"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _cell_index(cells: list[str], marker: str) -> int:
    hits = [index for index, src in enumerate(cells) if marker in src]
    assert len(hits) == 1, f"expected exactly one code cell containing {marker!r}, found {len(hits)}"
    return hits[0]


def _cell(marker: str) -> str:
    cells = _code_cells()
    return cells[_cell_index(cells, marker)]


def _branch_source(src: str, nodes: list[ast.stmt]) -> str:
    """The source text spanned by a list of statements (an ``if`` body or ``orelse``)."""
    lines = src.splitlines()
    first, last = nodes[0], nodes[-1]
    assert last.end_lineno is not None
    return "\n".join(lines[first.lineno - 1 : last.end_lineno])


def _top_level_if(src: str, test_source: str) -> ast.If:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.If) and ast.get_source_segment(src, node.test) == test_source:
            return node
    raise AssertionError(f"resume cell has no top-level `if {test_source}:`")


def _call_segment(src: str, func_name: str) -> str:
    """Source of the (single) call to ``func_name`` in ``src``."""
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name
    ]
    assert len(calls) == 1, f"expected exactly one {func_name}(...) call, found {len(calls)}"
    segment = ast.get_source_segment(src, calls[0])
    assert segment is not None
    return segment


class TestSameStageResumeIsAContinuation:
    """JX2: ``RESUME_FROM`` restores params + opt_state + obs_rms, undecayed, and forwards opt_state."""

    def test_resume_cell_restores_the_full_train_state(self):
        src = _cell(RESUME_CELL_MARKER)
        assert "restore_train_state(" in src, "same-stage RESUME_FROM must go through restore_train_state"
        assert "_resume_opt_state" in src.split("restore_train_state(")[0].rsplit("\n", 1)[-1], (
            "restore_train_state's opt_state must be captured as _resume_opt_state for the optimizer cell"
        )
        imports = _cell(IMPORT_CELL_MARKER)
        assert "restore_train_state" in imports, "the setup cell must import restore_train_state"

    def test_restored_opt_state_reaches_the_trainer(self):
        cells = _code_cells()
        resume_at = _cell_index(cells, RESUME_CELL_MARKER)
        optimizer_at = _cell_index(cells, OPTIMIZER_CELL_MARKER)
        train_at = _cell_index(cells, TRAIN_CELL_MARKER)
        assert resume_at < optimizer_at < train_at

        optimizer_src = cells[optimizer_at]
        assert "if _resume_opt_state is not None:" in optimizer_src
        assert "opt_state = jax.device_put(_resume_opt_state)" in optimizer_src, (
            "the optimizer cell must adopt the restored state instead of unconditionally re-initialising"
        )
        # A structure mismatch (e.g. learning_rate_end changed) must stop, not
        # silently fall through to a fresh optimizer.
        assert "jax.tree.structure(_resume_opt_state)" in optimizer_src
        assert "raise RuntimeError(" in optimizer_src

        train_call = _call_segment(cells[train_at], "train")
        assert "opt_state=opt_state" in train_call, "train() must receive the (possibly restored) opt_state"

    def test_same_stage_path_skips_the_obs_rms_decay_and_cross_stage_keeps_it(self):
        src = _cell(RESUME_CELL_MARKER)
        cross_stage = _top_level_if(src, "_ckpt_path is not None and _auto_resume")
        cross_src = _branch_source(src, cross_stage.body)
        assert len(cross_stage.orelse) == 1 and isinstance(cross_stage.orelse[0], ast.If), (
            "the cross-stage `if` must be followed by the same-stage `elif`"
        )
        same_stage = cross_stage.orelse[0]
        assert ast.get_source_segment(src, same_stage.test) == "_ckpt_path is not None"
        same_src = _branch_source(src, same_stage.body)

        # Cross-stage handoff keeps its documented fresh-optimizer + decay semantics.
        assert "load_checkpoint(" in cross_src
        assert "decay_running_stats(" in cross_src
        assert "obs_rms_decay_on_resume" in cross_src
        assert "restore_train_state(" not in cross_src
        assert "_resume_opt_state" not in cross_src

        # Same-stage continuation restores everything as saved and never decays.
        assert "restore_train_state(" in same_src
        assert "_resume_opt_state" in same_src
        assert "decay_running_stats(" not in same_src
        assert "obs_rms_decay_on_resume" not in same_src
        assert "no decay" in same_src, "the operator notice must say the obs_rms count was not decayed"

    def test_same_stage_resume_trains_the_remaining_budget_only(self):
        """A continuation must not run NUM_UPDATES *more*: the schedule, checkpoint
        numbering and stage record all assume NUM_UPDATES in total."""
        resume_src = _cell(RESUME_CELL_MARKER)
        assert "_same_stage_resume = False" in resume_src
        cross_stage = _top_level_if(resume_src, "_ckpt_path is not None and _auto_resume")
        assert "_same_stage_resume = True" not in _branch_source(resume_src, cross_stage.body)
        assert "_same_stage_resume = True" in _branch_source(resume_src, cross_stage.orelse)

        train_src = _cell(TRAIN_CELL_MARKER)
        assert (
            "_updates_this_session = max(NUM_UPDATES - _resume_update, 0) if _same_stage_resume else NUM_UPDATES"
            in train_src
        )
        config_call = _call_segment(train_src, "TrainConfig")
        assert "num_updates=_updates_this_session" in config_call
        assert "num_updates=NUM_UPDATES" not in config_call

    def test_continuation_notice_names_the_update_count(self):
        src = _cell(RESUME_CELL_MARKER)
        same_src = _branch_source(src, _top_level_if(src, "_ckpt_path is not None and _auto_resume").orelse)
        assert "Continuing Stage {CURRENT_STAGE}" in same_src
        assert "at update {_resume_update}" in same_src


class TestRewardPanelReceivesActionLags:
    def test_the_detail_wrapper_forwards_the_step_kwargs(self):
        """Without **step_kwargs the panel's jerk row is scored against zero lags."""
        cells = _code_cells()
        cell = next(src for src in cells if "def _reward_detail_fn(" in src)
        assert "def _reward_detail_fn(data, action, **step_kwargs):" in cell
        assert "compute_reward_detailed(data, action, reward_cfg, **step_kwargs)" in cell


class TestResumeGuardsAndScheduleSizing:
    def test_a_resume_with_nothing_left_stops_before_training(self):
        """A 0-update train() overwrote models/params.pkl with an empty history."""
        train_src = _cell(TRAIN_CELL_MARKER)
        guard = train_src.index("if _updates_this_session == 0:")
        assert "raise RuntimeError(" in train_src[guard : guard + 400]
        assert train_src.index("result = train(") > guard

    def test_the_lr_schedule_is_sized_to_the_real_minibatch_count(self):
        """PPOConfig's default n_minibatches=4 decayed the LR to its floor after ~8 of 500 updates."""
        optimizer_src = _cell(OPTIMIZER_CELL_MARKER)
        config_call = _call_segment(optimizer_src, "PPOConfig")
        assert "n_minibatches=max(1, (NUM_ENVS * ROLLOUT_LEN) // MINIBATCH_SIZE)" in config_call
        # The same formula train_jax uses, so both paths decay over the same steps.
        training_src = (REPO_ROOT / "environments" / "shared" / "jax_training.py").read_text(encoding="utf-8")
        assert "(num_envs * rollout_len) // int(minibatch_size)" in training_src


class TestAutoResumeFindsTheGatedPreviousStage:
    """JX4: the previous stage comes from the manifest order and its on-disk name from stage_dir_candidates."""

    def test_previous_stage_resolves_through_the_manifest(self):
        src = _cell(RESUME_CELL_MARKER)
        auto_src = _branch_source(src, _top_level_if(src, "_auto_resume").body)
        assert "load_stage_manifest(" in auto_src
        assert ".advancing_stages" in auto_src
        assert "stage_dir_candidates(" in auto_src
        # Comments may mention the old arithmetic; the CODE must not do it.
        subtractions = [
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and isinstance(node.left, ast.Name)
            and node.left.id == "CURRENT_STAGE"
        ]
        assert not subtractions, "the previous stage is the manifest's prior advancing entry, not CURRENT_STAGE - 1"

    def test_no_stage_n_literal_remains_in_any_code_cell(self):
        for index, src in enumerate(_code_cells()):
            assert 'f"stage{' not in src and "f'stage{" not in src, (
                f"code cell {index} rebuilds a stage{{N}} directory name; use stage_dir_candidates"
            )

    def test_publication_gate_check_stays_on_the_auto_resume_path(self):
        src = _cell(RESUME_CELL_MARKER)
        auto_src = _branch_source(src, _top_level_if(src, "_auto_resume").body)
        candidates_at = auto_src.index("stage_dir_candidates(")
        result_at = auto_src.index("stage_result.json")
        gate_at = auto_src.index('"publication_gate_passed") is not True')
        model_at = auto_src.index("best_model.pkl")
        assert candidates_at < result_at < gate_at < model_at, (
            "auto-resume must locate the stage dir, read its stage_result.json, and check the "
            "publication gate BEFORE loading best_model.pkl"
        )
        assert "raise RuntimeError(" in auto_src[gate_at:model_at]

    def test_the_manifest_mechanism_the_notebook_relies_on(self):
        """The trex manifest has recovery at position 2: prior-advancing != CURRENT_STAGE - 1."""
        from environments.shared.stage_manifest import load_stage_manifest, stage_dir_candidates

        manifest = load_stage_manifest("trex")
        advancing = list(manifest.advancing_stages)
        current_index = advancing.index(manifest.resolve(2))
        previous = advancing[current_index - 1]
        assert previous.id == "stance"
        assert manifest.by_position(2).id == "recovery", "position arithmetic would have picked recovery"
        candidates = stage_dir_candidates("trex", previous.id)
        assert candidates[0] == "01_stance", "the current NN_id layout must be tried first"
        assert "stage1" in candidates, "runs written before 2026-08-20 must still be found"


class TestNotebookNetworkFollowsNetArch:
    """JX9 (notebook half): the network is built by make_network(ctx), the one factory
    train_jax and every load/eval path share, so [jax.policy_kwargs] net_arch is honored."""

    def test_network_is_built_by_the_shared_factory(self):
        cells = _code_cells()
        assert any("network = make_network(ctx)" in src for src in cells)
        assert not any("make_actor_critic(" in src for src in cells), (
            "the notebook must not size the network itself; make_network(ctx) reads net_arch"
        )

    def test_the_factory_reads_the_stage_net_arch(self):
        from environments.shared.jax_curriculum import network_hidden_dims

        assert network_hidden_dims({"policy_kwargs": {"net_arch": [256, 128]}}) == (256, 128)
        assert network_hidden_dims({}) == (512, 256)


class TestVideoUsesThePlantActionFilter:
    """JX7 (notebook half): the evaluation video runs on the training plant's command low-pass."""

    def test_video_call_passes_the_env_action_filter(self):
        call = _call_segment(_cell(VIDEO_CELL_MARKER), "record_training_video")
        assert "action_filter_cutoff_hz=env.config.action_filter_cutoff_hz" in call

    def test_the_env_config_exposes_the_attribute_the_notebook_reads(self):
        mjx_env = pytest.importorskip("environments.shared.mjx_env")
        fields = {field.name for field in dataclasses.fields(mjx_env.MJXEnvConfig)}
        assert "action_filter_cutoff_hz" in fields

    def test_record_training_video_accepts_the_kwarg(self):
        from environments.shared.jax_viz import record_training_video

        params = inspect.signature(record_training_video).parameters
        assert "action_filter_cutoff_hz" in params, (
            "record_training_video must accept action_filter_cutoff_hz (JX7 library half) or the video cell raises"
        )
