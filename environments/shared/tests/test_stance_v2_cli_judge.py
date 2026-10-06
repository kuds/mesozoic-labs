"""The CLI curriculum judges a ``stance_quality/v2`` node after training (decision D-D23; gate map item 16).

The in-training manager refuses the kind -- its certificate is the
floor-truth panel on the handoff pair, which an EvalCallback cannot roll --
so without a post-training judge ``train_curriculum`` would write the fixed
"stage budget exhausted" FAIL for every v2 node and never advance one.
``_post_training_stance_v2_verdict`` rolls the stance report on the handoff
pair through the same ``stage_artifacts._write_stance_gate_report`` the
post-stage pipeline uses and judges it through ``evaluate_stage_gate``; the
verdict records a distinct ``judged_by``.  SB3-free: the report builder is
stubbed with the synthetic v2 report, the judge is real.
"""

from __future__ import annotations

import ast
import inspect

import environments.shared.reporting.stage_artifacts as stage_artifacts
from environments.shared import train_base

from .stance_v2_helpers import V2_CURRICULUM, v2_report


def _config() -> dict:
    return {"name": "Balance", "description": "Stand", "curriculum_kwargs": dict(V2_CURRICULUM), "env_kwargs": {}}


def test_a_passing_report_on_the_handoff_pair_passes_the_node(tmp_path, monkeypatch):
    calls = []

    def fake_write(*, species, stage, stage_config, stage_dir, model_dir):
        calls.append((species, stage, model_dir))
        return v2_report(stage_dir)

    monkeypatch.setattr(stage_artifacts, "_write_stance_gate_report", fake_write)
    passed, failures, stage_result = train_base._post_training_stance_v2_verdict(
        "trex", 1, _config(), stage_dir=tmp_path, model_dir=tmp_path / "models"
    )
    assert calls == [("trex", 1, tmp_path / "models")]
    assert (passed, failures) == (True, [])
    assert stage_result["gate_kind"] == "stance_quality/v2" and stage_result["gate_passed"] is True
    assert stage_result["stance_clean_count"] == 40 and stage_result["stance_n_episodes"] == 40


def test_no_report_is_a_named_fail_never_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_artifacts, "_write_stance_gate_report", lambda **kwargs: None)
    passed, failures, stage_result = train_base._post_training_stance_v2_verdict(
        "trex", 1, _config(), stage_dir=tmp_path, model_dir=tmp_path / "models"
    )
    assert passed is False and any("no stance gate report was produced" in failure for failure in failures)
    assert "stance_clean_count" not in stage_result


def test_a_failing_panel_fails_the_node_with_its_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage_artifacts,
        "_write_stance_gate_report",
        lambda *, stage_dir, **kwargs: v2_report(stage_dir, n_clean=36, defect={"touchdown_rate": 9.9}),
    )
    passed, failures, stage_result = train_base._post_training_stance_v2_verdict(
        "trex", 1, _config(), stage_dir=tmp_path, model_dir=tmp_path / "models"
    )
    assert passed is False and any("36/40 episodes clean" in failure for failure in failures)
    assert stage_result["stance_clean_count"] == 36  # an admitted report's numbers travel with its FAIL


def test_train_curriculum_uses_the_post_training_judge_for_a_v2_node():
    """The wiring, by AST: under a ``gate_kind == STANCE_GATE_V2_KIND`` test the node's ``passed`` comes from
    ``_post_training_stance_v2_verdict`` and the verdict's ``judged_by`` from a variable that branch sets to
    the distinct ``STANCE_V2_POST_TRAINING_JUDGED_BY`` -- never the manager's fixed budget-exhausted FAIL."""
    tree = ast.parse(inspect.getsource(train_base.train_curriculum))
    v2_branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "STANCE_GATE_V2_KIND" in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        and "interrupted" in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
    ]
    assert len(v2_branches) == 1
    branch = v2_branches[0]
    calls = {c.func.id for c in ast.walk(branch) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "_post_training_stance_v2_verdict" in calls
    assigned = {
        target.id: node.value
        for node in ast.walk(branch)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert isinstance(assigned["verdict_judged_by"], ast.Name)
    assert assigned["verdict_judged_by"].id == "STANCE_V2_POST_TRAINING_JUDGED_BY"
    verdict_calls = [
        c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "write_gate_verdict"
    ]
    keywords = {kw.arg: kw.value for kw in verdict_calls[0].keywords}
    assert isinstance(keywords["judged_by"], ast.Name) and keywords["judged_by"].id == "verdict_judged_by"
    assert isinstance(keywords["failures"], ast.Name) and keywords["failures"].id == "verdict_failures"
    assert train_base.STANCE_V2_POST_TRAINING_JUDGED_BY != train_base.CURRICULUM_MANAGER_JUDGED_BY
