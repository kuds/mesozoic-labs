"""Saved reports preserve paired data and distinguish measurement from selection."""

import copy
import json
from pathlib import Path

import pytest

from environments.shared.certified_comparison import save_head_to_head


def _scores(values, keys=("0", "1")):
    return {
        "protocol": {"version": "test/v1", "episode_seeds": [0, 1]},
        "model_sha256": "sha256:" + "a" * 64,
        "normalization_sha256": "sha256:" + "b" * 64,
        "metrics": [{"name": "time", "values": values, "direction": "lower", "margin": 0.1, "sample_keys": list(keys)}],
        "episodes": [{"seed": int(key), "time": value} for key, value in zip(keys, values, strict=True)],
    }


def _publication(previous="old", *, eligible=True):
    return {
        "version": "new",
        "recommended_version": "new" if eligible else "old",
        "eligible": eligible,
        "status": "eligible" if eligible else "provisional",
        "decision_reason": "paired_improvement" if eligible else "provisional",
        "decision": {"previous_version": previous, "decisions": [], "version": "new" if eligible else "old"},
    }


def test_report_preserves_scores_and_actual_decision_and_uses_paired_keys(tmp_path):
    candidate, incumbent, publication = _scores([2, 4]), _scores([7, 5], ("1", "0")), _publication()
    before = copy.deepcopy((candidate, incumbent, publication))
    files = save_head_to_head(
        tmp_path,
        candidate=candidate,
        incumbent=incumbent,
        publication=publication,
        incumbent_version="old",
        incumbent_reused=True,
    )
    report = json.loads(Path(files["report"]).read_text())
    assert report["head_to_head"]["promote"]
    assert report["head_to_head"]["metrics"][0]["mean_difference"] == 3
    assert report["candidate"]["scores"]["episodes"] == candidate["episodes"]
    assert report["incumbent"]["scores"]["episodes"] == incumbent["episodes"]
    assert report["incumbent"]["source"] == "saved_scores"
    assert report["selection_snapshot_matches"]
    assert json.loads(Path(files["decision"]).read_text()) == publication["decision"]
    assert (candidate, incumbent, publication) == before
    summary = Path(files["summary"]).read_text()
    assert "| time | 3 | 6 | 3 | [3, 3] | 0.1 |" in summary
    assert "reused verified panel" in summary


def test_no_incumbent_records_benchmark_without_claiming_a_win(tmp_path):
    files = save_head_to_head(tmp_path, candidate=_scores([2, 4]), incumbent=None, publication=_publication(None))
    report = json.loads(Path(files["report"]).read_text())
    assert report["incumbent"] is None
    assert report["head_to_head"] == {"promote": False, "reason": "no_incumbent"}
    assert "incumbent" not in files
    assert "no head-to-head win is claimed" in Path(files["summary"]).read_text()


def test_provisional_and_concurrent_selection_are_explicit(tmp_path):
    publication = _publication("different", eligible=False)
    files = save_head_to_head(
        tmp_path, candidate=_scores([2, 4]), incumbent=_scores([5, 7]), publication=publication, incumbent_version="old"
    )
    report = json.loads(Path(files["report"]).read_text())
    assert report["head_to_head"]["promote"]
    assert not report["publication"]["eligible"]
    assert not report["selection_snapshot_matches"]
    summary = Path(files["summary"]).read_text()
    assert "changed during publication" in summary
    assert "needs independent replication" in summary


def test_different_protocol_does_not_display_a_paired_win(tmp_path):
    incumbent = _scores([5, 7])
    incumbent["protocol"]["version"] = "old/v1"
    files = save_head_to_head(
        tmp_path, candidate=_scores([2, 4]), incumbent=incumbent, publication=_publication(), incumbent_version="old"
    )
    report = json.loads(Path(files["report"]).read_text())
    assert not report["head_to_head"]["promote"]
    assert report["head_to_head"]["reason"] == "different_comparison_protocol"


def test_invalid_measurements_fail_before_writing_report(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        save_head_to_head(
            tmp_path, candidate=_scores([float("nan"), 4]), incumbent=None, publication=_publication(None)
        )
    assert not list(tmp_path.iterdir())
