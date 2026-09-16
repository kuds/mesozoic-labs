"""Portable, run-local evidence for a candidate's head-to-head comparison."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from environments.shared.certified_library import _comparison, _paired_comparison


def _scores(value: dict[str, Any]) -> dict[str, Any]:
    scores = _comparison({key: item for key, item in value.items() if key != "episodes"})
    assert scores is not None
    if "episodes" in value:
        scores["episodes"] = value["episodes"]
    return scores


def _write(path: Path, contents: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path: Path, value: dict[str, Any]) -> None:
    _write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _label(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def save_head_to_head(
    output_dir: Path,
    *,
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    publication: dict[str, Any],
    incumbent_version: str | None = None,
    incumbent_reused: bool = False,
) -> dict[str, str]:
    """Save both panels and the actual publication decision without external reads.

    The direct comparison uses the library's existing selection policy. The
    publication decision is also retained verbatim: replication requirements
    or other eligible versions can affect the final recommendation separately.
    """
    candidate = _scores(candidate)
    incumbent = _scores(incumbent) if incumbent is not None else None
    comparison: dict[str, Any] = (
        _paired_comparison({"comparison": candidate}, {"comparison": incumbent})
        if incumbent is not None
        else {"promote": False, "reason": "no_incumbent"}
    )
    selection_matches = publication["decision"]["previous_version"] == incumbent_version
    report = {
        "schema": "mesozoic.head-to-head/v1",
        "candidate": {"version": publication["version"], "scores": candidate},
        "incumbent": {
            "version": incumbent_version,
            "source": "saved_scores" if incumbent_reused else "reevaluated",
            "scores": incumbent,
        }
        if incumbent is not None
        else None,
        "head_to_head": comparison,
        "selection_snapshot_matches": selection_matches,
        "publication": publication,
    }
    lines = [
        "# Head-to-head comparison",
        "",
        f"Candidate version: `{publication['version']}`",
        f"Episodes per model: {len(candidate['protocol']['episode_seeds'])}",
        f"Publication status: **{_label(publication['status'])}**",
        f"Recommendation decision: `{publication['decision_reason']}`",
        f"Recommended version: `{publication['recommended_version'] or 'none'}`",
        "",
    ]
    if incumbent is None:
        lines += [
            "No incumbent was available. This report records the candidate's benchmark; no head-to-head win is claimed.",
            "",
        ]
    else:
        lines += [
            f"Incumbent version: `{incumbent_version}`",
            "Incumbent scores: " + ("reused verified panel." if incumbent_reused else "evaluated on this panel."),
            f"Head-to-head result: `{comparison['reason']}`",
            "",
        ]
    if not selection_matches:
        lines += [
            "The library recommendation changed during publication. The head-to-head result describes the recorded opponent; `decision.json` contains the actual selection decision.",
            "",
        ]
    if incumbent is not None and not publication["eligible"]:
        lines += ["The candidate still needs independent replication before it can become recommended.", ""]
    lines += [
        "## Metrics",
        "",
        "| Metric | Candidate mean | Incumbent mean | Paired improvement | Interval | Margin |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    previous = {metric["name"]: metric for metric in incumbent["metrics"]} if incumbent else {}
    paired = {metric["name"]: metric for metric in comparison.get("metrics", [])}
    for metric in candidate["metrics"]:
        name = metric["name"]
        mean = float(np.mean(metric["values"]))
        old = f"{float(np.mean(previous[name]['values'])):.6g}" if name in previous else "—"
        difference = f"{paired[name]['mean_difference']:.6g}" if name in paired else "—"
        interval = f"[{paired[name]['lower']:.6g}, {paired[name]['upper']:.6g}]" if name in paired else "—"
        lines.append(f"| {_label(name)} | {mean:.6g} | {old} | {difference} | {interval} | {metric['margin']:.6g} |")
    lines += [
        "",
        "Paired improvement is oriented so positive means better, including metrics where lower is better. The interval gives the paired bootstrap's 5th and 95th percentiles. Metrics appear in selection priority order.",
        "",
        "## Saved evidence",
        "",
        "- [Complete report](head_to_head.json): both scored panels, model hashes, protocol, per-episode metric samples and decision.",
        "- [Candidate scores](candidate.json)",
        "- [Publication decision](decision.json)",
    ]
    if incumbent is not None:
        lines.append("- [Incumbent scores](incumbent.json)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "report": output_dir / "head_to_head.json",
        "summary": output_dir / "summary.md",
        "candidate": output_dir / "candidate.json",
        "decision": output_dir / "decision.json",
    }
    _json(artifacts["candidate"], candidate)
    if incumbent is not None:
        artifacts["incumbent"] = output_dir / "incumbent.json"
        _json(artifacts["incumbent"], incumbent)
    elif (output_dir / "incumbent.json").exists():
        (output_dir / "incumbent.json").unlink()
    _json(artifacts["decision"], publication["decision"])
    _json(artifacts["report"], report)
    _write(artifacts["summary"], "\n".join(lines) + "\n")
    return {key: str(path) for key, path in artifacts.items()}
