"""Runs that never count: not as certified, replicates, reusable ancestors, widening parents or committed results.

The maintainer's stance-landing decision of 2026-11-06 (cited by date until the
next records change assigns its decision id; its record is §30 of
``docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md``): the T. rex
stance recipe that the r10 phase-critic screen runs passed the six-heading
certificate with (the stance stagger penalty, the per-foot centre-of-pressure
band, the lateral lean, the settled leg reference and mirror-KL PPO, task
``1e5964b6``) landed on ``main`` for its certification by two NEW seeds.  The
nine r10 stance screen runs trained that recipe on branch commits that were
never merged, and their task, gate, plant and recipe digests equal the landed
recipe's byte for byte, so without this module replicate discovery would count
them as replicates of a new seed and trunk selection could reuse one as the
certified stance.  They never count:

* ``stance6_r10LTSB_s42``, ``_s60`` and ``_s80``, trained at
  ``8737cd3d86431de1bdd0fcdb49a0d956a45e084d`` (``claude/screen-trex-r10-armltsb``);
* ``stance7_r10LTSBoff_s42``, ``_s60`` and ``_s80`` and
  ``stance7_r10LTSBpc_s42``, ``_s60`` and ``_s80``, trained at
  ``1dfe2327f773007fa7fd68460dc0c53740936ff9``
  (``claude/screen-trex-r10-phasecritic``).

A run is excluded when ANY of three records names one of them, so neither a
missing ``provenance.json`` (a stage directory copied into a run folder without
it, which ``ancestors.run_id_for`` names by its directory) nor a same-stage
resume on ``main`` (which rewrites ``stage_config.json`` with ``main``'s commit
while ``provenance.json`` keeps the original) lets one count:

(a) the run's ``provenance.json`` records a ``repository_commit`` in
    :data:`EXCLUDED_COMMITS`;
(b) a stage directory's ``stage_config.json`` records a top-level
    ``git_commit`` in :data:`EXCLUDED_COMMITS`;
(c) a stage directory's ``gate_verdict.json`` hashes a handoff checkpoint in
    :data:`EXCLUDED_HANDOFF_SHA256` (``checkpoint_sha256``, in its
    ``sha256:`` form).

The handoff digests are the screens' own records: the three phase-critic runs'
``gate_verdict.json``, the three stance6 runs' offline judge files committed under
``docs/investigations/stance_2026_10/stance6_results/complete/``, and the three
switch-off runs' ``provenance.json`` (their ``robust_best_model.zip``
``model_hash``, the file their verdicts judge).  Readers here never raise on a
missing or malformed file: an unreadable record decides nothing, and the
callers' own rules still apply to it.

The callers: :func:`environments.shared.replication.discover_replicates` and
``discover_replicates_for_run`` (an excluded sibling is skipped, an excluded
run counts no replicate), :mod:`environments.shared.ancestors` (reuse rule 9,
and the older-interface widening hint), ``scripts/widen_checkpoint.py`` (an
excluded parent is refused), the result-bundle audit and the species catalog
(an excluded run is never a promotable or committed result).  A later screen
needs its own entry here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

#: The decision that keeps these runs out, cited by date (its record is §30 of
#: ``docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md``).
EXCLUSION_DECISION = "the maintainer's stance-landing decision of 2026-11-06"

#: Full commit -> the runs that trained at it.
EXCLUDED_COMMITS: Mapping[str, str] = {
    # The arm-LTSB screen branch (the module docstring names both branches).
    "8737cd3d86431de1bdd0fcdb49a0d956a45e084d": "the r10 stance screen runs stance6_r10LTSB_s42, _s60 and _s80",
    # The phase-critic screen branch.
    "1dfe2327f773007fa7fd68460dc0c53740936ff9": (
        "the r10 stance screen runs stance7_r10LTSBoff_s42, _s60 and _s80 and stance7_r10LTSBpc_s42, _s60 and _s80"
    ),
}

#: The screens' handoff checkpoints (``robust_best_model.zip``), ``sha256:``-prefixed -> the run that wrote it.
EXCLUDED_HANDOFF_SHA256: Mapping[str, str] = {
    "sha256:77959e30f4f20e1a3bb184d210b4430a14b6df1a046cdbc2e5b279487fbfc299": "stance7_r10LTSBpc_s42",
    "sha256:92ac93e518ccfd4bbfdc15a25817477517f58463e544065a12b0e12d1e9591ad": "stance7_r10LTSBpc_s60",
    "sha256:b03e12823f9003ee8ad9f73a7ef485ab8f9be4ee23946c395f7f0f333abe44f5": "stance7_r10LTSBpc_s80",
    "sha256:034a105f136dcef45381cd2948cbb1390133578914739d94ee780b1f9a426163": "stance7_r10LTSBoff_s42",
    "sha256:47ee71dcda386f5f833ff500054976d288a95ecc3e0e2845d09a4ce655e9bd1c": "stance7_r10LTSBoff_s60",
    "sha256:22a61b91daf0265938980bd4be550720a4de802d198bcf07e9c81f9efc547beb": "stance7_r10LTSBoff_s80",
    "sha256:ba5e46a9fb20f4edeaba05e8070f0bd868146a02a6eb0dd2866105bc09846866": "stance6_r10LTSB_s42",
    "sha256:23d9b74a5d3e39123cd8d32fa6d1e20c3b27203d33271ed5068b8e80b699f3ce": "stance6_r10LTSB_s60",
    "sha256:55ca16f498ccd7f4bc017459098c1e271fc4dfa9e3a65cc8be2ec7a721dbaba6": "stance6_r10LTSB_s80",
}

_SUFFIX = (
    f"; branch screen runs never count as certified, as replicates or as reusable ancestors ({EXCLUSION_DECISION})"
)


def excluded_commit_reason(commit: Any) -> str | None:
    """Why a run recorded at *commit* never counts, or None (any non-string, or a commit not listed)."""
    if not isinstance(commit, str):
        return None
    runs = EXCLUDED_COMMITS.get(commit.strip().lower())
    return None if runs is None else f"it was trained at commit {commit.strip()}, {runs}{_SUFFIX}"


def _normalized_sha256(digest: Any) -> str | None:
    if not isinstance(digest, str) or not digest.strip():
        return None
    value = digest.strip().lower()
    return value if value.startswith("sha256:") else f"sha256:{value}"


def excluded_handoff_reason(digest: Any) -> str | None:
    """Why a checkpoint hashing to *digest* (with or without the ``sha256:`` prefix) never counts, or None."""
    value = _normalized_sha256(digest)
    run = EXCLUDED_HANDOFF_SHA256.get(value) if value is not None else None
    return None if run is None else f"its handoff checkpoint {value} is the r10 stance screen run {run}'s{_SUFFIX}"


def _read_object(path: Path) -> Mapping[str, Any] | None:
    from .file_io import read_json_object

    return read_json_object(path)  # None for a missing, unreadable or non-object file


def excluded_stage_reason(stage_dir: "str | Path") -> str | None:
    """Rules (b) and (c) on one stage directory: its ``stage_config.json`` commit or its verdict's handoff."""
    path = Path(stage_dir)
    stage_config = _read_object(path / "stage_config.json")
    if stage_config is not None:
        reason = excluded_commit_reason(stage_config.get("git_commit"))
        if reason is not None:
            return f"{path.name}/stage_config.json: {reason}"
    verdict = _read_object(path / "gate_verdict.json")
    if verdict is not None:
        reason = excluded_handoff_reason(verdict.get("checkpoint_sha256"))
        if reason is not None:
            return f"{path.name}/gate_verdict.json: {reason}"
    return None


def excluded_run_reason(run_dir: "str | Path", stage_dir: "str | Path | None" = None) -> str | None:
    """Why the run at *run_dir* never counts, or None.

    Rule (a) on its ``provenance.json``; rules (b) and (c) on *stage_dir*
    when given, else on every immediate subdirectory of *run_dir* (every
    stage-directory layout generation is one level down).
    """
    from .result_bundle import DEFAULT_PROVENANCE_NAME

    path = Path(run_dir)
    provenance = _read_object(path / DEFAULT_PROVENANCE_NAME)
    if provenance is not None:
        reason = excluded_commit_reason(provenance.get("repository_commit"))
        if reason is not None:
            return f"{DEFAULT_PROVENANCE_NAME}: {reason}"
    if stage_dir is not None:
        return excluded_stage_reason(stage_dir)
    try:
        children = sorted(child for child in path.iterdir() if child.is_dir())
    except OSError:
        return None
    for child in children:
        reason = excluded_stage_reason(child)
        if reason is not None:
            return reason
    return None
