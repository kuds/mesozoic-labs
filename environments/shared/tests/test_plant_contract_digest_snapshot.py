"""The committed digest snapshot and the CI step that checks it (D-D22).

``configs/digest_snapshot.generated.txt`` is the full output of
``environments/shared/harnesses/digest_snapshot.py`` for the committed tree,
and the plant-contract CI job runs the full harness against it with
``--check`` on every pull request. That run builds every plant and one
environment per behavior recipe (about a minute), so it is CI's step, not a
test here. These tests pin that step, keep the committed golden complete
(no ERROR line, no section, stage or recipe missing, the bytes ``--write``
writes), and exercise ``--check``'s comparison and failure message and
``--write`` on tiny snapshots.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from environments.shared.config import SPECIES_NAMES
from environments.shared.harnesses import digest_snapshot
from environments.shared.stage_manifest import load_stage_manifest

from .ci_workflow_helpers import CI_WORKFLOW, glob_matches, path_filters

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPOSITORY_ROOT / digest_snapshot.GOLDEN
HARNESS = Path(digest_snapshot.__file__).resolve()
CI_COMMAND = "python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check"

#: Fields per line, by section (the first field).
_FIELDS = {"plant.check_plant_manifest": 2, "plant": 4, "policy": 4, "stage": 5, "recovery": 4, "behavior": 5}
#: The lines of one stage, in order: _stage_lines emits them.
_STAGE_NAMES = [
    "config_file",
    "task_sha256",
    "gate_sha256",
    "hyperparameters_sha256.PPO",
    "hyperparameters_sha256.SAC",
    "stage_config_view_sha256.PPO",
    "stage_config_view_sha256.SAC",
]


# -- the CI step --------------------------------------------------------------


def _ci_code() -> str:
    """python-ci.yml without its comment lines."""
    lines = CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#")) + "\n"


def _plant_contract_steps() -> list[str]:
    """The plant-contract job's steps, each from its ``- `` to the next."""
    text = _ci_code()
    # A workflow- or job-level `defaults: run: shell:` could replace the step's shell.
    assert not re.search(r"^[\"']?defaults[\"']?\s*:", text, re.MULTILINE), "python-ci.yml sets no run defaults"
    start = text.index("\n  plant-contract:\n")
    end = re.compile(r"^  \S", re.MULTILINE).search(text, start + 2)
    job = text[start : end.start() if end else len(text)]
    job_keys = re.compile(r"^    [\"']?(if|continue-on-error|defaults)[\"']?\s*:", re.MULTILINE)
    assert not job_keys.search(job), "the plant-contract job must always run, with the default shell"
    starts = [match.start() for match in re.finditer(r"^      - ", job, re.MULTILINE)]
    return [job[a:b] for a, b in zip(starts, [*starts[1:], len(job)])]


def test_plant_contract_job_checks_the_full_snapshot_against_the_golden() -> None:
    steps = [step for step in _plant_contract_steps() if "digest_snapshot" in step]
    assert len(steps) == 1, f"the plant-contract job needs one digest-snapshot step, found {len(steps)}"
    (step,) = steps
    # The whole step, so no `if:`, `continue-on-error:`, `shell:` or `env:` can skip, tolerate or replace
    # the run: every event runs it, and a skipped or tolerated failure would not pin anything.
    assert step.rstrip("\n").splitlines() == [
        "      - name: Verify the digest snapshot golden",
        f"        run: {CI_COMMAND}",
    ], step


def test_ci_never_skips_the_behavior_section() -> None:
    # D-D22: a change that moves only behavior identities is caught on its own pull request.
    assert "--skip-behaviors" not in _ci_code()


def test_the_golden_the_harness_and_the_line_endings_trigger_the_workflow() -> None:
    filters = path_filters()
    assert len(filters) == 2, f"expected the push and pull_request path filters, found {len(filters)}"
    # .gitattributes sets the checkout's line endings, and the digests hash file bytes.
    for path in (GOLDEN, HARNESS, REPOSITORY_ROOT / ".gitattributes"):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for patterns in filters:
            assert any(glob_matches(pattern, relative) for pattern in patterns), relative


# -- the committed golden -----------------------------------------------------


def _golden_lines() -> list[str]:
    return GOLDEN.read_text(encoding="utf-8").splitlines()


def test_golden_is_what_write_writes() -> None:
    text = GOLDEN.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert text == digest_snapshot.render(lines), "the golden must end in exactly one newline, with no blank line"
    assert text.isascii()
    for number, line in enumerate(lines, 1):
        fields = line.split("\t")
        assert all(field and field == field.strip() for field in fields), f"line {number}: {line!r}"
        assert len(fields) == _FIELDS.get(fields[0]), f"line {number} has {len(fields)} fields: {line!r}"


def test_golden_holds_no_error_line() -> None:
    errors = [line for line in _golden_lines() if line.split("\t")[1] == "ERROR"]
    assert not errors, f"the golden records values that could not be computed: {errors[:3]}"


def test_golden_holds_every_section_in_order() -> None:
    lines = _golden_lines()
    assert lines[0] == "plant.check_plant_manifest\tOK"
    sections = [line.split("\t")[0] for line in lines[1:]]
    blocks = [section for i, section in enumerate(sections) if i == 0 or section != sections[i - 1]]
    plants = sorted(json.loads((REPOSITORY_ROOT / "configs" / "plant_manifest.generated.json").read_text())["plants"])
    # Each plant's identity, then its policy-interface lines; then the other sections.
    assert blocks == ["plant", "policy"] * len(plants) + ["stage", "recovery", "behavior"]
    for kind, field in (("plant", "verify_generated"), ("plant", "physics_sha256"), ("policy", "WHOLE")):
        species = [line.split("\t")[1] for line in lines if line.startswith(f"{kind}\t") and f"\t{field}\t" in line]
        assert species == plants, (kind, field, species)


def test_golden_holds_every_stage() -> None:
    stages = [line.split("\t") for line in _golden_lines() if line.startswith("stage\t")]
    expected = [
        (species, entry.id, name)
        for species in SPECIES_NAMES
        for entry in load_stage_manifest(species).stages
        for name in _STAGE_NAMES
    ]
    assert [tuple(fields[1:4]) for fields in stages] == expected


def test_golden_holds_every_recovery_calibration_and_behavior_recipe() -> None:
    lines = [line.split("\t") for line in _golden_lines()]
    configs = REPOSITORY_ROOT / "configs"
    recoveries = sorted(path.parent.name for path in configs.glob("*/recovery_calibration.json"))
    assert recoveries, "no recovery calibration is committed"
    assert [fields[1:3] for fields in lines if fields[0] == "recovery"] == [
        [species, name] for species in recoveries for name in ("load_recovery_calibration", "file_sha256")
    ]
    recipes = sorted((path.parent.parent.name, path.stem) for path in configs.glob("*/behaviors/*.toml"))
    assert recipes, "no behavior recipe is committed"
    behavior = [fields for fields in lines if fields[0] == "behavior"]
    for name in ("recipe_sha256", "behavior_identity_sha256"):
        assert sorted((fields[1], fields[2]) for fields in behavior if fields[3] == name) == recipes, name
    with_sources = {(fields[1], fields[2]) for fields in behavior if fields[3].startswith("source:")}
    assert with_sources == set(recipes), "every recipe's identity records its source digests"


# -- --check and --write on tiny snapshots ------------------------------------

_LINES = [
    "plant.check_plant_manifest\tOK",
    "plant\ttrex\tphysics_sha256\tsha256:aaa",
    "stage\ttrex\tstance\tgate_sha256\tsha256:bbb",
    "behavior\ttrex\twalk\tbehavior_identity_sha256\tsha256:ccc",
]


def test_check_passes_only_an_exact_reproduction() -> None:
    golden = digest_snapshot.render(_LINES)
    assert digest_snapshot.golden_mismatch(golden, _LINES, 0, digest_snapshot.GOLDEN) is None
    report = digest_snapshot.golden_mismatch(golden.rstrip("\n"), _LINES, 0, digest_snapshot.GOLDEN)
    assert report is not None and "does not end in exactly one newline" in report
    report = digest_snapshot.golden_mismatch(golden, _LINES[::-1], 0, digest_snapshot.GOLDEN)
    assert report is not None and "No value moved, but the lines differ in order or number:" in report


def test_check_failure_names_every_moved_line_and_the_write_command() -> None:
    golden = digest_snapshot.render(_LINES)
    run = [
        _LINES[0],
        _LINES[1].replace("sha256:aaa", "sha256:ddd"),  # changed
        _LINES[3],  # the stage line is gone
        "behavior\ttrex\twalk\tsource:environments/trex/envs/trex_env.py\teee",  # added
    ]
    report = digest_snapshot.golden_mismatch(golden, run, 0, digest_snapshot.GOLDEN)
    assert report is not None
    lines = report.splitlines()
    assert "3 digest line(s) moved (1 changed, 1 added, 1 removed):" in lines
    assert "  changed  plant trex physics_sha256" in lines
    assert "  removed  stage trex stance gate_sha256" in lines
    assert "  added    behavior trex walk source:environments/trex/envs/trex_env.py" in lines
    assert not any("behavior_identity_sha256" in line for line in lines), "an unmoved line is not reported"
    assert f"--- {digest_snapshot.GOLDEN} (committed)" in lines and "+++ this checkout" in lines
    assert "-plant\ttrex\tphysics_sha256\tsha256:aaa" in lines
    assert "+plant\ttrex\tphysics_sha256\tsha256:ddd" in lines
    assert f"    {digest_snapshot.WRITE_COMMAND}" in lines
    assert "D-D22" in report


def test_check_fails_a_run_with_an_error_line_even_when_the_golden_holds_it() -> None:
    run = [*_LINES, "behavior.trex.walk\tERROR\tValueError\tgenerated plant manifest is stale"]
    report = digest_snapshot.golden_mismatch(digest_snapshot.render(run), run, 1, digest_snapshot.GOLDEN)
    assert report is not None
    assert f"  {run[-1]}" in report.splitlines()
    assert digest_snapshot.WRITE_COMMAND not in report, "--write refuses a run with an ERROR line"


def test_check_golden_and_write_golden(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "snapshot.txt"
    assert digest_snapshot.write_golden(path, "snapshot.txt", _LINES, 0) == 0
    assert path.read_bytes() == ("\n".join(_LINES) + "\n").encode()
    assert "Wrote snapshot.txt (4 lines)" in capsys.readouterr().out
    assert digest_snapshot.check_golden(path, "snapshot.txt", _LINES, 0) == 0
    assert "Digest snapshot is current: snapshot.txt (4 lines)" in capsys.readouterr().out
    assert digest_snapshot.check_golden(path, "snapshot.txt", _LINES[:-1], 0) == 1
    assert "  removed  behavior trex walk behavior_identity_sha256" in capsys.readouterr().err
    # A run with an ERROR line writes nothing.
    error = "plant.trex.identity\tERROR\tRuntimeError\tno MuJoCo"
    assert digest_snapshot.write_golden(path, "snapshot.txt", [*_LINES, error], 1) == 1
    assert path.read_text(encoding="utf-8") == digest_snapshot.render(_LINES)
    assert f"  {error}" in capsys.readouterr().err


def test_check_reads_a_crlf_checkout_of_the_golden(tmp_path: Path) -> None:
    # A Windows checkout may convert the golden's line endings; its text still compares equal.
    path = tmp_path / "snapshot.txt"
    path.write_bytes(digest_snapshot.render(_LINES).replace("\n", "\r\n").encode())
    assert digest_snapshot.check_golden(path, "snapshot.txt", _LINES, 0) == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--check", "--skip-behaviors"], "drop --skip-behaviors"),
        (["--write", "--skip-behaviors"], "drop --skip-behaviors"),
        (["--check", "no/such/snapshot.txt"], digest_snapshot.WRITE_COMMAND),
    ],
    ids=["check-skip-behaviors", "write-skip-behaviors", "check-no-golden"],
)
def test_refusals_come_before_the_run(extra: list[str], message: str) -> None:
    result = subprocess.run(
        [sys.executable, str(HARNESS), "--repo", str(REPOSITORY_ROOT), *extra],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2, result.stderr
    assert message in result.stderr and not result.stdout


@pytest.fixture
def stubbed_sections(monkeypatch: pytest.MonkeyPatch):
    """main() with its four sections replaced by one that emits _LINES, and its process-wide changes undone."""

    def plant_section(out: digest_snapshot._Snapshot) -> dict[str, object]:
        for line in _LINES:
            out.emit(*line.split("\t"))
        return {}

    monkeypatch.setattr(digest_snapshot, "plant_section", plant_section)
    for name in ("stage_section", "recovery_section", "behavior_section"):
        monkeypatch.setattr(digest_snapshot, name, lambda *args: None)
    # main() blocks the optional backends, changes into --repo and puts it on sys.path.
    monkeypatch.setattr(digest_snapshot, "BLOCKED_BACKENDS", ())
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(REPOSITORY_ROOT)
    yield
    logging.disable(logging.NOTSET)


def test_main_checks_without_writing_and_writes_what_check_accepts(
    stubbed_sections: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "snapshot.txt"
    stale = digest_snapshot.render(_LINES[:-1])
    path.write_text(stale, encoding="utf-8")
    repo = ["--repo", str(REPOSITORY_ROOT)]

    assert digest_snapshot.main([*repo, "--check", str(path)]) == 1
    assert path.read_text(encoding="utf-8") == stale, "--check never writes the golden"
    assert "  added    behavior trex walk behavior_identity_sha256" in capsys.readouterr().err
    assert digest_snapshot.main([*repo, "--write", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == digest_snapshot.render(_LINES)
    assert digest_snapshot.main([*repo, "--check", str(path)]) == 0
    assert f"Digest snapshot is current: {path} (4 lines)" in capsys.readouterr().out
