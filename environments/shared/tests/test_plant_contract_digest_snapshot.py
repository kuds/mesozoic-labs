"""The committed digest snapshot and the CI step that checks it (D-D22).

``configs/digest_snapshot.generated.txt`` is the full output of
``environments/shared/harnesses/digest_snapshot.py`` for the committed tree,
and the plant-contract CI job runs the full harness against it with
``--check`` on every pull request. That run builds every plant and one
environment per behavior recipe (about a minute), so it is CI's step, not a
test here. These tests pin that step, keep the committed golden complete
(no ERROR line, no section, stage or recipe missing, the bytes ``--write``
writes, every stage's reward capture and the ends its probes reach), and
exercise ``--check``'s comparison and failure message and ``--write`` on tiny
snapshots. The reward captures (CU-11) also get unit tests of their two
encodings, one model's capture in each shared leg of the test matrix, and a
check that no state probe sits on its tilt, height or nosedive threshold.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from environments.shared.harnesses import digest_snapshot
from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import load_stage_manifest

from .ci_workflow_helpers import ci_code, glob_matches, job_block, job_steps, path_filters

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPOSITORY_ROOT / digest_snapshot.GOLDEN
HARNESS = Path(digest_snapshot.__file__).resolve()
CI_COMMAND = "python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check"

#: Fields per line, by section (the first field); a model is two columns, species and embodiment (decision D-D33).
_FIELDS = {
    "plant.check_plant_manifest": 2,
    "plant": 5,
    "policy": 5,
    "stage": 6,
    "recovery": 5,
    "behavior": 6,
    "reward": 6,
}
#: The order of every section: the manifest models, sorted (species, then embodiment).
_MODELS = sorted(model_ids())
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
#: The lines of one stage's reward capture (CU-11), in order.
_REWARD_NAMES = ["summary", "poses", "shape_sha256", "rounded_values_sha256"]
#: The success reason each model's behavior stage ends the success probe with.
_SUCCESS = {
    ModelId("tyrannosaurus_rex", "robot"): "target_reach",
    ModelId("brachiosaurus_altithorax", "anatomical"): "food_reached",
    ModelId("compsognathus_longipes", "anatomical"): "target_reached",
    ModelId("compsognathus_longipes", "robot"): "target_reached",
    ModelId("dibothrosuchus_elaphros", "anatomical"): "snap_success",
    ModelId("tyrannosaurus_rex", "anatomical"): "bite_success",
    ModelId("velociraptor_mongoliensis", "anatomical"): "strike_success",
}
#: The reasons each model's state probes reach (the union over its stages' poses line).
_POSE_REASONS = {
    ModelId("tyrannosaurus_rex", "robot"): {"fallen", "too_high", "excessive_tilt", "body_contact"},
    ModelId("brachiosaurus_altithorax", "anatomical"): {
        "fallen",
        "too_high",
        "excessive_tilt",
        "tail_contact",
        "head_contact",
    },
    ModelId("compsognathus_longipes", "anatomical"): {
        "fallen",
        "too_high",
        "excessive_tilt",
        "body_contact",
        "nonfinite_state",
    },
    ModelId("compsognathus_longipes", "robot"): {
        "fallen",
        "too_high",
        "excessive_tilt",
        "body_contact",
        "nonfinite_state",
    },
    ModelId("dibothrosuchus_elaphros", "anatomical"): {
        "fallen",
        "too_high",
        "excessive_tilt",
        "nosedive",
        "tail_contact",
        "head_contact",
    },
    ModelId("tyrannosaurus_rex", "anatomical"): {"fallen", "too_high", "excessive_tilt", "nosedive", "tail_contact"},
    ModelId("velociraptor_mongoliensis", "anatomical"): {
        "fallen",
        "too_high",
        "excessive_tilt",
        "tail_contact",
        "body_contact",
    },
}


@pytest.mark.parametrize("name", ["_SUCCESS", "_POSE_REASONS"])
def test_the_per_model_tables_cover_every_model(name: str) -> None:
    """A new model fails here, naming the table, instead of going unpinned (review M4)."""
    assert set(globals()[name]) == set(model_ids()), f"{name} must pin every manifest model"


# -- the CI step --------------------------------------------------------------


def _plant_contract_steps() -> list[str]:
    """The plant-contract job's steps, each from its ``- `` to the next."""
    # A workflow- or job-level `defaults: run: shell:` could replace the step's shell.
    assert not re.search(r"^[\"']?defaults[\"']?\s*:", ci_code(), re.MULTILINE), "python-ci.yml sets no run defaults"
    job = job_block("plant-contract")
    job_keys = re.compile(r"^    [\"']?(if|continue-on-error|defaults)[\"']?\s*:", re.MULTILINE)
    assert not job_keys.search(job), "the plant-contract job must always run, with the default shell"
    # GitHub skips a job whose needed job was skipped, so the lint job it needs must always run too.
    needs = re.findall(r"^    [\"']?needs[\"']?\s*:.*$", job, re.MULTILINE)
    assert needs == ["    needs: lint"], f"the plant-contract job must need only lint: {needs}"
    assert not re.search(r"^    [\"']?(if|continue-on-error|needs)[\"']?\s*:", job_block("lint"), re.MULTILINE), (
        "the lint job the plant-contract job needs must always run"
    )
    return job_steps("plant-contract")


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
    assert "--skip-behaviors" not in ci_code()


def test_the_golden_the_harness_and_the_line_endings_trigger_the_workflow() -> None:
    filters = path_filters()
    assert len(filters) == 2, f"expected the push and pull_request path filters, found {len(filters)}"
    # .gitattributes sets the checkout's line endings, and the digests hash file bytes.
    for path in (GOLDEN, HARNESS, REPOSITORY_ROOT / ".gitattributes"):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for patterns in filters:
            assert any(glob_matches(pattern, relative) for pattern in patterns), relative


@pytest.mark.parametrize("item", ['"!configs/x.txt"', "'!configs/x.txt'"], ids=["double-quoted", "single-quoted"])
def test_the_trigger_pins_refuse_a_negated_path_in_either_quote_style(item: str) -> None:
    with pytest.raises(AssertionError, match="exclude"):
        path_filters(f'on:\n  push:\n    paths:\n      - "configs/**"\n      - {item}\n')


# -- the committed golden -----------------------------------------------------


def _golden_lines() -> list[str]:
    return GOLDEN.read_text(encoding="utf-8").splitlines()


def test_golden_is_what_write_writes() -> None:
    # Bytes, not newline-translated text: only a CRLF checkout of the LF file is accepted.
    data = GOLDEN.read_bytes().replace(b"\r\n", b"\n")
    assert b"\r" not in data, "the golden has a lone CR line ending; regenerate it with --write"
    text = data.decode("utf-8")
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
    manifest = json.loads((REPOSITORY_ROOT / "configs" / "plant_manifest.generated.json").read_text())
    plants = sorted(
        (species, embodiment) for species, embodiments in manifest["plants"].items() for embodiment in embodiments
    )
    assert plants == [(m.species, m.embodiment) for m in _MODELS]
    # Each plant's identity, then its policy-interface lines; then the other sections.
    assert blocks == ["plant", "policy"] * len(plants) + ["stage", "recovery", "behavior", "reward"]
    for kind, field in (("plant", "verify_generated"), ("plant", "physics_sha256"), ("policy", "WHOLE")):
        models = [
            tuple(line.split("\t")[1:3]) for line in lines if line.startswith(f"{kind}\t") and f"\t{field}\t" in line
        ]
        assert models == plants, (kind, field, models)


def test_golden_holds_every_stage() -> None:
    stages = [line.split("\t") for line in _golden_lines() if line.startswith("stage\t")]
    expected = [
        (model_id.species, model_id.embodiment, entry.id, name)
        for model_id in _MODELS
        for entry in load_stage_manifest(model_id).stages
        for name in _STAGE_NAMES
    ]
    assert expected and [tuple(fields[1:5]) for fields in stages] == expected


def test_golden_holds_every_stage_reward_capture() -> None:
    rewards = [line.split("\t") for line in _golden_lines() if line.startswith("reward\t")]
    expected = [
        (model_id.species, model_id.embodiment, entry.id, name)
        for model_id in _MODELS
        for entry in load_stage_manifest(model_id).stages
        for name in _REWARD_NAMES
    ]
    assert expected and [tuple(fields[1:5]) for fields in rewards] == expected


def test_golden_reward_probes_reach_the_ends_they_probe() -> None:
    # A probe that stops reaching its end (a renamed element, a moved attribute) would pass --check once regenerated.
    for fields in (line.split("\t") for line in _golden_lines() if line.startswith("reward\t")):
        if fields[4] != "summary":
            continue
        assert fields[5].split(" ")[1] == "second_reset=0:-:0.0000", fields
        model_id, stage = ModelId(fields[1], fields[2]), fields[3]
        ends = dict(part.split("=") for part in fields[5].split(" "))
        steps = {part: end.split(":")[:2] for part, end in ends.items()}
        pushes = ["zero"] if stage == "recovery" else []
        assert list(steps) == [
            "roll",
            "second_reset",
            *pushes,
            "success",
            "too_high",
            "tilt",
            "high_tilt",
            "truncation",
        ], fields
        assert steps["roll"][1] not in ("-", "truncated"), f"{model_id} {stage}: the roll must end the episode"
        if pushes:  # zero action to the episode's end (a termination, or truncation at the horizon), past the pushes
            assert steps["zero"][1] != "-", f"{model_id} {stage}: the zero part must run to the episode's end"
        success = [_SUCCESS[model_id] if stage == "behavior" else "-"]
        assert steps["success"] == ["1", *success], (model_id, stage, steps["success"])
        assert steps["too_high"] == ["1", "too_high"] and steps["high_tilt"] == ["1", "too_high"], steps
        assert steps["tilt"] == ["1", "excessive_tilt"] and steps["truncation"] == ["3", "truncated"], steps


def test_golden_state_probes_reach_every_branch_they_cover() -> None:
    reached: dict = {}
    for fields in (line.split("\t") for line in _golden_lines() if line.startswith("reward\t")):
        if fields[4] == "poses":
            ends = dict(item.split("=") for item in fields[5].split(" "))
            assert list(ends) == [name for name, *_ in digest_snapshot.REWARD_POSES] + ["nonfinite"], fields
            assert ends["low_roll"] == "fallen" and ends["high"] == "too_high", fields  # the prefix's precedence
            reached.setdefault(ModelId(fields[1], fields[2]), set()).update(ends.values())
    assert {model_id: reasons - {"-"} for model_id, reasons in reached.items()} == _POSE_REASONS


def test_golden_holds_every_recovery_calibration_and_behavior_recipe() -> None:
    lines = [line.split("\t") for line in _golden_lines()]
    configs = REPOSITORY_ROOT / "configs"
    # Coverage equality with the committed files: a model is configs/<species>/<embodiment>/.
    recoveries = sorted(
        (path.parent.parent.name, path.parent.name) for path in configs.glob("*/*/recovery_calibration.json")
    )
    assert recoveries, "no recovery calibration is committed"
    assert [tuple(fields[1:4]) for fields in lines if fields[0] == "recovery"] == [
        (*model, name) for model in recoveries for name in ("load_recovery_calibration", "file_sha256")
    ]
    recipes = sorted(
        (path.parents[2].name, path.parents[1].name, path.stem) for path in configs.glob("*/*/behaviors/*.toml")
    )
    assert recipes, "no behavior recipe is committed"
    behavior = [fields for fields in lines if fields[0] == "behavior"]
    for name in ("recipe_sha256", "task_sha256"):
        assert sorted(tuple(fields[1:4]) for fields in behavior if fields[4] == name) == recipes, name
    assert len(behavior) == 2 * len(recipes), "a recipe's identity is its task fingerprint, not source digests"
    recipe_tasks = [fields[5] for fields in behavior if fields[4] == "task_sha256"]
    stage_tasks = {fields[5] for fields in lines if fields[0] == "stage" and fields[4] == "task_sha256"}
    assert len(set(recipe_tasks)) == len(recipes) and stage_tasks and not stage_tasks & set(recipe_tasks)


# -- the reward captures ------------------------------------------------------


def test_reward_rounding_and_the_exact_encoding() -> None:
    rounded, exact = digest_snapshot._rounded, digest_snapshot._exact
    assert json.dumps(rounded(-4e-7)) == "0.0" and json.dumps(rounded(np.float64(-0.0))) == "0.0"
    assert rounded(np.float32(0.25)) == rounded(0.25) == 0.25 and rounded(1.23456789) == 1.234568
    assert rounded(1234.56789) == 1234.568 and rounded(-98765.4321) == -98765.43
    assert rounded(True) is True and rounded(7) == 7 and rounded("fallen") == "fallen"
    assert exact(np.float32(0.1)) != exact(0.1) and exact(float(np.nextafter(0.1, 1.0))) != exact(0.1)
    assert exact(np.float32(0.25)) != exact(0.25), "the dtype alone separates values every dtype holds exactly"


def _capture(x: float, **extra: float) -> list[tuple[str, object, list[object]]]:
    """A one-part, one-step capture as _capture returns it, with info value *x* (and *extra* keys)."""
    info = {"x": x, **extra, "termination_reason": "fallen"}
    row = (1.5, info, True, False, np.zeros(2, np.float32), np.ones(3), np.zeros(3))
    return [("roll", {"qpos": [0.5], "rng_state": {"state": 3}, "obs": np.zeros(2)}, [row])]


def test_exact_streams_see_an_ulp_that_the_rounded_golden_does_not() -> None:
    streams, exact, rounded = digest_snapshot._streams, digest_snapshot._exact, digest_snapshot._rounded
    ulp = float(np.nextafter(0.3, 1.0))
    moved = streams(_capture(ulp), exact)
    assert [key for key, digest in streams(_capture(0.3), exact).items() if moved[key] != digest] == [
        ("roll", "info:x")
    ]
    assert streams(_capture(ulp), rounded) == streams(_capture(0.3), rounded)
    assert streams(_capture(0.3001), rounded) != streams(_capture(0.3), rounded)
    # A new key, even an inert 0.0 reward term, moves the shape digest's streams.
    shape = streams(_capture(0.3, y=0.0), lambda value: "float")
    assert ("roll", "info:y") in shape and shape[("roll", "flags")] != streams(_capture(0.3), lambda value: "float")[
        ("roll", "flags")
    ]
    assert digest_snapshot._summary(_capture(0.3)) == "roll=1:fallen:1.5000"


def test_one_model_reproduces_its_golden_reward_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI's digest step runs every stage on Python 3.12; this runs one model in each shared leg of the test matrix,
    # where Python 3.11's float sum() moves the anatomical Compsognathus reward by an ulp that the golden rounds away.
    model_id = ModelId("compsognathus_longipes", "anatomical")
    monkeypatch.setattr(digest_snapshot, "_models", lambda: [model_id])
    out = digest_snapshot._Snapshot(REPOSITORY_ROOT, keep=True)
    digest_snapshot.reward_section(out)
    prefix = f"reward\t{model_id.species}\t{model_id.embodiment}\t"
    assert out.kept and out.kept == [line for line in _golden_lines() if line.startswith(prefix)]


def test_state_probes_keep_clear_of_the_thresholds() -> None:
    # A pose within an ulp of a threshold would flip its reason between machines; this keeps each pose's tilt, root
    # height and, where the species terminates on one, nosedive signal at least 1e-3 from its threshold (measured:
    # 0.15 rad, 2.9 mm, 0.015).
    out = digest_snapshot._Snapshot(REPOSITORY_ROOT, keep=True)
    probes = 0
    for model_id, stage, env in digest_snapshot._reward_envs(out, False):
        low, high = env.healthy_z_range
        # The T-Rex and Dibothrosuchus take the nosedive threshold as a parameter; velociraptor_mongoliensis/base_env.py
        # hard-codes its 0.5.
        nosedive = getattr(
            env, "nosedive_termination_threshold", 0.5 if model_id.species == "velociraptor_mongoliensis" else None
        )
        for part, _, rows in digest_snapshot._capture(env, digest_snapshot._state_parts()[:-1]):
            info = rows[0][1]
            height = info.get("torso_height", info.get("pelvis_height"))
            assert abs(info["tilt_angle"] - env.max_tilt_angle) > 1e-3, (model_id, stage, part)
            # The 113 mm mini uses a 0.5 mm margin; larger plants keep 1 mm.
            # Both are far above floating-point noise, the purpose of this guard.
            height_margin = 5e-4 if model_id == ModelId("tyrannosaurus_rex", "robot") else 1e-3
            assert min(abs(height - low), abs(height - high)) > height_margin, (model_id, stage, part)
            if nosedive is not None:
                assert abs(info["forward_z"] - (env._natural_forward_z - nosedive)) > 1e-3, (model_id, stage, part)
            probes += 1
    stages = sum(len(load_stage_manifest(model_id).stages) for model_id in model_ids())
    assert not out.kept and stages and probes == len(digest_snapshot.REWARD_POSES) * stages


# -- --check and --write on tiny snapshots ------------------------------------

_LINES = [
    "plant.check_plant_manifest\tOK",
    "plant\ttyrannosaurus_rex\tanatomical\tphysics_sha256\tsha256:aaa",
    "stage\ttyrannosaurus_rex\tanatomical\tstance\tgate_sha256\tsha256:bbb",
    "behavior\ttyrannosaurus_rex\tanatomical\twalk\tbehavior_identity_sha256\tsha256:ccc",
]


def test_check_passes_only_an_exact_reproduction() -> None:
    golden = digest_snapshot.render(_LINES)
    assert digest_snapshot.golden_mismatch(golden, _LINES, 0, digest_snapshot.GOLDEN) is None
    report = digest_snapshot.golden_mismatch(golden.rstrip("\n"), _LINES, 0, digest_snapshot.GOLDEN)
    assert report is not None and "is not the text --write writes" in report
    report = digest_snapshot.golden_mismatch(golden, _LINES[::-1], 0, digest_snapshot.GOLDEN)
    assert report is not None and "No value moved, but the lines differ in order or number:" in report


def test_check_failure_names_every_moved_line_and_the_write_command() -> None:
    golden = digest_snapshot.render(_LINES)
    run = [
        _LINES[0],
        _LINES[1].replace("sha256:aaa", "sha256:ddd"),  # changed
        _LINES[3],  # the stage line is gone
        "behavior\ttyrannosaurus_rex\tanatomical\twalk\tsource:environments/tyrannosaurus_rex/base_env.py\teee",  # added
    ]
    report = digest_snapshot.golden_mismatch(golden, run, 0, digest_snapshot.GOLDEN)
    assert report is not None
    lines = report.splitlines()
    assert "3 digest line(s) moved (1 changed, 1 added, 1 removed):" in lines
    assert "  changed  plant tyrannosaurus_rex anatomical physics_sha256" in lines
    assert "  removed  stage tyrannosaurus_rex anatomical stance gate_sha256" in lines
    assert (
        "  added    behavior tyrannosaurus_rex anatomical walk source:environments/tyrannosaurus_rex/base_env.py"
        in lines
    )
    assert not any("behavior_identity_sha256" in line for line in lines), "an unmoved line is not reported"
    assert f"--- {digest_snapshot.GOLDEN} (committed)" in lines and "+++ this checkout" in lines
    assert "-plant\ttyrannosaurus_rex\tanatomical\tphysics_sha256\tsha256:aaa" in lines
    assert "+plant\ttyrannosaurus_rex\tanatomical\tphysics_sha256\tsha256:ddd" in lines
    assert f"    {digest_snapshot.WRITE_COMMAND}" in lines
    assert "D-D22" in report


def test_check_fails_a_run_with_an_error_line_even_when_the_golden_holds_it() -> None:
    run = [*_LINES, "behavior.tyrannosaurus_rex.anatomical.walk\tERROR\tValueError\tgenerated plant manifest is stale"]
    report = digest_snapshot.golden_mismatch(digest_snapshot.render(run), run, 1, digest_snapshot.GOLDEN)
    assert report is not None
    assert f"  {run[-1]}" in report.splitlines()
    assert digest_snapshot.WRITE_COMMAND not in report, "--write refuses a run with an ERROR line"


def test_check_names_the_write_command_for_the_path_it_was_given() -> None:
    report = digest_snapshot.golden_mismatch(digest_snapshot.render(_LINES), _LINES[:-1], 0, "my golden.txt")
    assert report is not None
    assert f"    {digest_snapshot.WRITE_COMMAND} 'my golden.txt'" in report.splitlines()


def test_check_reports_a_byte_order_mark_or_extra_newlines_as_form_not_moves() -> None:
    golden = digest_snapshot.render(_LINES)
    for variant in ("\ufeff" + golden, golden + "\n"):
        report = digest_snapshot.golden_mismatch(variant, _LINES, 0, digest_snapshot.GOLDEN)
        assert report is not None and "is not the text --write writes" in report, report
        assert "moved" not in report.split("--write writes")[0], report


def test_check_counts_every_moved_line_once() -> None:
    error = "stage.tyrannosaurus_rex.anatomical.stance.view\tERROR\tTypeError\tbad env class"
    report = digest_snapshot.golden_mismatch(digest_snapshot.render(_LINES), [*_LINES, error, error], 2, "x")
    assert report is not None
    assert "2 digest line(s) moved (0 changed, 2 added, 0 removed):" in report.splitlines()


def test_check_golden_and_write_golden(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "snapshot.txt"
    assert digest_snapshot.write_golden(path, "snapshot.txt", _LINES, 0) == 0
    assert path.read_bytes() == ("\n".join(_LINES) + "\n").encode()
    assert "Wrote snapshot.txt (4 lines)" in capsys.readouterr().out
    assert digest_snapshot.check_golden(path, "snapshot.txt", _LINES, 0) == 0
    assert "Digest snapshot is current: snapshot.txt (4 lines)" in capsys.readouterr().out
    assert digest_snapshot.check_golden(path, "snapshot.txt", _LINES[:-1], 0) == 1
    assert "  removed  behavior tyrannosaurus_rex anatomical walk behavior_identity_sha256" in capsys.readouterr().err
    # A run with an ERROR line writes nothing.
    error = "plant.tyrannosaurus_rex.anatomical.identity\tERROR\tRuntimeError\tno MuJoCo"
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
        (["--check", "no/such/snapshot.txt"], f"`{digest_snapshot.WRITE_COMMAND} no/such/snapshot.txt`"),
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
    # Only the refusal: a run would have printed its "digest_snapshot: N lines, ..." summary first.
    assert all(line.startswith("digest_snapshot: refused: ") for line in result.stderr.splitlines()), result.stderr


@pytest.fixture
def stubbed_sections(monkeypatch: pytest.MonkeyPatch):
    """main() with its section functions replaced by one that emits _LINES, and its process-wide changes undone."""

    def plant_section(out: digest_snapshot._Snapshot) -> dict[str, object]:
        for line in _LINES:
            out.emit(*line.split("\t"))
        return {}

    monkeypatch.setattr(digest_snapshot, "plant_section", plant_section)
    for name in ("stage_section", "recovery_section", "behavior_section"):
        monkeypatch.setattr(digest_snapshot, name, lambda *args: None)
    monkeypatch.setattr(digest_snapshot, "reward_section", lambda out, exact=False, behaviors=False: None)
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
    assert "  added    behavior tyrannosaurus_rex anatomical walk behavior_identity_sha256" in capsys.readouterr().err
    assert digest_snapshot.main([*repo, "--write", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == digest_snapshot.render(_LINES)
    assert digest_snapshot.main([*repo, "--check", str(path)]) == 0
    assert f"Digest snapshot is current: {path} (4 lines)" in capsys.readouterr().out


def test_main_exact_prints_only_the_reward_captures(
    stubbed_sections: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def reward_section(out: digest_snapshot._Snapshot, exact: bool = False, behaviors: bool = False) -> None:
        calls.append((exact, behaviors))
        out.emit("reward-exact", "tyrannosaurus_rex", "anatomical", "stance", "roll", "reward", "sha256:e")

    monkeypatch.setattr(digest_snapshot, "reward_section", reward_section)
    repo = ["--repo", str(REPOSITORY_ROOT)]
    assert digest_snapshot.main([*repo, "--exact", "--skip-behaviors"]) == 0
    assert digest_snapshot.main([*repo, "--exact"]) == 0
    assert (
        capsys.readouterr().out == "reward-exact\ttyrannosaurus_rex\tanatomical\tstance\troll\treward\tsha256:e\n" * 2
    ), "no other section"
    assert calls == [(True, False), (True, True)]
    with pytest.raises(SystemExit):  # never a golden
        digest_snapshot.main([*repo, "--exact", "--check"])
