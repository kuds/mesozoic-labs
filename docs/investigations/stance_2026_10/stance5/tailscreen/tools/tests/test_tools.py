"""T6 (build tier; no torch, no environments, no training): run_tailscreen.py's arm mapping and its preflight
refusals, the CLI's refusal against the real SCREEN (no APPROVED in the build; no gate_pass.json at gate time),
calib_tailscreen.py's write refusals, and gate.sh / run_waves.sh refusing without APPROVED (build only).

The preflight cases start from a synthetic SCREEN root in tmp_path that passes every check (the real scratch clone
and start pairs, a synthetic APPROVED / criteria.txt / coef.json / gate_pass.json, the pins patched in memory), then
break one thing at a time: each refusal is caused by that one thing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RT = _load("run_tailscreen")
C = RT.C
SCREEN = Path(C.SCREEN)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coef_json(rms: float = 0.602, **override) -> dict:
    coef = RT.coef_by_rule(rms)
    states = {
        s: {"zip_sha256": C.STATES[s]["zip_sha256"], "pkl_sha256": C.STATES[s]["pkl_sha256"], "minibatches": 256,
            "digest_before": {"params": "p", "adam": "a"}, "digest_after": {"params": "p", "adam": "a"}}
        for s in C.STATES
    }
    out = {"schema": "tailscreen-coef/1", "reference_excess": 0.11, "unit_pull_per_coef": 0.22, "states": states,
           "pooled": {"minibatches": 512, "G_pg_rms": rms}, "COEF_90": coef, "COEF_60": coef,
           "coef_equal_by_formula": True, "stack": dict(C.STACK)}
    out.update(override)
    return out


def _gate(head: str, **status) -> dict:
    """A gate_pass.json with every GATE_REQUIRED test passed (status overrides by test name; None drops it)."""
    tests = {f"x.py::{t}": "passed" for t in C.GATE_REQUIRED}
    for name, st in status.items():
        node = next(k for k in tests if k.endswith("::" + name))
        if st is None:
            tests.pop(node)
        else:
            tests[node] = st
    return {"head": head, "passed": sum(v == "passed" for v in tests.values()), "failed": 0, "tests": tests}


@pytest.fixture
def screen(tmp_path, monkeypatch):
    """A synthetic SCREEN root on which preflight passes."""
    head = RT.clone_state(C.CLONE)["head"]
    assert head is not None
    (tmp_path / "criteria.txt").write_text("criteria (synthetic)\n")
    (tmp_path / "calib").mkdir()
    coef = _coef_json()
    (tmp_path / "calib" / "coef.json").write_text(json.dumps(coef))
    approved = {"criteria_sha256": _sha(tmp_path / "criteria.txt"), "runlog_line": "synthetic", "approved_utc": "x",
                "scratch_commit": head, "COEF_90": coef["COEF_90"], "COEF_60": coef["COEF_60"],
                "coef_json_sha256": _sha(tmp_path / "calib" / "coef.json"), "durable_branch": "x",
                "screen_common_sha256": _sha(TOOLS / "screen_common.py")}
    (tmp_path / "APPROVED").write_text(json.dumps(approved))
    (tmp_path / "tests_out").mkdir()
    (tmp_path / "tests_out" / "gate_pass.json").write_text(json.dumps(_gate(head)))
    monkeypatch.setattr(C, "SCRATCH_COMMIT", head)
    monkeypatch.setattr(C, "TOOL_SHA256", {t: _sha(TOOLS / t) for t in C.TOOL_FILES})
    return types.SimpleNamespace(root=tmp_path, head=head, coef=coef, approved=approved)


def _ns(**override):
    ns = {"state": "s42", "arm": "b60", "rep_seed": 101, "steps": 1048576}
    ns.update(override)
    return types.SimpleNamespace(**ns)


def _rewrite_approved(screen, **override):
    approved = dict(screen.approved, **override)
    (screen.root / "APPROVED").write_text(json.dumps(approved))


def _rewrite_coef(screen, coef: dict, *, repin: bool = True):
    (screen.root / "calib" / "coef.json").write_text(json.dumps(coef))
    if repin:
        _rewrite_approved(screen, COEF_90=coef.get("COEF_90"), COEF_60=coef.get("COEF_60"),
                          coef_json_sha256=_sha(screen.root / "calib" / "coef.json"))


# ---------------------------------------------------------------------------------------------------- arm mapping
def test_arm_config_maps_the_three_arms():
    coef = {"COEF_90": 2.7, "COEF_60": 3.1}  # unequal only to show which key each arm reads
    assert RT.arm_config("control", coef) == (0.0, 0.90)
    assert RT.arm_config("b90", coef) == (2.7, 0.90)
    assert RT.arm_config("b60", coef) == (3.1, 0.60)
    assert set(C.ARMS) == {"control", "b90", "b60"}
    with pytest.raises(ValueError):
        RT.arm_config("b75", coef)


def test_coef_rule():
    assert RT.coef_by_rule(0.602) == 2.7 and RT.coef_by_rule(0.58) == 2.6 and RT.coef_by_rule(0.6215) == 2.8
    assert RT.coef_rule_problems(_coef_json()) == []
    assert RT.coef_rule_problems(_coef_json(COEF_60=2.8)) != []
    assert RT.coef_rule_problems(_coef_json(COEF_90=2.737)) != []  # unrounded
    assert RT.coef_rule_problems(_coef_json(schema="x")) != []
    assert RT.coef_rule_problems(_coef_json(stack={})) != []


# ---------------------------------------------------------------------------------------------------- preflight
def test_preflight_passes_on_the_synthetic_root(screen):
    for arm in C.ARMS:
        for state in C.STATES:
            for seed in C.REP_SEEDS:
                assert RT.preflight(_ns(arm=arm, state=state, rep_seed=seed), screen.root) == []


def _break(screen, monkeypatch, case: str):
    root = screen.root
    if case == "approved_missing":
        (root / "APPROVED").unlink()
    elif case == "approved_unparsable":
        (root / "APPROVED").write_text("{not json")
    elif case == "approved_criteria_mismatch":
        (root / "criteria.txt").write_text("edited after approval\n")
    elif case == "criteria_missing":
        (root / "criteria.txt").unlink()
    elif case == "coef_missing":
        (root / "calib" / "coef.json").unlink()
    elif case == "coef_breaks_rule":
        _rewrite_coef(screen, _coef_json(COEF_60=2.8))
    elif case == "coef_unrounded":
        _rewrite_coef(screen, _coef_json(COEF_90=2.737, COEF_60=2.737))
    elif case == "coef_not_the_pin":
        _rewrite_approved(screen, COEF_90=2.6)
    elif case == "coef_sha_not_the_pin":
        _rewrite_coef(screen, dict(screen.coef, utc="later"), repin=False)
    elif case == "low_disk":
        monkeypatch.setattr(RT, "MIN_FREE_BYTES", 1 << 62)
    elif case == "run_exists":
        (root / "runs" / "s42_b60_r101").mkdir(parents=True)
    elif case == "start_sha":
        states = json.loads(json.dumps(C.STATES))
        states["s42"]["zip_sha256"] = "0" * 64
        monkeypatch.setattr(C, "STATES", states)
    elif case == "gate_missing":
        (root / "tests_out" / "gate_pass.json").unlink()
    elif case == "gate_other_head":
        (root / "tests_out" / "gate_pass.json").write_text(json.dumps(_gate("f" * 40)))
    elif case == "gate_failed":
        (root / "tests_out" / "gate_pass.json").write_text(json.dumps(dict(_gate(screen.head), failed=1)))
    elif case == "gate_t1b_skipped":
        (root / "tests_out" / "gate_pass.json").write_text(json.dumps(
            _gate(screen.head, test_tail3_t1b_coef_zero_trains_bit_for_bit_as_no_bound="skipped")))
    elif case == "gate_t2_absent":
        (root / "tests_out" / "gate_pass.json").write_text(json.dumps(
            _gate(screen.head, test_tail3_t2_same_seed_reproducibility=None)))
    elif case == "gate_t7b_deselected":
        (root / "tests_out" / "gate_pass.json").write_text(json.dumps(
            _gate(screen.head, test_tail3_t7b_a_bound_zip_loads_after_one_update="deselected")))
    elif case == "gate_counts_only":  # the old shape: counts with no per-test statuses
        (root / "tests_out" / "gate_pass.json").write_text(json.dumps({"head": screen.head, "passed": 9, "failed": 0}))
    elif case == "screen_common_not_the_pin":
        _rewrite_approved(screen, screen_common_sha256="0" * 64)
    elif case == "screen_common_pin_missing":
        approved = dict(screen.approved)
        approved.pop("screen_common_sha256")
        (root / "APPROVED").write_text(json.dumps(approved))
    elif case == "tool_pin_other":
        monkeypatch.setattr(C, "TOOL_SHA256", dict(C.TOOL_SHA256, **{"score_tailscreen.py": "0" * 64}))
    elif case == "approved_other_commit":
        _rewrite_approved(screen, scratch_commit="e" * 40)
    elif case == "pin_unfilled":
        monkeypatch.setattr(C, "SCRATCH_COMMIT", None)
    elif case == "tool_pin":
        monkeypatch.setattr(C, "TOOL_SHA256", {})
    elif case == "stack":
        monkeypatch.setattr(C, "STACK", dict(C.STACK, torch="0.0.0"))
    elif case.startswith("clone_"):
        real = RT.clone_state(C.CLONE)
        fake = dict(real)
        if case == "clone_dirty":
            fake["status"] = " M environments/shared/train_base.py"
        elif case == "clone_branch":
            fake["branch"] = "main"
        elif case == "clone_push":
            fake["push_url"] = ["https://github.com/kuds/mesozoic-labs"]
        elif case == "clone_files":
            fake["changed_files"] = sorted([*C.CHANGED_FILES, "environments/shared/train_base.py"])
        elif case == "clone_parent":
            fake["chain_ok"] = False
        elif case == "clone_ignored":
            fake["status"] = "!! environments/shared/__pycache__/"
        monkeypatch.setattr(RT, "clone_state", lambda clone: fake)
    else:
        raise AssertionError(case)


CASES = {
    "approved_missing": ("APPROVED:", "clone:"),  # no APPROVED: also no approved scratch commit
    "approved_unparsable": ("APPROVED:", "clone:"),
    "approved_criteria_mismatch": ("APPROVED:",),
    "criteria_missing": ("APPROVED:",),
    "coef_missing": ("coef.json:",),
    "coef_breaks_rule": ("coef.json:",),
    "coef_unrounded": ("coef.json:",),
    "coef_not_the_pin": ("coef.json:",),
    "coef_sha_not_the_pin": ("coef.json:",),
    "run_exists": ("runs:",),
    "start_sha": ("start:", "coef.json:"),  # coef.json records the start sha256 too
    "gate_missing": ("gate:",),
    "gate_other_head": ("gate:",),
    "gate_failed": ("gate:",),
    "gate_t1b_skipped": ("gate:",),
    "gate_t2_absent": ("gate:",),
    "gate_t7b_deselected": ("gate:",),
    "gate_counts_only": ("gate:",),
    "screen_common_not_the_pin": ("APPROVED:",),
    "screen_common_pin_missing": ("APPROVED:",),
    "tool_pin_other": ("pins:",),
    "clone_ignored": ("clone:",),
    "approved_other_commit": ("clone:",),
    "pin_unfilled": ("clone:",),
    "tool_pin": ("pins:",),
    "low_disk": ("disk:",),
    "stack": ("stack:", "coef.json:"),  # coef.json records the pinned stack, which no longer matches
    "clone_dirty": ("clone:",),
    "clone_branch": ("clone:",),
    "clone_push": ("clone:",),
    "clone_files": ("clone:",),
    "clone_parent": ("clone:",),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_preflight_refuses(screen, monkeypatch, case):
    assert RT.preflight(_ns(), screen.root) == []
    _break(screen, monkeypatch, case)
    problems = RT.preflight(_ns(), screen.root)
    prefixes = CASES[case]
    assert problems, case
    assert any(p.startswith(prefixes[0]) for p in problems), problems
    assert all(p.startswith(prefixes) for p in problems), problems


@pytest.mark.parametrize(
    ("override", "prefix"),
    [
        ({"arm": "b75"}, "args: arm"),
        ({"arm": "COEF_90"}, "args: arm"),
        ({"state": "s52"}, "args: state"),
        ({"rep_seed": 42}, "args: rep seed"),
        ({"rep_seed": 404}, "args: rep seed"),
        ({"steps": 1048575}, "args: steps"),
        ({"steps": 524288}, "args: steps"),
    ],
)
def test_preflight_refuses_arguments(screen, override, prefix):
    problems = RT.preflight(_ns(**override), screen.root)
    assert any(p.startswith(prefix) for p in problems), problems
    assert all(p.startswith(("args:", "start:")) for p in problems), problems  # an unknown state has no start pair


def test_preflight_creates_nothing(screen):
    before = sorted(str(p.relative_to(screen.root)) for p in screen.root.rglob("*"))
    RT.preflight(_ns(), screen.root)
    RT.preflight(_ns(arm="nope", state="nope"), screen.root)
    assert sorted(str(p.relative_to(screen.root)) for p in screen.root.rglob("*")) == before


# ---------------------------------------------------------------------------------------------------- CLI and import
def _run(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return subprocess.run([sys.executable, "-I", "-B", *args], capture_output=True, text=True, env=env, cwd=str(TOOLS),
                          timeout=600)


@pytest.mark.parametrize("extra", [["--preflight-only"], []])
def test_cli_refuses_against_the_real_screen_now(extra):
    """In the build there is no APPROVED; at gate time (gate.sh runs this file) APPROVED exists but gate_pass.json
    does not yet (gate.sh writes it only after these tests).  Either way the CLI refuses (exit 2) with or without
    --preflight-only, and creates nothing: no runs/ folder, no APPROVED."""
    approved_before = (SCREEN / "APPROVED").exists()
    assert not (SCREEN / "tests_out" / "gate_pass.json").exists(), "this test runs before any gate pass"
    runs_before = sorted(os.listdir(SCREEN / "runs")) if (SCREEN / "runs").exists() else None
    done = _run(str(TOOLS / "run_tailscreen.py"), "--state", "s42", "--arm", "b60", "--rep-seed", "101",
                "--steps", "1048576", *extra)
    assert done.returncode == 2, (done.stdout, done.stderr)
    verdict = json.loads(done.stdout)
    assert verdict["preflight"] == "refused"
    if approved_before:
        assert "gate: gate_pass.json missing" in verdict["problems"]
    else:
        assert "APPROVED: missing" in verdict["problems"]
    assert (SCREEN / "APPROVED").exists() == approved_before
    runs_after = sorted(os.listdir(SCREEN / "runs")) if (SCREEN / "runs").exists() else None
    assert runs_after == runs_before


@pytest.mark.skipif((SCREEN / "APPROVED").exists(), reason="build only: with APPROVED, gate.sh would run for real")
@pytest.mark.parametrize("script", ["gate.sh", "run_waves.sh"])
def test_gate_and_waves_refuse_without_approved_and_create_nothing(script):
    def tree():
        return sorted(str(p.relative_to(SCREEN)) for p in SCREEN.rglob("*")
                      if "review-code" not in p.parts and "scratch_tail3bound" not in p.parts)

    before = tree()
    done = subprocess.run(["bash", str(TOOLS / script)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert "refused: no SCREEN/APPROVED" in done.stderr
    assert tree() == before and not (SCREEN / "runs").exists() and not (SCREEN / "APPROVED").exists()


def test_run_waves_plan_is_the_pinned_waves():
    done = subprocess.run(["bash", str(TOOLS / "run_waves.sh"), "--plan"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    lines = [ln.split() for ln in done.stdout.splitlines() if "run_tailscreen.py" in ln]
    got = [(ln[ln.index("--state") + 1], ln[ln.index("--arm") + 1], int(ln[ln.index("--rep-seed") + 1]),
            int(ln[ln.index("--steps") + 1]), ln[1:3]) for ln in lines]
    want = [(st, arm, seed, C.STEPS, ["-I", "-B"]) for st, seed in C.WAVES for arm in C.ARMS]
    assert got == want and len(got) == 18


def test_importing_the_tools_imports_neither_torch_nor_environments():
    code = (
        "import importlib.util, sys, json\n"
        "for name in ('run_tailscreen', 'calib_tailscreen'):\n"
        f"    spec = importlib.util.spec_from_file_location(name, {str(TOOLS)!r} + '/' + name + '.py')\n"
        "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "heavy = sorted(k for k in sys.modules if k.split('.')[0] in "
        "('torch', 'environments', 'stable_baselines3', 'numpy', 'gymnasium', 'mujoco'))\n"
        "print(json.dumps(heavy))\n"
    )
    done = _run("-c", code)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []


# ---------------------------------------------------------------------------------------------------- calibration
def test_calib_refuses_to_overwrite_or_write_inside_calib(tmp_path, monkeypatch):
    calib = _load("calib_tailscreen")
    assert calib.refusal(tmp_path / "coef_dry.json") is None or (SCREEN / "runs").exists()
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    assert "exists" in calib.refusal(existing)
    assert "inside" in calib.refusal(SCREEN / "calib" / "coef_other.json")
    if (SCREEN / "calib" / "coef.json").exists():
        assert "exists" in calib.refusal(SCREEN / "calib" / "coef.json")
    fake_screen = tmp_path / "screen"
    (fake_screen / "runs" / "s42_control_r101").mkdir(parents=True)
    monkeypatch.setattr(calib.C, "SCREEN", str(fake_screen))
    assert "not empty" in calib.refusal(tmp_path / "coef_dry2.json")
