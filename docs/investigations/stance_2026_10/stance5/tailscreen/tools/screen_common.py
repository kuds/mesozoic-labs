"""Tail_3 bound screen: shared constants (BUILD_SPEC section 1).  Importable without torch or environments.

Pins (SCRATCH_COMMIT, DURABLE_BRANCH, TOOL_SHA256) are filled (2026-10-24, after the reviews); COEF_90 / COEF_60
live in SCREEN/calib/coef.json and SCREEN/APPROVED, never here.  This file's own sha256 is pinned in the criteria
text and in SCREEN/APPROVED (screen_common_sha256); preflight, gate.sh and the scorer check it.
"""
from __future__ import annotations

import importlib.metadata
import platform

S2 = "scratchpad2"
SP = "scratchpad"
SCREEN = S2 + "/tailround/screen"
FETCH = S2 + "/tailround/fetch"
DRAFT = S2 + "/tailround/criteria/draft"
PY = S2 + "/venv/bin/python"
RUNLOG = SP + "/foot/stance4/live/run_log.txt"

CLONE = SCREEN + "/scratch_tail3bound"
BRANCH = "scratch/tail3-upper-bound"
PARENT = "cf80ecb14d1630ce3b4dcdea28a05835d143470b"
SCRATCH_COMMIT = "4c0c8da76d14f4b482f1fc63e0cf57af79b0184f"  # pinned 2026-10-24 (HEAD of scratch/tail3-upper-bound)
CHANGED_FILES = ("environments/shared/mirror_kl_ppo.py", "environments/shared/tests/test_mirror_kl_ppo.py")

STATES = {
    "s42": {
        "zip": FETCH + "/stance5_LTSB_s42/A/stage1_4194304_steps.zip",
        "zip_sha256": "d3c90db8c107ce1fef6b63efa3e8cdd2271e6e437a3abdb5b30f57b3f46b8df2",
        "pkl": FETCH + "/stance5_LTSB_s42/A/stage1_vecnormalize_4194304_steps.pkl",
        "pkl_sha256": "3903cbcfe43559c8f5dccc0b2d6ed01f3c3989f039f966b12ed6aaa7ecbca118",
        "stage_config": SP + "/foot/stance5/runs/stance5_LTSB_s42/01_stance/stage_config.json",
        "own_seed": 42,
    },
    "s80": {
        "zip": FETCH + "/stance5_LTSB_s80/A/stage1_4194304_steps.zip",
        "zip_sha256": "649db94c026db3925a43643dfcd58299877fbe7783bca25e7b7b6ca504a5a3cc",
        "pkl": FETCH + "/stance5_LTSB_s80/A/stage1_vecnormalize_4194304_steps.pkl",
        "pkl_sha256": "98fe304328256720f59658da9a0c2b7ccae89dd0cb5ad5798a1768383adc06d9",
        "stage_config": SP + "/foot/stance5/replication/runs/stance5_LTSB_s80/01_stance/stage_config.json",
        "own_seed": 80,
    },
}

START_STEPS = 4194304
STEPS = 1048576
SAVE_EVERY = 131072
N_ENVS = 4
ROLLOUT = 8192
TAIL3 = 14

ARMS = {"control": ("COEF0", 0.90), "b90": ("COEF_90", 0.90), "b60": ("COEF_60", 0.60)}
REP_SEEDS = (101, 202, 303)
WAVES = [("s42", 101), ("s80", 101), ("s42", 202), ("s80", 202), ("s42", 303), ("s80", 303)]

R1 = 5111808
R2 = "final"
LABELS = ("4325376", "4456448", "4587520", "4718592", "4849664", "4980736", "5111808", "5242880", "final")
PANEL_SEEDS = range(3042, 3062)
SETTLE = 200

OVERLAYS = {
    "environments/shared/reporting/episode_trace.py":
        "96ced68e124ddf4fa225ce32d93831a3e3261a2ba5b3ef502a0b900ce7b16c0b",
    "environments/shared/scripts/episode_trace_report.py":
        "b5923e5cfbcafd9f6e108df207a268500313de9bff44572121c8e3b48aad0af2",
    "environments/shared/stance_diagnostics.py":
        "c5a684cd5b64e8eb41246c9bbc0dc07760e110895ab4022d42546414888c565e",
}

STACK = {"python": "3.13.17", "torch": "2.13.0+cpu", "numpy": "2.5.4", "gymnasium": "1.4.0",
         "stable_baselines3": "2.9.0", "mujoco": "3.10.0"}

# Criteria "Validations", V1: frac values in full, the others to 6 decimals; flagged = episodes flagged of 20.
V1_EXPECTED = {
    "s42": {"frac99": 0.33825, "frac90": 0.7736875, "cmd": 0.924539, "raw_max_window": 1.028092, "flagged": 10,
            "return_mean": 4379.390198},
    "s80": {"frac99": 0.0, "frac90": 0.0011875, "cmd": 0.628516, "raw_max_window": 0.902040, "flagged": 0,
            "return_mean": 4418.555849},
}

# The seven pinned scripts (criteria BUILD LIST item 7).  screen_common.py itself is pinned by its sha256 in the
# criteria text and in SCREEN/APPROVED (screen_common_sha256), since it holds these pins and cannot pin itself.
TOOL_FILES = ("run_tailscreen.py", "calib_tailscreen.py", "measure_tailscreen.py", "score_tailscreen.py",
              "copy_durable.sh", "gate.sh", "run_waves.sh")

# Tests that must be recorded "passed" (not skipped, not deselected, not absent) in tests_out/gate_pass.json at the
# pinned commit before any training step: the gate tier (T1b, T2, T7b take optimizer steps on a toy model) and the
# clone's build-tier tail_3 tests, re-run by gate.sh.  Matched on the node id's last '::' part.
GATE_REQUIRED = (
    "test_train_is_stock_ppo_train_plus_four_marked_lines",
    "test_tail3_config_and_enable_refuse_what_they_cannot_bound",
    "test_tail3_t1a_coef_zero_is_gradient_identical_and_takes_no_step[0.9]",
    "test_tail3_t1a_coef_zero_is_gradient_identical_and_takes_no_step[0.6]",
    "test_tail3_t3_scope_upper_side_of_index_14_only",
    "test_tail3_t4_the_penalty_is_mirror_invariant",
    "test_tail3_t5_no_shipped_file_sets_the_bound",
    "test_tail3_t7a_a_bound_zip_loads_with_stock_ppo",
    "test_tail3_t1b_coef_zero_trains_bit_for_bit_as_no_bound",
    "test_tail3_t2_same_seed_reproducibility",
    "test_tail3_t7b_a_bound_zip_loads_after_one_update",
    "test_t7a_screen_zip_carries_the_record_and_the_start_parameters",
    "test_t7a_screen_zip_runs_in_the_measuring_copy",
    "test_preflight_passes_on_the_synthetic_root",
)

DURABLE_BRANCH = "claude/handoff-tailscreen"  # pinned 2026-10-24 (absent on origin then); created after approval
TOOL_SHA256: dict = {  # pinned 2026-10-24 after the fixer round; screen_common.py itself is pinned in the criteria
    "run_tailscreen.py": "9527231283e4db295a603d79f2fb8d58524fbd2d4c5618307b1a03790e77b3f3",
    "calib_tailscreen.py": "c751931639ca41adf3539309854545f96c02fb0b7d6ab50e905690eb91abf6f2",
    "measure_tailscreen.py": "613212ce240e425d4c36b7dcfa76ad64d141bebb0a8f510852cb4f3410e153b1",
    "score_tailscreen.py": "3cd4724761fcdbde68d95b28d6e2afa5980f0acbdde5b8e27c8115816d932d19",
    "copy_durable.sh": "9a3932e121e97027d6057cd9307fbc98f63ad301c121d35a8ba6b693064376a6",
    "gate.sh": "4ef908ae8ac116e7a809e655caf56c1aaa3cd4a9e87ffd34c9a4bc1ad0b398b6",
    "run_waves.sh": "ffcbfe2749714abc308aab17c8f6ab0c373b04816b2e4bd3243fd0d25881e1ed",
}


def gate_problems(gate, head) -> list[str]:
    """Why a tests_out/gate_pass.json does not show the gate passed at HEAD `head` (empty when it does).  Every
    GATE_REQUIRED test must be present with status "passed"; no test may be failed or errored; failed == 0."""
    if not isinstance(gate, dict):
        return ["gate_pass.json is not a JSON object"]
    problems = []
    if not head or gate.get("head") != head:
        problems.append(f"gate_pass.json head {gate.get('head')!r} != HEAD {head!r}")
    if gate.get("failed") != 0:
        problems.append(f"gate_pass.json failed {gate.get('failed')!r} != 0")
    tests = gate.get("tests")
    if not isinstance(tests, dict) or not tests:
        return problems + ["gate_pass.json carries no per-test statuses"]
    by_name: dict = {}
    for node, status in tests.items():
        by_name.setdefault(str(node).split("::")[-1], []).append(status)
    bad = sorted(n for n, st in tests.items() if st not in ("passed", "skipped", "deselected", "xfailed"))
    if bad:
        problems.append(f"gate_pass.json: tests not passed: {bad[:5]}")
    for name in GATE_REQUIRED:
        statuses = by_name.get(name)
        if not statuses:
            problems.append(f"gate_pass.json: required test {name} absent")
        elif any(st != "passed" for st in statuses):
            problems.append(f"gate_pass.json: required test {name} is {statuses}, not passed")
    return problems


def stack_versions() -> dict:
    """The running interpreter's versions, read from package metadata (nothing is imported)."""
    out = {"python": platform.python_version()}
    for key, dist in (("torch", "torch"), ("numpy", "numpy"), ("gymnasium", "gymnasium"),
                      ("stable_baselines3", "stable_baselines3"), ("mujoco", "mujoco")):
        try:
            out[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[key] = None
    return out


def check_stack() -> list[str]:
    """Differences between the running versions and STACK; empty means equal."""
    got = stack_versions()
    return [f"{k}: running {got.get(k)} != pinned {v}" for k, v in STACK.items() if got.get(k) != v]
