"""Unit tests of PS's score_bs.py on synthetic records (no simulation, no training): each reading line, the boundary of
the test, the gate (0.20, inclusive, the median rule), the fill and its bounds, the missing and void classes,
outcome-blindness, the input errors (the critic form included), PROVISIONAL and final, the four secondaries in two
families (direction, fill, MISSING causes, the efficacy Bonferroni note, the guard's note, the note under line 1, never
the line), SEC-TAIL2's probe file (bound to check.json and the markers; changed, missing, not the markers' file), the
report-only readings (trajectory, G(E) > G(S), the phase swing, the per-actuator rises, BESIDE BS2), the constants
against their sources (the parent's F2 probe, the code copy's MJCF, the criteria text, SCREEN.json, PINS), a dry run of
the parent's own markers, and mutation tests (each mutant of score_bs.py is caught by the kill checks; the original
passes them).  Synthetic probe files are written with numpy (float32, as the run's probe).

    cd /tmp && TMPDIR=<a folder under PS> OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider \
        --basetemp=<a new folder under PS> <PS>/tools/tests/test_score_bs.py
(the synthetic roots, about 100 MB over the whole file, go under --basetemp: keep them inside <PS> and remove them
after; no test makes an optimizer step, simulates or opens a continuation's files.)
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import random
import re
import statistics
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]
SRC_PATH = TOOLS / "score_bs.py"
_spec = importlib.util.spec_from_file_location("score_bs", str(SRC_PATH))
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)
C, R = SB.C, SB.R
S, E, R8 = C.S, C.E, C.ROLLOUT
PHASE = SB.INTERVENTION
CTRL = C.CONTROL
B_S = SB.BASELINE_S
PS = Path(C.BS)
PARENT_MARKERS = PS / "validate" / "v3" / "parent_s42" / "markers.json"
G_S = -0.6363920116424561
PROBE_REL = Path(C.STAGE_DIR) / "policy_probe.npz"


def f32(x):
    """The float32 value of x as a double (the probe's columns are float32)."""
    return float(np.float32(x))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return path


def _criteria_text():
    """The installed <PS>/criteria.txt, or the draft it was installed from."""
    for p in (PS / "criteria.txt", PS / "criteria_work" / "criteria_v1.txt"):
        if p.exists():
            return " ".join(p.read_text(encoding="utf-8").split())
    return None


@pytest.fixture(autouse=True)
def _no_code_copy_measurement(monkeypatch):
    """The code copy is measured for real in test_code_copy_is_measured_at_bs_common_CODE only."""
    calls = []
    monkeypatch.setattr(SB, "code_seal_items", lambda d: calls.append(Path(d)) or [])
    return calls


# ------------------------------------------------------------------------------------------------ synthetic records
def _ledger(path, T, returns=4000.0, falls=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["evaluation,timesteps,episode,env,return,length,full_horizon,cause"]
    for k, t in enumerate(range(S + C.EVAL_FREQ, T + 1, C.EVAL_FREQ)):
        for ep in range(10):
            fell = k == 0 and ep < falls
            lines.append(f"{k},{t},{ep},0,{returns + k},{292 if fell else 1000},{0 if fell else 1},"
                         f"{'fall' if fell else 'truncated'}")
    path.write_text("\n".join(lines) + "\n")
    return path


OTHER_COL = tuple(0.0078125 * j - 0.0625 for j in range(15))      # dyadic: exact in float32


def _probe_file(path, value, T, nan_at=None, tail2=0.03125, tail2_nan=False, mu14_at=None):
    """A synthetic policy_probe.npz (update_timesteps, rollout_mu_mean (n, 15) float32): column 14 ramps from B_S to
    `value`, which the primary's 8 rows carry exactly; column 13 is `tail2` on the primary's rows (0.015625 elsewhere);
    the others constant.  Returns (steps, mu)."""
    steps = [S + R8 * k for k in range(1, (T - S) // R8)]
    n = len(steps)
    mu = np.empty((n, 15), dtype=np.float32)
    mu[:] = np.asarray(OTHER_COL, dtype=np.float32)
    mu[:, 13] = 0.015625
    mu[:, 14] = [B_S + (value - B_S) * (i + 1) / n for i in range(n)]
    for i, s in enumerate(steps):
        if s in SB.PRIMARY_STEPS:
            mu[i, 14] = value
            mu[i, 13] = np.nan if tail2_nan else tail2
    if nan_at is not None:
        mu[steps.index(nan_at), 14] = np.nan
    if mu14_at is not None:                                # (step, value): the file disagrees with the markers
        mu[steps.index(mu14_at[0]), 14] = mu14_at[1]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.savez(fh, update_timesteps=np.asarray(steps, dtype=np.int64), rollout_mu_mean=mu)
    return steps, mu


def _probe(path, value, T, nan_at=None, tail2=0.03125, tail2_nan=False):
    steps, mu = _probe_file(path, value, T, nan_at, tail2, tail2_nan)
    n = len(steps)
    trail = [None] * 7 + [0.05] * (n - 7)
    return {"file": str(path), "sha256": _sha(path), "label": "x", "S": S, "T": T, "n_rows": n, "steps": steps,
            "rollout_mu14_mean": mu[:, 14].astype(np.float64).tolist(), "tail3_statue_settled_mean": [0.05] * n,
            "trailing8_mean": trail, "trailing8_first_above": {"0.10": None}, "trailing8_max": 0.05,
            "std14": [0.19] * n, "obs6_mean_deg": [0.0] * n}


def _loop(arm, T, g=-0.5, b=1.0, mech=0.04, final=True, num_ts_E=None):
    """M-LOOP rows at S, the periodic pairs before T and (final) the final pair at T, as read_bs.saved_pairs gives, in
    the arm's critic form (read_bs.loop_markers_one): control "plain" (effective = raw); critic_phase
    "phase+raw+settled" with a phase record (raw, effective and settled differ: a reader of the wrong one is seen)."""
    phase = arm == PHASE
    rows = []
    for s in (S, *[x for x in C.SAVED_STEPS if x < T], *([T] if final else [])):
        end = s == T
        G = g if end else (G_S if s == S else -0.6)
        bstar = b if end else 1.4
        crit = {}
        for k in ("REF", "OWN", "ONPOLICY"):
            eff = mech if (end and k == "ONPOLICY") else 0.01
            if phase:
                crit[k] = {"raw": {"dV_dq3_per_deg": eff + 0.05, "V_mean": 4.0},
                           "effective": {"dV_dq3_per_deg": eff, "V_mean": 4.125},
                           "settled": {"dV_dq3_per_deg": eff + 0.07, "V_mean": 4.25}}
            else:
                crit[k] = {"raw": {"dV_dq3_per_deg": eff, "V_mean": 4.0},
                           "effective": {"dV_dq3_per_deg": eff, "V_mean": 4.0}}
        row = {"steps": s, "num_timesteps": num_ts_E if (end and num_ts_E is not None) else s,
               "critic_form": SB.CRITIC_FORM[arm],
               "OWN": {"P": {"G": G, "c0": 1.0 - G - bstar, "b_star": bstar}, "L": {"G": -0.8},
                       "own14_median": 0.0, "b_star_L": 1.8, "n_terminated": 0},
               "REF": {"P": {"G": -0.7, "c0": -0.1}}, "critic_slope_obs6": crit}
        if phase:
            row["phase"] = {"start_transform": s == S, "value_phase_cols_max_abs_read": 0.24 if s == S else 0.0125,
                            "value_phase_cols_max_abs_used": 0.0 if s == S else 0.0125,
                            "actor_phase_invariance_max_abs": 0.0}
        rows.append(row)
    return rows


def _direct(T, cmd=0.5, cmd_max=0.9, drop_E=False):
    per = {}
    for t in range(S + C.EVAL_FREQ, T + 1, C.EVAL_FREQ):
        if drop_E and t == E:
            continue
        at_T = t == T
        per[str(t)] = {"n": 10, "cmd_l": cmd if at_T else 0.2, "cmd_l_max": cmd_max if at_T else 0.25, "n_cmd": 10,
                       "n_over": 0, "n_tail_over": 0, "n_t3_over": 0, "over_actuators": []}
    vals = [e["cmd_l"] for e in per.values() if isinstance(e["cmd_l"], (int, float))]
    return {"per_evaluation": per, "firsts": {}, "evaluations_absent": [], "cmd_l_max": max(vals) if vals else None,
            "cmd_l_last": per[max(per, key=int)]["cmd_l"] if per else None}


def _meta(arm, seed, root, crit=None, screen=None, appr=None):
    phase = arm == PHASE
    ev = {"n": 175, "buffer_gae_lambda": [0.95], "max_adv_err_vs_gae095": 3e-6, "min_dmu_actor": 0.18,
          "min_dv_predict_values": 0.09, "max_dv_predict_values": 0.29, "max_buffer_values_vs_critic": 1e-6,
          "state_problems": [], "max_dmu_phase": 0.0 if phase else 6.7, "min_dmu_phase": 0.0 if phase else 4.0,
          "phase_rows_recorded": 175 if phase else 0}
    if phase:
        ev.update(phase_n_zero_init=1, phase_n_updated=174, zero_init_max_value_phase_cols=0.0,
                  zero_init_max_dv_phase=0.0, updated_min_value_phase_cols=1.7e-5,
                  updated_min_dv_phase={"dv_phase_predict_values": 9e-4, "dv_phase_forward": 9e-4,
                                        "dv_phase_evaluate_actions": 9e-4, "dv_phase_mirrored": 8e-4},
                  max_phase_buffer_max_err=0.0, phase_all_k_consistent=True, phase_all_phase_stats_frozen=True,
                  phase_all_mirror_phase_self_map=True, phase_all_policy_phase_cols_unchanged=True,
                  phase_all_envs_swapped=True, max_policy_phase_adam_max_abs=0.0, phase_rows_presettle_total=286720,
                  phase_k_max=999)
    aa = {"arm": arm, "mlp_extractor_class": "CriticPhaseMlpExtractor" if phase else "MlpExtractor",
          "obs_rms_class": "PhaseSlotsSymmetricRunningMeanStd" if phase else "SymmetricRunningMeanStd"}
    if phase:
        aa["transform_verified"] = True
    return {"schema": C.RUN_SCHEMA, "name": C.run_name(arm, seed), "arm": arm, "seed": seed, "arm_evidence": ev,
            "arm_applied": aa, "arm_state_end": "in force",
            "criteria_sha256": crit, "screen_sha256": screen, "approved_sha256": appr,
            "tool_sha256": {t: _sha(TOOLS / t) for t in C.TOOL_FILES}}


class Screen:
    """A synthetic <root> with SCREEN.json and, per entry, any of: check.json, run_meta.json, markers.json, ledger and
    the run's policy_probe.npz."""

    def __init__(self, root):
        self.root = Path(root)
        _write(self.root / "SCREEN.json", {"schema": C.SCREEN_SCHEMA, "intervention": PHASE,
                                           "queue": [list(q) for q in SB.QUEUE]})

    def run(self, arm, seed, value, *, stop="end", T=E, check=True, read=True, passed=True, nan_at=None,
            meta_kw=None, g=-0.5, cmd=0.5, cmd_max=0.9, b=1.0, mech=0.04, tail2=0.03125, tail2_nan=False,
            final=True, drop_E=False, num_ts_E=None):
        name = C.run_name(arm, seed)
        rd = self.root / "runs" / name
        meta = _write(rd / C.RUN_META, _meta(arm, seed, self.root, **(meta_kw or {})))
        led = _ledger(rd / C.STAGE_DIR / "eval_episodes.csv", T, falls=1 if arm == PHASE else 0)
        probe = _probe(rd / PROBE_REL, value, T, nan_at, tail2, tail2_nan)
        if check:
            _write(rd / C.CHECK_JSON, {
                "schema": R.CHECK_SCHEMA, "tool_sha256": _sha(TOOLS / "read_bs.py"), "name": name, "mode":
                "continuation", "passed": passed, "failures": [] if passed else ["x"], "arm": arm, "seed": seed,
                "S": S, "E": E, "T": T, "end": E, "stop_reason": stop,
                "sources": {"run_meta.json": [str(meta), _sha(meta)],
                            "policy_probe.npz": [probe["file"], probe["sha256"]]}})
        if read:
            direct = _direct(T, cmd, cmd_max, drop_E)
            _write(self.root / "reads" / name / "markers.json", {
                "schema": R.MARKERS_SCHEMA, "tool_sha256": _sha(TOOLS / "read_bs.py"), "label": name,
                "kind": "continuation", "arm": arm, "seed": seed, "S": S, "T": T,
                "stop_reason": stop, "probe": probe, "probe_phase": {"arm": arm, "installed": arm == PHASE},
                "ledger": {"file": str(led), "sha256": _sha(led), "direct": direct,
                           "m5": {"any_flagged": [], "tail_flagged": [], "any_alarms": [], "tail_alarms": [],
                                  "n_c6_counting_failures": 0}},
                "loop": _loop(arm, T, g, b, mech, final, num_ts_E)})
        return name

    def all(self, control, phase, per_seed=None, **kw):
        """per_seed: {(arm, seed): {run() keyword: value}} for the secondaries and the odd cases."""
        for seed, a, b in zip(SB.SEEDS, control, phase, strict=True):
            self.run(CTRL, seed, a, **{**kw, **(per_seed or {}).get((CTRL, seed), {})})
            self.run(PHASE, seed, b, **{**kw, **(per_seed or {}).get((PHASE, seed), {})})

    def lost(self, name, cause="crashes"):
        p = self.root / "lost.json"
        d = json.loads(p.read_text()) if p.exists() else {"schema": C.LOST_SCHEMA, "lost": {}}
        d["lost"][name] = {"cause": cause, "detail": "3 own crashes"}
        _write(p, d)


def _score(root, sb=SB):
    res, lines, code = sb.score(root)
    return res, "\n".join(lines), code


def _brute_p(higher, other):
    """An independent float implementation of the test (no fill): the share of relabelings whose `higher` sum is at
    least the observed."""
    vals = list(higher) + list(other)
    obs = sum(higher)
    subs = list(itertools.combinations(range(len(vals)), len(higher)))
    return sum(1 for s in subs if sum(vals[i] for i in s) >= obs - 1e-12) / len(subs)


def _sec(res, key):
    return next(s for s in res["secondaries"] if s["key"] == key)


# The main synthetic screen (line 2): control far above the phase arm on the primary; the secondaries chosen so that
# SEC-CMD reads p = 2/70 (SUPPORTED, no Bonferroni), SEC-MARGIN 24/70 (NOT SUPPORTED), SEC-MECH 1/70 (SUPPORTED,
# survives Bonferroni), SEC-TAIL2 3/70 (SUPPORTED: the guard's warning); control_2931's G at E equals G at S (G(E) >
# G(S) is strict).  The primaries are float32 values (the probe's).
HIGH = tuple(f32(B_S + x) for x in (0.30, 0.31, 0.32, 0.33))
LOW = tuple(f32(B_S + x) for x in (0.00, 0.01, 0.02, 0.03))
SEC_G = {CTRL: (-1.0, -0.9, -0.8, G_S), PHASE: (-0.4, -0.3, -0.2, -0.1)}
SEC_CMD = {CTRL: (0.80, 0.70, 0.60, 0.35), PHASE: (0.40, 0.30, 0.20, 0.10)}
SEC_CMD_MAX = {CTRL: (0.1, 0.2, 0.3, 0.4), PHASE: (0.9, 0.8, 0.7, 0.6)}   # reversed: reading it would differ
SEC_B = {CTRL: (1.0, 1.25, 1.5, 1.75), PHASE: (1.125, 1.375, 1.625, 1.875)}   # dyadic: no binary near-ties
SEC_MECH = {CTRL: (0.0625, 0.046875, 0.03125, 0.0390625), PHASE: (0.0078125, 0.0, -0.0078125, 0.015625)}
SEC_TAIL2 = {CTRL: (5 / 128, 4 / 128, 2 / 128, 1 / 128), PHASE: (8 / 128, 7 / 128, 6 / 128, 3.5 / 128)}


def _per_seed_main():
    out = {}
    for arm in (CTRL, PHASE):
        for i, seed in enumerate(SB.SEEDS):
            out[(arm, seed)] = {"g": SEC_G[arm][i], "cmd": SEC_CMD[arm][i], "cmd_max": SEC_CMD_MAX[arm][i],
                                "b": SEC_B[arm][i], "mech": SEC_MECH[arm][i], "tail2": SEC_TAIL2[arm][i]}
    return out


def _main_screen(root, per_seed_extra=None, **kw):
    s = Screen(root)
    ps = _per_seed_main()
    for k, v in (per_seed_extra or {}).items():
        ps[k] = {**ps.get(k, {}), **v}
    s.all(HIGH, LOW, per_seed=ps, **kw)
    return s


# ------------------------------------------------------------------------------------------------ constants
def test_constants_match_the_harness_and_screen():
    assert SB.QUEUE == ((CTRL, 2901), (PHASE, 2901), (CTRL, 2911), (PHASE, 2911), (CTRL, 2921), (PHASE, 2921),
                        (CTRL, 2931), (PHASE, 2931))
    assert PHASE == "critic_phase" and PHASE in C.INTERVENTIONS and C.ARMS == (CTRL, PHASE)
    assert (S, E) == (2097152, 3538944) and E == 27 * C.EVAL_FREQ
    assert SB.PRIMARY_STEPS == (3473408, 3481600, 3489792, 3497984, 3506176, 3514368, 3522560, 3530752)
    expected_rows = [S + R8 * k for k in range(1, C.expected_update_rows(S, E) + 1)]
    assert list(SB.PRIMARY_STEPS) == expected_rows[-8:] and len(expected_rows) == 175
    assert SB.EARLY_STEPS == tuple(expected_rows[:8]) and SB.EARLY_STEPS[0] == 2105344
    assert SB.BASELINE_STEPS == tuple(range(2039808, 2097152 + 1, R8))
    assert SB.EVAL_STEPS == tuple(range(2228224, 3538944 + 1, 131072)) and len(SB.EVAL_STEPS) == 11
    assert C.SAVED_STEPS == (2359296, 2621440, 2883584, 3145728, 3407872)
    assert SB.RISE_MIN == Fraction(1, 5) and SB.ALPHA == Fraction(1, 20) and SB.N_EFFICACY == 3
    assert Fraction(B_S) == Fraction(135930855, 1073741824)
    assert set(SB.PARENT_TRAJECTORY) == set(SB.EVAL_STEPS)
    assert SB.PARENT_TRAJECTORY[E] == 0.7053788751363754 and Fraction(SB.PARENT_TRAJECTORY[E]) == Fraction(47337175,
                                                                                                            67108864)
    assert [(s["key"], s["family"], s["higher"]) for s in SB.SECONDARIES] == [
        ("SEC-CMD", "efficacy", CTRL), ("SEC-MARGIN", "efficacy", PHASE), ("SEC-MECH", "efficacy", CTRL),
        ("SEC-TAIL2", "guard", PHASE)]
    assert sum(1 for s in SB.SECONDARIES if s["family"] == "efficacy") == SB.N_EFFICACY
    assert SB.CRITIC_FORM == {CTRL: "plain", PHASE: "phase+raw+settled"}
    assert SB.ACT_TAIL2 == 13 and SB.ACTUATORS[13] == "tail_2_pitch_act" and SB.ACTUATORS[C.ACT_TAIL3] == \
        "tail_3_pitch_act" and len(SB.ACTUATORS) == 15
    assert set(SB.PINNED_TOOLS) == set(C.TOOL_FILES) | {"score_bs.py"}
    assert C.FINAL_ZIP == "stage1_final.zip" and E not in C.SAVED_STEPS and E % C.SAVE_FREQ != 0
    assert not set(C.derived_seeds(s) for s in SB.SEEDS) & set(C.derived_seeds(v) for v in C.VALIDATION_SEEDS)
    assert C.MAX_CONTINUATIONS == len(SB.QUEUE) == 8
    real = PS / "SCREEN.json"
    if real.exists():
        d, problems = C.load_screen(str(real))
        assert problems == [] and C.queue_of(d) == SB.QUEUE and d["intervention"] == PHASE


def test_baseline_and_parent_values_recomputed_from_the_parent_probe():
    path = Path(C.f2_record("policy_probe.npz"))
    if not path.exists():
        pytest.skip("the parent's F2 probe is not on this machine")
    assert C.sha256_file(path) == C.START["records"]["policy_probe.npz"]
    with np.load(path, allow_pickle=False) as z:
        u = z["update_timesteps"].astype(np.int64)
        mu = z["rollout_mu_mean"].astype(np.float64)
    m = mu[:, C.ACT_TAIL3]
    idx = [int(np.flatnonzero(u == s)[0]) for s in SB.BASELINE_STEPS]
    assert idx == list(range(248, 256)) and idx[-1] == C.START["probe_row_at_S"]
    exact = sum((Fraction(float(m[i])) for i in idx), Fraction(0)) / 8
    assert exact == Fraction(135930855, 1073741824) and float(exact) == SB.BASELINE_S
    # the parent's trajectory, report only: the 8 rows ending at each evaluation step (at E the primary's rows)
    for t, want in SB.PARENT_TRAJECTORY.items():
        end = t if t < E else E - R8
        ii = [int(np.flatnonzero(u == end - R8 * k)[0]) for k in range(7, -1, -1)]
        got = sum((Fraction(float(m[i])) for i in ii), Fraction(0)) / 8
        assert got == Fraction(want), t
    assert all(math.isfinite(float(m[int(np.flatnonzero(u == s)[0])])) for s in range(S + R8, E, R8))
    # the scorer's own reading of the parent's per-actuator values at S (report only): b_14 = B_S, b_13 as measured
    pm, cause = SB.parent_actuator_means()
    assert cause is None and pm[C.ACT_TAIL3] == Fraction(135930855, 1073741824)
    assert pm[13] == sum((Fraction(float(mu[i, 13])) for i in idx), Fraction(0)) / 8
    assert round(float(pm[13]), 4) == 0.0233


def test_actuator_names_are_the_code_copy_mjcf_order():
    xml = Path(C.CODE) / "environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml"
    if not xml.exists():
        pytest.skip("the code copy is not on this machine")
    text = xml.read_text(encoding="utf-8")
    block = text.split("<actuator>")[1].split("</actuator>")[0]
    names = tuple(re.findall(r'name="([A-Za-z0-9_]+)"', block))
    assert names == SB.ACTUATORS


# ------------------------------------------------------------------------------------------------ the test itself
def test_perm_test_matches_an_independent_implementation():
    rng = random.Random(7)
    for _ in range(200):
        c = [rng.gauss(0.2, 0.05) for _ in range(4)]
        b = [rng.gauss(0.1, 0.05) for _ in range(4)]
        p, count, total = SB.perm_test([(0, Fraction(x)) for x in c], [(0, Fraction(x)) for x in b])
        assert total == 70 and p == Fraction(count, 70)
        assert float(p) == pytest.approx(_brute_p(c, b), abs=1e-12)
        p2 = SB.perm_test([(0, Fraction(x)) for x in b], [(0, Fraction(x)) for x in c])
        assert float(p2[0]) == pytest.approx(_brute_p(b, c), abs=1e-12)


def test_perm_test_ties_count_against_and_min_p():
    same = [(0, Fraction(1, 5))] * 4
    assert SB.perm_test(same, same)[:2] == (Fraction(1), 70)
    sep = SB.perm_test([(0, Fraction(x)) for x in (5, 6, 7, 8)], [(0, Fraction(x)) for x in (1, 2, 3, 4)])
    assert sep[:2] == (Fraction(1, 70), 1)


@pytest.mark.parametrize("c, b, count", [((8, 7, 6, 3.5), (5, 4, 2, 1), 3), ((8, 7, 6, 3), (5, 4, 2, 1), 4),
                                         ((8, 7, 6, 4), (5, 3, 2, 1), 2)])
def test_perm_test_boundary(c, b, count):
    p = SB.perm_test([(0, Fraction(x)) for x in c], [(0, Fraction(x)) for x in b])
    assert p[1] == count and (p[0] <= SB.ALPHA) == (count <= 3)
    assert float(p[0]) == pytest.approx(_brute_p(c, b))


def test_fill_is_the_supremum_over_completions():
    c = [(0, Fraction(x)) for x in (5, 6, 7, 8)]
    b3 = [(0, Fraction(x)) for x in (1, 2, 3)]
    filled = SB.perm_test(c, b3 + [SB.INF_PHASE])
    assert filled[:2] == (Fraction(36, 70), 36)
    worst = max(SB.perm_test(c, b3 + [(0, Fraction(v))])[0] for v in (-100, 0, 4.5, 6.5, 9, 100, 10**9))
    assert filled[0] >= worst and SB.perm_test(c, b3 + [(0, Fraction(10**12))])[0] == filled[0]
    filled_c = SB.perm_test(c[1:] + [SB.INF_CONTROL], b3 + [(0, Fraction(4))])
    assert filled_c[:2] == (Fraction(36, 70), 36)


def test_fill_bounds_in_the_criteria():
    """THE FILL: one MISSING -> p >= 36/70, one in each arm -> >= 51/70, two controls -> >= 56/70."""
    f = lambda xs: [(0, Fraction(x)) for x in xs]  # noqa: E731
    assert SB.perm_test(f((6, 7, 8)) + [SB.INF_CONTROL], f((1, 2, 3, 4)))[1] == 36
    assert SB.perm_test(f((5, 6, 7, 8)), f((1, 2, 3)) + [SB.INF_PHASE])[1] == 36
    assert SB.perm_test(f((6, 7, 8)) + [SB.INF_CONTROL], f((1, 2, 3)) + [SB.INF_PHASE])[1] == 51
    assert SB.perm_test(f((7, 8)) + [SB.INF_CONTROL] * 2, f((1, 2, 3, 4)))[1] == 56
    rng = random.Random(3)
    for _ in range(50):
        x = [rng.uniform(-1, 1) for _ in range(6)]
        assert SB.perm_test(f(x[:3]) + [SB.INF_CONTROL], f(x[3:]) + [SB.INF_PHASE])[1] >= 51


def test_sign_flip():
    assert SB.sign_flip_test([Fraction(x) for x in (1, 2, 3, 4)])[:2] == (Fraction(1, 16), 1)
    assert SB.sign_flip_test([Fraction(0)] * 4)[:2] == (Fraction(1), 16)
    assert SB.sign_flip_test([]) is None


# ------------------------------------------------------------------------------------------------ the reading lines
def test_line_reduced(tmp_path):
    _main_screen(tmp_path)
    res, text, code = _score(tmp_path)
    assert res["line"] == 2 and res["line_key"] == "REDUCED" and res["additions"] == []
    assert res["p_filled"]["count"] == 1 and res["p_filled"]["total"] == 70
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL        # no APPROVED in a synthetic root
    assert SB.LINE_TEXT[2] in text and "1/70 (0.0143)" in text and "s42's 2,097,152 STATE" in text
    assert "GIVING THE CRITIC THE EPISODE PHASE" in text
    assert "size 3/70 when the 8 primaries are exchangeable under no effect" in " ".join(text.split())
    assert "about 0.18 if the arms' spreads differ" in " ".join(text.split())
    assert res["paired_sign_flip"]["count"] == 1 and res["paired_sign_flip"]["total"] == 16
    assert [c["primary_exact"] for c in res["classes"]] == [str(Fraction(v)) for pair in zip(HIGH, LOW,
                                                                                             strict=True) for v in pair]
    assert res["control_rise"] == pytest.approx(statistics.median(HIGH) - B_S)


@pytest.mark.parametrize("c, b, line", [((8, 7, 6, 3.5), (5, 4, 2, 1), 2), ((8, 7, 6, 3), (5, 4, 2, 1), 3)])
def test_line_at_the_boundary(tmp_path, c, b, line):
    s = Screen(tmp_path)
    s.all([0.5 + x / 64 for x in c], [0.5 + x / 64 for x in b])           # exact in float32: no rounding ties
    res, _text, _code = _score(tmp_path)
    assert res["line"] == line and res["informative"] is True


def test_line_no_reduction_detected(tmp_path):
    s = Screen(tmp_path)
    c, b = [0.5 + x / 128 for x in (26, 19, 32, 23)], [0.5 + x / 128 for x in (27, 22, 24, 31)]   # exact; ties
    s.all(c, b)
    res, text, _ = _score(tmp_path)
    assert res["line"] == 3 and res["additions"] == []
    assert SB.LINE_TEXT[3] in text and "NO REDUCTION DETECTED IN THE ON-POLICY TAIL_3 COMMAND" in text
    assert res["p_filled"]["p_float"] == pytest.approx(_brute_p(c, b))


def test_ties_are_exact_on_the_recorded_values():
    c = [(0, Fraction(x)) for x in (0.20, 0.15, 0.25, 0.18)]
    b = [(0, Fraction(x)) for x in (0.21, 0.17, 0.19, 0.24)]
    gap = sum(map(Fraction, (0.15, 0.18, 0.21, 0.24))) - sum(map(Fraction, (0.20, 0.15, 0.25, 0.18)))
    assert gap == -Fraction(1, 2**55)
    p = SB.perm_test(c, b)
    assert p[:2] == (Fraction(45, 70), 45) and _brute_p([0.20, 0.15, 0.25, 0.18], [0.21, 0.17, 0.19, 0.24]) == 46 / 70


def test_line_not_informative_comes_first(tmp_path):
    s = Screen(tmp_path)
    low = [f32(B_S + x) for x in (0.05, 0.06, 0.07, 0.08)]                  # control median rise about 0.065 < 0.20
    s.all(low, [f32(x - 0.2) for x in low])                                 # the phase arm far lower: p = 1/70
    res, text, _ = _score(tmp_path)
    assert res["line"] == 1 and res["line_key"] == "NOT INFORMATIVE" and res["p_filled"]["count"] == 1
    assert res["informative"] is False and res["control_rise"] == pytest.approx(0.065, abs=1e-6)
    assert res["additions"] == [SB.ADDITION_NOT_INFORMATIVE]
    assert SB.LINE_TEXT[1] in text and "rose by less than 0.20" in text and "critic's phase input" in text


def _classes():
    return [{"name": C.run_name(a, sd), "arm": a, "seed": sd} for a, sd in SB.QUEUE]


def test_informativeness_threshold_is_inclusive_and_0_20():
    edge = Fraction(B_S) + Fraction(1, 5)
    for ctrl, informative in ((edge, True), (edge - Fraction(1, 10**12), False),
                              (Fraction(B_S) + Fraction(15, 100), False), (Fraction(B_S) + Fraction(1, 10), False)):
        prim = {C.run_name(a, sd): ctrl for a, sd in SB.QUEUE}
        dec = SB.decide(_classes(), prim)
        assert dec["informative"] is informative and dec["line"] == (3 if informative else 1)
        assert dec["control_rise"] == ctrl - Fraction(B_S)


def test_informativeness_uses_the_median_not_the_mean():
    rises = dict(zip(SB.SEEDS, (Fraction(5, 100), Fraction(6, 100), Fraction(7, 100), Fraction(90, 100)), strict=True))
    prim = {C.run_name(a, sd): (Fraction(B_S) + rises[sd] if a == CTRL else Fraction(B_S)) for a, sd in SB.QUEUE}
    dec = SB.decide(_classes(), prim)
    assert dec["medians"][CTRL] - Fraction(B_S) == Fraction(65, 1000)
    assert dec["informative"] is False and dec["line"] == 1
    rises = dict(zip(SB.SEEDS, (Fraction(0), Fraction(21, 100), Fraction(22, 100), Fraction(23, 100)), strict=True))
    prim = {C.run_name(a, sd): (Fraction(B_S) + rises[sd] if a == CTRL else Fraction(B_S)) for a, sd in SB.QUEUE}
    dec = SB.decide(_classes(), prim)
    assert dec["control_rise"] == Fraction(215, 1000) and dec["informative"] is True


@pytest.mark.parametrize("n_read, want_rise", [(4, Fraction(215, 1000)), (3, Fraction(22, 100)),
                                               (2, Fraction(225, 1000)), (1, Fraction(23, 100)), (0, None)])
def test_the_median_rule_for_4_3_2_1_0_controls_read(n_read, want_rise):
    rises = (Fraction(0), Fraction(21, 100), Fraction(22, 100), Fraction(23, 100))
    prim = {}
    for i, sd in enumerate(SB.SEEDS):
        prim[C.run_name(CTRL, sd)] = Fraction(B_S) + rises[i] if i >= 4 - n_read else None
        prim[C.run_name(PHASE, sd)] = Fraction(B_S)
    dec = SB.decide(_classes(), prim)
    assert dec["control_rise"] == want_rise
    assert dec["informative"] is (want_rise is not None and want_rise >= Fraction(1, 5))
    assert SB.median_rule([]) is None and SB.median_rule([Fraction(1), Fraction(2)]) == Fraction(3, 2)


# ------------------------------------------------------------------------------------------------ missing and the fill
def test_fill_phase_lost(tmp_path):
    s = _main_screen(tmp_path)
    s.lost(C.run_name(PHASE, 2931))
    res, text, _ = _score(tmp_path)
    assert res["line"] == 3 and res["p_filled"]["count"] == 36
    assert res["p_as_counted"]["count"] == 1 and res["p_as_counted"]["total"] == 35
    assert res["additions"] == ["(the as-counted p, report only, is <= 0.05; the filled p decides)"]
    assert "MISSING, filled at +inf: lost.json: crashes" in text


def test_fill_control_stopped(tmp_path):
    s = Screen(tmp_path)
    for seed, a, b in zip(SB.SEEDS, HIGH, LOW, strict=True):
        if seed == 2901:
            s.run(CTRL, seed, a, stop="collapse", T=3407872)
        else:
            s.run(CTRL, seed, a)
        s.run(PHASE, seed, b)
    res, text, _ = _score(tmp_path)
    cls = {c["name"]: c for c in res["classes"]}
    assert cls["control_2901"]["cls"] == "STOPPED" and cls["control_2901"]["primary"] is None
    assert res["p_filled"]["count"] == 36 and res["line"] == 3
    assert "MISSING, filled at -inf" in text and "the collapse stop at T = 3,407,872" in text
    for s_ in res["secondaries"]:
        assert s_["values"]["control_2901"] is None and "collapse" in s_["causes"]["control_2901"]


def test_primary_unreadable_is_missing(tmp_path):
    s = Screen(tmp_path)
    for seed, a, b in zip(SB.SEEDS, HIGH, LOW, strict=True):
        s.run(CTRL, seed, a)
        s.run(PHASE, seed, b, nan_at=SB.PRIMARY_STEPS[3] if seed == 2911 else None)
    res, _text, _ = _score(tmp_path)
    cls = {c["name"]: c for c in res["classes"]}
    assert cls[f"{PHASE}_2911"]["primary"] is None and "primary unreadable" in cls[f"{PHASE}_2911"]["primary_cause"]
    assert res["line"] == 3 and res["n_missing"] == 1


def test_all_missing_in_one_arm_is_still_decided(tmp_path):
    s = Screen(tmp_path)
    for seed, a in zip(SB.SEEDS, HIGH, strict=True):
        s.run(CTRL, seed, a)
        s.lost(C.run_name(PHASE, seed))
    res, _t, _c = _score(tmp_path)
    assert res["line"] == 3 and res["p_filled"]["count"] == 70 and res["p_as_counted"] is None


def test_no_control_counted_is_not_informative(tmp_path):
    s = Screen(tmp_path)
    for seed, b in zip(SB.SEEDS, LOW, strict=True):
        s.lost(C.run_name(CTRL, seed))
        s.run(PHASE, seed, b)
    res, _t, _c = _score(tmp_path)
    assert res["line"] == 1 and res["control_median"] is None


# ------------------------------------------------------------------------------------------------ the secondaries
def test_secondaries_direction_exact_p_families_and_notes(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    cmd, mar, mech, t2 = (_sec(res, k) for k in ("SEC-CMD", "SEC-MARGIN", "SEC-MECH", "SEC-TAIL2"))
    # SEC-CMD, phase LOWER (control higher): 2/70, SUPPORTED, does not survive Bonferroni (2/70 > 0.05 / 3)
    assert cmd["p_filled"]["count"] == 2 and cmd["verdict"] == SB.SEC_SUPPORTED and cmd["notes"] == []
    assert cmd["p_filled"]["p_float"] == pytest.approx(_brute_p(SEC_CMD[CTRL], SEC_CMD[PHASE]))
    # SEC-MARGIN, phase HIGHER: 24/70, NOT SUPPORTED
    assert mar["p_filled"]["count"] == 24 and mar["verdict"] == SB.SEC_NOT_SUPPORTED and mar["notes"] == []
    assert mar["p_filled"]["p_float"] == pytest.approx(_brute_p(SEC_B[PHASE], SEC_B[CTRL]))
    # SEC-MECH, phase LOWER (control higher): 1/70, SUPPORTED, survives Bonferroni over the 3 efficacy secondaries
    assert mech["p_filled"]["count"] == 1 and mech["verdict"] == SB.SEC_SUPPORTED
    assert mech["notes"] == [SB.SEC_BONFERRONI] and mech["family"] == "efficacy"
    # SEC-TAIL2, the guard, phase HIGHER: 3/70, SUPPORTED with the warning note (never the Bonferroni note)
    assert t2["p_filled"]["count"] == 3 and t2["verdict"] == SB.SEC_SUPPORTED and t2["notes"] == [SB.SEC_GUARD]
    assert t2["family"] == "guard"
    assert t2["p_filled"]["p_float"] == pytest.approx(_brute_p(SEC_TAIL2[PHASE], SEC_TAIL2[CTRL]))
    for key, vals in (("SEC-CMD", SEC_CMD), ("SEC-MARGIN", SEC_B), ("SEC-MECH", SEC_MECH), ("SEC-TAIL2", SEC_TAIL2)):
        s = _sec(res, key)
        for arm in (CTRL, PHASE):
            for i, seed in enumerate(SB.SEEDS):
                assert s["values"][C.run_name(arm, seed)] == vals[arm][i], (key, arm, seed)
        assert s["p_as_counted"] == s["p_filled"]
    assert "SUPPORTED (p <= 0.05, unadjusted) (survives Bonferroni over the 3 efficacy secondaries)" in text
    assert ("SUPPORTED (p <= 0.05, unadjusted) (guard: the habit may be moving to tail_2; a warning, not support for "
            "the phase input)") in text
    assert "NOT SUPPORTED" in text and res["line"] == 2


def test_sec_mech_reads_the_effective_onpolicy_slope_at_E(tmp_path):
    """The phase arm's loop rows carry raw = effective + 0.05 and settled = effective + 0.07 and the other sets 0.01:
    SEC-MECH's values are the effective ONPOLICY ones at E exactly (main screen)."""
    _main_screen(tmp_path)
    res, _t, _ = _score(tmp_path)
    mech = _sec(res, "SEC-MECH")
    assert [mech["values"][C.run_name(PHASE, sd)] for sd in SB.SEEDS] == list(SEC_MECH[PHASE])
    m = json.loads((tmp_path / "reads" / f"{PHASE}_2901" / "markers.json").read_text())
    row = SB.loop_rows_at(m, E)[0]["critic_slope_obs6"]["ONPOLICY"]
    assert row["raw"]["dV_dq3_per_deg"] != row["effective"]["dV_dq3_per_deg"] != row["settled"]["dV_dq3_per_deg"]


def test_sec_tail2_reads_column_13_of_the_markers_probe_file(tmp_path):
    _main_screen(tmp_path)
    res, _t, _ = _score(tmp_path)
    t2 = _sec(res, "SEC-TAIL2")
    for arm in (CTRL, PHASE):
        for i, sd in enumerate(SB.SEEDS):
            name = C.run_name(arm, sd)
            path = tmp_path / "runs" / name / PROBE_REL
            with np.load(path) as z:
                u, mu = z["update_timesteps"], z["rollout_mu_mean"].astype(np.float64)
            idx = [int(np.flatnonzero(u == s)[0]) for s in SB.PRIMARY_STEPS]
            want = sum((Fraction(float(mu[j, 13])) for j in idx), Fraction(0)) / 8
            assert t2["values_exact"][name] == str(want) and float(want) == SEC_TAIL2[arm][i]
            assert res["probe_files"][name] == {"file": str(path), "sha256": _sha(path)}


def test_secondaries_never_change_the_line(tmp_path):
    """The same primaries with the secondaries reversed: the line, its p and its additions are identical."""
    _main_screen(tmp_path / "a")
    rev = {}
    for arm, other in ((CTRL, PHASE), (PHASE, CTRL)):
        for i, seed in enumerate(SB.SEEDS):
            rev[(arm, seed)] = {"cmd": SEC_CMD[other][i], "b": SEC_B[other][i], "mech": SEC_MECH[other][i],
                                "tail2": SEC_TAIL2[other][i]}
    _main_screen(tmp_path / "b", per_seed_extra=rev)
    ra, _ta, _ = _score(tmp_path / "a")
    rb, _tb, _ = _score(tmp_path / "b")
    assert [s["verdict"] for s in ra["secondaries"]] != [s["verdict"] for s in rb["secondaries"]]
    for k in ("line", "line_key", "line_text", "additions", "p_filled", "p_as_counted", "informative", "control_rise"):
        assert ra[k] == rb[k], k
    assert list(SB.decide.__code__.co_varnames[:SB.decide.__code__.co_argcount]) == ["classes", "primaries"]


def test_secondary_fill_works_against_its_direction(tmp_path):
    s = _main_screen(tmp_path)
    s.lost(C.run_name(PHASE, 2931))
    res, _t, _ = _score(tmp_path)
    mar, cmd, mech, t2 = (_sec(res, k) for k in ("SEC-MARGIN", "SEC-CMD", "SEC-MECH", "SEC-TAIL2"))
    f = lambda xs: [(0, Fraction(x)) for x in xs]  # noqa: E731
    # phase HIGHER: a MISSING phase value is -inf
    assert mar["values"][f"{PHASE}_2931"] is None and t2["values"][f"{PHASE}_2931"] is None
    assert mar["p_filled"]["count"] == SB.perm_test(f(SEC_B[PHASE][:3]) + [SB.NEG_INF], f(SEC_B[CTRL]))[1]
    assert t2["p_filled"]["count"] == SB.perm_test(f(SEC_TAIL2[PHASE][:3]) + [SB.NEG_INF], f(SEC_TAIL2[CTRL]))[1] == 36
    assert not t2["supported"] and t2["notes"] == []
    # phase LOWER: a MISSING phase value is +inf
    assert cmd["p_filled"]["count"] == SB.perm_test(f(SEC_CMD[CTRL]), f(SEC_CMD[PHASE][:3]) + [SB.POS_INF])[1] == 37
    assert mech["p_filled"]["count"] == SB.perm_test(f(SEC_MECH[CTRL]), f(SEC_MECH[PHASE][:3]) + [SB.POS_INF])[1] == 36
    # a MISSING control value: in SEC-MECH (control higher) -inf, in SEC-TAIL2 (phase higher) +inf
    s2 = _main_screen(tmp_path / "c")
    s2.lost(C.run_name(CTRL, 2901))
    r2 = _score(tmp_path / "c")[0]
    assert _sec(r2, "SEC-MECH")["p_filled"]["count"] == SB.perm_test(f(SEC_MECH[CTRL][1:]) + [SB.NEG_INF],
                                                                     f(SEC_MECH[PHASE]))[1] == 36
    assert _sec(r2, "SEC-TAIL2")["p_filled"]["count"] == SB.perm_test(f(SEC_TAIL2[PHASE]),
                                                                      f(SEC_TAIL2[CTRL][1:]) + [SB.POS_INF])[1]


def test_secondary_missing_causes(tmp_path):
    extra = {(PHASE, 2911): {"nan_at": SB.PRIMARY_STEPS[0]},                 # UNREADABLE: every secondary MISSING
             (CTRL, 2921): {"final": False, "drop_E": True},                 # no loop row at E, no evaluation at E
             (CTRL, 2931): {"cmd": None, "mech": float("nan"), "tail2_nan": True}}   # not finite
    _main_screen(tmp_path, per_seed_extra=extra)
    res, text, _ = _score(tmp_path)
    for s in res["secondaries"]:
        assert s["values"][f"{PHASE}_2911"] is None and s["causes"][f"{PHASE}_2911"].startswith("UNREADABLE")
    cmd, mar, mech, t2 = (_sec(res, k) for k in ("SEC-CMD", "SEC-MARGIN", "SEC-MECH", "SEC-TAIL2"))
    assert mar["causes"]["control_2921"] == "0 M-LOOP rows at E = 3,538,944 (the final pair)"
    assert mech["causes"]["control_2921"] == mar["causes"]["control_2921"]
    assert cmd["causes"]["control_2921"] == "the evaluation at E = 3,538,944 is absent from the ledger markers"
    assert t2["values"]["control_2921"] == SEC_TAIL2[CTRL][2]                # the probe file does not need the row
    assert cmd["causes"]["control_2931"] == "CMD_L at E is None (no finite value)"
    assert mech["causes"]["control_2931"] == "the effective on-policy critic slope at E is nan (no finite value)"
    assert t2["causes"]["control_2931"] == "rollout_mu_mean[:, 13] is not finite on the primary's rows"
    assert mar["values"]["control_2931"] == SEC_B[CTRL][3]
    assert res["line"] == 3 and res["n_missing"] == 1
    assert "MISSING, filled at -inf: 0 M-LOOP rows at E" in text


def test_secondaries_carry_the_note_under_line_1(tmp_path):
    rises = (0.05, 0.06, 0.07, 0.90)
    s = Screen(tmp_path)
    s.all([f32(B_S + x) for x in rises], [f32(B_S - 0.2 + x / 100) for x in range(4)], per_seed=_per_seed_main())
    res, text, _ = _score(tmp_path)
    assert res["line"] == 1 and res["additions"] == [SB.ADDITION_NOT_INFORMATIVE]
    for sec in res["secondaries"]:
        assert sec["notes"][-1] == SB.SEC_UNDER_LINE1
    assert _sec(res, "SEC-MECH")["notes"] == [SB.SEC_BONFERRONI, SB.SEC_UNDER_LINE1]
    assert _sec(res, "SEC-TAIL2")["notes"] == [SB.SEC_GUARD, SB.SEC_UNDER_LINE1]
    assert text.count(SB.SEC_UNDER_LINE1) == 4


def test_bonferroni_note_only_at_1_over_70_and_only_in_the_efficacy_family():
    classes = _classes()
    for count, ctrl, phase in ((1, (5, 6, 7, 8), (1, 2, 3, 4)), (2, (4.5, 6, 7, 8), (1, 2, 3, 5.5)),
                               (3, (8, 7, 6, 3.5), (5, 4, 2, 1))):
        vals = {}
        for i, sd in enumerate(SB.SEEDS):
            vals[C.run_name(CTRL, sd)] = Fraction(ctrl[i])
            vals[C.run_name(PHASE, sd)] = Fraction(phase[i])
        mech = SB.secondary_test(SB.SECONDARIES[2], classes, vals, {}, 2)        # SEC-MECH: control higher
        assert mech["filled"][1] == count and mech["supported"] is True
        assert mech["notes"] == ([SB.SEC_BONFERRONI] if count == 1 else [])
        swapped = {n: (vals[C.run_name(PHASE if n.startswith(CTRL) else CTRL, int(n.rsplit("_", 1)[1]))])
                   for n in vals}
        guard = SB.secondary_test(SB.SECONDARIES[3], classes, swapped, {}, 2)    # SEC-TAIL2: phase higher
        assert guard["filled"][1] == count and guard["supported"] is True and guard["notes"] == [SB.SEC_GUARD]


def test_counted_without_the_final_pair_is_provisional(tmp_path):
    Screen(tmp_path)
    crit, screen, appr = _approve(tmp_path)
    _main_screen(tmp_path, per_seed_extra={(CTRL, 2921): {"final": False}},
                 meta_kw={"crit": crit, "screen": screen, "appr": appr})
    res, _text, code = _score(tmp_path)
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL
    assert res["provisional"] == ["control_2921: COUNTED but its markers hold 0 loop rows at E = 3,538,944 (the final "
                                  "pair was not read: a tool or file defect; SEC-MARGIN and SEC-MECH MISSING)"]
    assert _sec(res, "SEC-MECH")["values"]["control_2921"] is None and res["line"] == 2
    root2 = tmp_path / "with_row"
    Screen(root2)
    crit2, screen2, appr2 = _approve(root2)
    _main_screen(root2, meta_kw={"crit": crit2, "screen": screen2, "appr": appr2})
    assert _score(root2)[0]["provisional"] == []


def test_secondary_needs_its_row_at_E_to_be_the_model_at_E(tmp_path):
    _main_screen(tmp_path, per_seed_extra={(CTRL, 2901): {"num_ts_E": E - R8}})
    with pytest.raises(SB.InputError, match="num_timesteps"):
        SB.score(tmp_path)


# ------------------------------------------------------------------------------------------------ SEC-TAIL2's file
def test_probe_file_changed_since_the_read_is_missing_and_provisional(tmp_path):
    Screen(tmp_path)
    crit, screen, appr = _approve(tmp_path)
    _main_screen(tmp_path, meta_kw={"crit": crit, "screen": screen, "appr": appr})
    path = tmp_path / "runs" / f"{PHASE}_2911" / PROBE_REL
    _probe_file(path, LOW[1], E, tail2=0.5)                 # column 14 as before, column 13 changed
    res, text, code = _score(tmp_path)
    t2 = _sec(res, "SEC-TAIL2")
    assert t2["values"][f"{PHASE}_2911"] is None and "changed since the read" in t2["causes"][f"{PHASE}_2911"]
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL
    assert [x for x in res["provisional"] if "changed since the read" in x and "SEC-TAIL2 MISSING" in x]
    f = lambda xs: [(0, Fraction(x)) for x in xs]  # noqa: E731
    vals = [v for i, v in enumerate(SEC_TAIL2[PHASE]) if i != 1]
    assert t2["p_filled"]["count"] == SB.perm_test(f(vals) + [SB.NEG_INF], f(SEC_TAIL2[CTRL]))[1]
    assert res["line"] == 2 and _sec(res, "SEC-MECH")["values"][f"{PHASE}_2911"] == SEC_MECH[PHASE][1]
    path.unlink()                                           # missing: the same, unreadable
    res2, _t2, _c2 = _score(tmp_path)
    assert "unreadable" in _sec(res2, "SEC-TAIL2")["causes"][f"{PHASE}_2911"]
    assert [x for x in res2["provisional"] if "is unreadable" in x and "SEC-TAIL2 MISSING" in x]


def test_probe_file_that_is_not_the_markers_is_an_input_error(tmp_path):
    _main_screen(tmp_path)
    name = f"{PHASE}_2921"
    path = tmp_path / "runs" / name / PROBE_REL
    _probe_file(path, LOW[2], E, tail2=SEC_TAIL2[PHASE][2], mu14_at=(SB.PRIMARY_STEPS[5], 0.875))
    new = _sha(path)
    for p, keys in ((tmp_path / "reads" / name / "markers.json", ("probe", "sha256")),
                    (tmp_path / "runs" / name / C.CHECK_JSON, None)):
        d = json.loads(p.read_text())
        if keys:
            d["probe"]["sha256"] = new
        else:
            d["sources"]["policy_probe.npz"][1] = new
        _write(p, d)
    with pytest.raises(SB.InputError, match="not the markers' rollout_mu14_mean"):
        SB.score(tmp_path)


def test_probe_file_with_the_wrong_shape_is_an_input_error(tmp_path):
    _main_screen(tmp_path)
    name = f"{CTRL}_2911"
    path = tmp_path / "runs" / name / PROBE_REL
    with open(path, "wb") as fh:
        np.savez(fh, update_timesteps=np.arange(3, dtype=np.int64), rollout_mu_mean=np.zeros((3, 14), np.float32))
    for p in (tmp_path / "reads" / name / "markers.json", tmp_path / "runs" / name / C.CHECK_JSON):
        d = json.loads(p.read_text())
        if "probe" in d:
            d["probe"]["sha256"] = _sha(path)
        else:
            d["sources"]["policy_probe.npz"][1] = _sha(path)
        _write(p, d)
    with pytest.raises(SB.InputError, match="shape"):
        SB.score(tmp_path)


# ------------------------------------------------------------------------------------------------ outcome-blind
def test_incomplete_opens_no_markers_run_meta_or_probe_and_prints_no_outcome(tmp_path):
    _main_screen(tmp_path)
    (tmp_path / "runs" / f"{PHASE}_2931" / C.CHECK_JSON).unlink()           # still running
    (tmp_path / "reads" / "control_2901" / "markers.json").write_text("not json {")   # never opened
    (tmp_path / "runs" / "control_2911" / C.RUN_META).write_text("not json {")      # never opened
    (tmp_path / "runs" / "control_2921" / PROBE_REL).write_bytes(b"not an npz")     # never opened
    res, text, code = _score(tmp_path)
    assert res["status"] == "incomplete" and code == SB.EXIT_NOT_FINAL
    body = text.split("STATUS")[1].split("PROVISIONAL")[0]
    assert "THE READING" not in text and "SEC-" not in text and "0.4" not in body and "0.1" not in body
    assert f"{PHASE}_2931" in text and "undecided" in text and "line" not in res


def test_decided_but_unread_is_incomplete(tmp_path):
    _main_screen(tmp_path)
    (tmp_path / "reads" / "control_2911" / "markers.json").unlink()
    (tmp_path / "runs" / "control_2901" / C.RUN_META).write_text("not json {")
    res, text, code = _score(tmp_path)
    assert res["status"] == "incomplete" and "decided, markers not read yet" in text and code == 3


def test_check_failed_is_undecided(tmp_path):
    s = _main_screen(tmp_path)
    s.run(PHASE, 2921, LOW[2], passed=False)
    res, text, _ = _score(tmp_path)
    assert res["status"] == "incomplete" and "CHECK failed" in text


# ------------------------------------------------------------------------------------------------ input errors
def test_input_errors(tmp_path):
    _main_screen(tmp_path)
    assert _score(tmp_path)[2] in (0, 3)
    d = json.loads((tmp_path / "SCREEN.json").read_text())
    d["queue"] = d["queue"][:7]
    _write(tmp_path / "SCREEN.json", d)
    with pytest.raises(SB.InputError, match="not the criteria's"):
        SB.score(tmp_path)
    assert SB.main(["--root", str(tmp_path)]) == SB.EXIT_INPUT
    d["queue"] = [list(q) for q in ((a, s - 400) for a, s in SB.QUEUE)]        # LB's draft seeds 2501-2531
    _write(tmp_path / "SCREEN.json", d)
    with pytest.raises(SB.InputError, match="not the criteria's"):
        SB.score(tmp_path)


@pytest.mark.parametrize("who, mutate, match", [
    (PHASE, lambda m: m.update(arm="control"), "arm"),
    (PHASE, lambda m: m["probe"].update(sha256="cd" * 32), "another policy_probe.npz"),
    (PHASE, lambda m: m["loop"][0].update(critic_form="plain"), "critic form"),
    (PHASE, lambda m: m["loop"][1].pop("phase"), "no phase record"),
    (PHASE, lambda m: m["loop"][-1]["critic_slope_obs6"]["ONPOLICY"].pop("settled"), "settled"),
    (PHASE, lambda m: m["probe_phase"].update(installed=False), "probe_phase"),
    (CTRL, lambda m: m["loop"][0].update(critic_form="phase+raw+settled"), "critic form"),
    (CTRL, lambda m: m["loop"][1].update(phase={"start_transform": False}), "carries a phase record"),
    (CTRL, lambda m: m["loop"][-1]["critic_slope_obs6"]["ONPOLICY"]["effective"].update(dV_dq3_per_deg=0.003),
     "not the stock"),
    (CTRL, lambda m: m["loop"][2]["critic_slope_obs6"]["OWN"].update(settled={"dV_dq3_per_deg": 0.1}), "not the stock"),
    (CTRL, lambda m: m.pop("probe_phase"), "probe_phase"),
    (CTRL, lambda m: m["loop"][0]["critic_slope_obs6"].update(EXTRA={}), "sets"),
    (PHASE, lambda m: m["probe"]["steps"].pop(), "probe rows"),
    (PHASE, lambda m: m.update(T=E - R8), "T"),
    (PHASE, lambda m: m.update(S=1048576), "S"),
    (PHASE, lambda m: m["loop"][-1].update(num_timesteps=E + R8), "num_timesteps"),
])
def test_markers_that_disagree_are_input_errors(tmp_path, who, mutate, match):
    _main_screen(tmp_path)
    p = tmp_path / "reads" / f"{who}_2901" / "markers.json"
    m = json.loads(p.read_text())
    mutate(m)
    _write(p, m)
    with pytest.raises(SB.InputError, match=match):
        SB.score(tmp_path)


def test_lost_outside_queue_and_check_mismatch(tmp_path):
    s = _main_screen(tmp_path)
    s.lost("control_9999")
    with pytest.raises(SB.InputError, match="outside the queue"):
        SB.score(tmp_path)
    (tmp_path / "lost.json").unlink()
    p = tmp_path / "runs" / "control_2901" / C.CHECK_JSON
    c = json.loads(p.read_text())
    c["seed"] = 2911
    _write(p, c)
    with pytest.raises(SB.InputError, match="seed"):
        SB.score(tmp_path)
    c["seed"], c["E"] = 2901, 2490368
    _write(p, c)
    with pytest.raises(SB.InputError, match="E"):
        SB.score(tmp_path)


def test_out_outside_root_refused(tmp_path):
    _main_screen(tmp_path / "root")
    assert SB.main(["--root", str(tmp_path / "root"), "--out", str(tmp_path / "elsewhere")]) == SB.EXIT_INPUT
    assert not (tmp_path / "elsewhere").exists()


# ------------------------------------------------------------------------------------------------ PROVISIONAL / final
def _approve(root, scorer_sha=None):
    crit = root / "criteria.txt"
    crit.write_text(f"criteria ... tools/score_bs.py {scorer_sha or _sha(TOOLS / 'score_bs.py')}\n")
    appr = {"criteria_sha256": _sha(crit), "screen_sha256": _sha(root / "SCREEN.json"),
            "code_tree_digest": C.TREE_DIGEST,
            "tool_sha256": {t: _sha(TOOLS / t) for t in SB.PINNED_TOOLS}, "approved_by": "test"}
    _write(root / "APPROVED", appr)
    return _sha(crit), _sha(root / "SCREEN.json"), _sha(root / "APPROVED")


def test_final_when_every_pin_holds(tmp_path):
    Screen(tmp_path)
    crit, screen, appr = _approve(tmp_path)
    _main_screen(tmp_path, meta_kw={"crit": crit, "screen": screen, "appr": appr})
    res, _text, code = _score(tmp_path)
    assert res["provisional"] == [] and res["status"] == "final" and code == SB.EXIT_FINAL
    assert SB.main(["--root", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "score" / "score.json").read_text())
    assert out["line_key"] == "REDUCED" and (tmp_path / "score" / "score.txt").read_text().startswith(SB.VERSION)
    assert [s["key"] for s in out["secondaries"]] == ["SEC-CMD", "SEC-MARGIN", "SEC-MECH", "SEC-TAIL2"]
    assert out["secondaries"][2]["p_filled"]["count"] == 1 and out["secondaries"][3]["notes"] == [SB.SEC_GUARD]


@pytest.mark.parametrize("breakage, match", [
    ("criteria", "criteria.txt sha256"), ("scorer_pin", "does not name this scorer"),
    ("run_meta_criteria", "ran under criteria.txt"), ("tool", "tools/score_bs.py"), ("ledger", "ledger"),
    ("approved_missing", "no "), ("run_meta_changed", "run_meta.json changed since CHECK"),
    ("markers_tool", "markers.json made by read_bs.py"), ("check_tool", "check.json made by read_bs.py"),
    ("code_copy", "code copy"), ("probe_changed", "SEC-TAIL2 MISSING"),
])
def test_provisional_items(tmp_path, monkeypatch, breakage, match):
    Screen(tmp_path)
    crit, screen, appr = _approve(tmp_path, scorer_sha="0" * 64 if breakage == "scorer_pin" else None)
    _main_screen(tmp_path, meta_kw={"crit": "f" * 64 if breakage == "run_meta_criteria" else crit,
                                    "screen": screen, "appr": appr})
    if breakage == "criteria":
        (tmp_path / "criteria.txt").write_text("edited after approval\n" + (tmp_path / "criteria.txt").read_text())
    if breakage == "tool":
        a = json.loads((tmp_path / "APPROVED").read_text())
        a["tool_sha256"]["score_bs.py"] = "e" * 64
        _write(tmp_path / "APPROVED", a)
    if breakage == "ledger":
        with open(tmp_path / "runs" / "control_2901" / C.STAGE_DIR / "eval_episodes.csv", "a") as fh:
            fh.write("99,3538944,10,0,1.0,1000,1,truncated\n")
    if breakage == "approved_missing":
        (tmp_path / "APPROVED").unlink()
    if breakage == "run_meta_changed":
        p = tmp_path / "runs" / f"{PHASE}_2911" / C.RUN_META
        p.write_text(p.read_text() + " ")
    if breakage == "markers_tool":
        p = tmp_path / "reads" / "control_2921" / "markers.json"
        m = json.loads(p.read_text())
        m["tool_sha256"] = "d" * 64
        _write(p, m)
    if breakage == "check_tool":
        p = tmp_path / "runs" / "control_2921" / C.CHECK_JSON
        m = json.loads(p.read_text())
        m["tool_sha256"] = "d" * 64
        _write(p, m)
    if breakage == "code_copy":
        monkeypatch.setattr(SB, "code_seal_items", lambda d: [f"code copy tree digest x != pinned ({d})"])
    if breakage == "probe_changed":
        with open(tmp_path / "runs" / "control_2931" / PROBE_REL, "ab") as fh:
            fh.write(b"\0")
    res, _text, code = _score(tmp_path)
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL
    assert any(match in x for x in res["provisional"]), res["provisional"]
    assert res["line_key"] == "REDUCED"


def test_code_copy_is_measured_at_bs_common_CODE(tmp_path, _no_code_copy_measurement):
    """PS reuses BS's code copy read only: the scorer measures bs_common.CODE, never <root>/code."""
    _main_screen(tmp_path)
    _score(tmp_path)
    assert _no_code_copy_measurement and set(_no_code_copy_measurement) == {Path(C.CODE)}
    if not Path(C.CODE).is_dir():
        pytest.skip("the code copy is not on this machine")
    spec = importlib.util.spec_from_file_location("score_bs_fresh", str(SRC_PATH))
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.code_seal_items(Path(C.CODE)) == []
    assert fresh.code_seal_items(tmp_path / "code") == [f"code copy {tmp_path / 'code'} missing: not measured"]


# ------------------------------------------------------------------------------------------------ report only
def test_report_only_readings(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    ro = res["report_only"][f"{PHASE}_2901"]
    assert ro["ledger"]["status"] == "read" and ro["ledger"]["n_falls"] == 1 and ro["ledger"]["n_episodes"] == 110
    assert ro["ledger"]["mean_return_at_T"] == pytest.approx(4010.0)
    assert "mean return per evaluation: 2,228,224 4000.0" in text
    mp = ro["manipulation"]
    assert mp["max_dmu_phase"] == 0.0 and mp["phase_n_zero_init"] == 1 and mp["phase_n_updated"] == 174
    assert mp["mlp_extractor_class"] == "CriticPhaseMlpExtractor" and mp["transform_verified"] is True
    assert res["report_only"]["control_2901"]["manipulation"]["phase_rows_recorded"] == 0
    assert res["report_only"]["control_2901"]["ledger"]["n_falls"] == 0
    assert "MANIPULATION" in text and "CriticPhaseMlpExtractor" in text and "PAIRED DIFFERENCES" in text
    assert [p["diff"] for p in res["paired"]] == pytest.approx([0.30, 0.30, 0.30, 0.30], abs=1e-6)
    sd = statistics.stdev(HIGH)
    assert res["arm_sd_report_only"]["control"] == pytest.approx(sd)
    assert f"SD {sd:.5f} (report only)" in text
    cmd = ro["cmd_l_per_evaluation"]
    assert sorted(cmd) == list(SB.EVAL_STEPS) and cmd[E] == SEC_CMD[PHASE][0]


def test_report_only_phase_swing_and_slopes(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    loop = res["report_only"][f"{PHASE}_2911"]["loop"]
    assert [lr["steps"] for lr in loop] == [S, *C.SAVED_STEPS, E]
    end = loop[-1]
    assert end["swing_onpolicy"] == 0.25 and end["swing_own_phase_onpolicy"] == 0.125
    assert end["slope_onpolicy_eff"] == SEC_MECH[PHASE][1]
    assert end["slope_onpolicy_raw"] == pytest.approx(SEC_MECH[PHASE][1] + 0.05)
    assert end["slope_onpolicy_settled"] == pytest.approx(SEC_MECH[PHASE][1] + 0.07)
    assert loop[0]["phase_start_transform"] is True and loop[0]["phase_cols_used"] == 0.0
    ctrl_end = res["report_only"]["control_2911"]["loop"][-1]
    assert ctrl_end["swing_onpolicy"] is None and ctrl_end["slope_onpolicy_settled"] is None
    assert "phase swing (V settled - V slots 0, on-policy) +0.2500" in text


def test_report_only_actuator_rises(tmp_path):
    pm, cause = SB.parent_actuator_means()
    if pm is None:
        pytest.skip(f"the parent's F2 probe is not readable here ({cause})")
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    act = res["report_only"][f"{PHASE}_2901"]["actuators"]
    assert act["rise"][13] == Fraction(SEC_TAIL2[PHASE][0]) - pm[13]
    assert act["rise"][14] == Fraction(LOW[0]) - pm[14]
    assert act["rise"][0] == Fraction(OTHER_COL[0]) - pm[0]
    arise = [abs(Fraction(OTHER_COL[j] if j not in (13, 14) else (SEC_TAIL2[PHASE][0] if j == 13 else LOW[0])))
             - abs(pm[j]) for j in range(15)]
    jmax = max((j for j in range(15) if j != 14), key=lambda j: arise[j])
    assert act["largest_other_name"] == SB.ACTUATORS[jmax] and act["largest_other_abs_rise"] == arise[jmax]
    assert "largest |.|-rise among the 14 others" in text
    assert res["actuator_rise_arm_medians"]["control"][13] == pytest.approx(
        statistics.median([float(Fraction(v) - pm[13]) for v in SEC_TAIL2[CTRL]]))


def test_report_only_beside_bs2(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    b = res["beside_bs2"]
    path = Path(C.BS2_DIR) / "score" / "score.json"
    if not path.exists():
        assert b["status"].endswith("absent")
        return
    d = json.loads(path.read_text())
    assert b["status"] == "read" and b["sha256"] == _sha(path) and b["line_key"] == d["line_key"] == "REDUCED"
    assert b["primaries"] == {c["name"]: c["primary"] for c in d["classes"]}
    assert "REPORT ONLY, BESIDE BS2" in text and "BS2 control_2501" in text


def test_trajectory_and_g_count(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    ro = res["report_only"]["control_2911"]
    traj = dict(ro["trajectory"])
    assert sorted(traj) == list(SB.EVAL_STEPS)
    assert traj[E] == Fraction(HIGH[1])
    m = json.loads((tmp_path / "reads" / "control_2911" / "markers.json").read_text())
    probe = m["probe"]
    idx = [probe["steps"].index(2752512 - R8 * k) for k in range(8)]
    assert traj[2752512] == sum((Fraction(probe["rollout_mu14_mean"][i]) for i in idx), Fraction(0)) / 8
    assert "REPORT ONLY, THE ON-POLICY TRAJECTORY" in text and f"{SB.PARENT_TRAJECTORY[E]:+.4f}" in text
    assert res["g_end_above_start"] == {"control": {"above": 0, "of": 4}, PHASE: {"above": 4, "of": 4}}
    assert "control 0 of 4; critic_phase 4 of 4" in text
    assert res["trajectory_arm_medians"][str(E)]["control"] == pytest.approx(statistics.median(HIGH))
    assert ro["end_pair"]["steps"] == E and ro["end_pair"]["OWN_G"] == SEC_G[CTRL][1]


def test_report_only_never_fails_the_score(tmp_path):
    _main_screen(tmp_path)
    p = tmp_path / "reads" / f"{PHASE}_2901" / "markers.json"
    m = json.loads(p.read_text())
    m["loop"][0]["critic_slope_obs6"]["REF"]["effective"] = {"dV_dq3_per_deg": None}
    led = Path(m["ledger"]["file"])
    led.write_bytes(b"\xff\xfe not utf-8")
    m["ledger"]["sha256"] = _sha(led)
    _write(p, m)
    res, _text, _code = _score(tmp_path)
    assert res["line"] == 2
    assert res["report_only"][f"{PHASE}_2901"]["ledger"]["status"].startswith("unreadable (UnicodeDecodeError")


# ------------------------------------------------------------------------------------------------ the criteria text
def test_line_texts_are_the_criteria_text():
    """The scorer's line texts, additions and secondary wordings appear verbatim (whitespace collapsed) in
    <PS>/criteria.txt (or the draft it is installed from)."""
    text = _criteria_text()
    if text is None:
        pytest.skip("no criteria text in this build")
    for line, t in SB.LINE_TEXT.items():
        assert f'"{" ".join(t.split())}"' in text, line
    for t in (SB.ADDITION_AS_COUNTED, SB.ADDITION_NOT_INFORMATIVE, SB.SEC_SUPPORTED, SB.SEC_BONFERRONI, SB.SEC_GUARD,
              SB.SEC_UNDER_LINE1, '"NOT SUPPORTED"'):
        assert t in text, t
    for s in SB.SECONDARIES:
        assert s["key"] in text
    assert "B_S = 0.1265954738482833 (exactly 135930855/1073741824)" in text and "at least 0.20" in text
    assert "2901, 2911, 2921, 2931" in text
    assert "critic_slope_obs6.ONPOLICY.effective.dV_dq3_per_deg" in text


PIN_FILES = ("SCREEN.json", "tools/score_bs.py", "tools/tests/test_score_bs.py", "tools/bs_common.py",
             "tools/bs_hooks.py", "tools/run_bs.py", "tools/run_waves_bs.sh", "tools/read_bs.py")


def test_criteria_pins_name_the_current_files():
    """PINS: the installed criteria.txt names each pinned file's current sha256 (a change to a pinned file changes its
    line; P2: the re-reviewed versions are the pinned ones)."""
    crit = PS / "criteria.txt"
    if not crit.exists():
        pytest.skip("criteria.txt not installed")
    raw = crit.read_text(encoding="utf-8")
    pins = raw.split("\nPINS")[1].split("\nHOW THE COORDINATOR PINS")[0]
    for rel in PIN_FILES:
        m = re.search(rf"^\s+{re.escape(rel)}\s+([0-9a-f]{{64}})\b", pins, re.M)
        assert m, rel
        assert m.group(1) == _sha(PS / rel), rel


# ------------------------------------------------------------------------------------------------ the dry run's form
def _parent_as(m, arm, seed):
    """The parent's reference markers relabelled as one continuation (the criteria's dry run): its probe, ledger and
    loop rows unchanged, except the phase arm's loop rows carry the arm's form (critic_form, a phase record, the
    settled slope = the raw one: the start transform's zero columns) and every entry its probe_phase."""
    m = json.loads(json.dumps(m))
    name = C.run_name(arm, seed)
    m.update(kind="continuation", label=name, arm=arm, seed=seed, stop_reason="end",
             probe_phase={"arm": arm, "installed": arm == PHASE})
    m.pop("reproduction", None)
    if arm == PHASE:
        for row in m["loop"]:
            row["critic_form"] = "phase+raw+settled"
            row["phase"] = {"start_transform": row["steps"] == S, "value_phase_cols_max_abs_used": 0.0,
                            "actor_phase_invariance_max_abs": 0.0, "dry_run": "relabelled parent row"}
            for d in row["critic_slope_obs6"].values():
                d["settled"] = dict(d["raw"])
    return m


def build_parent_root(root):
    """The dry run's root: the parent's markers as every queue entry (both arms)."""
    pm = json.loads(PARENT_MARKERS.read_text())
    s = Screen(root)
    for arm, seed in SB.QUEUE:
        name = C.run_name(arm, seed)
        rd = Path(root) / "runs" / name
        meta = _write(rd / C.RUN_META, _meta(arm, seed, root))
        _write(rd / C.CHECK_JSON, {
            "schema": R.CHECK_SCHEMA, "tool_sha256": _sha(TOOLS / "read_bs.py"), "name": name, "mode": "continuation",
            "passed": True, "failures": [], "arm": arm, "seed": seed, "S": S, "E": E, "T": E, "end": E,
            "stop_reason": "end", "sources": {"run_meta.json": [str(meta), _sha(meta)],
                                              "policy_probe.npz": [pm["probe"]["file"], pm["probe"]["sha256"]]}})
        _write(Path(root) / "reads" / name / "markers.json", _parent_as(pm, arm, seed))
    return s, pm


def test_dry_run_on_the_parents_markers(tmp_path):
    if not PARENT_MARKERS.exists():
        pytest.skip("the builder's reference read is not on this machine")
    _s, pm = build_parent_root(tmp_path)
    res, _text, _ = _score(tmp_path)
    assert res["line"] == 3 and res["p_filled"]["count"] == 70 and res["informative"] is True
    assert res["control_rise"] == float(Fraction(47337175, 67108864) - Fraction(135930855, 1073741824))
    assert round(res["control_rise"], 5) == 0.57878
    assert all(c["primary"] == SB.PARENT_TRAJECTORY[E] for c in res["classes"])
    for key in ("SEC-CMD", "SEC-TAIL2"):
        sec = _sec(res, key)
        assert sec["verdict"] == SB.SEC_NOT_SUPPORTED and sec["p_filled"]["count"] == 70
        assert len(set(sec["values"].values())) == 1 and None not in sec["values"].values()
    assert all(v == pm["ledger"]["direct"]["per_evaluation"][str(E)]["cmd_l"]
               for v in _sec(res, "SEC-CMD")["values"].values())
    assert round(next(iter(_sec(res, "SEC-TAIL2")["values"].values())), 4) == 0.0334     # the parent's tail_2 at E
    for key in ("SEC-MARGIN", "SEC-MECH"):
        assert set(_sec(res, key)["values"].values()) == {None}
        assert set(_sec(res, key)["causes"].values()) == {"0 M-LOOP rows at E = 3,538,944 (the final pair)"}
        assert _sec(res, key)["p_filled"]["count"] == 70
    final_items = [x for x in res["provisional"] if "the final pair was not read" in x]
    assert sorted(x.split(":")[0] for x in final_items) == sorted(C.run_name(a, sd) for a, sd in SB.QUEUE)


# ------------------------------------------------------------------------------------------------ mutation tests
SRC = SRC_PATH.read_text(encoding="utf-8")


def _variant(src, tag):
    mod = types.ModuleType(f"score_bs_mutant_{tag}")
    mod.__file__ = str(SRC_PATH)
    exec(compile(src, str(SRC_PATH), "exec"), mod.__dict__)  # noqa: S102 - the scorer's own source, one edit
    return mod


def _markers_edit(root, name, fn):
    p = Path(root) / "reads" / name / "markers.json"
    m = json.loads(p.read_text())
    fn(m)
    _write(p, m)


@pytest.fixture(scope="module")
def kill_roots(tmp_path_factory):
    """The synthetic screens the kill checks read (built once; the mutants only read them)."""
    base = tmp_path_factory.mktemp("kill")
    roots = {}
    roots["main"] = base / "main"                       # line 2; CMD 2/70, MARGIN 24/70, MECH 1/70, TAIL2 3/70
    _main_screen(roots["main"])
    roots["notinf"] = base / "notinf"                   # line 1 (median 0.065, mean 0.27)
    Screen(roots["notinf"]).all([f32(B_S + x) for x in (0.05, 0.06, 0.07, 0.90)],
                                [f32(B_S - 0.2 + x / 100) for x in range(4)], per_seed=_per_seed_main())
    roots["lost"] = base / "lost"                       # line 3, 36/70, the as-counted addition
    _main_screen(roots["lost"]).lost(C.run_name(PHASE, 2931))
    roots["odd"] = base / "odd"                         # UNREADABLE, rows absent, not finite
    _main_screen(roots["odd"], per_seed_extra={(PHASE, 2911): {"nan_at": SB.PRIMARY_STEPS[0]},
                                               (CTRL, 2921): {"final": False, "drop_E": True},
                                               (CTRL, 2931): {"cmd": None, "mech": float("nan")}})
    roots["unread"] = base / "unread"
    _main_screen(roots["unread"])
    (roots["unread"] / "reads" / "control_2911" / "markers.json").unlink()
    roots["undecided"] = base / "undecided"
    _main_screen(roots["undecided"])
    (roots["undecided"] / "runs" / f"{PHASE}_2931" / C.CHECK_JSON).unlink()
    (roots["undecided"] / "reads" / "control_2901" / "markers.json").write_text("not json {")
    roots["badrow"] = base / "badrow"                   # the row at E is not the model at E
    _main_screen(roots["badrow"], per_seed_extra={(CTRL, 2901): {"num_ts_E": E - R8}})
    roots["changed"] = base / "changed"                 # a probe file changed after the read (column 13 only)
    _main_screen(roots["changed"])
    _probe_file(roots["changed"] / "runs" / f"{PHASE}_2911" / PROBE_REL, LOW[1], E, tail2=0.5)
    roots["notmarkers"] = base / "notmarkers"           # the probe file is not the markers' (column 14 differs)
    _main_screen(roots["notmarkers"])
    nm = f"{PHASE}_2921"
    path = roots["notmarkers"] / "runs" / nm / PROBE_REL
    _probe_file(path, LOW[2], E, tail2=SEC_TAIL2[PHASE][2], mu14_at=(SB.PRIMARY_STEPS[5], 0.875))
    _markers_edit(roots["notmarkers"], nm, lambda m: m["probe"].update(sha256=_sha(path)))
    cj = roots["notmarkers"] / "runs" / nm / C.CHECK_JSON
    d = json.loads(cj.read_text())
    d["sources"]["policy_probe.npz"][1] = _sha(path)
    _write(cj, d)
    form_cases = {"form_phase_plain": (PHASE, lambda m: m["loop"][0].update(critic_form="plain")),
                  "form_ctrl_eff": (CTRL, lambda m: m["loop"][-1]["critic_slope_obs6"]["ONPOLICY"]["effective"]
                                    .update(dV_dq3_per_deg=0.003)),
                  "form_probe_phase": (PHASE, lambda m: m["probe_phase"].update(installed=False)),
                  "form_no_phase_record": (PHASE, lambda m: m["loop"][2].pop("phase")),
                  "form_no_settled": (PHASE, lambda m: m["loop"][-1]["critic_slope_obs6"]["ONPOLICY"].pop("settled")),
                  "form_ctrl_phase_record": (CTRL, lambda m: m["loop"][1].update(phase={}))}
    for key, (who, fn) in form_cases.items():
        roots[key] = base / key
        _main_screen(roots[key])
        _markers_edit(roots[key], f"{who}_2911", fn)
    return roots


def _kill_checks(sb, roots):
    """Decisive checks of score_bs.py's behaviour; returns the names of those that fail (an exception fails)."""
    failed = []
    seen = []
    saved = sb.code_seal_items
    sb.code_seal_items = lambda d: seen.append(Path(d)) or []

    def chk(name, fn):
        try:
            ok = fn()
        except Exception:  # noqa: BLE001 - any exception is a caught mutant
            ok = False
        if not ok:
            failed.append(name)

    def secs(res):
        return {s["key"]: s for s in res["secondaries"]}

    def main_ok():
        res, text, _ = _score(roots["main"], sb)
        s = secs(res)
        traj = dict(res["report_only"]["control_2911"]["trajectory"])
        ph_end = res["report_only"][f"{PHASE}_2901"]["loop"][-1]
        vals_ok = all([s[k]["values"][C.run_name(a, sd)] for a, sd in SB.QUEUE] ==
                      [v[a][SB.SEEDS.index(sd)] for a, sd in SB.QUEUE]
                      for k, v in (("SEC-CMD", SEC_CMD), ("SEC-MARGIN", SEC_B), ("SEC-MECH", SEC_MECH),
                                   ("SEC-TAIL2", SEC_TAIL2)))
        return (res["line"] == 2 and res["p_filled"]["count"] == 1
                and [c["primary_exact"] for c in res["classes"]] == [str(Fraction(v)) for pr in
                                                                     zip(HIGH, LOW, strict=True) for v in pr]
                and s["SEC-CMD"]["p_filled"]["count"] == 2 and s["SEC-CMD"]["notes"] == [] and s["SEC-CMD"]["supported"]
                and s["SEC-MARGIN"]["p_filled"]["count"] == 24 and not s["SEC-MARGIN"]["supported"]
                and s["SEC-MECH"]["p_filled"]["count"] == 1 and s["SEC-MECH"]["notes"] == [SB.SEC_BONFERRONI]
                and s["SEC-TAIL2"]["p_filled"]["count"] == 3 and s["SEC-TAIL2"]["notes"] == [SB.SEC_GUARD]
                and vals_ok and res["provisional"] and not [x for x in res["provisional"] if "SEC-" in x]
                and res["g_end_above_start"]["control"] == {"above": 0, "of": 4}
                and ph_end["swing_onpolicy"] == 0.25 and ph_end["swing_own_phase_onpolicy"] == 0.125
                and traj[E] == Fraction(HIGH[1]) and set(seen) == {Path(C.CODE)}
                and sb.LINE_TEXT[2] in text)

    def notinf_ok():
        res, _t, _ = _score(roots["notinf"], sb)
        return (res["line"] == 1 and res["additions"] == [SB.ADDITION_NOT_INFORMATIVE]
                and all(s["notes"][-1] == SB.SEC_UNDER_LINE1 for s in res["secondaries"])
                and res["control_rise"] == pytest.approx(0.065, abs=1e-6))

    def lost_ok():
        res, _t, _ = _score(roots["lost"], sb)
        s = secs(res)
        return (res["line"] == 3 and res["p_filled"]["count"] == 36 and res["additions"] == [SB.ADDITION_AS_COUNTED]
                and s["SEC-TAIL2"]["p_filled"]["count"] == 36 and s["SEC-MECH"]["p_filled"]["count"] == 36)

    def odd_ok():
        res, _t, _ = _score(roots["odd"], sb)
        s = secs(res)
        return (all(x["values"][f"{PHASE}_2911"] is None and x["causes"][f"{PHASE}_2911"].startswith("UNREADABLE")
                    for x in s.values())
                and s["SEC-MECH"]["values"]["control_2921"] is None
                and s["SEC-MARGIN"]["values"]["control_2921"] is None
                and s["SEC-TAIL2"]["values"]["control_2921"] == SEC_TAIL2[CTRL][2]
                and s["SEC-CMD"]["values"]["control_2931"] is None
                and s["SEC-MECH"]["values"]["control_2931"] is None
                and [x.split(":")[0] for x in res["provisional"] if "the final pair was not read" in x]
                == ["control_2921"])

    def unread_ok():
        res, _t, _ = _score(roots["unread"], sb)
        return res["status"] == "incomplete"

    def undecided_ok():
        res, text, _ = _score(roots["undecided"], sb)
        return res["status"] == "incomplete" and "THE READING" not in text

    def raises(root, words):
        try:
            sb.score(roots[root])
        except sb.InputError as exc:
            return words in str(exc)
        return False

    def changed_ok():
        res, _t, _ = _score(roots["changed"], sb)
        t2 = secs(res)["SEC-TAIL2"]
        return (t2["values"][f"{PHASE}_2911"] is None and "changed since the read" in t2["causes"][f"{PHASE}_2911"]
                and [x for x in res["provisional"] if "SEC-TAIL2 MISSING" in x])

    def decide_ok():
        cl = _classes()
        edge = Fraction(B_S) + Fraction(1, 5)
        out = []
        for ctrl, inf in ((edge, True), (edge - Fraction(1, 10**12), False),
                          (Fraction(B_S) + Fraction(15, 100), False)):
            prim = {C.run_name(a, sd): (ctrl if a == CTRL else Fraction(-1)) for a, sd in SB.QUEUE}
            out.append(sb.decide(cl, prim)["informative"] is inf)
        for c, b, line in (((8, 7, 6, 3.5), (5, 4, 2, 1), 2), ((8, 7, 6, 3), (5, 4, 2, 1), 3)):
            prim = {}
            for i, sd in enumerate(SB.SEEDS):
                prim[C.run_name(CTRL, sd)] = Fraction(0.5 + c[i] / 64)
                prim[C.run_name(PHASE, sd)] = Fraction(0.5 + b[i] / 64)
            out.append(sb.decide(cl, prim)["line"] == line)
        return all(out)

    def guard_ok():
        cl = _classes()
        vals = {C.run_name(a, sd): Fraction(1 if a == CTRL else 2) + Fraction(i, 8)
                for i, (a, sd) in enumerate(SB.QUEUE)}
        t = sb.secondary_test(sb.SECONDARIES[3], cl, vals, {}, 2)
        return t["filled"][1] == 1 and t["notes"] == [SB.SEC_GUARD]

    def ties_ok():
        same = [(0, Fraction(1, 5))] * 4
        return sb.perm_test(same, same)[1] == 70

    def texts_ok():
        text = _criteria_text()
        return text is not None and all(f'"{" ".join(t.split())}"' in text for t in sb.LINE_TEXT.values())

    def baseline_ok():
        return Fraction(sb.BASELINE_S) == Fraction(135930855, 1073741824)

    try:
        for name, fn in (("main", main_ok), ("notinf", notinf_ok), ("lost", lost_ok), ("odd", odd_ok),
                         ("unread", unread_ok), ("undecided", undecided_ok),
                         ("badrow", lambda: raises("badrow", "num_timesteps")), ("changed", changed_ok),
                         ("notmarkers", lambda: raises("notmarkers", "not the markers' rollout_mu14_mean")),
                         ("form_phase_plain", lambda: raises("form_phase_plain", "critic form")),
                         ("form_ctrl_eff", lambda: raises("form_ctrl_eff", "not the stock")),
                         ("form_probe_phase", lambda: raises("form_probe_phase", "probe_phase")),
                         ("form_no_phase_record", lambda: raises("form_no_phase_record", "no phase record")),
                         ("form_no_settled", lambda: raises("form_no_settled", "settled")),
                         ("form_ctrl_phase_record", lambda: raises("form_ctrl_phase_record", "carries a phase")),
                         ("decide", decide_ok), ("guard", guard_ok), ("ties", ties_ok), ("texts", texts_ok),
                         ("baseline", baseline_ok)):
            chk(name, fn)
    finally:
        sb.code_seal_items = saved
    return failed


MUTANTS = [
    ("gate_0_10", "RISE_MIN = Fraction(1, 5)", "RISE_MIN = Fraction(1, 10)"),
    ("gate_exclusive", "informative = rise is not None and rise >= RISE_MIN",
     "informative = rise is not None and rise > RISE_MIN"),
    ("mean_not_median", "med = {arm: median_rule(counted[arm]) for arm in ARM_ORDER}",
     "med = {arm: (sum(counted[arm], Fraction(0)) / len(counted[arm]) if counted[arm] else None) "
     "for arm in ARM_ORDER}"),
    ("alpha_4_70", "ALPHA = Fraction(1, 20)", "ALPHA = Fraction(4, 70)"),
    ("fill_swapped", "INF_CONTROL, INF_PHASE = NEG_INF, POS_INF", "INF_CONTROL, INF_PHASE = POS_INF, NEG_INF"),
    ("ties_for_the_reading", "count = sum(1 for s in sums.values() if s >= obs)",
     "count = sum(1 for s in sums.values() if s > obs)"),
    ("bs_baseline", "BASELINE_S = 0.1265954738482833", "BASELINE_S = -0.03157372563146055"),
    ("primary_rows_shifted", "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K, 0, -1))",
     "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K + 1, 1, -1))"),
    ("line1_not_first", "    if not informative:\n        line = 1", "    if False:\n        line = 1"),
    ("secCMD_direction", '{"key": "SEC-CMD", "family": EFFICACY, "higher": CONTROL',
     '{"key": "SEC-CMD", "family": EFFICACY, "higher": INTERVENTION'),
    ("secMARGIN_direction", '{"key": "SEC-MARGIN", "family": EFFICACY, "higher": INTERVENTION',
     '{"key": "SEC-MARGIN", "family": EFFICACY, "higher": CONTROL'),
    ("secMECH_direction", '{"key": "SEC-MECH", "family": EFFICACY, "higher": CONTROL',
     '{"key": "SEC-MECH", "family": EFFICACY, "higher": INTERVENTION'),
    ("secTAIL2_direction", '{"key": "SEC-TAIL2", "family": GUARD, "higher": INTERVENTION',
     '{"key": "SEC-TAIL2", "family": GUARD, "higher": CONTROL'),
    ("secTAIL2_in_efficacy", '{"key": "SEC-TAIL2", "family": GUARD,', '{"key": "SEC-TAIL2", "family": EFFICACY,'),
    ("sec_fill_swapped", 'vals[c["arm"]].append(NEG_INF if c["arm"] == higher else POS_INF)',
     'vals[c["arm"]].append(POS_INF if c["arm"] == higher else NEG_INF)'),
    ("bonferroni_at_alpha", "if supported and spec[\"family\"] == EFFICACY and filled[0] <= ALPHA / N_EFFICACY:",
     "if supported and spec[\"family\"] == EFFICACY and filled[0] <= ALPHA:"),
    ("bonferroni_for_the_guard", "if supported and spec[\"family\"] == EFFICACY and filled[0] <= ALPHA / N_EFFICACY:",
     "if supported and filled[0] <= ALPHA / N_EFFICACY:"),
    ("bonferroni_over_4", "N_EFFICACY = 3 ", "N_EFFICACY = 4 "),
    ("guard_note_lost", "    if supported and spec[\"family\"] == GUARD:\n        notes.append(SEC_GUARD)",
     "    if False:\n        notes.append(SEC_GUARD)"),
    ("secCMD_reads_max", 'v, what = e.get("cmd_l"), "CMD_L at E"', 'v, what = e.get("cmd_l_max"), "CMD_L at E"'),
    ("secMARGIN_reads_c0", 'v, what = _get(rows[0], "OWN", "P", "b_star"), "OWN.P.b_star at E"',
     'v, what = _get(rows[0], "OWN", "P", "c0"), "OWN.P.b_star at E"'),
    ("secMECH_reads_raw", 'v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")',
     'v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "raw", "dV_dq3_per_deg")'),
    ("secMECH_reads_settled", 'v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")',
     'v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "settled", "dV_dq3_per_deg")'),
    ("secMECH_reads_OWN", 'v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")',
     'v = _get(rows[0], "critic_slope_obs6", "OWN", "effective", "dV_dq3_per_deg")'),
    ("sec_rows_at_S", "        rows = loop_rows_at(m, C.E)\n        if len(rows) != 1:",
     "        rows = loop_rows_at(m, C.S)\n        if len(rows) != 1:"),
    ("tail2_reads_col12", "ACT_TAIL2 = 13 ", "ACT_TAIL2 = 12 "),
    ("tail2_reads_col14", "ACT_TAIL2 = 13 ", "ACT_TAIL2 = 14 "),
    ("tail2_rows_early", "rec, cause = _npz_rows_mean(path, PRIMARY_STEPS)",
     "rec, cause = _npz_rows_mean(path, EARLY_STEPS)"),
    ("tail2_sha_unchecked", "    if got != want:\n        return (None, f\"the probe file {path} changed",
     "    if False:\n        return (None, f\"the probe file {path} changed"),
    ("tail2_change_not_provisional", "f\"{c['name']}: the probe file {path} changed since the read ({got} != {want}); "
     "SEC-TAIL2 MISSING\")", "None)"),
    ("tail2_col14_unbound", "        if not (_finite(mk) and math.isfinite(row[C.ACT_TAIL3]) and Fraction(mk) == "
     "Fraction(row[C.ACT_TAIL3])):", "        if False:"),
    ("form_unchecked", "        if row.get(\"critic_form\") != form:", "        if False:"),
    ("ctrl_effective_unchecked", "if not isinstance(d, dict) or \"settled\" in d or _canon(d.get(\"effective\")) != "
     "_canon(d.get(\"raw\")):", "if not isinstance(d, dict) or \"settled\" in d:"),
    ("probe_phase_unchecked", "    if not isinstance(pp, dict) or pp.get(\"installed\") is not phase:",
     "    if False:"),
    ("phase_record_unchecked", "            if not isinstance(row.get(\"phase\"), dict):", "            if False:"),
    ("settled_unchecked", "if not isinstance(d, dict) or not {\"raw\", \"effective\", \"settled\"} <= set(d):",
     "if not isinstance(d, dict):"),
    ("ctrl_phase_record_unchecked", "            if \"phase\" in row:", "            if False:"),
    ("swing_reversed", 'lr["swing_onpolicy"] = _diff(lr["V_onpolicy_settled"], lr["V_onpolicy_raw"])',
     'lr["swing_onpolicy"] = _diff(lr["V_onpolicy_raw"], lr["V_onpolicy_settled"])'),
    ("swing_against_effective", 'lr["swing_onpolicy"] = _diff(lr["V_onpolicy_settled"], lr["V_onpolicy_raw"])',
     'lr["swing_onpolicy"] = _diff(lr["V_onpolicy_settled"], lr["V_onpolicy_eff"])'),
    ("sec_unreadable_read", '    if primary_cause is not None:\n        return None, f"UNREADABLE',
     '    if False:\n        return None, f"UNREADABLE'),
    ("no_note_under_line1", "    if line == 1:\n        notes.append(SEC_UNDER_LINE1)",
     "    if line == 0:\n        notes.append(SEC_UNDER_LINE1)"),
    ("secondary_sets_the_line", 'status=status, provisional=prov, line=dec["line"],',
     'status=status, provisional=prov, line=(2 if secs[2]["supported"] else dec["line"]),'),
    ("blind_to_unread_only", "    if undecided or unread:", "    if undecided:"),
    ("blind_to_undecided_only", "    if undecided or unread:", "    if unread:"),
    ("code_copy_under_root", "items.extend(code_seal_items(Path(C.CODE)))",
     'items.extend(code_seal_items(root / "code"))'),
    ("line1_text_0_10", '"primary rose by less than 0.20 from the value at S',
     '"primary rose by less than 0.10 from the value at S'),
    ("line2_text_blinding", '"REDUCED: GIVING THE CRITIC THE EPISODE PHASE', '"REDUCED: BLINDING THE CRITIC'),
    ("line3_power_text", '"THE EPISODE PHASE (not excluded: a reduction of about half the rise or smaller (power about '
     '0.3-1.0 at half "', '"THE EPISODE PHASE (not excluded: a reduction of about half the rise or smaller (power '
     'about 0.3-0.9 at half "'),
    ("sec_missing_row_unchecked", "        if len(rows) != 1:\n            return None, f\"{len(rows)} M-LOOP",
     "        if len(rows) > 1:\n            return None, f\"{len(rows)} M-LOOP"),
    ("row_at_E_unchecked", "        if row.get(\"num_timesteps\") != C.E:", "        if False:"),
    ("as_counted_addition_lost", "if line == 3 and as_counted is not None and as_counted[0] <= ALPHA:",
     "if line == 3 and as_counted is not None and as_counted[0] < 0:"),
    ("not_informative_addition_lost", "    if line == 1 and filled[0] <= ALPHA:\n        additions.append",
     "    if False:\n        additions.append"),
    ("trajectory_at_T_no_row", "end = t if t < T else T - C.ROLLOUT", "end = t if t <= T else T - C.ROLLOUT"),
    ("g_count_not_strict", "    return ge > gs", "    return ge >= gs"),
    ("final_pair_item_lost", 'if c["cls"] == COUNTED and mk is not None and len(loop_rows_at(mk, C.E)) != 1:',
     "if False:"),
]


def test_the_original_passes_every_kill_check(kill_roots):
    assert _kill_checks(SB, kill_roots) == []


@pytest.mark.parametrize("tag, old, new", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_each_mutant_is_caught(kill_roots, tag, old, new):
    assert SRC.count(old) == 1, f"mutant {tag}: the edited text must occur exactly once"
    mutant = _variant(SRC.replace(old, new), tag)
    failed = _kill_checks(mutant, kill_roots)
    assert failed, f"mutant {tag} survived every kill check"
