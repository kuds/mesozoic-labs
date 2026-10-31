"""Unit tests of BS2's score_bs.py on synthetic records (no simulation, no training): each reading line, the boundary of
the test, the gate (0.20, inclusive, the median rule), the fill and its bounds, the missing and void classes,
outcome-blindness, the input errors, PROVISIONAL and final, the three secondaries (direction, fill, MISSING causes, the
Bonferroni note, the note under line 1, never the line), the report-only readings (trajectory, G(E) > G(S)), the
constants against their sources (the parent's F2 probe, the criteria text, SCREEN.json, PINS), a dry run of the parent's
own markers, and mutation tests (each mutant of score_bs.py is caught by the kill checks; the original passes them).

    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS2>/tools/tests/test_score_bs.py
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

import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]
SRC_PATH = TOOLS / "score_bs.py"
_spec = importlib.util.spec_from_file_location("score_bs", str(SRC_PATH))
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)
C, R = SB.C, SB.R
S, E, R8 = C.S, C.E, C.ROLLOUT
BLIND = SB.INTERVENTION
B_S = SB.BASELINE_S
FAKE_PROBE_SHA = "ab" * 32
BS2 = Path(C.BS)
PARENT_MARKERS = BS2 / "validate" / "v3final" / "parent_s42" / "markers.json"
G_S = -0.6363920116424561


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return path


def _criteria_text():
    """The installed <BS2>/criteria.txt, or the draft it was installed from."""
    for p in (BS2 / "criteria.txt", BS2 / "criteria_work" / "criteria_v1.txt"):
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


def _probe(value, T, nan_at=None):
    steps = [S + R8 * k for k in range(1, (T - S) // R8)]
    mu = [float(B_S) + (value - B_S) * (i + 1) / len(steps) for i in range(len(steps))]
    for i, s in enumerate(steps):                      # the last 8 rows carry the value exactly
        if s in SB.PRIMARY_STEPS:
            mu[i] = value
    if nan_at is not None:
        mu[steps.index(nan_at)] = float("nan")
    ss = [0.05] * len(steps)
    trail = [None] * 7 + [0.05] * (len(steps) - 7)
    return {"file": "x", "sha256": FAKE_PROBE_SHA, "label": "x", "S": S, "T": T, "n_rows": len(steps), "steps": steps,
            "rollout_mu14_mean": mu, "tail3_statue_settled_mean": ss, "trailing8_mean": trail,
            "trailing8_first_above": {"0.10": None}, "trailing8_max": 0.05, "std14": [0.19] * len(steps),
            "obs6_mean_deg": [0.0] * len(steps)}


def _loop(arm, T, g=-0.5, b=1.0, final=True, num_ts_E=None):
    """M-LOOP rows at S, the periodic pairs before T and (final) the final pair at T, as read_bs.saved_pairs gives."""
    blind = arm == BLIND
    rows = []
    for s in (S, *[x for x in C.SAVED_STEPS if x < T], *([T] if final else [])):
        end = s == T
        G = g if end else (G_S if s == S else -0.6)
        bstar = b if end else 1.4
        crit = {k: {"raw": {"dV_dq3_per_deg": 0.01}, "effective": {"dV_dq3_per_deg": 0.0 if blind else 0.01}}
                for k in ("REF", "OWN", "ONPOLICY")}
        rows.append({"steps": s, "num_timesteps": num_ts_E if (end and num_ts_E is not None) else s,
                     "critic_form": "blind+raw" if blind else "plain",
                     "OWN": {"P": {"G": G, "c0": 1.0 - G - bstar, "b_star": bstar}, "L": {"G": -0.8},
                             "own14_median": 0.0, "b_star_L": 1.8, "n_terminated": 0},
                     "REF": {"P": {"G": -0.7, "c0": -0.1}}, "critic_slope_obs6": crit})
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
    blind = arm == BLIND
    ev = {"n": 175, "buffer_gae_lambda": [0.95], "max_adv_err_vs_gae095": 3e-6, "min_dmu_actor": 0.18,
          "max_dv_predict_values": 0.0 if blind else 0.29, "max_dv_forward": 0.0 if blind else 0.29,
          "max_dv_evaluate_actions": 0.0 if blind else 0.29, "max_dv_mirrored": 0.0 if blind else 0.30,
          "min_dv_predict_values": 0.0 if blind else 0.09, "min_buffer_values_vs_raw_value_mlp": 0.005 if blind
          else 9e-7, "state_problems": []}
    return {"schema": C.RUN_SCHEMA, "name": C.run_name(arm, seed), "arm": arm, "seed": seed, "arm_evidence": ev,
            "arm_applied": {"arm": arm, "mlp_extractor_class": "CriticBlindTail3MlpExtractor" if blind
                            else "MlpExtractor"}, "arm_state_end": "in force",
            "criteria_sha256": crit, "screen_sha256": screen, "approved_sha256": appr,
            "tool_sha256": {t: _sha(TOOLS / t) for t in C.TOOL_FILES}}


class Screen:
    """A synthetic <root> with SCREEN.json and, per entry, any of: check.json, run_meta.json, markers.json, ledger."""

    def __init__(self, root):
        self.root = Path(root)
        _write(self.root / "SCREEN.json", {"schema": C.SCREEN_SCHEMA, "intervention": BLIND,
                                           "queue": [list(q) for q in SB.QUEUE]})

    def run(self, arm, seed, value, *, stop="end", T=E, check=True, read=True, passed=True, nan_at=None,
            meta_kw=None, g=-0.5, cmd=0.5, cmd_max=0.9, b=1.0, final=True, drop_E=False, num_ts_E=None):
        name = C.run_name(arm, seed)
        rd = self.root / "runs" / name
        meta = _write(rd / C.RUN_META, _meta(arm, seed, self.root, **(meta_kw or {})))
        led = _ledger(rd / "01_stance" / "eval_episodes.csv", T, falls=1 if arm == BLIND else 0)
        if check:
            _write(rd / C.CHECK_JSON, {
                "schema": R.CHECK_SCHEMA, "tool_sha256": _sha(TOOLS / "read_bs.py"), "name": name, "mode":
                "continuation", "passed": passed, "failures": [] if passed else ["x"], "arm": arm, "seed": seed,
                "S": S, "E": E, "T": T, "end": E, "stop_reason": stop,
                "sources": {"run_meta.json": [str(meta), _sha(meta)], "policy_probe.npz": ["p", FAKE_PROBE_SHA]}})
        if read:
            direct = _direct(T, cmd, cmd_max, drop_E)
            _write(self.root / "reads" / name / "markers.json", {
                "schema": R.MARKERS_SCHEMA, "tool_sha256": _sha(TOOLS / "read_bs.py"), "label": name,
                "kind": "continuation", "arm": arm, "seed": seed, "S": S, "T": T,
                "stop_reason": stop, "probe": _probe(value, T, nan_at),
                "ledger": {"file": str(led), "sha256": _sha(led), "direct": direct,
                           "m5": {"any_flagged": [], "tail_flagged": [], "any_alarms": [], "tail_alarms": [],
                                  "n_c6_counting_failures": 0}},
                "loop": _loop(arm, T, g, b, final, num_ts_E)})
        return name

    def all(self, control, blind, per_seed=None, **kw):
        """per_seed: {(arm, seed): {run() keyword: value}} for the secondaries and the odd cases."""
        for seed, a, b in zip(SB.SEEDS, control, blind, strict=True):
            self.run(C.CONTROL, seed, a, **{**kw, **(per_seed or {}).get((C.CONTROL, seed), {})})
            self.run(BLIND, seed, b, **{**kw, **(per_seed or {}).get((BLIND, seed), {})})

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


# The main synthetic screen (line 2): control far above blind on the primary; the secondaries chosen so that SEC-G reads
# p = 1/70 (SUPPORTED, survives Bonferroni), SEC-CMD p = 2/70 (SUPPORTED, no Bonferroni), SEC-MARGIN p = 24/70 (NOT
# SUPPORTED); control_2531's G at E equals G at S (G(E) > G(S) is strict).
HIGH = tuple(B_S + x for x in (0.30, 0.31, 0.32, 0.33))
LOW = tuple(B_S + x for x in (0.00, 0.01, 0.02, 0.03))
SEC_G = {C.CONTROL: (-1.0, -0.9, -0.8, G_S), BLIND: (-0.4, -0.3, -0.2, -0.1)}
SEC_CMD = {C.CONTROL: (0.80, 0.70, 0.60, 0.35), BLIND: (0.40, 0.30, 0.20, 0.10)}
SEC_CMD_MAX = {C.CONTROL: (0.1, 0.2, 0.3, 0.4), BLIND: (0.9, 0.8, 0.7, 0.6)}   # reversed: reading it would differ
SEC_B = {C.CONTROL: (1.0, 1.25, 1.5, 1.75), BLIND: (1.125, 1.375, 1.625, 1.875)}   # dyadic: no binary near-ties


def _per_seed_main():
    out = {}
    for arm in (C.CONTROL, BLIND):
        for i, seed in enumerate(SB.SEEDS):
            out[(arm, seed)] = {"g": SEC_G[arm][i], "cmd": SEC_CMD[arm][i], "cmd_max": SEC_CMD_MAX[arm][i],
                                "b": SEC_B[arm][i]}
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
    assert SB.QUEUE == (("control", 2501), (BLIND, 2501), ("control", 2511), (BLIND, 2511), ("control", 2521),
                        (BLIND, 2521), ("control", 2531), (BLIND, 2531))
    assert (S, E) == (2097152, 3538944) and E == 27 * C.EVAL_FREQ
    assert SB.PRIMARY_STEPS == (3473408, 3481600, 3489792, 3497984, 3506176, 3514368, 3522560, 3530752)
    expected_rows = [S + R8 * k for k in range(1, C.expected_update_rows(S, E) + 1)]
    assert list(SB.PRIMARY_STEPS) == expected_rows[-8:] and len(expected_rows) == 175
    assert SB.EARLY_STEPS == tuple(expected_rows[:8]) and SB.EARLY_STEPS[0] == 2105344
    assert SB.BASELINE_STEPS == tuple(range(2039808, 2097152 + 1, R8))
    assert SB.EVAL_STEPS == tuple(range(2228224, 3538944 + 1, 131072)) and len(SB.EVAL_STEPS) == 11
    assert SB.RISE_MIN == Fraction(1, 5) and SB.ALPHA == Fraction(1, 20) and SB.N_SECONDARIES == 3
    assert Fraction(B_S) == Fraction(135930855, 1073741824)
    assert set(SB.PARENT_TRAJECTORY) == set(SB.EVAL_STEPS)
    assert SB.PARENT_TRAJECTORY[E] == 0.7053788751363754 and Fraction(SB.PARENT_TRAJECTORY[E]) == Fraction(47337175,
                                                                                                            67108864)
    assert [(s["key"], s["higher"]) for s in SB.SECONDARIES] == [("SEC-G", BLIND), ("SEC-CMD", "control"),
                                                                 ("SEC-MARGIN", BLIND)]
    assert set(SB.PINNED_TOOLS) == set(C.TOOL_FILES) | {"score_bs.py"}
    assert C.FINAL_ZIP == "stage1_final.zip" and E not in C.SAVED_STEPS and E % C.SAVE_FREQ != 0
    assert not set(C.derived_seeds(s) for s in SB.SEEDS) & set(C.derived_seeds(v) for v in C.VALIDATION_SEEDS)
    real = BS2 / "SCREEN.json"
    if real.exists():
        d, problems = C.load_screen(str(real))
        assert problems == [] and C.queue_of(d) == SB.QUEUE and d["intervention"] == BLIND


def test_baseline_recomputed_from_the_parent_probe():
    path = Path(C.f2_record("policy_probe.npz"))
    if not path.exists():
        pytest.skip("the parent's F2 probe is not on this machine")
    np = pytest.importorskip("numpy")
    assert C.sha256_file(path) == C.START["records"]["policy_probe.npz"]
    with np.load(path, allow_pickle=False) as z:
        u = z["update_timesteps"].astype(np.int64)
        m = z["rollout_mu_mean"][:, C.ACT_TAIL3].astype(np.float64)
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


# ------------------------------------------------------------------------------------------------ the test itself
def test_perm_test_matches_an_independent_implementation():
    rng = random.Random(7)
    for _ in range(200):
        c = [rng.gauss(0.2, 0.05) for _ in range(4)]
        b = [rng.gauss(0.1, 0.05) for _ in range(4)]
        p, count, total = SB.perm_test([(0, Fraction(x)) for x in c], [(0, Fraction(x)) for x in b])
        assert total == 70 and p == Fraction(count, 70)
        assert float(p) == pytest.approx(_brute_p(c, b), abs=1e-12)
        # the same test with the other arm as the hypothesis arm (the blind-HIGHER secondaries)
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
    filled = SB.perm_test(c, b3 + [SB.INF_BLIND])
    assert filled[:2] == (Fraction(36, 70), 36)
    worst = max(SB.perm_test(c, b3 + [(0, Fraction(v))])[0] for v in (-100, 0, 4.5, 6.5, 9, 100, 10**9))
    assert filled[0] >= worst and SB.perm_test(c, b3 + [(0, Fraction(10**12))])[0] == filled[0]
    filled_c = SB.perm_test(c[1:] + [SB.INF_CONTROL], b3 + [(0, Fraction(4))])
    assert filled_c[:2] == (Fraction(36, 70), 36)


def test_fill_bounds_in_the_criteria():
    """THE FILL: one MISSING -> p >= 36/70, one in each arm -> >= 51/70, two controls -> >= 56/70 (the minimum over
    every finite completion is reached when the read controls are far above the read blinds)."""
    f = lambda xs: [(0, Fraction(x)) for x in xs]  # noqa: E731
    assert SB.perm_test(f((6, 7, 8)) + [SB.INF_CONTROL], f((1, 2, 3, 4)))[1] == 36
    assert SB.perm_test(f((5, 6, 7, 8)), f((1, 2, 3)) + [SB.INF_BLIND])[1] == 36
    assert SB.perm_test(f((6, 7, 8)) + [SB.INF_CONTROL], f((1, 2, 3)) + [SB.INF_BLIND])[1] == 51
    assert SB.perm_test(f((7, 8)) + [SB.INF_CONTROL] * 2, f((1, 2, 3, 4)))[1] == 56
    rng = random.Random(3)
    for _ in range(50):                                             # never below the bound, whatever the values
        x = [rng.uniform(-1, 1) for _ in range(6)]
        assert SB.perm_test(f(x[:3]) + [SB.INF_CONTROL], f(x[3:]) + [SB.INF_BLIND])[1] >= 51


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
    assert "size 3/70 when the 8 primaries are exchangeable under no effect" in " ".join(text.split())   # D1
    assert "about 0.18 if the arms' spreads differ" in " ".join(text.split())                           # D1
    assert res["paired_sign_flip"]["count"] == 1 and res["paired_sign_flip"]["total"] == 16
    assert [c["primary_exact"] for c in res["classes"]] == [str(Fraction(v)) for pair in zip(HIGH, LOW,
                                                                                             strict=True) for v in pair]
    assert res["control_rise"] == pytest.approx(statistics.median(HIGH) - B_S)


@pytest.mark.parametrize("c, b, line", [((8, 7, 6, 3.5), (5, 4, 2, 1), 2), ((8, 7, 6, 3), (5, 4, 2, 1), 3)])
def test_line_at_the_boundary(tmp_path, c, b, line):
    s = Screen(tmp_path)
    s.all([0.5 + x / 64 for x in c], [0.5 + x / 64 for x in b])           # exact in binary: no rounding ties
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
    # {0.15, 0.18, 0.21, 0.24} and the observed {0.20, 0.15, 0.25, 0.18} both sum to 0.78 in decimal, but as the
    # recorded binary doubles the first is 2**-55 lower: the scorer compares the exact rationals of the recorded values
    # (criteria THE TEST), so that relabeling is not counted (a tolerance would count it: 46/70 against 45/70)
    c = [(0, Fraction(x)) for x in (0.20, 0.15, 0.25, 0.18)]
    b = [(0, Fraction(x)) for x in (0.21, 0.17, 0.19, 0.24)]
    gap = sum(map(Fraction, (0.15, 0.18, 0.21, 0.24))) - sum(map(Fraction, (0.20, 0.15, 0.25, 0.18)))
    assert gap == -Fraction(1, 2**55)
    p = SB.perm_test(c, b)
    assert p[:2] == (Fraction(45, 70), 45) and _brute_p([0.20, 0.15, 0.25, 0.18], [0.21, 0.17, 0.19, 0.24]) == 46 / 70


def test_line_not_informative_comes_first(tmp_path):
    s = Screen(tmp_path)
    low = [B_S + 0.05, B_S + 0.06, B_S + 0.07, B_S + 0.08]                  # control median rise 0.065 < 0.20
    s.all(low, [x - 0.2 for x in low])                                      # the blind arm far lower: p = 1/70
    res, text, _ = _score(tmp_path)
    assert res["line"] == 1 and res["line_key"] == "NOT INFORMATIVE" and res["p_filled"]["count"] == 1
    assert res["informative"] is False and res["control_rise"] == pytest.approx(0.065)
    assert res["additions"] == [SB.ADDITION_NOT_INFORMATIVE]
    assert SB.LINE_TEXT[1] in text and "rose by less than 0.20" in text


def _classes():
    return [{"name": C.run_name(a, sd), "arm": a, "seed": sd} for a, sd in SB.QUEUE]


def test_informativeness_threshold_is_inclusive_and_0_20():
    edge = Fraction(B_S) + Fraction(1, 5)
    for ctrl, informative in ((edge, True), (edge - Fraction(1, 10**12), False),
                              (Fraction(B_S) + Fraction(15, 100), False), (Fraction(B_S) + Fraction(1, 10), False)):
        prim = {C.run_name(a, sd): ctrl for a, sd in SB.QUEUE}                 # all tied: p = 70/70
        dec = SB.decide(_classes(), prim)
        assert dec["informative"] is informative and dec["line"] == (3 if informative else 1)
        assert dec["control_rise"] == ctrl - Fraction(B_S)
    # with recorded doubles the rise is dyadic and can never equal 1/5 exactly: the inclusive edge is the exact one
    assert Fraction(B_S + 0.2) - Fraction(B_S) != Fraction(1, 5)


def test_informativeness_uses_the_median_not_the_mean():
    # control rises 0.05, 0.06, 0.07, 0.90 from B_S: median 0.065 (< 0.20, NOT INFORMATIVE), mean 0.27 (>= 0.20)
    rises = dict(zip(SB.SEEDS, (Fraction(5, 100), Fraction(6, 100), Fraction(7, 100), Fraction(90, 100)), strict=True))
    prim = {C.run_name(a, sd): (Fraction(B_S) + rises[sd] if a == C.CONTROL else Fraction(B_S)) for a, sd in SB.QUEUE}
    dec = SB.decide(_classes(), prim)
    assert dec["medians"]["control"] - Fraction(B_S) == Fraction(65, 1000)
    assert dec["informative"] is False and dec["line"] == 1
    # and the other way round: 0.00, 0.21, 0.22, 0.23 -> median 0.215 informative although the mean 0.165 is below
    rises = dict(zip(SB.SEEDS, (Fraction(0), Fraction(21, 100), Fraction(22, 100), Fraction(23, 100)), strict=True))
    prim = {C.run_name(a, sd): (Fraction(B_S) + rises[sd] if a == C.CONTROL else Fraction(B_S)) for a, sd in SB.QUEUE}
    dec = SB.decide(_classes(), prim)
    assert dec["control_rise"] == Fraction(215, 1000) and dec["informative"] is True


@pytest.mark.parametrize("n_read, want_rise", [(4, Fraction(215, 1000)), (3, Fraction(22, 100)),
                                               (2, Fraction(225, 1000)), (1, Fraction(23, 100)), (0, None)])
def test_the_median_rule_for_4_3_2_1_0_controls_read(n_read, want_rise):
    """STAT-2: 4 -> midpoint of the two middle values; 3 -> the middle one; 2 -> the midpoint; 1 -> the value; 0 -> not
    informative.  The controls read are the last n_read of rises 0.00, 0.21, 0.22, 0.23 (the others MISSING)."""
    rises = (Fraction(0), Fraction(21, 100), Fraction(22, 100), Fraction(23, 100))
    prim = {}
    for i, sd in enumerate(SB.SEEDS):
        prim[C.run_name(C.CONTROL, sd)] = Fraction(B_S) + rises[i] if i >= 4 - n_read else None
        prim[C.run_name(BLIND, sd)] = Fraction(B_S)
    dec = SB.decide(_classes(), prim)
    assert dec["control_rise"] == want_rise
    assert dec["informative"] is (want_rise is not None and want_rise >= Fraction(1, 5))
    assert SB.median_rule([]) is None and SB.median_rule([Fraction(1), Fraction(2)]) == Fraction(3, 2)


# ------------------------------------------------------------------------------------------------ missing and the fill
def test_fill_blind_lost(tmp_path):
    s = _main_screen(tmp_path)
    s.lost(C.run_name(BLIND, 2531))
    res, text, _ = _score(tmp_path)
    assert res["line"] == 3 and res["p_filled"]["count"] == 36
    assert res["p_as_counted"]["count"] == 1 and res["p_as_counted"]["total"] == 35
    assert res["additions"] == [SB.ADDITION_AS_COUNTED]
    assert res["additions"] == ["(the as-counted p, report only, is <= 0.05; the filled p decides)"]
    assert "MISSING, filled at +inf: lost.json: crashes" in text


def test_fill_control_stopped(tmp_path):
    s = Screen(tmp_path)
    for seed, a, b in zip(SB.SEEDS, HIGH, LOW, strict=True):
        if seed == 2501:
            s.run(C.CONTROL, seed, a, stop="collapse", T=3407872)
        else:
            s.run(C.CONTROL, seed, a)
        s.run(BLIND, seed, b)
    res, text, _ = _score(tmp_path)
    cls = {c["name"]: c for c in res["classes"]}
    assert cls["control_2501"]["cls"] == "STOPPED" and cls["control_2501"]["primary"] is None
    assert res["p_filled"]["count"] == 36 and res["line"] == 3
    assert "MISSING, filled at -inf" in text and "the collapse stop at T = 3,407,872" in text
    for s_ in res["secondaries"]:                                        # STOPPED: every secondary MISSING too
        assert s_["values"]["control_2501"] is None and "collapse" in s_["causes"]["control_2501"]


def test_primary_unreadable_is_missing(tmp_path):
    s = Screen(tmp_path)
    for seed, a, b in zip(SB.SEEDS, HIGH, LOW, strict=True):
        s.run(C.CONTROL, seed, a)
        s.run(BLIND, seed, b, nan_at=SB.PRIMARY_STEPS[3] if seed == 2511 else None)
    res, _text, _ = _score(tmp_path)
    cls = {c["name"]: c for c in res["classes"]}
    assert cls[f"{BLIND}_2511"]["primary"] is None and "primary unreadable" in cls[f"{BLIND}_2511"]["primary_cause"]
    assert res["line"] == 3 and res["n_missing"] == 1


def test_all_missing_in_one_arm_is_still_decided(tmp_path):
    s = Screen(tmp_path)
    for seed, a in zip(SB.SEEDS, HIGH, strict=True):
        s.run(C.CONTROL, seed, a)
        s.lost(C.run_name(BLIND, seed))
    res, _t, _c = _score(tmp_path)
    assert res["line"] == 3 and res["p_filled"]["count"] == 70 and res["p_as_counted"] is None


def test_no_control_counted_is_not_informative(tmp_path):
    s = Screen(tmp_path)
    for seed, b in zip(SB.SEEDS, LOW, strict=True):
        s.lost(C.run_name(C.CONTROL, seed))
        s.run(BLIND, seed, b)
    res, _t, _c = _score(tmp_path)
    assert res["line"] == 1 and res["control_median"] is None


# ------------------------------------------------------------------------------------------------ the secondaries
def test_secondaries_direction_exact_p_and_bonferroni(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    g, cmd, mar = _sec(res, "SEC-G"), _sec(res, "SEC-CMD"), _sec(res, "SEC-MARGIN")
    # SEC-G, blind HIGHER: blind G all above control G -> 1/70, SUPPORTED, survives Bonferroni
    assert g["p_filled"]["count"] == 1 and g["verdict"] == SB.SEC_SUPPORTED and g["notes"] == [SB.SEC_BONFERRONI]
    # SEC-CMD, blind LOWER (control higher): 2/70, SUPPORTED, does not survive Bonferroni (2/70 > 0.05 / 3)
    assert cmd["p_filled"]["count"] == 2 and cmd["verdict"] == SB.SEC_SUPPORTED and cmd["notes"] == []
    assert cmd["p_filled"]["p_float"] == pytest.approx(_brute_p(SEC_CMD[C.CONTROL], SEC_CMD[BLIND]))
    # SEC-MARGIN, blind HIGHER: 24/70, NOT SUPPORTED
    assert mar["p_filled"]["count"] == 24 and mar["verdict"] == SB.SEC_NOT_SUPPORTED and mar["notes"] == []
    assert mar["p_filled"]["p_float"] == pytest.approx(_brute_p(SEC_B[BLIND], SEC_B[C.CONTROL]))
    for key, vals in (("SEC-G", SEC_G), ("SEC-CMD", SEC_CMD), ("SEC-MARGIN", SEC_B)):
        s = _sec(res, key)
        for arm in (C.CONTROL, BLIND):
            for i, seed in enumerate(SB.SEEDS):
                assert s["values"][C.run_name(arm, seed)] == vals[arm][i]
        assert s["p_as_counted"] == s["p_filled"]                         # nothing missing
    assert "SUPPORTED (p <= 0.05, unadjusted) (survives Bonferroni over the 3 secondaries)" in text
    assert "SEC-CMD" in text and "NOT SUPPORTED" in text and res["line"] == 2


def test_secondaries_never_change_the_line(tmp_path):
    """The same primaries with the secondaries reversed: the line, its p and its additions are identical."""
    _main_screen(tmp_path / "a")
    rev = {}
    for arm, other in ((C.CONTROL, BLIND), (BLIND, C.CONTROL)):
        for i, seed in enumerate(SB.SEEDS):
            rev[(arm, seed)] = {"g": SEC_G[other][i], "cmd": SEC_CMD[other][i], "b": SEC_B[other][i]}
    _main_screen(tmp_path / "b", per_seed_extra=rev)
    ra, _ta, _ = _score(tmp_path / "a")
    rb, _tb, _ = _score(tmp_path / "b")
    assert [s["verdict"] for s in ra["secondaries"]] != [s["verdict"] for s in rb["secondaries"]]
    for k in ("line", "line_key", "line_text", "additions", "p_filled", "p_as_counted", "informative", "control_rise"):
        assert ra[k] == rb[k], k
    # decide() does not take the secondaries at all
    assert list(SB.decide.__code__.co_varnames[:SB.decide.__code__.co_argcount]) == ["classes", "primaries"]


def test_secondary_fill_works_against_its_direction(tmp_path):
    s = _main_screen(tmp_path)
    s.lost(C.run_name(BLIND, 2531))
    res, _t, _ = _score(tmp_path)
    g, cmd = _sec(res, "SEC-G"), _sec(res, "SEC-CMD")
    # blind HIGHER: a MISSING blind value is -inf -> 36/70 (would be 1/70 if filled the other way)
    assert g["values"][f"{BLIND}_2531"] is None and g["p_filled"]["count"] == 36 and not g["supported"]
    assert g["p_as_counted"]["count"] == 1 and g["p_as_counted"]["total"] == 35
    # blind LOWER: a MISSING blind value is +inf
    vals = [(0, Fraction(x)) for x in SEC_CMD[C.CONTROL]]
    want = SB.perm_test(vals, [(0, Fraction(x)) for x in SEC_CMD[BLIND][:3]] + [SB.POS_INF])
    assert cmd["p_filled"]["count"] == want[1] == 37
    # a MISSING control value in SEC-G (blind HIGHER) is +inf
    s2 = _main_screen(tmp_path / "c")
    s2.lost(C.run_name(C.CONTROL, 2501))
    g2 = _sec(_score(tmp_path / "c")[0], "SEC-G")
    want2 = SB.perm_test([(0, Fraction(x)) for x in SEC_G[BLIND]],
                         [(0, Fraction(x)) for x in SEC_G[C.CONTROL][1:]] + [SB.POS_INF])
    assert g2["p_filled"]["count"] == want2[1] == 36


def test_secondary_missing_causes(tmp_path):
    extra = {(BLIND, 2511): {"nan_at": SB.PRIMARY_STEPS[0]},                 # UNREADABLE: every secondary MISSING
             (C.CONTROL, 2521): {"final": False, "drop_E": True},           # no loop row at E, no evaluation at E
             (C.CONTROL, 2531): {"cmd": None, "g": float("nan")}}           # not finite
    _main_screen(tmp_path, per_seed_extra=extra)
    res, text, _ = _score(tmp_path)
    for s in res["secondaries"]:
        assert s["values"][f"{BLIND}_2511"] is None and s["causes"][f"{BLIND}_2511"].startswith("UNREADABLE")
    g, cmd, mar = _sec(res, "SEC-G"), _sec(res, "SEC-CMD"), _sec(res, "SEC-MARGIN")
    assert g["causes"]["control_2521"] == "0 M-LOOP rows at E = 3,538,944 (the final pair)"
    assert mar["causes"]["control_2521"] == g["causes"]["control_2521"]
    assert cmd["causes"]["control_2521"] == "the evaluation at E = 3,538,944 is absent from the ledger markers"
    assert cmd["causes"]["control_2531"] == "CMD_L at E is None (no finite value)"
    assert g["causes"]["control_2531"] == "OWN.P.G at E is nan (no finite value)"
    assert mar["values"]["control_2531"] == SEC_B[C.CONTROL][3]           # b* itself is finite
    assert res["line"] == 3 and res["n_missing"] == 1                     # the primary: only the UNREADABLE one
    assert "MISSING, filled at +inf: 0 M-LOOP rows at E" in text


def test_secondaries_carry_the_note_under_line_1(tmp_path):
    rises = (0.05, 0.06, 0.07, 0.90)                                        # median 0.065: line 1
    ps = _per_seed_main()
    s = Screen(tmp_path)
    s.all([B_S + x for x in rises], [B_S - 0.2 + x / 100 for x in range(4)], per_seed=ps)
    res, text, _ = _score(tmp_path)
    assert res["line"] == 1 and res["additions"] == [SB.ADDITION_NOT_INFORMATIVE]
    for sec in res["secondaries"]:
        assert sec["notes"][-1] == SB.SEC_UNDER_LINE1
    assert _sec(res, "SEC-G")["notes"] == [SB.SEC_BONFERRONI, SB.SEC_UNDER_LINE1]
    assert text.count(SB.SEC_UNDER_LINE1) == 3


def test_bonferroni_note_only_at_1_over_70():
    classes = _classes()
    for count, ctrl, blind in ((1, (1, 2, 3, 4), (5, 6, 7, 8)), (2, (1, 2, 3, 5.5), (4.5, 6, 7, 8)),
                               (3, (5, 4, 2, 1), (8, 7, 6, 3.5))):
        vals = {}
        for i, sd in enumerate(SB.SEEDS):
            vals[C.run_name(C.CONTROL, sd)] = Fraction(ctrl[i])
            vals[C.run_name(BLIND, sd)] = Fraction(blind[i])
        t = SB.secondary_test(SB.SECONDARIES[0], classes, vals, {}, 2)
        assert t["filled"][1] == count and t["supported"] is True
        assert t["notes"] == ([SB.SEC_BONFERRONI] if count == 1 else [])


def test_counted_without_the_final_pair_is_provisional(tmp_path):
    """Review finding D7: a COUNTED continuation whose markers hold no loop row at E (the final pair not read: a tool
    or file defect) reads SEC-G / SEC-MARGIN MISSING and is a PROVISIONAL item, so the reading is held, never final."""
    Screen(tmp_path)
    crit, screen, appr = _approve(tmp_path)
    _main_screen(tmp_path, per_seed_extra={(C.CONTROL, 2521): {"final": False}},
                 meta_kw={"crit": crit, "screen": screen, "appr": appr})
    res, text, code = _score(tmp_path)
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL
    assert res["provisional"] == ["control_2521: COUNTED but its markers hold 0 loop rows at E = 3,538,944 (the final "
                                  "pair was not read: a tool or file defect; SEC-G and SEC-MARGIN MISSING)"]
    assert _sec(res, "SEC-G")["values"]["control_2521"] is None and res["line"] == 2
    # the same screen with the row present is final (the item comes from the missing row alone)
    root2 = tmp_path / "with_row"
    Screen(root2)
    crit2, screen2, appr2 = _approve(root2)
    _main_screen(root2, meta_kw={"crit": crit2, "screen": screen2, "appr": appr2})
    assert _score(root2)[0]["provisional"] == []


def test_secondary_needs_its_row_at_E_to_be_the_model_at_E(tmp_path):
    _main_screen(tmp_path, per_seed_extra={(C.CONTROL, 2501): {"num_ts_E": E - R8}})
    with pytest.raises(SB.InputError, match="num_timesteps"):
        SB.score(tmp_path)


# ------------------------------------------------------------------------------------------------ outcome-blind
def test_incomplete_opens_no_markers_or_run_meta_and_prints_no_outcome(tmp_path):
    _main_screen(tmp_path)
    (tmp_path / "runs" / f"{BLIND}_2531" / C.CHECK_JSON).unlink()           # still running
    (tmp_path / "reads" / "control_2501" / "markers.json").write_text("not json {")   # never opened
    (tmp_path / "runs" / "control_2511" / C.RUN_META).write_text("not json {")      # never opened
    res, text, code = _score(tmp_path)
    assert res["status"] == "incomplete" and code == SB.EXIT_NOT_FINAL
    body = text.split("STATUS")[1].split("PROVISIONAL")[0]
    assert "THE READING" not in text and "SEC-" not in text and "0.4" not in body and "0.1" not in body
    assert f"{BLIND}_2531" in text and "undecided" in text and "line" not in res


def test_decided_but_unread_is_incomplete(tmp_path):
    _main_screen(tmp_path)
    (tmp_path / "reads" / "control_2511" / "markers.json").unlink()
    (tmp_path / "runs" / "control_2501" / C.RUN_META).write_text("not json {")      # never opened
    res, text, code = _score(tmp_path)
    assert res["status"] == "incomplete" and "decided, markers not read yet" in text and code == 3


def test_check_failed_is_undecided(tmp_path):
    s = _main_screen(tmp_path)
    s.run(BLIND, 2521, LOW[2], passed=False)
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
    d["queue"] = [list(q) for q in ((a, s - 400) for a, s in SB.QUEUE)]        # BS's seeds 2101-2131
    _write(tmp_path / "SCREEN.json", d)
    with pytest.raises(SB.InputError, match="not the criteria's"):
        SB.score(tmp_path)


@pytest.mark.parametrize("mutate, match", [
    (lambda m: m.update(arm="control"), "arm"),
    (lambda m: m["probe"].update(sha256="cd" * 32), "another policy_probe.npz"),
    (lambda m: m["loop"][1]["critic_slope_obs6"]["ONPOLICY"]["effective"].update(dV_dq3_per_deg=0.003),
     "must be 0.0"),
    (lambda m: m["loop"][0].update(critic_form="plain"), "critic form"),
    (lambda m: m["probe"]["steps"].pop(), "probe rows"),
    (lambda m: m.update(T=E - R8), "T"),
    (lambda m: m.update(S=1048576), "S"),
    (lambda m: m["loop"][-1].update(num_timesteps=E + R8), "num_timesteps"),
])
def test_markers_that_disagree_are_input_errors(tmp_path, mutate, match):
    _main_screen(tmp_path)
    p = tmp_path / "reads" / f"{BLIND}_2501" / "markers.json"
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
    p = tmp_path / "runs" / "control_2501" / C.CHECK_JSON
    c = json.loads(p.read_text())
    c["seed"] = 2511
    _write(p, c)
    with pytest.raises(SB.InputError, match="seed"):
        SB.score(tmp_path)
    c["seed"], c["E"] = 2501, 2490368                                       # BS's window
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
    assert [s["key"] for s in out["secondaries"]] == ["SEC-G", "SEC-CMD", "SEC-MARGIN"]
    assert out["secondaries"][0]["p_filled"]["count"] == 1


@pytest.mark.parametrize("breakage, match", [
    ("criteria", "criteria.txt sha256"), ("scorer_pin", "does not name this scorer"),
    ("run_meta_criteria", "ran under criteria.txt"), ("tool", "tools/score_bs.py"), ("ledger", "ledger"),
    ("approved_missing", "no "), ("run_meta_changed", "run_meta.json changed since CHECK"),
    ("markers_tool", "markers.json made by read_bs.py"), ("check_tool", "check.json made by read_bs.py"),
    ("code_copy", "code copy"),
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
        with open(tmp_path / "runs" / "control_2501" / "01_stance" / "eval_episodes.csv", "a") as fh:
            fh.write("99,3538944,10,0,1.0,1000,1,truncated\n")
    if breakage == "approved_missing":
        (tmp_path / "APPROVED").unlink()
    if breakage == "run_meta_changed":
        p = tmp_path / "runs" / f"{BLIND}_2511" / C.RUN_META
        p.write_text(p.read_text() + " ")
    if breakage == "markers_tool":                                          # markers made by another read_bs.py (R6)
        p = tmp_path / "reads" / "control_2521" / "markers.json"
        m = json.loads(p.read_text())
        m["tool_sha256"] = "d" * 64
        _write(p, m)
    if breakage == "check_tool":
        p = tmp_path / "runs" / "control_2521" / C.CHECK_JSON
        m = json.loads(p.read_text())
        m["tool_sha256"] = "d" * 64
        _write(p, m)
    if breakage == "code_copy":
        monkeypatch.setattr(SB, "code_seal_items", lambda d: [f"code copy tree digest x != pinned ({d})"])
    res, _text, code = _score(tmp_path)
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL
    assert any(match in x for x in res["provisional"]), res["provisional"]
    assert res["line_key"] == "REDUCED"                                     # printed, not acted on


def test_code_copy_is_measured_at_bs_common_CODE(tmp_path, _no_code_copy_measurement):
    """BS2 reuses BS's code copy read only: the scorer measures bs_common.CODE, never <root>/code."""
    _main_screen(tmp_path)
    _score(tmp_path)
    assert _no_code_copy_measurement and set(_no_code_copy_measurement) == {Path(C.CODE)}
    if not Path(C.CODE).is_dir():
        pytest.skip("the code copy is not on this machine")
    spec = importlib.util.spec_from_file_location("score_bs_fresh", str(SRC_PATH))
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.code_seal_items(Path(C.CODE)) == []                       # the real measurement (M1's check_copy)
    assert fresh.code_seal_items(tmp_path / "code") == [f"code copy {tmp_path / 'code'} missing: not measured"]


# ------------------------------------------------------------------------------------------------ report only
def test_report_only_readings(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    ro = res["report_only"][f"{BLIND}_2501"]
    assert ro["ledger"]["status"] == "read" and ro["ledger"]["n_falls"] == 1 and ro["ledger"]["n_episodes"] == 110
    assert ro["ledger"]["mean_return_at_T"] == pytest.approx(4010.0)
    assert ro["ledger"]["mean_return_per_evaluation"] == {t: pytest.approx(4000.0 + k)
                                                          for k, t in enumerate(SB.EVAL_STEPS)}
    assert "mean return per evaluation: 2,228,224 4000.0" in text
    assert ro["manipulation"]["max_dv_predict_values"] == 0.0
    assert res["report_only"]["control_2501"]["ledger"]["n_falls"] == 0
    assert "MANIPULATION" in text and "CriticBlindTail3MlpExtractor" in text and "PAIRED DIFFERENCES" in text
    assert [p["diff"] for p in res["paired"]] == pytest.approx([0.30, 0.30, 0.30, 0.30])
    sd = statistics.stdev(HIGH)
    assert res["arm_sd_report_only"]["control"] == pytest.approx(sd)
    assert f"SD {sd:.5f} (report only)" in text
    # CMD_L per evaluation (11 evaluations; at E the value read by SEC-CMD)
    cmd = ro["cmd_l_per_evaluation"]
    assert sorted(cmd) == list(SB.EVAL_STEPS) and cmd[E] == SEC_CMD[BLIND][0]


def test_trajectory_and_g_count(tmp_path):
    _main_screen(tmp_path)
    res, text, _ = _score(tmp_path)
    ro = res["report_only"]["control_2511"]
    traj = dict(ro["trajectory"])
    assert sorted(traj) == list(SB.EVAL_STEPS)
    assert traj[E] == Fraction(HIGH[1])                                     # at E: the primary's rows
    probe = _probe(HIGH[1], E)                                              # an independent mean at 2,752,512
    idx = [probe["steps"].index(2752512 - R8 * k) for k in range(8)]
    assert traj[2752512] == sum((Fraction(probe["rollout_mu14_mean"][i]) for i in idx), Fraction(0)) / 8
    assert "REPORT ONLY, THE ON-POLICY TRAJECTORY" in text and f"{SB.PARENT_TRAJECTORY[E]:+.4f}" in text
    # G(E) > G(S), strict: control G at E -1.0, -0.9, -0.8 and exactly G(S) -> 0 of 4; blind all above -> 4 of 4
    assert res["g_end_above_start"] == {"control": {"above": 0, "of": 4}, BLIND: {"above": 4, "of": 4}}
    assert "control 0 of 4; critic_blind_tail3 4 of 4" in text
    tm = res["trajectory_arm_medians"][str(E)]
    assert tm["control"] == pytest.approx(statistics.median(HIGH))
    # the end pair is the loop row at T (the final pair), not merely the last row
    assert ro["end_pair"]["steps"] == E and ro["end_pair"]["OWN_G"] == SEC_G[C.CONTROL][1]
    assert f"at the end pair (the loop row at T) OWN G {statistics.median(SEC_G[C.CONTROL]):+.3f}" in text


def test_report_only_never_fails_the_score(tmp_path):
    _main_screen(tmp_path)
    p = tmp_path / "reads" / f"{BLIND}_2501" / "markers.json"
    m = json.loads(p.read_text())
    m["loop"][0]["critic_slope_obs6"]["REF"]["effective"] = None          # a null entry is not an error
    led = Path(m["ledger"]["file"])
    led.write_bytes(b"\xff\xfe not utf-8")
    m["ledger"]["sha256"] = _sha(led)
    _write(p, m)
    res, _text, _code = _score(tmp_path)
    assert res["line"] == 2
    assert res["report_only"][f"{BLIND}_2501"]["ledger"]["status"].startswith("unreadable (UnicodeDecodeError")


# ------------------------------------------------------------------------------------------------ the criteria text
def test_line_texts_are_the_criteria_text():
    """The scorer's line texts, additions and secondary wordings appear verbatim (whitespace collapsed) in
    <BS2>/criteria.txt (or the draft it is installed from)."""
    text = _criteria_text()
    if text is None:
        pytest.skip("no criteria text in this build")
    for line, t in SB.LINE_TEXT.items():
        assert f'"{" ".join(t.split())}"' in text, line
    for t in (SB.ADDITION_AS_COUNTED, SB.ADDITION_NOT_INFORMATIVE, SB.SEC_SUPPORTED, SB.SEC_BONFERRONI,
              SB.SEC_UNDER_LINE1, '"NOT SUPPORTED"'):
        assert t in text, t
    for s in SB.SECONDARIES:
        assert s["key"] in text
    assert "B_S = 0.1265954738482833 (exactly 135930855/1073741824)" in text and "at least 0.20" in text
    assert "2501, 2511, 2521, 2531" in text


PIN_FILES = ("SCREEN.json", "tools/score_bs.py", "tools/tests/test_score_bs.py", "tools/bs_common.py",
             "tools/bs_hooks.py", "tools/run_bs.py", "tools/run_waves_bs.sh", "tools/read_bs.py")


def test_criteria_pins_name_the_current_files():
    """PINS: the installed criteria.txt names each pinned file's current sha256 (a change to a pinned file changes its
    line; P2: the re-reviewed versions are the pinned ones)."""
    crit = BS2 / "criteria.txt"
    if not crit.exists():
        pytest.skip("criteria.txt not installed")
    raw = crit.read_text(encoding="utf-8")
    pins = raw.split("\nPINS")[1].split("\nHOW THE COORDINATOR PINS")[0]
    for rel in PIN_FILES:
        m = re.search(rf"^\s+{re.escape(rel)}\s+([0-9a-f]{{64}})\b", pins, re.M)
        assert m, rel
        assert m.group(1) == _sha(BS2 / rel), rel


# ------------------------------------------------------------------------------------------------ the dry run's form
def _parent_as(m, arm, seed, root):
    """The parent's reference markers relabelled as one continuation (the criteria's dry run): its probe, ledger and
    loop rows unchanged, except the blind arm's loop rows carry the arm's form (critic_form, effective slope 0)."""
    m = json.loads(json.dumps(m))
    name = C.run_name(arm, seed)
    m.update(kind="continuation", label=name, arm=arm, seed=seed, stop_reason="end")
    m.pop("reproduction", None)
    if arm == BLIND:
        for row in m["loop"]:
            row["critic_form"] = "blind+raw"
            for d in row["critic_slope_obs6"].values():
                d["effective"] = {**d["effective"], "dV_dq3_per_deg": 0.0}
    return m


def test_dry_run_on_the_parents_markers(tmp_path):
    if not PARENT_MARKERS.exists():
        pytest.skip("the builder's reference read is not on this machine")
    pm = json.loads(PARENT_MARKERS.read_text())
    s = Screen(tmp_path)
    for arm, seed in SB.QUEUE:
        name = C.run_name(arm, seed)
        rd = tmp_path / "runs" / name
        meta = _write(rd / C.RUN_META, _meta(arm, seed, tmp_path))
        _write(rd / C.CHECK_JSON, {
            "schema": R.CHECK_SCHEMA, "tool_sha256": _sha(TOOLS / "read_bs.py"), "name": name, "mode": "continuation",
            "passed": True, "failures": [], "arm": arm, "seed": seed, "S": S, "E": E, "T": E, "end": E,
            "stop_reason": "end", "sources": {"run_meta.json": [str(meta), _sha(meta)],
                                              "policy_probe.npz": ["p", pm["probe"]["sha256"]]}})
        _write(tmp_path / "reads" / name / "markers.json", _parent_as(pm, arm, seed, tmp_path))
    assert s.root == tmp_path
    res, text, _ = _score(tmp_path)
    assert res["line"] == 3 and res["p_filled"]["count"] == 70 and res["informative"] is True
    assert res["control_rise"] == float(Fraction(47337175, 67108864) - Fraction(135930855, 1073741824))
    assert round(res["control_rise"], 5) == 0.57878
    assert all(c["primary"] == SB.PARENT_TRAJECTORY[E] for c in res["classes"])
    for sec in res["secondaries"]:
        assert sec["verdict"] == SB.SEC_NOT_SUPPORTED and sec["p_filled"]["count"] == 70
    assert all(v == pm["ledger"]["direct"]["per_evaluation"][str(E)]["cmd_l"]
               for v in _sec(res, "SEC-CMD")["values"].values())
    for key in ("SEC-G", "SEC-MARGIN"):
        assert set(_sec(res, key)["values"].values()) == {None}
        assert set(_sec(res, key)["causes"].values()) == {"0 M-LOOP rows at E = 3,538,944 (the final pair)"}
    assert res["g_end_above_start"] == {"control": {"above": 0, "of": 0}, BLIND: {"above": 0, "of": 0}}
    assert all(ro["end_pair"] is None for ro in res["report_only"].values())     # no loop row at E in the parent's
    final_items = [x for x in res["provisional"] if "the final pair was not read" in x]     # D7: one per entry
    assert sorted(x.split(":")[0] for x in final_items) == sorted(C.run_name(a, sd) for a, sd in SB.QUEUE)


# ------------------------------------------------------------------------------------------------ mutation tests
SRC = SRC_PATH.read_text(encoding="utf-8")


def _variant(src, tag):
    mod = types.ModuleType(f"score_bs_mutant_{tag}")
    mod.__file__ = str(SRC_PATH)
    exec(compile(src, str(SRC_PATH), "exec"), mod.__dict__)  # noqa: S102 - the scorer's own source, one edit
    return mod


@pytest.fixture(scope="module")
def kill_roots(tmp_path_factory):
    """The synthetic screens the kill checks read (built once; the mutants only read them)."""
    base = tmp_path_factory.mktemp("kill")
    roots = {}
    roots["main"] = base / "main"                                         # line 2; SEC-G 1/70, SEC-CMD 2/70
    _main_screen(roots["main"])
    roots["notinf"] = base / "notinf"                                     # line 1 (median 0.065, mean 0.27)
    Screen(roots["notinf"]).all([B_S + x for x in (0.05, 0.06, 0.07, 0.90)],
                                [B_S - 0.2 + x / 100 for x in range(4)], per_seed=_per_seed_main())
    roots["lost"] = base / "lost"                                         # line 3, 36/70, the as-counted addition
    _main_screen(roots["lost"]).lost(C.run_name(BLIND, 2531))
    roots["odd"] = base / "odd"                                           # UNREADABLE, rows absent, not finite
    _main_screen(roots["odd"], per_seed_extra={(BLIND, 2511): {"nan_at": SB.PRIMARY_STEPS[0]},
                                               (C.CONTROL, 2521): {"final": False, "drop_E": True},
                                               (C.CONTROL, 2531): {"cmd": None}})
    roots["unread"] = base / "unread"                                     # all decided, one not read
    _main_screen(roots["unread"])
    (roots["unread"] / "reads" / "control_2511" / "markers.json").unlink()
    roots["undecided"] = base / "undecided"                               # one undecided, garbage markers elsewhere
    _main_screen(roots["undecided"])
    (roots["undecided"] / "runs" / f"{BLIND}_2531" / C.CHECK_JSON).unlink()
    (roots["undecided"] / "reads" / "control_2501" / "markers.json").write_text("not json {")
    roots["badrow"] = base / "badrow"                                     # the row at E is not the model at E
    _main_screen(roots["badrow"], per_seed_extra={(C.CONTROL, 2501): {"num_ts_E": E - R8}})
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

    def main_ok():
        res, text, _ = _score(roots["main"], sb)
        g, cmd, mar = (next(s for s in res["secondaries"] if s["key"] == k) for k in ("SEC-G", "SEC-CMD",
                                                                                         "SEC-MARGIN"))
        traj = dict(res["report_only"]["control_2511"]["trajectory"])
        return (res["line"] == 2 and res["p_filled"]["count"] == 1
                and [c["primary_exact"] for c in res["classes"]] == [str(Fraction(v)) for pr in
                                                                     zip(HIGH, LOW, strict=True) for v in pr]
                and g["p_filled"]["count"] == 1 and g["notes"] == [SB.SEC_BONFERRONI]
                and cmd["p_filled"]["count"] == 2 and cmd["notes"] == [] and cmd["supported"]
                and mar["p_filled"]["count"] == 24 and not mar["supported"]
                and [mar["values"][C.run_name(a, sd)] for a, sd in SB.QUEUE] == [SEC_B[a][SB.SEEDS.index(sd)]
                                                                               for a, sd in SB.QUEUE]
                and res["g_end_above_start"]["control"] == {"above": 0, "of": 4}
                and traj[E] == Fraction(HIGH[1]) and set(seen) == {Path(C.CODE)}
                and sb.LINE_TEXT[2] in text)

    def notinf_ok():
        res, _t, _ = _score(roots["notinf"], sb)
        return (res["line"] == 1 and res["additions"] == [SB.ADDITION_NOT_INFORMATIVE]
                and all(s["notes"][-1] == SB.SEC_UNDER_LINE1 for s in res["secondaries"])
                and res["control_rise"] == pytest.approx(0.065))

    def lost_ok():
        res, _t, _ = _score(roots["lost"], sb)
        g = next(s for s in res["secondaries"] if s["key"] == "SEC-G")
        return (res["line"] == 3 and res["p_filled"]["count"] == 36 and res["additions"] == [SB.ADDITION_AS_COUNTED]
                and g["p_filled"]["count"] == 36)

    def odd_ok():
        res, _t, _ = _score(roots["odd"], sb)
        secs = {s["key"]: s for s in res["secondaries"]}
        return (all(s["values"][f"{BLIND}_2511"] is None and s["causes"][f"{BLIND}_2511"].startswith("UNREADABLE")
                    for s in secs.values())
                and secs["SEC-G"]["values"]["control_2521"] is None
                and secs["SEC-CMD"]["values"]["control_2531"] is None
                and [x.split(":")[0] for x in res["provisional"] if "the final pair was not read" in x]
                == ["control_2521"])

    def unread_ok():
        res, _t, _ = _score(roots["unread"], sb)
        return res["status"] == "incomplete"

    def undecided_ok():
        res, text, _ = _score(roots["undecided"], sb)
        return res["status"] == "incomplete" and "THE READING" not in text

    def badrow_ok():
        try:
            sb.score(roots["badrow"])
        except sb.InputError as exc:
            return "num_timesteps" in str(exc)
        return False

    def decide_ok():
        cl = _classes()
        edge = Fraction(B_S) + Fraction(1, 5)
        out = []
        for ctrl, inf in ((edge, True), (edge - Fraction(1, 10**12), False),
                          (Fraction(B_S) + Fraction(15, 100), False)):
            prim = {C.run_name(a, sd): (ctrl if a == C.CONTROL else Fraction(-1)) for a, sd in SB.QUEUE}
            out.append(sb.decide(cl, prim)["informative"] is inf)
        for c, b, line in (((8, 7, 6, 3.5), (5, 4, 2, 1), 2), ((8, 7, 6, 3), (5, 4, 2, 1), 3)):
            prim = {}
            for i, sd in enumerate(SB.SEEDS):
                prim[C.run_name(C.CONTROL, sd)] = Fraction(0.5 + c[i] / 64)
                prim[C.run_name(BLIND, sd)] = Fraction(0.5 + b[i] / 64)
            out.append(sb.decide(cl, prim)["line"] == line)
        return all(out)

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
                         ("unread", unread_ok), ("undecided", undecided_ok), ("badrow", badrow_ok),
                         ("decide", decide_ok), ("ties", ties_ok), ("texts", texts_ok), ("baseline", baseline_ok)):
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
    ("fill_swapped", "INF_CONTROL, INF_BLIND = NEG_INF, POS_INF", "INF_CONTROL, INF_BLIND = POS_INF, NEG_INF"),
    ("ties_for_the_reading", "count = sum(1 for s in sums.values() if s >= obs)",
     "count = sum(1 for s in sums.values() if s > obs)"),
    ("bs_baseline", "BASELINE_S = 0.1265954738482833", "BASELINE_S = -0.03157372563146055"),
    ("primary_rows_shifted", "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K, 0, -1))",
     "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K + 1, 1, -1))"),
    ("line1_not_first", "    if not informative:\n        line = 1", "    if False:\n        line = 1"),
    ("secG_direction", '{"key": "SEC-G", "higher": INTERVENTION', '{"key": "SEC-G", "higher": CONTROL'),
    ("secCMD_direction", '{"key": "SEC-CMD", "higher": CONTROL', '{"key": "SEC-CMD", "higher": INTERVENTION'),
    ("secMARGIN_direction", '{"key": "SEC-MARGIN", "higher": INTERVENTION', '{"key": "SEC-MARGIN", "higher": CONTROL'),
    ("sec_fill_swapped", 'vals[c["arm"]].append(NEG_INF if c["arm"] == higher else POS_INF)',
     'vals[c["arm"]].append(POS_INF if c["arm"] == higher else NEG_INF)'),
    ("bonferroni_at_alpha", "if supported and filled[0] <= ALPHA / N_SECONDARIES:",
     "if supported and filled[0] <= ALPHA:"),
    ("secCMD_reads_max", 'v, what = e.get("cmd_l"), "CMD_L at E"', 'v, what = e.get("cmd_l_max"), "CMD_L at E"'),
    ("secG_reads_row_at_S", "        rows = loop_rows_at(m, C.E)", "        rows = loop_rows_at(m, C.S)"),
    ("secMARGIN_reads_c0", 'field = "G" if key == "SEC-G" else "b_star"', 'field = "G" if key == "SEC-G" else "c0"'),
    ("sec_unreadable_read", '    if primary_cause is not None:\n        return None, f"UNREADABLE',
     '    if False:\n        return None, f"UNREADABLE'),
    ("no_note_under_line1", "    if line == 1:\n        notes.append(SEC_UNDER_LINE1)",
     "    if line == 0:\n        notes.append(SEC_UNDER_LINE1)"),
    ("secondary_sets_the_line", 'status=status, provisional=prov, line=dec["line"],',
     'status=status, provisional=prov, line=(2 if secs[0]["supported"] else dec["line"]),'),
    ("blind_to_unread_only", "    if undecided or unread:", "    if undecided:"),
    ("blind_to_undecided_only", "    if undecided or unread:", "    if unread:"),
    ("code_copy_under_root", "items.extend(code_seal_items(Path(C.CODE)))",
     'items.extend(code_seal_items(root / "code"))'),
    ("line1_text_0_10", '"primary rose by less than 0.20 from the value at S',
     '"primary rose by less than 0.10 from the value at S'),
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
    ("line3_power_text_0_96", '"0.25-1.00 at half the parent', '"0.25-0.96 at half the parent'),
]


def test_the_original_passes_every_kill_check(kill_roots):
    assert _kill_checks(SB, kill_roots) == []


@pytest.mark.parametrize("tag, old, new", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_each_mutant_is_caught(kill_roots, tag, old, new):
    assert SRC.count(old) == 1, f"mutant {tag}: the edited text must occur exactly once"
    mutant = _variant(SRC.replace(old, new), tag)
    failed = _kill_checks(mutant, kill_roots)
    assert failed, f"mutant {tag} survived every kill check"
