"""Unit tests of score_bs.py on synthetic records (no simulation, no training): each reading line, the boundary of the
test, the fill, the missing classes, outcome-blindness, the input errors, PROVISIONAL and final, the report-only
readings and the constants against their sources.

    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS>/tools/tests/test_score_bs.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import random
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_bs", str(TOOLS / "score_bs.py"))
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)
C, R = SB.C, SB.R
S, E, R8 = C.S, C.E, C.ROLLOUT
BLIND = SB.INTERVENTION
B_S = SB.BASELINE_S
FAKE_PROBE_SHA = "ab" * 32


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return path


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


def _loop(arm, T):
    blind = arm == BLIND
    rows = []
    for s in (S, T):
        crit = {k: {"raw": {"dV_dq3_per_deg": 0.01}, "effective": {"dV_dq3_per_deg": 0.0 if blind else 0.01}}
                for k in ("REF", "OWN", "ONPOLICY")}
        rows.append({"steps": s, "critic_form": "blind+raw" if blind else "plain",
                     "OWN": {"P": {"G": -0.7, "c0": 0.0, "b_star": 1.7}, "L": {"G": -0.8}, "own14_median": 0.0,
                             "b_star_L": 1.8, "n_terminated": 0},
                     "REF": {"P": {"G": -0.7, "c0": -0.1}}, "critic_slope_obs6": crit})
    return rows


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
            meta_kw=None):
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
            _write(self.root / "reads" / name / "markers.json", {
                "schema": R.MARKERS_SCHEMA, "tool_sha256": _sha(TOOLS / "read_bs.py"), "label": name,
                "kind": "continuation", "arm": arm, "seed": seed, "S": S, "T": T,
                "stop_reason": stop, "probe": _probe(value, T, nan_at),
                "ledger": {"file": str(led), "sha256": _sha(led), "direct": {"cmd_l_last": 0.2, "cmd_l_max": 0.2,
                                                                             "firsts": {}},
                           "m5": {"any_flagged": [], "tail_flagged": [], "any_alarms": [], "tail_alarms": [],
                                  "n_c6_counting_failures": 0}},
                "loop": _loop(arm, T)})
        return name

    def all(self, control, blind, **kw):
        for seed, a, b in zip(SB.SEEDS, control, blind, strict=True):
            self.run(C.CONTROL, seed, a, **kw)
            self.run(BLIND, seed, b, **kw)

    def lost(self, name, cause="crashes"):
        p = self.root / "lost.json"
        d = json.loads(p.read_text()) if p.exists() else {"schema": C.LOST_SCHEMA, "lost": {}}
        d["lost"][name] = {"cause": cause, "detail": "3 own crashes"}
        _write(p, d)


def _score(root):
    res, lines, code = SB.score(root)
    return res, "\n".join(lines), code


def _brute_p(control, blind):
    """An independent float implementation of the test (no fill)."""
    vals = list(control) + list(blind)
    obs = sum(control)
    subs = list(itertools.combinations(range(len(vals)), len(control)))
    return sum(1 for s in subs if sum(vals[i] for i in s) >= obs - 1e-12) / len(subs)


# ------------------------------------------------------------------------------------------------ constants
def test_constants_match_the_scaffold_and_screen():
    assert SB.QUEUE == (("control", 2101), (BLIND, 2101), ("control", 2111), (BLIND, 2111), ("control", 2121),
                        (BLIND, 2121), ("control", 2131), (BLIND, 2131))
    assert SB.PRIMARY_STEPS == (2424832, 2433024, 2441216, 2449408, 2457600, 2465792, 2473984, 2482176)
    expected_rows = [S + R8 * k for k in range(1, C.expected_update_rows(S, E) + 1)]
    assert list(SB.PRIMARY_STEPS) == expected_rows[-8:] and len(expected_rows) == 175
    assert SB.BASELINE_STEPS == tuple(range(991232, 1048576 + 1, R8))
    assert SB.RISE_MIN == Fraction(1, 10) and SB.ALPHA == Fraction(1, 20)
    real = Path(C.BS) / "SCREEN.json"
    if real.exists():
        d, problems = C.load_screen(str(real))
        assert problems == [] and C.queue_of(d) == SB.QUEUE and d["intervention"] == BLIND
    assert set(SB.PINNED_TOOLS) == set(C.TOOL_FILES) | {"score_bs.py"}


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
    assert idx == list(range(120, 128))
    assert float(np.mean(m[idx])) == SB.BASELINE_S


# ------------------------------------------------------------------------------------------------ the test itself
def test_perm_test_matches_an_independent_implementation():
    rng = random.Random(7)
    for _ in range(200):
        c = [rng.gauss(0.2, 0.05) for _ in range(4)]
        b = [rng.gauss(0.1, 0.05) for _ in range(4)]
        p, count, total = SB.perm_test([(0, Fraction(x)) for x in c], [(0, Fraction(x)) for x in b])
        assert total == 70 and p == Fraction(count, 70)
        assert float(p) == pytest.approx(_brute_p(c, b), abs=1e-12)


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


def test_sign_flip():
    assert SB.sign_flip_test([Fraction(x) for x in (1, 2, 3, 4)])[:2] == (Fraction(1, 16), 1)
    assert SB.sign_flip_test([Fraction(0)] * 4)[:2] == (Fraction(1), 16)
    assert SB.sign_flip_test([]) is None


# ------------------------------------------------------------------------------------------------ the reading lines
HIGH = (0.20, 0.21, 0.22, 0.23)


def test_line_reduced(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    res, text, code = _score(tmp_path)
    assert res["line"] == 2 and res["line_key"] == "REDUCED"
    assert res["p_filled"]["count"] == 1 and res["p_filled"]["total"] == 70
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL        # no APPROVED in a synthetic root
    assert "REDUCED: BLINDING THE CRITIC" in text and "1/70 (0.0143)" in text
    assert res["paired_sign_flip"]["count"] == 1 and res["paired_sign_flip"]["total"] == 16


@pytest.mark.parametrize("c, b, line", [((8, 7, 6, 3.5), (5, 4, 2, 1), 2), ((8, 7, 6, 3), (5, 4, 2, 1), 3)])
def test_line_at_the_boundary(tmp_path, c, b, line):
    s = Screen(tmp_path)
    s.all([0.125 + x / 64 for x in c], [0.125 + x / 64 for x in b])        # exact in binary: no rounding ties
    res, _text, _code = _score(tmp_path)
    assert res["line"] == line and res["informative"] is True


def test_line_no_reduction_detected(tmp_path):
    s = Screen(tmp_path)
    c, b = [x / 128 for x in (26, 19, 32, 23)], [x / 128 for x in (27, 22, 24, 31)]   # exact; 26+19+24+31 ties
    s.all(c, b)
    res, text, _ = _score(tmp_path)
    assert res["line"] == 3 and res["additions"] == []
    assert "NO REDUCTION DETECTED IN THE ON-POLICY TAIL_3 COMMAND" in text
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
    exact = sum(1 for sub in itertools.combinations(range(8), 4)
                if sum(([v for v in c + b][i][1] for i in sub), Fraction(0)) >= sum(v[1] for v in c))
    assert p[1] == exact


def test_line_not_informative_comes_first(tmp_path):
    s = Screen(tmp_path)
    low = [B_S + 0.05, B_S + 0.06, B_S + 0.07, B_S + 0.08]                  # control median rise 0.065 < 0.10
    s.all(low, [x - 0.2 for x in low])                                      # the blind arm far lower: p = 1/70
    res, text, _ = _score(tmp_path)
    assert res["line"] == 1 and res["line_key"] == "NOT INFORMATIVE" and res["p_filled"]["count"] == 1
    assert res["informative"] is False and res["control_rise"] == pytest.approx(0.065)
    assert any("did not rise" in a for a in res["additions"])
    assert "NOT INFORMATIVE: THE CONTROL ARM DID NOT REPRODUCE" in text


def test_informativeness_threshold_is_inclusive():
    classes = [{"name": C.run_name(a, sd), "arm": a, "seed": sd} for a, sd in SB.QUEUE]
    edge = Fraction(B_S) + Fraction(1, 10)
    for ctrl, informative in ((edge, True), (edge - Fraction(1, 10**12), False)):
        prim = {C.run_name(a, sd): (ctrl if a == C.CONTROL else Fraction(1, 2)) for a, sd in SB.QUEUE}
        dec = SB.decide(classes, prim)
        assert dec["informative"] is informative and dec["line"] == (3 if informative else 1)
        assert dec["control_rise"] == ctrl - Fraction(B_S)


def test_informativeness_uses_the_median_not_the_mean():
    # control rises 0.05, 0.06, 0.07, 0.40 from B_S: median 0.065 (< 0.10, NOT INFORMATIVE), mean 0.145 (>= 0.10)
    classes = [{"name": C.run_name(a, sd), "arm": a, "seed": sd} for a, sd in SB.QUEUE]
    rises = dict(zip(SB.SEEDS, (Fraction(5, 100), Fraction(6, 100), Fraction(7, 100), Fraction(40, 100)), strict=True))
    prim = {C.run_name(a, sd): (Fraction(B_S) + rises[sd] if a == C.CONTROL else Fraction(B_S)) for a, sd in SB.QUEUE}
    dec = SB.decide(classes, prim)
    assert dec["medians"]["control"] - Fraction(B_S) == Fraction(65, 1000)
    assert dec["informative"] is False and dec["line"] == 1
    # and the other way round: 0.00, 0.11, 0.12, 0.13 -> median 0.115 informative although the mean 0.09 is below
    # 0.10 (the zero ties with the blind values: p = 5/70, line 3)
    rises = dict(zip(SB.SEEDS, (Fraction(0), Fraction(11, 100), Fraction(12, 100), Fraction(13, 100)), strict=True))
    prim = {C.run_name(a, sd): (Fraction(B_S) + rises[sd] if a == C.CONTROL else Fraction(B_S)) for a, sd in SB.QUEUE}
    dec = SB.decide(classes, prim)
    assert dec["control_rise"] == Fraction(115, 1000) and dec["informative"] is True and dec["line"] == 3


# ------------------------------------------------------------------------------------------------ missing and the fill
def test_fill_blind_lost(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    s.lost(C.run_name(BLIND, 2131))
    res, text, _ = _score(tmp_path)
    assert res["line"] == 3 and res["p_filled"]["count"] == 36
    assert res["p_as_counted"]["count"] == 1 and res["p_as_counted"]["total"] == 35
    assert res["additions"] == [SB.ADDITION_AS_COUNTED]
    assert res["additions"] == ["(the as-counted p, report only, is <= 0.05; the filled p decides)"]
    assert "REDUCED holds only" not in text                                  # not worded as partial support (R2)
    assert "MISSING, filled at +inf: lost.json: crashes" in text


def test_fill_control_stopped(tmp_path):
    s = Screen(tmp_path)
    for seed, a, b in zip(SB.SEEDS, HIGH, (0.00, 0.01, 0.02, 0.03), strict=True):
        if seed == 2101:
            s.run(C.CONTROL, seed, a, stop="collapse", T=E - C.EVAL_FREQ)
        else:
            s.run(C.CONTROL, seed, a)
        s.run(BLIND, seed, b)
    res, text, _ = _score(tmp_path)
    cls = {c["name"]: c for c in res["classes"]}
    assert cls["control_2101"]["cls"] == "STOPPED" and cls["control_2101"]["primary"] is None
    assert res["p_filled"]["count"] == 36 and res["line"] == 3
    assert "MISSING, filled at -inf" in text and "the collapse stop at T = 2,359,296" in text


def test_primary_unreadable_is_missing(tmp_path):
    s = Screen(tmp_path)
    for seed, a, b in zip(SB.SEEDS, HIGH, (0.00, 0.01, 0.02, 0.03), strict=True):
        s.run(C.CONTROL, seed, a)
        s.run(BLIND, seed, b, nan_at=SB.PRIMARY_STEPS[3] if seed == 2111 else None)
    res, text, _ = _score(tmp_path)
    cls = {c["name"]: c for c in res["classes"]}
    assert cls[f"{BLIND}_2111"]["primary"] is None and "primary unreadable" in cls[f"{BLIND}_2111"]["primary_cause"]
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
    for seed, b in zip(SB.SEEDS, (0.0, 0.01, 0.02, 0.03), strict=True):
        s.lost(C.run_name(C.CONTROL, seed))
        s.run(BLIND, seed, b)
    res, _t, _c = _score(tmp_path)
    assert res["line"] == 1 and res["control_median"] is None


# ------------------------------------------------------------------------------------------------ outcome-blind
def test_incomplete_opens_no_markers_and_prints_no_outcome(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    (tmp_path / "runs" / f"{BLIND}_2131" / C.CHECK_JSON).unlink()           # still running
    (tmp_path / "reads" / "control_2101" / "markers.json").write_text("not json {")   # never opened
    res, text, code = _score(tmp_path)
    assert res["status"] == "incomplete" and code == SB.EXIT_NOT_FINAL
    assert "THE READING" not in text and "0.2" not in text.split("STATUS")[1].split("PROVISIONAL")[0]
    assert f"{BLIND}_2131" in text and "undecided" in text


def test_decided_but_unread_is_incomplete(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    (tmp_path / "reads" / "control_2111" / "markers.json").unlink()
    res, text, code = _score(tmp_path)
    assert res["status"] == "incomplete" and "decided, markers not read yet" in text and code == 3


def test_check_failed_is_undecided(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    s.run(BLIND, 2121, 0.02, passed=False)
    res, text, _ = _score(tmp_path)
    assert res["status"] == "incomplete" and "CHECK failed" in text


# ------------------------------------------------------------------------------------------------ input errors
def test_input_errors(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    assert _score(tmp_path)[2] in (0, 3)
    # SCREEN.json not the criteria's queue
    d = json.loads((tmp_path / "SCREEN.json").read_text())
    d["queue"] = d["queue"][:7]
    _write(tmp_path / "SCREEN.json", d)
    with pytest.raises(SB.InputError, match="not the criteria's"):
        SB.score(tmp_path)
    assert SB.main(["--root", str(tmp_path)]) == SB.EXIT_INPUT


@pytest.mark.parametrize("mutate, match", [
    (lambda m: m.update(arm="control"), "arm"),
    (lambda m: m["probe"].update(sha256="cd" * 32), "another policy_probe.npz"),
    (lambda m: m["loop"][1]["critic_slope_obs6"]["ONPOLICY"]["effective"].update(dV_dq3_per_deg=0.003),
     "must be 0.0"),
    (lambda m: m["loop"][0].update(critic_form="plain"), "critic form"),
    (lambda m: m["probe"]["steps"].pop(), "probe rows"),
    (lambda m: m.update(T=E - R8), "T"),
])
def test_markers_that_disagree_are_input_errors(tmp_path, mutate, match):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    p = tmp_path / "reads" / f"{BLIND}_2101" / "markers.json"
    m = json.loads(p.read_text())
    mutate(m)
    _write(p, m)
    with pytest.raises(SB.InputError, match=match):
        SB.score(tmp_path)


def test_lost_outside_queue_and_check_mismatch(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    s.lost("control_9999")
    with pytest.raises(SB.InputError, match="outside the queue"):
        SB.score(tmp_path)
    (tmp_path / "lost.json").unlink()
    p = tmp_path / "runs" / "control_2101" / C.CHECK_JSON
    c = json.loads(p.read_text())
    c["seed"] = 2111
    _write(p, c)
    with pytest.raises(SB.InputError, match="seed"):
        SB.score(tmp_path)


def test_out_outside_root_refused(tmp_path):
    s = Screen(tmp_path / "root")
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    assert SB.main(["--root", str(tmp_path / "root"), "--out", str(tmp_path / "elsewhere")]) == SB.EXIT_INPUT
    assert not (tmp_path / "elsewhere").exists()


# ------------------------------------------------------------------------------------------------ PROVISIONAL / final
def _approve(root, monkeypatch, scorer_sha=None):
    crit = root / "criteria.txt"
    crit.write_text(f"criteria ... tools/score_bs.py {scorer_sha or _sha(TOOLS / 'score_bs.py')}\n")
    appr = {"criteria_sha256": _sha(crit), "screen_sha256": _sha(root / "SCREEN.json"),
            "code_tree_digest": C.TREE_DIGEST,
            "tool_sha256": {t: _sha(TOOLS / t) for t in SB.PINNED_TOOLS}, "approved_by": "test"}
    _write(root / "APPROVED", appr)
    monkeypatch.setattr(SB, "code_seal_items", lambda d: [])
    return _sha(crit), _sha(root / "SCREEN.json"), _sha(root / "APPROVED")


def test_final_when_every_pin_holds(tmp_path, monkeypatch):
    s = Screen(tmp_path)
    crit, screen, appr = _approve(tmp_path, monkeypatch)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03), meta_kw={"crit": crit, "screen": screen, "appr": appr})
    res, text, code = _score(tmp_path)
    assert res["provisional"] == [] and res["status"] == "final" and code == SB.EXIT_FINAL
    assert SB.main(["--root", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "score" / "score.json").read_text())
    assert out["line_key"] == "REDUCED" and (tmp_path / "score" / "score.txt").read_text().startswith(SB.VERSION)


@pytest.mark.parametrize("breakage, match", [
    ("criteria", "criteria.txt sha256"), ("scorer_pin", "does not name this scorer"),
    ("run_meta_criteria", "ran under criteria.txt"), ("tool", "tools/score_bs.py"), ("ledger", "ledger"),
    ("approved_missing", "no "), ("run_meta_changed", "run_meta.json changed since CHECK"),
    ("markers_tool", "markers.json made by read_bs.py"),
])
def test_provisional_items(tmp_path, monkeypatch, breakage, match):
    s = Screen(tmp_path)
    crit, screen, appr = _approve(tmp_path, monkeypatch, scorer_sha="0" * 64 if breakage == "scorer_pin" else None)
    if breakage == "scorer_pin":
        crit, screen, appr = _sha(tmp_path / "criteria.txt"), screen, appr
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03), meta_kw={"crit": "f" * 64 if breakage == "run_meta_criteria" else crit,
                                                   "screen": screen, "appr": appr})
    if breakage == "criteria":
        (tmp_path / "criteria.txt").write_text("edited after approval\n" + (tmp_path / "criteria.txt").read_text())
    if breakage == "tool":
        a = json.loads((tmp_path / "APPROVED").read_text())
        a["tool_sha256"]["score_bs.py"] = "e" * 64
        _write(tmp_path / "APPROVED", a)
    if breakage == "ledger":
        with open(tmp_path / "runs" / "control_2101" / "01_stance" / "eval_episodes.csv", "a") as fh:
            fh.write("99,2490368,10,0,1.0,1000,1,truncated\n")
    if breakage == "approved_missing":
        (tmp_path / "APPROVED").unlink()
    if breakage == "run_meta_changed":
        p = tmp_path / "runs" / f"{BLIND}_2111" / C.RUN_META
        p.write_text(p.read_text() + " ")
    if breakage == "markers_tool":                                          # markers made by another read_bs.py (R6)
        p = tmp_path / "reads" / "control_2121" / "markers.json"
        m = json.loads(p.read_text())
        m["tool_sha256"] = "d" * 64
        _write(p, m)
    res, _text, code = _score(tmp_path)
    assert res["status"] == "PROVISIONAL" and code == SB.EXIT_NOT_FINAL
    assert any(match in x for x in res["provisional"]), res["provisional"]
    assert res["line_key"] == "REDUCED"                                     # printed, not acted on


# ------------------------------------------------------------------------------------------------ report only
def test_report_only_readings(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    res, text, _ = _score(tmp_path)
    ro = res["report_only"][f"{BLIND}_2101"]
    assert ro["ledger"]["status"] == "read" and ro["ledger"]["n_falls"] == 1 and ro["ledger"]["n_episodes"] == 110
    assert ro["ledger"]["mean_return_at_T"] == pytest.approx(4010.0)
    assert ro["manipulation"]["max_dv_predict_values"] == 0.0
    assert res["report_only"]["control_2101"]["ledger"]["n_falls"] == 0
    assert "MANIPULATION" in text and "CriticBlindTail3MlpExtractor" in text and "PAIRED DIFFERENCES" in text
    assert [p["diff"] for p in res["paired"]] == pytest.approx([0.20, 0.20, 0.20, 0.20])
    assert float(Fraction(ro["early8_rollout_mu14"])) < 0.01
    # each arm's SD of the primaries read, report only (R2): printed beside the mean and median, in score.json
    sd = statistics.stdev(HIGH)
    assert res["arm_sd_report_only"]["control"] == pytest.approx(sd)
    assert res["arm_sd_report_only"][BLIND] == pytest.approx(statistics.stdev((0.00, 0.01, 0.02, 0.03)))
    assert f"SD {sd:.5f} (report only)" in text


def test_arm_sd_needs_two_values_and_never_decides():
    classes = [{"name": C.run_name(a, sd), "arm": a, "seed": sd} for a, sd in SB.QUEUE]
    prim = {C.run_name(a, sd): (Fraction(1, 4) if a == C.CONTROL else Fraction(1, 8)) for a, sd in SB.QUEUE}
    prim[C.run_name(BLIND, 2101)] = prim[C.run_name(BLIND, 2111)] = prim[C.run_name(BLIND, 2121)] = None
    dec = SB.decide(classes, prim)
    assert dec["sds"] == {"control": 0.0, BLIND: None}
    # three blind values filled at +inf: 65 relabelings hold one of them, plus the observed split itself
    assert dec["line"] == 3 and dec["filled"][1] == 66 and dec["additions"] == []


def test_line_texts_are_the_criteria_text():
    """The scorer's line texts and the line-3 addition appear verbatim (whitespace collapsed) in <BS>/criteria.txt."""
    real = Path(C.BS) / "criteria.txt"
    if not real.exists():
        pytest.skip("no criteria.txt in this build")
    text = " ".join(real.read_text(encoding="utf-8").split())
    for line, t in SB.LINE_TEXT.items():
        assert " ".join(t.split()) in text, line
    assert SB.ADDITION_AS_COUNTED in text


def test_report_only_never_fails_the_score(tmp_path):
    s = Screen(tmp_path)
    s.all(HIGH, (0.00, 0.01, 0.02, 0.03))
    p = tmp_path / "reads" / f"{BLIND}_2101" / "markers.json"
    m = json.loads(p.read_text())
    m["loop"][0]["critic_slope_obs6"]["REF"]["effective"] = None          # a null entry is not an error
    led = Path(m["ledger"]["file"])
    led.write_bytes(b"\xff\xfe not utf-8")
    m["ledger"]["sha256"] = _sha(led)
    _write(p, m)
    res, _text, _code = _score(tmp_path)
    assert res["line"] == 2
    assert res["report_only"][f"{BLIND}_2101"]["ledger"]["status"].startswith("unreadable (UnicodeDecodeError")
