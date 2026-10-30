"""Lane R tests: read_ecs.py (build list v2 item 9, the reader's part).  No optimizer step, no simulation: synthetic
records, the parents' F2 records read only, and the pinned M5 (a ledger reader) as a subprocess."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("test_read_score_fixtures",
                                               Path(__file__).resolve().parent / "test_read_score_fixtures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
E, R = F.E, F.R
G = F.GRID
S42, E42 = E.window("s42")
S80, E80 = E.window("s80")


# ================================================================================================ batteries
# Each battery raises AssertionError when the module under test reads wrongly; the mutant tests run them on mutants.
def m5_json(S, T, any_counts=None, tail_counts=None, failures=(), last=None):
    """A synthetic M5 JSON (the keys ledger_r10round.py writes) on the grid up to `last` (default T)."""
    any_counts, tail_counts = any_counts or {}, tail_counts or {}
    steps = F.grid_to(last or T)
    per = {str(s): dict(n=10, any=any_counts.get(s, 0), tail=tail_counts.get(s, 0), unres=0, tail_by={}, any_by={},
                        reasons={}, any_missing=[], tail_missing=[], rail=[]) for s in steps}
    af = [s for s in steps if any_counts.get(s, 0) >= 5]
    tf = [s for s in steps if tail_counts.get(s, 0) >= 5]
    aa = [s for s in steps if s in af and s - G in af]
    ta = [s for s in steps if s in tf and s - G in tf]
    return dict(per_evaluation=per, any_flagged=af, tail_flagged=tf, any_alarms=aa, tail_alarms=ta,
                needs_resolution=[], c6=dict(passed=not failures, failures=list(failures), E=T))


def battery_window(Rm):
    S, T = 3 * G, 8 * G
    w = Rm.window_reading(m5_json(S, T, {S: 6}), S, T)
    assert w["positive"] is False and w["first_any_flag"] is None, "a flag at S (the parent's) is not in (S, T]"
    assert w["n_any_flagged"] == 0, "the flagged count is the window's (review mutant r_n_flagged_all)"
    assert w["flagged_at_or_before_S"] == [S]
    w = Rm.window_reading(m5_json(S, T, {S + G: 5}), S, T)
    assert w["positive"] is True and w["first_any_flag"] == S + G and w["n_any_flagged"] == 1
    w = Rm.window_reading(m5_json(S, T, {S + G: 4}), S, T)
    assert w["positive"] is False, "4 of 10 rows is not ANY-flagged"
    w = Rm.window_reading(m5_json(S, T, {T: 7}), S, T)
    assert w["positive"] is True and w["first_any_flag"] == T, "the evaluation at T is in (S, T]"
    w = Rm.window_reading(m5_json(S, T, {T + G: 7}, last=T + G), S, T)
    assert w["positive"] is False, "an evaluation after T is not read"
    w = Rm.window_reading(m5_json(S, T, {S: 6, S + G: 6}), S, T)
    assert w["any_alarms"] == [] and w["any_alarms_straddling_S"] == [S + G], "an alarm straddling S is not counted"
    assert w["positive"] is True and w["first_any_flag"] == S + G
    w = Rm.window_reading(m5_json(S, T, {S + G: 6, S + 2 * G: 6, S + 4 * G: 9}), S, T)
    assert w["any_alarms"] == [S + 2 * G] and w["any_flagged"] == [S + G, S + 2 * G, S + 4 * G]
    assert w["n_any_flagged"] == 3
    w = Rm.window_reading(m5_json(S, T, {S: 5, S + G: 5}, {S: 5, S + G: 5, S + 2 * G: 5, T: 5}), S, T)
    assert w["tail_alarms"] == [S + 2 * G] and w["tail_alarms_straddling_S"] == [S + G]
    assert w["tail_flagged"] == [S + G, S + 2 * G, T]
    with pytest.raises(Rm.InputError):
        bad = m5_json(S, T, {S + G: 6})
        bad["any_flagged"] = []
        Rm.window_reading(bad, S, T)
    with pytest.raises(Rm.InputError):
        Rm.window_reading(m5_json(S, T + G), S, T)            # C6 run with another E
    ok_tail = m5_json(S, T, failures=[f"evaluation {S + G} episode 9: TAIL input missing (fraction over the bar and "
                                      "saturated_actuator blank or naming no actuator)"])
    w = Rm.window_reading(ok_tail, S, T)
    assert w["c6_ok"] is True and len(w["c6_tail_missing"]) == 1 and w["c6_passed_m5"] is False
    for f in (f"evaluation {S + G} episode 9: ANY input missing (screen unmeasured and fraction missing)",
              f"evaluation {S + G}: 9 rows, episodes ['0', '1']", f"evaluation {S + 2 * G} absent",
              f"evaluation {T + G} beyond E = {T}", "row 7 timesteps '131073' is not on the grid", "anything else"):
        w = Rm.window_reading(m5_json(S, T, failures=[f]), S, T)
        assert w["c6_ok"] is False and len(w["c6_counting_failures"]) == 1, f
    return True


def battery_c6(Rm):
    cases = {
        "row 12 timesteps '131073' is not on the grid": ("grid", None),
        "evaluation 1179648 beyond E = 1048576": ("grid", 1179648),
        "evaluation 524288 absent": ("grid", 524288),
        "evaluation 524288: 9 rows, episodes ['0', '1', '2']": ("rows", 524288),
        "evaluation 524288 episode 9: ANY input missing (screen unmeasured and fraction missing)": ("any", 524288),
        "evaluation 524288 episode 3: TAIL input missing (fraction over the bar and saturated_actuator blank or naming "
        "no actuator)": ("tail", 524288),
        "evaluation 524288 episode 3: something new": ("other", None),
    }
    for f, want in cases.items():
        assert Rm.classify_c6(f) == want, (f, Rm.classify_c6(f))
    return True


_PARENT = {}


def parent_cols(start):
    if start not in _PARENT:
        _PARENT[start] = F.parent_arrays(start, E.PROBE_CHECK_COLUMNS)
    return _PARENT[start]


def window_of(start, T, arm="A", edit=None):
    """Continuation probe columns built from the parent's rows (seeded rows <= S kept, rows in (S, T))."""
    S, _ = E.window(start)
    p = parent_cols(start)
    u = p["update_timesteps"].astype(np.int64)
    keep = [i for i, x in enumerate(u.tolist()) if x < T]
    c = {k: np.array(v[keep]) for k, v in p.items()}
    win = [j for j, x in enumerate(c["update_timesteps"].tolist()) if S < x < T]
    if arm == "B":
        c["ent_coef"][win] = np.float32(0.0)
    if edit:
        c = edit(c, win) or c
    return c


def battery_r4(Rm):
    for start in ("s42", "s80"):
        S, Eend = E.window(start)
        p = parent_cols(start)
        fails, det = Rm.r4(window_of(start, Eend), p, "A", S, Eend)
        assert fails == [] and det["rows_in_window"] == 191 and det["first_step"] == S + E.ROLLOUT, fails
        assert det["last_step"] == Eend - E.ROLLOUT
        fails, _ = Rm.r4(window_of(start, Eend, "B"), p, "B", S, Eend)
        assert fails == [], fails
    S, Eend, p = S42, E42, parent_cols("s42")
    assert Rm.r4(window_of("s42", Eend, "B"), p, "A", S, Eend)[0], "zeros are not arm A's values"
    assert Rm.r4(window_of("s42", Eend), p, "B", S, Eend)[0], "the recorded values are not arm B's zeros"

    def one_ulp(c, win):
        c["ent_coef"][win[7]] = np.nextafter(c["ent_coef"][win[7]], np.float32(1.0))
    fails, det = Rm.r4(window_of("s42", Eend, edit=one_ulp), p, "A", S, Eend)
    assert fails and det["mismatches"][0]["step"] == S + 8 * E.ROLLOUT, "a row changed by one float32 ulp fails"

    def lr(c, win):
        c["learning_rate"][win[0]] = c["learning_rate"][win[0]] * np.float32(1.5)
    assert Rm.r4(window_of("s42", Eend, edit=lr), p, "A", S, Eend)[0], "learning_rate is checked from the first row"

    def b_one(c, win):
        c["ent_coef"][win[-1]] = np.float32(1e-9)
    assert Rm.r4(window_of("s42", Eend, "B", edit=b_one), p, "B", S, Eend)[0]

    def drop(c, win):
        keep = [i for i in range(len(c["update_timesteps"])) if i != win[3]]
        return {k: v[keep] for k, v in c.items()}
    fails, det = Rm.r4(window_of("s42", Eend, edit=drop), p, "A", S, Eend)
    assert fails and det["rows_in_window"] == 190, "a wrong row count fails"

    def dup(c, win):
        idx = list(range(len(c["update_timesteps"])))
        idx.insert(win[5], win[5])
        return {k: v[idx] for k, v in c.items()}
    assert Rm.r4(window_of("s42", Eend, edit=dup), p, "A", S, Eend)[0], "a repeated row fails"

    def at_T(c, win):
        out = {k: np.concatenate([v, v[win[-1]:win[-1] + 1]]) for k, v in c.items()}
        out["update_timesteps"][-1] = Eend
        return out
    fails, det = Rm.r4(window_of("s42", Eend, edit=at_T), p, "A", S, Eend)
    assert fails and det["n_rows_at_or_after_T"] == 1, "a row at T (an update on the end rollout) fails"
    T = S + 10 * G
    fails, det = Rm.r4(window_of("s42", T), p, "A", S, T)
    assert fails == [] and det["rows_in_window"] == (T - S) // E.ROLLOUT - 1, "a STOPPED run's (S, T) rows"
    return True


def test_window_reading():
    assert battery_window(R)


def test_c6_classification():
    assert battery_c6(R)


def test_r4_parent_probe():
    assert battery_r4(R)


def test_parent_probe_is_pinned(monkeypatch):
    for start in ("s42", "s80"):
        path, sha, cols = R.parent_probe(start)
        assert sha == E.STARTS[start]["records"]["policy_probe.npz"] and set(cols) == set(E.PROBE_CHECK_COLUMNS)
    # review RR-3 (mutant r_parent_probe_unpinned): a parent probe that is not FMAN's is refused, never read
    monkeypatch.setitem(E.STARTS["s42"]["records"], "policy_probe.npz", "0" * 64)
    with pytest.raises(R.InputError, match="sha256 .* != FMAN's"):
        R.parent_probe("s42")


# ================================================================================================ CHECK on run folders
def _check(root, run):
    code, rec = R.check_run(run, root)
    on_disk = json.loads((Path(run) / R.CHECK_NAME).read_text())
    assert on_disk["passed"] == rec["passed"]
    return code, rec


def test_check_passes_and_fails_on_run_meta():
    root = F.work_dir("check_meta")
    cases = {
        "s42_A_1101": (dict(), True),
        "s80_B_1401": (dict(), True),
        "s42_A_1111": (dict(meta_over=dict(stop_reason="other")), False),
        "s42_A_1121": (dict(meta_over=dict(code=dict(tree_digest="0" * 64, commit=E.COMMIT))), False),
        "s42_A_1131": (dict(meta_over=dict(start_pair=dict(zip_sha256="0" * 64, pkl_sha256="1" * 64))), False),
        "s42_B_1201": (dict(meta_over=dict(arm_ppo={"ent_coef": 0.005, "ent_coef_end": 0.0})), False),
        "s42_B_1211": (dict(meta_over=dict(ent_coef_after_load=0.005)), False),
        "s42_A_1141": (dict(meta_over=dict(status="running", exit=None)), False),
        "s42_A_1151": (dict(meta_over=dict(stop_at_end=dict(target=E42, flag=False, step=None))), False),
        "s80_A_1301": (dict(meta_over=dict(parent_records={})), False),
        "s80_A_1311": (dict(meta_over=dict(schema="ecs-run/0")), False),
        "s80_A_1321": (dict(meta_over=dict(mode="no_update")), False),
        "s80_A_1341": (dict(meta_over=dict(arm_ppo={"ent_coef": 0.005, "ent_coef_end": 0.0})), False),
        "s80_A_1351": (dict(meta_over=dict(recipe_v5={"problems": ["hyperparameters differ"]})), False),
        "s80_B_1411": (dict(meta_over=dict(arm_ppo="unchanged")), False),
    }
    for name, (kw, want) in cases.items():
        p = E.parse_run_name(name)
        run = F.make_run(root / "runs" / name, p["start"], p["arm"], p["seed"], **kw)
        code, rec = _check(root, run)
        assert rec["passed"] is want and code == (0 if want else 4), (name, rec["failures"])
    # a seed outside the arm's list (T1)
    run = F.make_run(root / "runs" / "s42_A_1102", "s42", "A", 1102)
    code, rec = _check(root, run)
    assert code == 4 and any("seed" in f for f in rec["failures"])
    # run_meta disagreeing with its folder's name
    run = F.make_run(root / "runs" / "s80_A_1331", "s80", "A", 1331, meta_over=dict(seed=1341, name="s80_A_1341"))
    code, rec = _check(root, run)
    assert code == 4 and any("folder" in f for f in rec["failures"])


def test_check_stop_reasons():
    root = F.work_dir("check_stop")
    T = S42 + 10 * G
    run = F.make_run(root / "runs" / "s42_A_1101", "s42", "A", 1101, T=T, stop="collapse")
    code, rec = _check(root, run)
    assert code == 0 and rec["T"] == T and rec["stop_reason"] == "collapse"
    run = F.make_run(root / "runs" / "s42_A_1111", "s42", "A", 1111, T=T, stop="collapse",
                     meta_over=dict(collapse=dict(found=1, consecutive_drops=3, patience=10, tripped=False)))
    code, rec = _check(root, run)
    assert code == 4, "a collapse stop without the callback's state is VOID"
    assert rec["failures"] and all("without the collapse callback's state" in f for f in rec["failures"]), \
        rec["failures"]
    run = F.make_run(root / "runs" / "s42_A_1121", "s42", "A", 1121, T=T, stop="end")
    code, rec = _check(root, run)
    assert code == 4 and any("not at" in f for f in rec["failures"]), "'end' must be at E"
    run = F.make_run(root / "runs" / "s42_A_1131", "s42", "A", 1131, T=T, stop="other")
    assert _check(root, run)[0] == 4
    run = F.make_run(root / "runs" / "s42_B_1201", "s42", "B", 1201, T=E42 + G, stop="end", target=E42)
    assert _check(root, run)[0] == 4, "T past E fails"
    # review RR-3 (mutant r_collapse_at_E_allowed): a collapse stop at T == E is not a STOPPED continuation
    run = F.make_run(root / "runs" / "s42_B_1211", "s42", "B", 1211, T=E42, stop="collapse")
    code, rec = _check(root, run)
    assert code == 4 and any("'collapse' at T" in f for f in rec["failures"])


def test_check_review_conditions():
    """Review RR-3 (target, status, identity), H6 (torch threads) and I6 (updates made, the collapse stop present)."""
    root = F.work_dir("check_review")
    d0 = dict(params="p0", adam="a0", vecnormalize="v0", all="x0", n_updates=3458)
    cases = {
        "s42_A_1101": (dict(target=E42 - G), "target"),                       # r_target_unchecked
        "s42_A_1111": (dict(status="running", exit=0), "status"),             # r_status_unchecked
        "s42_A_1121": (dict(torch_threads=2), "torch_threads"),
        "s42_A_1131": (dict(torch_threads=None), "torch_threads"),
        "s42_A_1141": (dict(collapse=dict(found=0, consecutive_drops=0, patience=None, tripped=False)),
                       "collapse stop"),
        "s42_A_1151": (dict(collapse=dict(found=1, consecutive_drops=0, patience=9, tripped=False)), "collapse stop"),
        "s42_B_1201": (dict(digest_end=dict(d0, vecnormalize="v1", all="x1")), "no update made"),
        "s42_B_1211": (dict(digest_end=dict(d0, params="p1")), "did not grow"),
        "s42_B_1221": (dict(digest_end=dict(d0, params="p1", n_updates=None)), "did not grow"),
    }
    for name, (over, text) in cases.items():
        p = E.parse_run_name(name)
        run = F.make_run(root / "runs" / name, p["start"], p["arm"], p["seed"], meta_over=over)
        code, rec = _check(root, run)
        assert code == 4 and any(text in f for f in rec["failures"]), (name, rec["failures"])
    # a continuation's identity is its folder's name (mutant r_identity_from_meta)
    run = F.make_run(root / "runs" / "s42_B_1231", "s42", "B", 1231, meta_over=dict(seed=1241, name="s42_B_1241"))
    code, rec = _check(root, run)
    assert code == 4 and rec["seed"] == 1231 and any("folder" in f for f in rec["failures"])
    # a validation (no-update) run needs the thread count and the collapse stop too, and no update
    run = F.make_run(root / "validate" / "v2" / "s80_A_1301", "s80", "A", 1301, mode="no_update",
                     target=S80 + E.VALIDATE_LENGTH, meta_over=dict(torch_threads=4))
    assert _check(root, run)[0] == 4


def test_decided_requires_a_passing_check():
    """Review RR-3 (mutant r_decided_ignores_passed): a failing check.json in runs/<name> keeps the outcome-blind
    gate closed."""
    root = F.work_dir("decided_gate")
    for start, arm, seed in E.QUEUE:
        F.passing_check(root, start, arm, seed)
    assert R.decided(root) == (True, [])
    p = root / "runs" / "s80_B_1451" / R.CHECK_NAME
    d = json.loads(p.read_text())
    d["passed"] = False
    p.write_text(json.dumps(d))
    assert R.decided(root) == (False, ["s80_B_1451"])
    name = "s42_A_1101"
    run = F.make_run(root / "runs" / name, "s42", "A", 1101)
    assert R.check_run(run, root)[0] == 0
    with pytest.raises(R.InputError, match="not decided"):
        R.full_read(run, root)


def test_check_probe_failures():
    root = F.work_dir("check_probe")

    def change(a, win):
        a["ent_coef"][win[100]] = a["ent_coef"][win[100]] * np.float32(2.0)
    run = F.make_run(root / "runs" / "s80_A_1301", "s80", "A", 1301, probe_mutate=change)
    code, rec = _check(root, run)
    assert code == 4 and rec["r4"]["mismatches"][0]["step"] == S80 + 101 * E.ROLLOUT

    def drop_all(a, win):
        keep = [i for i in range(len(a["update_timesteps"])) if i not in set(win[:5])]
        return {k: v[keep] for k, v in a.items()}
    run = F.make_run(root / "runs" / "s80_A_1311", "s80", "A", 1311, probe_mutate=drop_all)
    assert _check(root, run)[0] == 4
    run = F.make_run(root / "runs" / "s80_B_1401", "s80", "B", 1401)
    (run / R.PROBE_REL).unlink()
    code, rec = _check(root, run)
    assert code == 4 and any("policy_probe.npz" in f for f in rec["failures"])


def test_check_validation_no_update():
    root = F.work_dir("check_v2")
    target = S42 + E.VALIDATE_LENGTH
    run = F.make_run(root / "validate" / "v2" / "s42_A_1101", "s42", "A", 1101, mode="no_update", target=target)
    code, rec = _check(root, run)
    assert code == 0 and rec["mode"] == "validation" and rec["r4"]["rows_in_window"] == 31, rec["failures"]
    run = F.make_run(root / "validate" / "v2" / "s42_B_1201", "s42", "B", 1201, mode="no_update", target=target,
                     meta_over=dict(digest_end=dict(params="changed", adam="a0", vecnormalize="v1", all="x1")))
    code, rec = _check(root, run)
    assert code == 4 and any("params digest" in f for f in rec["failures"]), "an optimizer step is caught"
    run = F.make_run(root / "validate" / "v2" / "s80_A_1301", "s80", "A", 1301, mode="continuation", target=S80 + G * 2)
    assert _check(root, run)[0] == 4, "a validation run must be a no-update run"


def test_check_input_errors_and_main():
    root = F.work_dir("check_main")
    run = F.make_run(root / "runs" / "s42_A_1101_crashed1", "s42", "A", 1101)
    with pytest.raises(R.InputError):
        R.check_run(run, root)
    assert R.main(["--root", str(root), "--check", str(run)]) == 2
    assert R.main(["--root", str(root), "--check", str(root / "runs" / "absent")]) == 2
    other = F.work_dir("check_main_outside")
    assert R.main(["--root", str(root), "--check", str(other)]) == 2
    run = F.make_run(root / "runs" / "s42_A_1111", "s42", "A", 1111)
    assert R.main(["--root", str(root), "--check", str(run)]) == 0
    run = F.make_run(root / "runs" / "s42_A_1121", "s42", "A", 1121, meta_over=dict(stop_reason="other"))
    assert R.main(["--root", str(root), "--check", str(run)]) == 4


# ================================================================================================ M5 on ledgers (R1)
def _ledger_case(name, special, S=3 * G, T=8 * G, extra=(), steps=None):
    d = F.work_dir(f"ledger_{name}")
    led = F.write_ledger(d / "eval_episodes.csv", steps or F.grid_to(T), special=special, extra_rows=extra)
    rec, lines = R.ledger_read(led, S, T, d / "out", d)
    assert (d / "out" / R.READ_NAME).exists() and (d / "out" / "m5.json").exists()
    return rec["r1"], "\n".join(lines)


def test_m5_window_edges_and_straddle():
    S, T = 3 * G, 8 * G
    r1, text = _ledger_case("edges", {S: dict(n_over=6), S + G: dict(n_over=5),
                                      T: dict(n_over=10, act="neck_pitch_act")})
    assert r1["c6_ok"] and r1["c6_passed_m5"]
    assert r1["positive"] and r1["first_any_flag"] == S + G and r1["any_flagged"] == [S + G, T]
    assert r1["any_alarms"] == [] and r1["any_alarms_straddling_S"] == [S + G]
    assert r1["tail_flagged"] == [S + G] and r1["tail_alarms_straddling_S"] == [S + G] and r1["tail_alarms"] == []
    assert r1["flagged_at_or_before_S"] == [S]
    assert r1["m5_sha256"] == E.PIN_SHA256[E.M5] and r1["m5_pins_ok"]
    assert "straddling S (not counted): [%d]" % (S + G) in text
    r1, _ = _ledger_case("negative", {S: dict(n_over=9), S + G: dict(n_over=4)})
    assert r1["c6_ok"] and not r1["positive"] and r1["first_any_flag"] is None


def test_m5_c6_failures_split():
    S, T = 3 * G, 8 * G
    r1, text = _ledger_case("tail_only", {S + 2 * G: dict(kind="tail_missing")})
    assert r1["c6_ok"] and not r1["c6_passed_m5"] and len(r1["c6_tail_missing"]) == 1
    assert r1["c6_tail_missing"][0]["where"] == "window" and "report only" in text
    r1, _ = _ledger_case("any_missing", {S + 2 * G: dict(kind="any_missing")})
    assert not r1["c6_ok"] and [f["category"] for f in r1["c6_counting_failures"]] == ["any"]
    r1, _ = _ledger_case("any_missing_seeded", {S - G: dict(kind="any_missing")})
    assert not r1["c6_ok"] and r1["c6_counting_failures"][0]["where"] == "seeded"
    r1, _ = _ledger_case("rows", {S + 2 * G: dict(n_rows=9)})
    assert not r1["c6_ok"] and r1["c6_counting_failures"][0]["category"] == "rows"
    r1, _ = _ledger_case("absent", {S + 2 * G: None})
    assert not r1["c6_ok"] and r1["c6_counting_failures"][0]["category"] == "grid"
    assert r1["window_evaluations_absent"] == [S + 2 * G]
    r1, _ = _ledger_case("beyond", {}, steps=F.grid_to(T + G))
    assert not r1["c6_ok"] and "beyond" in r1["c6_counting_failures"][0]["failure"]
    off = F.clean_row(S + 2 * G + 1, 0)
    r1, _ = _ledger_case("offgrid", {}, extra=[off])
    assert not r1["c6_ok"] and "not on the grid" in r1["c6_counting_failures"][0]["failure"]


@pytest.mark.parametrize("start,S,T,first,n", [("s42", S42, E42, 3_801_088, 8), ("s80", S80, E80, 4_849_664, 4)])
def test_v4_truncated_parent_ledgers(start, S, T, first, n):
    d = F.work_dir(f"v4_{start}")
    src = E.f2_record(start, "eval_episodes.csv")
    assert E.sha256_file(src) == E.STARTS[start]["records"]["eval_episodes.csv"]
    rec, _ = R.ledger_read(src, S, T, d / "out", d, truncate=True)
    r1 = rec["r1"]
    assert r1["c6_ok"] and r1["c6_passed_m5"], r1["c6_counting_failures"][:3]
    assert r1["positive"] and r1["first_any_flag"] == first == E.STARTS[start]["first_any_flag"]
    assert r1["n_any_flagged"] == n == E.STARTS[start]["flagged_to_E"]
    assert r1["flagged_at_or_before_S"] == [] and len(r1["window_evaluations"]) == 12


def test_v4_s60_negative():
    d = F.work_dir("v4_s60")
    src = f"{E.F2_ROOT}/{E.V4_NEGATIVE['run_id']}/copies/{E.STAGE_DIR}/eval_episodes.csv.copy1"
    assert E.sha256_file(src) == E.V4_NEGATIVE["eval_episodes_sha256"]
    T = E.V4_NEGATIVE["truncate_at"]
    for S in (6_291_456, 0):
        rec, _ = R.ledger_read(src, S, T, d / f"out_{S}", d, truncate=True)
        assert rec["r1"]["c6_ok"] and rec["r1"]["positive"] is False and rec["r1"]["n_any_flagged"] == 0


def test_full_ledger_without_truncation_fails_c6():
    """M5 --E E on a parent's whole ledger: C6 fails on the evaluations beyond E (RF F-5), which count."""
    d = F.work_dir("v4_full")
    rec, _ = R.ledger_read(E.f2_record("s42", "eval_episodes.csv"), S42, E42, d / "out", d)
    assert not rec["r1"]["c6_ok"] and all("beyond" in f["failure"] for f in rec["r1"]["c6_counting_failures"])


# ================================================================================================ the full read
def test_full_read_refusals_and_gate():
    root = F.work_dir("full_gate")
    name = "s42_A_1101"
    run = F.make_run(root / "runs" / name, "s42", "A", 1101,
                     special={S42 + 2 * G: dict(n_over=6), S42 + 3 * G: dict(n_over=6)},
                     trace_root=root / "traces" / name)
    with pytest.raises(R.InputError, match="no readable check.json"):
        R.full_read(run, root)
    assert R.main(["--root", str(root), str(run)]) == 2
    assert R.check_run(run, root)[0] == 0
    with pytest.raises(R.InputError, match="not decided"):
        R.full_read(run, root)
    for start, arm, seed in E.QUEUE[1:]:
        F.passing_check(root, start, arm, seed)
    assert R.decided(root) == (True, [])
    rec, lines = R.full_read(run, root)
    out = R.read_dir(root, name)
    assert (out / R.READ_NAME).exists() and (out / "m5.json").exists() and (out / "r2_arrays.json").exists()
    r1 = rec["r1"]
    assert r1["positive"] and r1["first_any_flag"] == S42 + 2 * G and r1["any_alarms"] == [S42 + 3 * G]
    assert r1["c6_ok"] and rec["check_sha256"] == E.sha256_file(run / R.CHECK_NAME)
    assert rec["r2"]["steps"][0] == S42 + E.ROLLOUT and len(rec["r2"]["steps"]) == 191
    assert rec["r3"]["status"] == "read" and rec["r3"]["n_tail_flags"] == 1
    assert rec["returns"]["mean_return_last"] == 100.0
    text = "\n".join(lines)
    assert E.sha256_file(run / R.LEDGER_REL) in text and "R3 (report only)" in text
    # a check.json made before the run's files changed is not read
    meta = json.loads((run / E.RUN_META).read_text())
    meta["wall_seconds"] = 2.0
    (run / E.RUN_META).write_text(json.dumps(meta))
    with pytest.raises(R.InputError, match="changed since check.json"):
        R.full_read(run, root)
    # a failed check is never read
    bad = F.make_run(root / "validate" / "v2" / "s42_B_1201", "s42", "B", 1201, mode="no_update",
                     target=S42 + E.VALIDATE_LENGTH, meta_over=dict(exit=1))
    assert R.check_run(bad, root)[0] == 4
    with pytest.raises(R.InputError, match="did not pass"):
        R.full_read(bad, root)


def test_full_read_validation_run_needs_no_gate_and_r3_cases():
    root = F.work_dir("full_validate")
    target = S80 + E.VALIDATE_LENGTH
    run = F.make_run(root / "validate" / "v2" / "s80_B_1401", "s80", "B", 1401, mode="no_update", target=target,
                     special={target: dict(n_over=5)})
    assert R.check_run(run, root)[0] == 0
    rec, _ = R.full_read(run, root)
    assert rec["r1"]["positive"] and rec["r1"]["first_any_flag"] == target and rec["r3"]["status"] == "missing"
    assert (run / "read" / R.READ_NAME).exists()
    assert R.main(["--root", str(root), str(run)]) == 0
    F.write_trace(run / "trace" / R.TRACE_NAME, R.end_zip(run, "end", target), target,
                  checkpoint=dict(sha256="0" * 64, num_timesteps=target))
    rec, _ = R.full_read(run, root)
    assert rec["r3"]["status"].startswith("not of the end pair")
    F.write_trace(run / "trace" / R.TRACE_NAME, R.end_zip(run, "end", target), target)
    rec, _ = R.full_read(run, root)
    assert rec["r3"]["status"] == "read"


def test_full_read_c6_failure_writes_lost():
    root = F.work_dir("full_lost")
    name = "s80_A_1301"
    run = F.make_run(root / "runs" / name, "s80", "A", 1301, special={S80 + 5 * G: dict(kind="any_missing")})
    assert R.check_run(run, root)[0] == 0
    for start, arm, seed in E.QUEUE:
        if E.run_name(start, arm, seed) != name:
            F.passing_check(root, start, arm, seed)
    rec, _ = R.full_read(run, root)
    assert not rec["r1"]["c6_ok"] and rec["lost_written"]
    lost = R.load_lost(root / "lost.json")
    assert lost[name]["cause"] == "C6" and lost[name]["by"] == "read_ecs.py"
    rec, _ = R.full_read(run, root)
    assert rec["lost_written"] is False and list(R.load_lost(root / "lost.json")) == [name]


def test_stopped_run_reads_to_T_and_final_pair():
    root = F.work_dir("full_stopped")
    T = S42 + 10 * G
    name = "s42_B_1211"
    run = F.make_run(root / "runs" / name, "s42", "B", 1211, T=T, stop="collapse", trace_root=root / "traces" / name)
    assert R.end_zip(run, "collapse", T).name == "stage1_final.zip"
    assert R.check_run(run, root)[0] == 0
    for start, arm, seed in E.QUEUE:
        if E.run_name(start, arm, seed) != name:
            F.passing_check(root, start, arm, seed)
    rec, _ = R.full_read(run, root)
    assert rec["T"] == T and rec["r1"]["c6_ok"] and not rec["r1"]["positive"]
    assert len(rec["r1"]["window_evaluations"]) == 10 and rec["r3"]["status"] == "read"


def test_lost_json_schema():
    d = F.work_dir("lost_schema")
    p = d / "lost.json"
    assert R.load_lost(p) == {}
    p.write_text(json.dumps({"lost": {}}))
    with pytest.raises(R.InputError):
        R.load_lost(p)
    p.write_text(json.dumps({"schema": E.LOST_SCHEMA, "lost": {"s42_A_1101_crashed1": {}}}))
    with pytest.raises(R.InputError):
        R.load_lost(p)
    p.unlink()
    assert R.add_lost(p, "s42_A_1101", "C6", "x") and not R.add_lost(p, "s42_A_1101", "C6", "x")
    assert json.loads(p.read_text())["schema"] == E.LOST_SCHEMA


def test_outputs_stay_under_root():
    d = F.work_dir("under_root")
    with pytest.raises(R.InputError):
        R.require_under("/tmp/elsewhere", d)
    assert R.main(["--root", str(d), "--ledger", E.f2_record("s42", "eval_episodes.csv"), "--S", str(S42), "--T",
                   str(E42), "--out", "/tmp/not_under_root"]) == 2


def test_local_constants_agree_with_the_pinned_m5():
    """read_ecs's local copies (actuator names, the flag threshold's source) and ecs_common's M6 columns agree with the
    pinned M5 (loaded under private names; the files are only read)."""
    assert E.sha256_file(E.M5) == E.PIN_SHA256[E.M5] and E.sha256_file(E.M5_COMMON) == E.PIN_SHA256[E.M5_COMMON]
    spec = importlib.util.spec_from_file_location("lane_r_m5_common", E.M5_COMMON)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    assert tuple(R.ACTUATOR_NAMES) == tuple(C.ACTUATOR_NAMES) and C.GRID == R.GRID
    src = Path(E.M5).read_text(encoding="utf-8")
    keys = src.split("M6_KEYS = (", 1)[1].split(")", 1)[0]
    m6 = tuple(k.strip().strip('"') for k in keys.replace("\n", " ").split(",") if k.strip())
    assert tuple(E.M6_COLUMNS) == (*m6, "std_per_actuator")
    common = Path(E.M5_COMMON).read_text(encoding="utf-8")
    assert 'd["any_flagged"] = d["any"] >= 5' in common and 'd["tail_flagged"] = d["tail"] >= 5' in common
    assert R.FLAG_ROWS == 5


# ================================================================================================ R1 cannot run (RR-1)
def _r1_root(tag, name="s80_A_1311"):
    root = F.work_dir(f"r1_{tag}")
    p = E.parse_run_name(name)
    run = F.make_run(root / "runs" / name, p["start"], p["arm"], p["seed"])
    assert R.check_run(run, root)[0] == 0
    for start, arm, seed in E.QUEUE:
        if E.run_name(start, arm, seed) != name:
            F.passing_check(root, start, arm, seed)
    return root, run, name


def _assert_r1_lost(root, run, name, text):
    rec, lines = R.full_read(run, root)
    assert rec["lost_written"] is True and rec["lost_cause"] == "R1" and rec["r1"]["positive"] is None
    lost = R.load_lost(root / "lost.json")
    assert lost[name]["cause"] == "R1" and text in lost[name]["detail"] and lost[name]["by"] == "read_ecs.py"
    assert any("R1 CANNOT RUN" in x for x in lines)
    check, csha = R.read_json_file(run / R.CHECK_NAME)
    check["_sha256"] = csha
    read, _ = R.read_json_file(R.read_dir(root, name) / R.READ_NAME)
    c = F.S.classify(name, check, read, lost[name])
    assert c["cls"] == "LOST" and c["cause"] == "R1"
    assert R.main(["--root", str(root), str(run)]) == 0  # read again: already LOST, nothing else changes
    assert list(R.load_lost(root / "lost.json")) == [name]


def test_r1_missing_ledger_is_lost():
    root, run, name = _r1_root("missing")
    (run / R.LEDGER_REL).unlink()
    _assert_r1_lost(root, run, name, "missing")


def test_r1_ledger_not_utf8_is_lost_through_m5s_input_error():
    root, run, name = _r1_root("not_utf8")
    (run / R.LEDGER_REL).write_bytes(b"timesteps,episode\n\xff\xfe\x00 not utf-8\n")
    _assert_r1_lost(root, run, name, "INPUT ERROR")


def test_r1_m5_timeout_twice_is_lost_once_is_retried(monkeypatch):
    import subprocess

    root, run, name = _r1_root("timeout")
    real, calls = subprocess.run, []

    def once(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(a[0], R.M5_TIMEOUT_S)
        return real(*a, **k)
    monkeypatch.setattr(R.subprocess, "run", once)
    rec, _ = R.full_read(run, root)
    assert len(calls) == 2 and rec["r1"]["c6_ok"] and rec["lost_written"] is False and not (root / "lost.json").exists()

    def always(*a, **k):
        raise subprocess.TimeoutExpired(a[0], R.M5_TIMEOUT_S)
    monkeypatch.setattr(R.subprocess, "run", always)
    _assert_r1_lost(root, run, name, "did not finish")


def test_r1_other_m5_failures_are_tool_defects_not_lost(monkeypatch):
    """An M5 exit other than its INPUT ERROR, or an M5 that cannot start, is a tool defect: exit 2, no lost.json (the
    criteria's VOID route), and a validation run is never written to lost.json."""
    import subprocess

    root, run, name = _r1_root("tool")
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "Traceback"))
    with pytest.raises(R.InputError, match="M5 exit 1") as ei:
        R.full_read(run, root)
    assert not isinstance(ei.value, R.R1Unreadable) and not (root / "lost.json").exists()
    # an exit 2 without M5's own "INPUT ERROR:" line (Python's "can't open file", a usage error) is a tool defect too
    for out, err in (("", "python: can't open file '/x/ledger_r10round.py': [Errno 2] No such file or directory"),
                     ("", "usage: ledger_r10round.py [-h] ledger\nerror: unrecognized arguments: --x"),
                     ("INPUT ERROR elsewhere in the text", "")):
        monkeypatch.setattr(R.subprocess, "run",
                            lambda *a, o=out, e=err, **k: subprocess.CompletedProcess(a[0], 2, o, e))
        with pytest.raises(R.InputError, match="M5 exit 2") as ei:
            R.full_read(run, root)
        assert not isinstance(ei.value, R.R1Unreadable) and not (root / "lost.json").exists()

    def cannot_start(*a, **k):
        raise OSError("no such interpreter")
    monkeypatch.setattr(R.subprocess, "run", cannot_start)
    with pytest.raises(R.InputError, match="M5 did not run"):
        R.full_read(run, root)
    monkeypatch.undo()
    vroot = F.work_dir("r1_validation")
    target = S42 + E.VALIDATE_LENGTH
    vrun = F.make_run(vroot / "validate" / "v2" / "s42_A_1101", "s42", "A", 1101, mode="no_update", target=target)
    assert R.check_run(vrun, vroot)[0] == 0
    (vrun / R.LEDGER_REL).unlink()
    with pytest.raises(R.R1Unreadable):
        R.full_read(vrun, vroot)
    assert not (vroot / "lost.json").exists()


def test_ledger_mode_is_gated_for_a_continuations_ledger():
    """Review RR-7: --ledger refuses a ledger under <root>/runs/ until all 24 are decided."""
    root = F.work_dir("ledger_gate")
    name = "s42_A_1101"
    run = F.make_run(root / "runs" / name, "s42", "A", 1101)
    led = run / R.LEDGER_REL
    args = ["--root", str(root), "--ledger", str(led), "--S", str(S42), "--T", str(E42), "--out", str(root / "v4x")]
    with pytest.raises(R.InputError, match="not decided"):
        R.ledger_read(led, S42, E42, root / "v4x", root)
    assert R.main(args) == 2 and not (root / "v4x").exists()
    for start, arm, seed in E.QUEUE:
        F.passing_check(root, start, arm, seed)
    assert R.main(args) == 0
    copy = root / "validate" / "copy.csv"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(led.read_bytes())
    (root / "runs" / "s80_B_1451" / R.CHECK_NAME).unlink()
    rec, _ = R.ledger_read(copy, S42, E42, root / "v4y", root)  # a copy outside runs/ is read (V4's form)
    assert rec["r1"]["c6_ok"]
