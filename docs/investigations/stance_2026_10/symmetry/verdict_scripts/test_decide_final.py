"""Unit tests of decide_final.py (r2) on synthetic records (no data): the outcome table is exhaustive and exclusive over
every reachable label combination; fa_needed needs >= 2 distinct evidence seeds (count evidence or even-series CI
evidence, each with the creep class) and ignores falls / imputed / one-pad / single-seed / exploratory evidence; the
exploratory overrides; the 12M ladder adopts L first, C only through multi-seed creep COP evidence (path 1) or a
family L failed at both 6M and 12M (path 2), never on single-seed COP or GENERALISATION failures.
Run: python test_decide_final.py <frozen_params.json>"""
import copy
import itertools
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide_final as D  # noqa: E402

P = json.load(open(sys.argv[1]))
SEEDS = P["seeds"]


def rec(arm, state, evidence=None, imputed=None, comparator=False, exploratory=False):
    """state: HS / HnS / FAIL_COP / FAIL_ROLL / STOP_I3 / STOP_I1."""
    r = {"arm": arm, "seeds": SEEDS, "comparator": comparator, "seeds_imputed": imputed or {}, "exploratory": exploratory,
         "exploratory_reason": ["test"] if exploratory else [],
         "interim": {"I1": True, "I2": True, "I3": True}, "reached_6M": True, "families_failed": [],
         "H": {k: True for k in ("H1", "H2", "H3", "H3a", "H3b", "H4", "H5", "H6")}, "S1": True, "S2": True,
         "cop_evidence": evidence or {"per_seed": {}, "h3b_even_ci_fail": {}, "class_6M": {}}}
    if state == "HnS":
        r["S1"] = False
    elif state == "FAIL_COP":
        r["H"]["H3"] = r["H"]["H3a"] = False
        r["families_failed"] = ["COP"]
    elif state == "FAIL_ROLL":
        r["H"]["H2"] = False
        r["families_failed"] = ["ROLL"]
    elif state == "STOP_I3":
        r["interim"]["I3"], r["reached_6M"], r["H"] = False, False, None
        r["families_failed"] = ["COP"]
    elif state == "STOP_I1":
        r["interim"]["I1"], r["reached_6M"], r["H"] = False, False, None
        r["families_failed"] = ["ROLL"]
    return r


def ev(count_seeds=(), cls=D.CREEP, ck="4M", ci_seeds=(), n=6, ci_cls=None):
    e = {"per_seed": {}, "h3b_even_ci_fail": {}, "class_6M": {}}
    for s in SEEDS:
        e["per_seed"][str(s)] = {"3M": {"observed": True, "cop_fh": 0, "class": D.CLUSTERED}}
        e["per_seed"][str(s)][ck] = {"observed": True, "cop_fh": n if s in count_seeds else 0, "class": cls if s in count_seeds else D.CLUSTERED}
        e["h3b_even_ci_fail"][str(s)] = s in ci_seeds
        e["class_6M"][str(s)] = (ci_cls or cls) if (s in ci_seeds or s in count_seeds) else D.CLUSTERED
    return e


def test_fa_rule():
    fa = lambda r: D.fa_needed(r, P)["fa_needed"]  # noqa: E731
    assert fa(rec("L", "FAIL_COP", ev((42, 52))))
    assert not fa(rec("L", "FAIL_COP", ev((42,)))), "one count seed is not enough"
    assert not fa(rec("L", "FAIL_COP", ev((42, 52), cls=D.ONE_PAD))), "one-pad creep is not evidence"
    assert not fa(rec("L", "FAIL_COP", ev((42, 52), cls=D.CLUSTERED)))
    assert not fa(rec("L", "FAIL_COP", ev((42, 52), n=P["fa_needed"]["count_bar"]))), "count must exceed the bar"
    assert not fa(rec("L", "FAIL_COP", ev(ci_seeds=(60,)))), "a single even-series CI seed is not enough (r2: no single-seed path)"
    assert fa(rec("L", "FAIL_COP", ev(ci_seeds=(52, 60)))), "two even-series CI seeds with creep suffice"
    assert fa(rec("L", "FAIL_COP", ev(count_seeds=(42,), ci_seeds=(60,)))), "one count seed + one CI seed = two evidence seeds"
    assert not fa(rec("L", "FAIL_COP", ev(count_seeds=(42,), ci_seeds=(42,)))), "count and CI on the same seed = one evidence seed"
    assert not fa(rec("L", "FAIL_COP", ev(ci_seeds=(52, 60), ci_cls=D.ONE_PAD))), "CI evidence needs the creep class at 6M"
    assert not fa(rec("L", "FAIL_COP", ev(ci_seeds=(52, 60)), imputed={"60": ["6M"]})), "an imputed 6M is not CI evidence"
    e = ev((42, 52))
    assert not fa(rec("L", "FAIL_COP", e, imputed={"52": ["4M", "5M", "6M"]})), "imputed checkpoints are not evidence"
    e2 = copy.deepcopy(e)
    e2["per_seed"]["52"]["4M"]["observed"] = False
    assert not fa(rec("L", "FAIL_COP", e2))
    r = D.fa_needed(rec("L", "FAIL_COP", ev((42, 52)), exploratory=True), P)
    assert not r["fa_needed"] and r["fa_needed_if_not_exploratory"], "an exploratory L gives no binding evidence"


def test_outcome_table_exhaustive():
    states_L = {"HS": rec("L", "HS"), "HnS": rec("L", "HnS"), "FAIL": rec("L", "FAIL_COP")}
    states_C = {"HS": rec("C", "HS"), "HnS": rec("C", "HnS"), "FAIL": rec("C", "FAIL_ROLL")}
    seen = {}
    for (sl, L), (sc, C), fa in itertools.product(states_L.items(), states_C.items(), (False, True)):
        if fa and sl != "FAIL":
            continue  # unreachable: an H-PASS L passes the COP family, which excludes all fa evidence
        L2 = copy.deepcopy(L)
        if fa:
            L2["cop_evidence"] = ev((42, 52))
        o = D.outcome(L2, C, P)
        seen[(sl, sc, fa)] = o["outcome"]
        if o["pr_arm"] == "C" or "C" in o["arms_12M"]:
            assert "L" in o["arms_12M"], "L is the 12M ablation whenever C can be adopted"
        if o["pr_arm"] == "C":
            assert fa
    assert len(seen) == 12, seen
    assert seen[("HS", "HS", False)] == "O1" and seen[("FAIL", "HS", True)] == "O2" and seen[("FAIL", "FAIL", True)] == "O9"
    return seen


def test_exploratory():
    o = D.outcome(rec("L", "HS", exploratory=True), rec("C", "HS"), P)
    assert o["outcome"] == "O1" and o["pr_arm"] is None and o["pr_withheld_exploratory"] == "L" and o["arms_12M"] == ["L"]
    assert o["labels"]["L"].startswith("EXPLORATORY")
    o = D.outcome(rec("L", "FAIL_COP", ev((42, 52))), rec("C", "HS", exploratory=True), P)
    assert o["outcome"] == "O2" and o["pr_arm"] is None and o["arms_12M"] == ["C", "L"], "an exploratory C is never the PR arm"
    o = D.outcome(rec("L", "FAIL_COP", ev((42, 52)), exploratory=True), rec("C", "HS"), P)
    assert o["outcome"] == "O4" and o["pr_arm"] is None, "an exploratory L cannot establish fa_needed (O2 -> O4)"
    o = D.outcome(rec("L", "HnS"), rec("C", "FAIL_ROLL", exploratory=True), P)
    assert o["outcome"] == "O8" and o["L_6M_families"] == ["SYMMETRY"]


def test_interim_and_comparator():
    L, C = rec("L", "STOP_I1"), rec("C", "HS")
    assert D.comparator_continuation(L, C, P), "L stopped on roll, C continues -> comparator"
    L3 = rec("L", "STOP_I3", ev((42, 52), ck="3M"))
    assert not D.comparator_continuation(L3, C, P), "fa_needed already established at 3M"
    L4 = rec("L", "STOP_I3", ev((42, 52), ck="3M", cls=D.ONE_PAD))
    assert D.comparator_continuation(L4, C, P), "an I3 stop without creep evidence still runs the comparator"
    assert not D.comparator_continuation(L, rec("C", "STOP_I1"), P)
    Lc = rec("L", "STOP_I1", comparator=True)
    Lc["reached_6M"] = True
    assert D.arm_state(Lc) == "FAIL", "a comparator is never adoptable"


def r12(fail=(), c1_fam=(), fh_seeds=(), c3_seeds=(), cls=D.CREEP):
    seeds12 = P["confirmation_12M"]["seeds"]
    x = {f"C{i}": f"C{i}" not in fail for i in range(1, 8)}
    x["C1_families"] = list(c1_fam)
    x["per_seed"] = {str(s): {"C2_fh_fail": s in fh_seeds, "C3": s not in c3_seeds, "class_12M": cls if (s in fh_seeds or s in c3_seeds) else D.CLUSTERED}
                     for s in seeds12}
    return x


def test_ladder_12M():
    s1, s2 = P["confirmation_12M"]["seeds"][:2]
    ok = r12()
    lad = lambda L, C, fam6=(), expl=False: D.ladder_12M({"L": L, "C": C, "L_6M": {"families": list(fam6), "exploratory": expl}}, P)  # noqa: E731
    assert lad(ok, ok)["adopt"] == "L"
    r = lad(r12(("C2",), fh_seeds=(s1, s2)), ok)
    assert r["adopt"] == "C" and r["fore_aft_needed_for_COP"] and r["path"] == "1_COP_multi_seed"
    r = lad(r12(("C3",), c3_seeds=(s1, s2)), ok)
    assert r["adopt"] == "C" and r["fore_aft_needed_for_COP"], "C3 failures count per seed"
    r = lad(r12(("C2", "C3"), fh_seeds=(s1,), c3_seeds=(s2,)), ok)
    assert r["adopt"] == "C" and r["cop_seeds"] == sorted([s1, s2]), "C2_fh on one seed + C3 on another = two seeds"
    assert lad(r12(("C2",), fh_seeds=(s1,)), ok, fam6=("COP",))["adopt"] is None, "single-seed COP never adopts C"
    assert lad(r12(("C2",)), ok, fam6=("COP",))["adopt"] is None, "an any-length C2 failure from falls is not COP evidence"
    assert lad(r12(("C2",), fh_seeds=(s1, s2), cls=D.ONE_PAD), ok)["adopt"] is None, "one-pad creep at 12M is not path-1 evidence"
    r = lad(r12(("C4",)), ok, fam6=("ROBUSTNESS",))
    assert r["adopt"] == "C" and not r["fore_aft_needed_for_COP"] and r["families"] == ["ROBUSTNESS"] and r["path"] == "2_replicated_family"
    assert lad(r12(("C4",)), ok, fam6=())["adopt"] is None, "an unreplicated non-COP failure never adopts C"
    assert lad(r12(("C4",)), ok, fam6=("ROBUSTNESS",), expl=True)["adopt"] is None, "an exploratory L's 6M families are not replication"
    assert lad(r12(("C5",)), ok, fam6=("ROLL", "COP", "ROBUSTNESS", "SYMMETRY"))["adopt"] is None, "GENERALISATION has no 6M counterpart"
    assert lad(r12(("C7",)), ok, fam6=("SYMMETRY",))["adopt"] == "C"
    assert lad(r12(("C1",), c1_fam=("COP",)), ok, fam6=("COP",))["adopt"] is None, "COP via C1's rail is not replicable"
    assert lad(r12(("C1",), c1_fam=("ROLL",)), ok, fam6=("ROLL",))["adopt"] == "C"
    r = lad(r12(("C2", "C4"), fh_seeds=(s1,)), ok, fam6=("ROBUSTNESS",))
    assert r["adopt"] == "C" and r["path"] == "2_replicated_family" and not r["fore_aft_needed_for_COP"], "a single-seed COP failure does not block a replicated family"
    bad = r12(("C4",))
    assert lad(r12(("C2",), fh_seeds=(s1, s2)), bad)["adopt"] is None, "C must pass C1-C7"
    assert lad(ok, None)["adopt"] == "L"
    assert lad(r12(("C2",), fh_seeds=(s1, s2)), None)["adopt"] is None
    try:
        D.ladder_12M({"L": None, "C": ok, "L_6M": {"families": [], "exploratory": False}}, P)
        raise SystemExit("C without L must be refused")
    except AssertionError:
        pass


if __name__ == "__main__":
    test_fa_rule()
    seen = test_outcome_table_exhaustive()
    test_exploratory()
    test_interim_and_comparator()
    test_ladder_12M()
    print(json.dumps({"ok": True, "outcome_by_combination": {f"L={k[0]} C={k[1]} fa={k[2]}": v for k, v in sorted(seen.items(), key=lambda kv: kv[1])}}, indent=1))
