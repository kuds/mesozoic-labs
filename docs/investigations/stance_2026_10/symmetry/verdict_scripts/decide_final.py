"""foot/symmetry screen (final design, r2): the binding decision logic, as pure functions of per-arm criterion records.

No I/O except the CLI.  verify_final.py and recount_final.py compute the per-arm records from the raw files; this module
turns them into: the interim decision, the comparator continuation, the after-interim COP class, fa_needed (the
tightened A_fa necessity rule), the per-arm labels, the outcome class O1-O9 (a total order), the PR arm, the arms of
the 12M confirmation, the exploratory overrides, and the 12M ladder.  Every threshold is read from frozen_params.json
(sha256 asserted by the callers), never typed here.

Per-arm record (dict), produced by verify_final.py (and independently by recount_final.py):
  arm 'L' | 'C'; seeds [42, 52, 60]
  interim: {I1, I2, I3: bool}                       (3M; I3 on any-length CoP>0.8 counts, unchanged)
  reached_6M: bool                                  (main run or comparator continuation)
  comparator: bool                                  (L resumed after an interim stop; not adoptable)
  H: {H1..H6: bool} or None                         (6M; H3 = H3a and H3b)
  S1, S2: bool or None
  families_failed: [..]                             (ROLL / COP / ROBUSTNESS, from verify)
  seeds_imputed: {seed: [checkpoint labels]}        (method failure / collapse backstop / 2nd infra failure)
  exploratory: bool, exploratory_reason: [..]       (validity.json of any seed; decision_rules.pre_registration)
  cop_evidence: {"per_seed": {seed: {ckpt: {"cop_fh": int, "class": str, "observed": bool}}},
                 "h3b_even_ci_fail": {seed: bool},  (EVEN series (g_r + g_l)/2: OLS 90% CI lower bound > bar, 24 evals,
                                                     6M observed)
                 "class_6M": {seed: str}}
12M ladder input (ladder_12M): see its docstring.
"""
from __future__ import annotations

import json
import sys

CREEP, ONE_PAD, CLUSTERED = "MIRROR-EVEN CREEP", "ONE-PAD CREEP", "SPAWN-CLUSTERED COP"


def classify_cop(median_even_g: float, pad_bias: float, evals_even_mean: float | None, evals_pad_gap: float | None, P: dict) -> dict:
    """after_interim_stop / fa co-condition class of one seed at one checkpoint (panel decides; evals corroborate).
    median_even_g = median over the seed's full-horizon panel episodes of (g_r + g_l)/2; pad_bias = |median over the same
    episodes of (g_r - g_l)| (the persistent pad bias: a mirror-symmetric policy's spawn-driven pad differences change
    sign with the spawn and cancel, a one-pad creep keeps its sign); evals_* = the mean over the last 8 in-training evals
    of (gate_r + gate_l)/2 and |mean gate_r - mean gate_l|."""
    c = P["classification"]
    if pad_bias >= c["pad_bias_max"]:
        cls = ONE_PAD
    elif median_even_g >= c["creep_even_g"]:
        cls = CREEP
    else:
        cls = CLUSTERED
    agree = None
    if evals_even_mean is not None and evals_pad_gap is not None:
        ev = ONE_PAD if evals_pad_gap >= c["pad_bias_max"] else (CREEP if evals_even_mean >= c["creep_even_g"] else CLUSTERED)
        agree = ev == cls
    return {"class": cls, "evals_agree": agree}


def interim(rec: dict) -> bool:
    i = rec["interim"]
    return bool(i["I1"] and i["I2"] and i["I3"])


def fa_needed(L: dict, P: dict) -> dict:
    """Tightened A_fa (binding, r2).  Evidence comes only from OBSERVED checkpoints of L (3M, and 4M-6M if L reached them
    in its main run or as the comparator); an imputed checkpoint is never evidence.  A seed is an EVIDENCE SEED if
      (i)  at some observed checkpoint its full-horizon CoP>0.8 count exceeds the bar AND its class there is
           MIRROR-EVEN CREEP, or
      (ii) its EVEN in-training series (g_r + g_l)/2 has an OLS 90% CI lower bound of the 12M projection above the H3(b)
           bar on the complete, observed 3-6M evals AND its 6M class is MIRROR-EVEN CREEP.
    fa_needed iff >= min_seeds DISTINCT evidence seeds (no single-seed path).  An exploratory L gives no binding
    evidence: fa_needed is False."""
    a = P["fa_needed"]
    ev = L.get("cop_evidence", {})
    imputed = {int(k): set(v) for k, v in L.get("seeds_imputed", {}).items()}
    count_seeds, ci_seeds, detail = [], [], {}
    for s in P["seeds"]:
        per = ev.get("per_seed", {}).get(str(s), {})
        hits = []
        for ck, x in per.items():
            if not x.get("observed", False) or ck in imputed.get(s, set()):
                continue
            if x["cop_fh"] > a["count_bar"] and x["class"] == CREEP:
                hits.append(ck)
        if hits:
            count_seeds.append(s)
        even_ci = bool(ev.get("h3b_even_ci_fail", {}).get(str(s), False))
        c6 = ev.get("class_6M", {}).get(str(s))
        ci_ok = even_ci and c6 == CREEP and "6M" not in imputed.get(s, set())
        if ci_ok:
            ci_seeds.append(s)
        detail[s] = {"count_hits": hits, "even_ci_fail": even_ci, "class_6M": c6, "ci_evidence": ci_ok}
    seeds = sorted(set(count_seeds) | set(ci_seeds))
    need = len(seeds) >= a["min_seeds"]
    out = {"fa_needed": need, "evidence_seeds": seeds, "count_evidence_seeds": count_seeds, "ci_evidence_seeds": ci_seeds, "per_seed": detail}
    if L.get("exploratory"):
        out.update({"fa_needed": False, "fa_needed_if_not_exploratory": need, "reason": "L is exploratory: its evidence is not binding"})
    return out


def comparator_continuation(L: dict, C: dict, P: dict) -> bool:
    """L stops at the interim, C continues, and fa_needed is not already established by L's 3M evidence."""
    if interim(L) or not interim(C):
        return False
    return not fa_needed(L, P)["fa_needed"]


def h_pass(rec: dict) -> bool:
    if rec.get("comparator") or not interim(rec) or not rec.get("reached_6M") or rec.get("H") is None:
        return False
    return all(bool(rec["H"][k]) for k in ("H1", "H2", "H3", "H4", "H5", "H6"))


def arm_state(rec: dict) -> str:
    """'HS' (H-PASS with S1 and S2), 'HnS' (H-PASS, S1 or S2 fails), 'FAIL'."""
    if not h_pass(rec):
        return "FAIL"
    return "HS" if (rec.get("S1") and rec.get("S2")) else "HnS"


def labels(L: dict, C: dict, P: dict) -> dict:
    fa = fa_needed(L, P)
    sL, sC = arm_state(L), arm_state(C)
    lab_L = {"HS": "PASS", "HnS": "PASS_UNATTRIBUTED(sym)", "FAIL": "FAIL"}[sL]
    if sC == "HS":
        lab_C = "PASS" if fa["fa_needed"] else "PASS_UNATTRIBUTED(fa)"
    elif sC == "HnS":
        lab_C = "PASS_UNATTRIBUTED(sym)" if fa["fa_needed"] else "PASS_UNATTRIBUTED(sym,fa)"
    else:
        lab_C = "FAIL"
    if L.get("exploratory") and lab_L != "FAIL":
        lab_L = "EXPLORATORY " + lab_L
    if C.get("exploratory") and lab_C != "FAIL":
        lab_C = "EXPLORATORY " + lab_C
    return {"L": lab_L, "C": lab_C, "L_state": sL, "C_state": sC, "fa": fa}


# Total order over the label combinations (first = most preferred).  fa_needed can only be true with L FAIL (an
# H-PASS L passes the COP family, which excludes every kind of fa evidence), so 12 combinations exist.
OUTCOMES = [
    # id, L state, C state, fa_needed (None = any), PR arm, 12M arms, wording key
    ("O1", "HS", None, False, "L", ["L"], "L_pass"),
    ("O2", "FAIL", "HS", True, "C", ["C", "L"], "C_pass"),
    ("O3", "HnS", "HS", False, None, ["L", "C"], "pu"),
    ("O4", "FAIL", "HS", False, None, ["C", "L"], "pu"),
    ("O5", "FAIL", "HnS", True, None, ["C", "L"], "pu"),
    ("O6", "HnS", "HnS", False, None, ["L", "C"], "pu"),
    ("O7", "FAIL", "HnS", False, None, ["C", "L"], "pu"),
    ("O8", "HnS", "FAIL", False, None, ["L"], "pu"),
    ("O9", "FAIL", "FAIL", None, None, [], "fail"),
]


def outcome(L: dict, C: dict, P: dict) -> dict:
    """The outcome class, then the exploratory overrides (predeclared, frozen_params 'exploratory'): an exploratory arm is
    never the PR arm (the class and its 12M arms are kept: eligible for the 12M confirmation only); an exploratory L gives
    no binding fa evidence (already applied inside fa_needed, so O2 / O5 cannot follow from it); at 12M an exploratory L's
    6M families cannot serve as replication (ladder_12M)."""
    lab = labels(L, C, P)
    fa = lab["fa"]["fa_needed"]
    for oid, sl, sc, f, pr, arms, wk in OUTCOMES:
        if lab["L_state"] != sl:
            continue
        if sc is not None and lab["C_state"] != sc:
            continue
        if f is not None and f != fa:
            continue
        expl = {"L": bool(L.get("exploratory")), "C": bool(C.get("exploratory"))}
        withheld = pr is not None and expl[pr]
        return {"outcome": oid, "pr_arm": None if withheld else pr, "pr_withheld_exploratory": pr if withheld else None,
                "arms_12M": arms, "wording": "exploratory_no_pr" if withheld else wk, "labels": {"L": lab["L"], "C": lab["C"]},
                "states": {"L": lab["L_state"], "C": lab["C_state"]}, "exploratory": expl,
                "L_6M_families": sorted(set(L.get("families_failed", [])) | ({"SYMMETRY"} if lab["L_state"] == "HnS" else set())),
                "fa_needed": lab["fa"], "rank": [o[0] for o in OUTCOMES].index(oid) + 1}
    raise AssertionError(f"no outcome for {lab}")  # unreachable: the table is exhaustive (tested)


C_FAMILY = {"C1": "BY_FAILING_RAIL", "C2": "COP", "C3": "COP", "C4": "ROBUSTNESS", "C5": "GENERALISATION", "C6": "GENERALISATION", "C7": "SYMMETRY"}
REPLICABLE = ("ROLL", "ROBUSTNESS", "SYMMETRY")   # families with a binding 6M counterpart other than COP


def ladder_12M(res: dict, P: dict) -> dict:
    """The 12M ladder (binding, r2).  res = {
         'L': None (not run) | {'C1'..'C7': bool, 'C1_families': [ROLL/COP/ROBUSTNESS of the failing rails],
                                'per_seed': {seed: {'C2_fh_fail': bool, 'C3': bool, 'class_12M': str}}},
              C2_fh_fail = more than 3 FULL-HORIZON CoP>0.8 episodes at one of the seed's judged checkpoints (falls are not
              COP evidence, as in fa_needed); C3 = the seed's both-pad in-training C3 (frozen_params confirmation_12M).
         'C': the same or None,
         'L_6M': {'families': [the outcome's L_6M_families], 'exploratory': bool}}   (from the 6M verdict)
    1. L passes C1-C7 -> adopt L.
    2. else, C was run and passes C1-C7:
       path 1 (COP, multi-seed): on >= min_cop_seeds of the 12M seeds L fails C2 on the full-horizon count or fails C3,
              each such seed in the MIRROR-EVEN CREEP class at 12M -> adopt C, 'the fore-aft term is needed for COP' (the
              only path with that wording);
       path 2 (replication): L failed, at 12M, a criterion of a family in REPLICABLE (C1 by its failing rails, C4,
              C7) that L ALSO failed at 6M on seeds 42/52/60 (L_6M families; not available if L was exploratory) ->
              adopt C, 'C passed where L failed <families> at 6M and 12M; necessity for COP not shown'.
              COP failures that do not meet path 1 and GENERALISATION failures (C5, C6; no 6M counterpart) never adopt C.
    3. else no adoption; any landed PR is rolled back by config revert (on_fail by family)."""
    keys = [f"C{i}" for i in range(1, 8)]
    lad = P["ladder_12M"]
    Lr, Cr = res.get("L"), res.get("C")
    if Cr is not None:
        assert Lr is not None, "C at 12M is always run with L as the ablation"
    if Lr is not None and all(Lr[k] for k in keys):
        return {"adopt": "L", "path": "L", "reason": "L passes C1-C7", "fore_aft_needed_for_COP": False}
    if Cr is not None and all(Cr[k] for k in keys):
        cop_seeds = sorted(int(s) for s, x in Lr.get("per_seed", {}).items() if (x["C2_fh_fail"] or not x["C3"]) and x.get("class_12M") == CREEP)
        if len(cop_seeds) >= lad["min_cop_seeds"]:
            return {"adopt": "C", "path": "1_COP_multi_seed", "fore_aft_needed_for_COP": True, "cop_seeds": cop_seeds,
                    "reason": f"C passes C1-C7; L fails C2/C3 with mirror-even creep on seeds {cop_seeds}"}
        fam12 = set()
        for k in keys:
            if not Lr[k]:
                fam12 |= set(Lr.get("C1_families", [])) if k == "C1" else {C_FAMILY[k]}
        l6 = res.get("L_6M") or {}
        fam6 = set() if l6.get("exploratory") else set(l6.get("families", []))
        rep = sorted((fam12 & set(REPLICABLE)) & fam6)
        if rep:
            return {"adopt": "C", "path": "2_replicated_family", "fore_aft_needed_for_COP": False, "families": rep,
                    "reason": f"C passes C1-C7; L failed {', '.join(rep)} at both 6M and 12M; necessity of the fore-aft term for COP not shown"}
        return {"adopt": None, "path": "none", "fore_aft_needed_for_COP": False, "cop_seeds_creep": cop_seeds, "families_12M": sorted(fam12),
                "families_6M": sorted(fam6), "reason": "C passes but L's failures are neither multi-seed creep COP nor a replicated family: no adoption; rollback"}
    return {"adopt": None, "path": "none", "fore_aft_needed_for_COP": False, "reason": "no arm adopted at 12M; any landed PR is rolled back by config revert"}


def main():
    """CLI: decide_final.py frozen_params.json verdict_records.json -> prints the decision JSON."""
    P = json.load(open(sys.argv[1]))
    R = json.load(open(sys.argv[2]))
    print(json.dumps(outcome(R["L"], R["C"], P), indent=1, default=str))


if __name__ == "__main__":
    main()
