"""Bias screen 2 (BS2): the scorer.  Report only: reads files, never trains, never simulates.

    PY -I -B score_bs.py [--root <BS2>] [--out <root>/score]

It applies <BS2>/criteria.txt (BS2 criteria v1) exactly (THE PRIMARY OUTCOME, THE TEST, THE INFORMATIVENESS CONDITION,
THE READING, THE SECONDARIES, REPORT ONLY, RUN HANDLING, WHEN THE READING IS MADE); the constants below are the text's
and are pinned with it (criteria.txt names this file's sha256 in PINS; <BS2>/APPROVED pins it as
tool_sha256["score_bs.py"]).  Derived from BS's score_bs.py (score_bs 1.1, sha256 869898c5...): the window, B_S, the
gate (0.20), the queue's seeds and the line texts are BS2's; the three pre-registered secondaries, the report-only
trajectory and the count of G(E) > G(S) are new; the code copy is measured at bs_common.CODE (BS's copy, reused).

What it reads: <root>/SCREEN.json (it must be the criteria's queue); <root>/lost.json; for each queue entry
<root>/runs/<name>/check.json (read_bs.py --check).  Only once every entry is decided (COUNTED, STOPPED or LOST) and
every COUNTED or STOPPED one has its full read: <root>/reads/<name>/markers.json (read_bs.py's markers), the run's
run_meta.json (the per-rollout manipulation evidence summary and the pins it ran under) and the ledger the markers name
(returns and falls, report only).  For PROVISIONAL: <root>/APPROVED, <root>/criteria.txt, <root>/SCREEN.json, the five
BS2 tools and this file (in this file's folder) against APPROVED, M5's pins, and the code copy bs_common.CODE measured
by M1's own r10m_common.check_copy (loaded only at its pin).
What it writes: <out>/score.json and <out>/score.txt (default <root>/score; refused outside <root>).

Classes: LOST (named in lost.json); COUNTED (check.json passed, stop reason "end" at E); STOPPED (check.json passed,
stop reason "collapse" at T < E); UNDECIDED otherwise (running, queued, crashed or VOID).  While any entry is UNDECIDED,
or a COUNTED / STOPPED one has no markers.json yet, the screen is "incomplete": no markers file and no run_meta.json is
opened and no outcome is printed (outcome-blind).

THE PRIMARY (one number per COUNTED continuation): the mean of the probe's rollout_mu14_mean (policy_probe.npz
rollout_mu_mean[:, 14], the actor's raw Gaussian mean for tail_3_pitch_act averaged over the rollout's own 8,192
states) over the 8 probe rows with update_timesteps E - 65,536 ... E - 8,192 (3,473,408 ... 3,530,752), in exact
rational arithmetic.  MISSING (filled): a LOST or STOPPED continuation, or a COUNTED one whose 8 rows are not each
present once and finite ("primary unreadable", UNREADABLE).
THE TEST: exact one-sided permutation test of control against critic_blind_tail3 on the difference of arm means, over
all C(8, 4) = 70 relabelings; p = (relabelings whose control-arm sum is >= the observed) / 70, exact rationals (a tie
counts against the reading).
THE FILL: a missing control value counts as -infinity and a missing critic_blind_tail3 value as +infinity (the supremum
of p over every value the missing ones could have had: it never helps REDUCED).
THE READING, the first line that applies: 1 NOT INFORMATIVE (the median control primary rose by less than 0.20 from
B_S, or no control primary was read; the median rule: the midpoint of the two middle values for 4, the middle one for
3, the midpoint for 2, the value for 1, exact); 2 REDUCED (filled p <= 1/20); 3 NO REDUCTION DETECTED, with "(the
as-counted p, report only, is <= 0.05; the filled p decides)" when the as-counted p is <= 1/20 and the filled one is
not.
THE SECONDARIES (tested, never change the line): SEC-G (OWN G at the final pair, E; blind HIGHER), SEC-CMD (CMD_L at the
evaluation at E; blind LOWER), SEC-MARGIN (OWN b*_P = 1 - G - c0 at the final pair, E; blind HIGHER).  Each: the same
exact test on its own values with the hypothesis arm's sum as the statistic, its own fill (a MISSING value counts as the
infinity that works against the secondary's direction), SUPPORTED when the filled p <= 0.05 (unadjusted), with
"(survives Bonferroni over the 3 secondaries)" when the filled p <= 0.05 / 3 (only 1/70 can), and, under line 1,
"(the control arm did not climb on the primary)".  A secondary is MISSING for a LOST, STOPPED or UNREADABLE
continuation and when its marker is absent or not finite (cause printed); a COUNTED continuation whose markers lack the
loop row at E (the final pair not read: a tool or file defect) is also a PROVISIONAL item (score_bs 2.1, finding D7).
Each arm's sample SD of the primaries read is printed beside its mean and median (report only: it places the power).

Exit codes (as score_ecs.py): 0 final; 3 provisional or incomplete; 2 input error.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
VERSION = "score_bs 2.1 (2026-10-30; bias screen 2 criteria v1 as fixed; derived from score_bs 1.1)"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = sys.modules.get("bs_common") or _load("bs_common")
R = sys.modules.get("read_bs") or _load("read_bs")

EXIT_FINAL, EXIT_NOT_FINAL, EXIT_INPUT = C.EXIT_OK, C.EXIT_PROVISIONAL, C.EXIT_REFUSED      # 0, 3, 2

# ------------------------------------------------------------------------------------------------ the criteria's
CONTROL = C.CONTROL
INTERVENTION = "critic_blind_tail3"
ARM_ORDER = (CONTROL, INTERVENTION)
SEEDS = (2501, 2511, 2521, 2531)                                          # criteria THE QUEUE
QUEUE = tuple((arm, seed) for seed in SEEDS for arm in ARM_ORDER)        # interleaved, paired
PRIMARY_K = 8
PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K, 0, -1))   # 3,473,408 ... 3,530,752
EARLY_STEPS = tuple(C.S + C.ROLLOUT * k for k in range(1, PRIMARY_K + 1))     # 2,105,344 ... 2,162,688 (report only)
# B_S: the parent's own value at S in the primary's form: the mean of the parent F2 probe's rollout_mu_mean[:, 14] over
# its 8 rows with update_timesteps 2,039,808 ... 2,097,152 (rows 248-255; policy_probe.npz.copy1 sha256 4e0e7220...),
# exactly 135930855/1073741824 [measured 2026-10-30, criteria_work/parent_window.out and the builder's
# validate/logs/parent_schedule.out; recomputed by tests/test_score_bs.py when the file is present].
BASELINE_S = 0.1265954738482833
BASELINE_STEPS = tuple(C.S - C.ROLLOUT * k for k in range(PRIMARY_K - 1, -1, -1))
RISE_MIN = Fraction(1, 5)                                                 # the gate: 0.20, inclusive, exact
ALPHA = Fraction(1, 20)
N_SECONDARIES = 3
STATUE_LINES = ("0.10", "0.14", "0.15", "0.16")
EVAL_STEPS = tuple(range(C.S + C.EVAL_FREQ, C.E + 1, C.EVAL_FREQ))       # 2,228,224 ... 3,538,944 (11)
# The parent's on-policy trajectory (report only): the mean of its F2 probe's rollout_mu_mean[:, 14] over the 8 rows
# ending at each evaluation step (at E, which has no row, the 8 rows ending at E - 8,192: the primary's rows), exact
# dyadic values [measured 2026-10-30 from the F2 probe; = criteria_work/parent_window.out "Trajectory"; recomputed by
# tests/test_score_bs.py when the file is present].
PARENT_TRAJECTORY = {
    2228224: 0.15514186304062605, 2359296: 0.16859042644500732, 2490368: 0.19547326490283012,
    2621440: 0.19018659926950932, 2752512: 0.24137981049716473, 2883584: 0.29176359437406063,
    3014656: 0.32772068679332733, 3145728: 0.4317394308745861, 3276800: 0.6135170012712479,
    3407872: 0.6913527101278305, 3538944: 0.7053788751363754,
}
PARENT_G_S = -0.6363920116424561      # the parent's OWN G at S (read_bs REPRO 2097152, ew_r10.jsonl line 3; report)

COUNTED, STOPPED, LOST, UNDECIDED = "COUNTED", "STOPPED", "LOST", "UNDECIDED"
LINE_KEYS = {1: "NOT INFORMATIVE", 2: "REDUCED", 3: "NO REDUCTION DETECTED"}
LINE_TEXT = {
    1: ("NOT INFORMATIVE: THE CONTROL ARM DID NOT REPRODUCE THE CLIMB OF THE ON-POLICY TAIL_3 COMMAND (its median "
        "primary rose by less than 0.20 from the value at S, or no control primary was read); no reading about the "
        "critic is made"),
    2: ("REDUCED: BLINDING THE CRITIC TO THE TAIL_3 ANGLE (observation 6) REDUCED THE ON-POLICY TAIL_3 COMMAND "
        "OVER THE LAST 8 UPDATES BEFORE E, FROM s42's 2,097,152 STATE (exact one-sided permutation test, p <= 0.05)"),
    3: ("NO REDUCTION DETECTED IN THE ON-POLICY TAIL_3 COMMAND OVER THE LAST 8 UPDATES BEFORE E FROM BLINDING THE "
        "CRITIC TO THE TAIL_3 ANGLE (not excluded: a reduction of about half the rise or smaller (power about "
        "0.25-1.00 at half the parent's rise, by the between-continuation spread; about 0.1-0.3 if blinding makes "
        "climbs rarer rather than smaller), other inputs that carry the angle, the phase after E, the rail)"),
}
ADDITION_AS_COUNTED = "(the as-counted p, report only, is <= 0.05; the filled p decides)"
ADDITION_NOT_INFORMATIVE = "(the filled p is <= 0.05, but the control arm did not climb: no reading about the critic)"
NEG_INF, POS_INF = (-1, Fraction(0)), (1, Fraction(0))   # THE FILL (lexicographic: the infinite part first)
INF_CONTROL, INF_BLIND = NEG_INF, POS_INF                # the primary: control -inf, critic_blind_tail3 +inf

# THE SECONDARIES (criteria: pre-registered; tested; none changes the line)
SEC_SUPPORTED = "SUPPORTED (p <= 0.05, unadjusted)"
SEC_NOT_SUPPORTED = "NOT SUPPORTED"
SEC_BONFERRONI = "(survives Bonferroni over the 3 secondaries)"
SEC_UNDER_LINE1 = "(the control arm did not climb on the primary)"
SECONDARIES = (
    {"key": "SEC-G", "higher": INTERVENTION, "marker": "OWN.P.G at the loop row with steps == E (the final pair)",
     "title": "the end loop gain: OWN G at the final pair (E)"},
    {"key": "SEC-CMD", "higher": CONTROL, "marker": "ledger.direct.per_evaluation[E].cmd_l",
     "title": "the ledger tail command CMD_L at the evaluation at E"},
    {"key": "SEC-MARGIN", "higher": INTERVENTION, "marker": "OWN.P.b_star at the loop row with steps == E (the final "
     "pair)", "title": "the margin to the rail: OWN b*_P = 1 - G - c0 at the final pair (E)"},
)


class InputError(Exception):
    """A record the scorer cannot read as the criteria require (exit 2)."""


# ------------------------------------------------------------------------------------------------ the test
def subset_sums(values, k):
    """{subset (tuple of indices): sum} over all k-subsets; values are (infinite part, finite Fraction) pairs."""
    out = {}
    for sub in itertools.combinations(range(len(values)), k):
        inf = sum(values[i][0] for i in sub)
        fin = sum((values[i][1] for i in sub), Fraction(0))
        out[sub] = (inf, fin)
    return out


def perm_test(higher, other):
    """The exact one-sided permutation test that the arm `higher` is higher (for the primary: control higher = the
    blind arm lower).  higher, other: lists of (infinite part, Fraction) pairs.  Returns (p, count, total): p =
    #{relabelings whose `higher`-arm sum is >= the observed} / total (a tie counts against the reading), as an exact
    Fraction; the sum of one arm is equivalent to the difference of arm means for any arm sizes.  None when an arm is
    empty."""
    if not higher or not other:
        return None
    values = list(higher) + list(other)
    k = len(higher)
    obs = (sum(v[0] for v in higher), sum((v[1] for v in higher), Fraction(0)))
    sums = subset_sums(values, k)
    count = sum(1 for s in sums.values() if s >= obs)
    return Fraction(count, len(sums)), count, len(sums)


def sign_flip_test(diffs):
    """Exact one-sided sign-flip test of paired differences (control - blind; report only).  (p, count, total)."""
    if not diffs:
        return None
    obs = sum(diffs, Fraction(0))
    total = 0
    count = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if sum((s * d for s, d in zip(signs, diffs, strict=True)), Fraction(0)) >= obs:
            count += 1
    return Fraction(count, total), count, total


def fnum(x, fmt="+.4f"):
    return "n/a" if x is None else format(float(x), fmt)


def fmt_p(t):
    if t is None:
        return "n/a"
    p, count, total = t
    return f"{count}/{total} ({float(p):.4f})"


# ------------------------------------------------------------------------------------------------ records
def _read_json(path):
    """(object, sha256) or (None, None) when the file does not exist."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InputError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(d, dict):
        raise InputError(f"{path} is not a JSON object")
    return d, hashlib.sha256(raw).hexdigest()


def screen_of(root):
    """SCREEN.json, valid (bs_common.screen_problems) and equal to the criteria's queue."""
    d, problems = C.load_screen(str(Path(root) / "SCREEN.json"))
    if problems:
        raise InputError("SCREEN.json: " + "; ".join(problems))
    if d.get("intervention") != INTERVENTION:
        raise InputError(f"SCREEN.json intervention {d.get('intervention')!r} is not the criteria's {INTERVENTION}")
    if C.queue_of(d) != QUEUE:
        raise InputError(f"SCREEN.json queue {C.queue_of(d)} is not the criteria's {QUEUE}")
    return d


def lost_of(root):
    try:
        lost = R.load_lost(Path(root) / "lost.json")
    except R.InputError as exc:
        raise InputError(str(exc)) from exc
    names = {C.run_name(a, s) for a, s in QUEUE}
    unknown = sorted(set(lost) - names)
    if unknown:
        raise InputError(f"lost.json names continuations outside the queue: {unknown}")
    return lost


def classify(name, arm, seed, check, csha, lost_entry):
    """One entry's class from check.json and its lost.json entry (criteria RUN HANDLING)."""
    out = dict(name=name, arm=arm, seed=seed, cls=UNDECIDED, T=None, cause=None, check_sha256=csha)
    if lost_entry is not None:
        cause = lost_entry.get("cause") if isinstance(lost_entry, dict) else None
        detail = lost_entry.get("detail") if isinstance(lost_entry, dict) else None
        out.update(cls=LOST, cause=f"lost.json: {cause or 'listed'}" + (f" ({detail})" if detail else ""))
        return out
    if check is None:
        out["cause"] = "no check.json (queued, running, crashed or VOID)"
        return out
    want = dict(name=name, arm=arm, seed=seed, S=C.S, E=C.E, mode="continuation")
    for key, value in want.items():
        if check.get(key) != value:
            raise InputError(f"{name}: check.json {key} = {check.get(key)!r}, expected {value!r}")
    if check.get("passed") is not True or check.get("failures"):
        out["cause"] = "CHECK failed (VOID until the cause is fixed and it reruns)"
        return out
    T, stop = check.get("T"), check.get("stop_reason")
    if stop == "end" and T == C.E:
        out.update(cls=COUNTED, T=T)
    elif stop == "collapse" and isinstance(T, int) and not isinstance(T, bool) and C.S < T < C.E:
        out.update(cls=STOPPED, T=T, cause=f"the collapse stop at T = {T:,} (no rows before E: filled)")
    else:
        raise InputError(f"{name}: a passing check.json with stop reason {stop!r} at T = {T!r}")
    return out


def gather(root):
    """[(class dict, check)] in the criteria's queue order."""
    root = Path(root)
    lost = lost_of(root)
    out = []
    for arm, seed in QUEUE:
        name = C.run_name(arm, seed)
        check, csha = _read_json(root / "runs" / name / C.CHECK_JSON)
        out.append((classify(name, arm, seed, check, csha, lost.get(name)), check))
    return out


def markers_path(root, name):
    return Path(root) / "reads" / name / "markers.json"


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def loop_rows_at(m, steps):
    """The markers' M-LOOP rows whose steps equal `steps` (the final pair at E has steps == T: read_bs.saved_pairs)."""
    return [r for r in (m.get("loop") or []) if isinstance(r, dict) and type(r.get("steps")) is int
            and r.get("steps") == steps]


def check_markers(m, c, check):
    """markers.json against its class and check.json (raises InputError on a record that disagrees)."""
    name = c["name"]
    if not isinstance(m, dict):
        raise InputError(f"{name}: no markers.json")
    for key, value in (("schema", R.MARKERS_SCHEMA), ("kind", "continuation"), ("label", name), ("arm", c["arm"]),
                       ("seed", c["seed"]), ("S", C.S), ("T", c["T"]),
                       ("stop_reason", "end" if c["cls"] == COUNTED else "collapse")):
        if m.get(key) != value:
            raise InputError(f"{name}: markers.json {key} = {m.get(key)!r}, expected {value!r}")
    probe = m.get("probe")
    if not isinstance(probe, dict):
        raise InputError(f"{name}: markers.json has no probe markers")
    src = (check.get("sources") or {}).get("policy_probe.npz")
    if not isinstance(src, list) or len(src) != 2 or probe.get("sha256") != src[1]:
        raise InputError(f"{name}: markers.json was read from another policy_probe.npz than the one CHECK passed "
                         f"({probe.get('sha256')!r} vs {src!r})")
    steps = probe.get("steps")
    want = [C.S + C.ROLLOUT * k for k in range(1, (c["T"] - C.S) // C.ROLLOUT)]
    if steps != want or probe.get("n_rows") != len(want):
        raise InputError(f"{name}: markers' probe rows are not S + 8,192 k in (S, T) ({probe.get('n_rows')!r} rows)")
    for key in ("rollout_mu14_mean", "tail3_statue_settled_mean", "trailing8_mean"):
        if not isinstance(probe.get(key), list) or len(probe[key]) != len(want):
            raise InputError(f"{name}: markers' probe {key} is not one value per row")
    blind = c["arm"] == INTERVENTION
    form = "blind+raw" if blind else "plain"
    for row in m.get("loop") or []:
        if row.get("critic_form") != form:
            raise InputError(f"{name}: markers' loop row {row.get('steps')} critic form {row.get('critic_form')!r} "
                             f"is not {form!r} (the arm's reading)")
        if blind:
            for states, d in (row.get("critic_slope_obs6") or {}).items():
                eff = (((d or {}).get("effective")) or {}).get("dV_dq3_per_deg")
                if eff not in (None, 0.0):
                    raise InputError(f"{name}: a critic_blind_tail3 run's effective critic slope on observation 6 "
                                     f"is {eff} on the {states} states at {row.get('steps')} (must be 0.0)")
    # the final pair's loop row (SEC-G, SEC-MARGIN) is the model at E: its recorded num_timesteps must be E
    for row in loop_rows_at(m, C.E):
        if row.get("num_timesteps") != C.E:
            raise InputError(f"{name}: markers' loop row at E = {C.E:,} records num_timesteps "
                             f"{row.get('num_timesteps')!r} (the final pair must be the model at E)")


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def mean_over(probe, key, steps):
    """(Fraction mean, None) of probe[key] over the rows at the given steps, or (None, cause)."""
    st = probe.get("steps") or []
    vals = []
    for s in steps:
        idx = [i for i, x in enumerate(st) if x == s]
        if len(idx) != 1:
            return None, f"{len(idx)} probe rows at update_timesteps {s:,}"
        v = probe[key][idx[0]]
        if not _finite(v):
            return None, f"{key} at {s:,} is {v!r}"
        vals.append(Fraction(v))
    return sum(vals, Fraction(0)) / len(vals), None


def primary_of(c, m):
    """(Fraction, None) or (None, cause): THE PRIMARY of one COUNTED continuation."""
    if c["cls"] != COUNTED:
        return None, c.get("cause") or c["cls"]
    v, cause = mean_over(m["probe"], "rollout_mu14_mean", PRIMARY_STEPS)
    if v is None:
        return None, f"primary unreadable: {cause}"
    return v, None


def secondary_value(key, c, m, primary_cause):
    """(Fraction, None) or (None, cause): one secondary's value for one continuation (criteria THE SECONDARIES).
    MISSING for a LOST, STOPPED or UNREADABLE continuation, and when its marker is absent or not finite."""
    if c["cls"] != COUNTED:
        return None, c.get("cause") or c["cls"]
    if primary_cause is not None:
        return None, f"UNREADABLE ({primary_cause})"
    if key in ("SEC-G", "SEC-MARGIN"):
        rows = loop_rows_at(m, C.E)
        if len(rows) != 1:
            return None, f"{len(rows)} M-LOOP rows at E = {C.E:,} (the final pair)"
        field = "G" if key == "SEC-G" else "b_star"
        v, what = _get(rows[0], "OWN", "P", field), f"OWN.P.{field} at E"
    elif key == "SEC-CMD":
        per = _get(m, "ledger", "direct", "per_evaluation")
        e = per.get(str(C.E)) if isinstance(per, dict) else None
        if not isinstance(e, dict):
            return None, f"the evaluation at E = {C.E:,} is absent from the ledger markers"
        v, what = e.get("cmd_l"), "CMD_L at E"
    else:
        raise ValueError(key)
    if not _finite(v):
        return None, f"{what} is {v!r} (no finite value)"
    return Fraction(v), None


def ledger_report(m):
    """Returns and falls in (S, T] from the ledger the markers name (report only; never fails the score).
    (dict, provisional item or None)."""
    lm = m.get("ledger") or {}
    path, want = lm.get("file"), lm.get("sha256")
    if not path:
        return {"status": "no ledger named"}, None
    try:
        got = C.sha256_file(path)
    except OSError as exc:
        return {"status": f"unreadable ({exc})"}, None
    if got != want:
        return ({"status": "changed since the read (sha256 differs)"},
                f"{m.get('label')}: the ledger {path} changed since the read ({got} != {want})")
    per = {}
    falls = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return {"status": f"unreadable ({exc.__class__.__name__}: {exc})"}, None
    for r in rows:
        try:
            t = float(r.get("timesteps"))
            ret = float(r.get("return"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(t) and t == int(t) and C.S < int(t) <= m["T"]):
            continue
        per.setdefault(int(t), []).append(ret)
        if str(r.get("full_horizon")).strip() not in ("1", "1.0", "True", "true"):
            falls.append({"timesteps": int(t), "episode": r.get("episode"), "length": r.get("length"),
                          "cause": r.get("cause")})
    rets = [x for v in per.values() for x in v]
    return {"status": "read", "n_episodes": len(rets), "n_evaluations": len(per),
            "mean_return_window": statistics.fmean(rets) if rets else None,
            "mean_return_at_T": statistics.fmean(per[max(per)]) if per else None,
            "mean_return_first": statistics.fmean(per[min(per)]) if per else None,
            "mean_return_per_evaluation": {t: statistics.fmean(v) for t, v in sorted(per.items())},
            "n_falls": len(falls), "falls": falls[:20]}, None


def loop_rows(m):
    out = []
    for row in m.get("loop") or []:
        def g(*ks, row=row):
            return _get(row, *ks)
        out.append({"steps": row.get("steps"), "OWN_G": g("OWN", "P", "G"), "OWN_c0": g("OWN", "P", "c0"),
                    "OWN_b_star": g("OWN", "P", "b_star"), "OWN_G_L": g("OWN", "L", "G"),
                    "OWN_b_star_L": g("OWN", "b_star_L"), "own14": g("OWN", "own14_median"),
                    "OWN_falls": g("OWN", "n_terminated"), "REF_G": g("REF", "P", "G"), "REF_c0": g("REF", "P", "c0"),
                    "slope_onpolicy_eff": g("critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg"),
                    "slope_onpolicy_raw": g("critic_slope_obs6", "ONPOLICY", "raw", "dV_dq3_per_deg"),
                    "slope_own_eff": g("critic_slope_obs6", "OWN", "effective", "dV_dq3_per_deg"),
                    "slope_own_raw": g("critic_slope_obs6", "OWN", "raw", "dV_dq3_per_deg"),
                    "slope_ref_eff": g("critic_slope_obs6", "REF", "effective", "dV_dq3_per_deg"),
                    "slope_ref_raw": g("critic_slope_obs6", "REF", "raw", "dV_dq3_per_deg")})
    return out


def trajectory(probe, T):
    """[(evaluation step t, Fraction or None)] for t in (S, T] on the 131,072 grid: the mean of rollout_mu14_mean over
    the 8 probe rows ending at t (at t = T, which has no row, the 8 rows ending at T - 8,192: at E the primary's rows).
    Report only."""
    out = []
    for t in range(C.S + C.EVAL_FREQ, T + 1, C.EVAL_FREQ):
        end = t if t < T else T - C.ROLLOUT
        v, _ = mean_over(probe, "rollout_mu14_mean", tuple(end - C.ROLLOUT * k for k in range(PRIMARY_K - 1, -1, -1)))
        out.append((t, v))
    return out


def g_end_above_start(m):
    """True / False: OWN G at the loop row at E above OWN G at the loop row at S (the start pair); None if either row
    or value is absent (report only)."""
    rs, re_ = loop_rows_at(m, C.S), loop_rows_at(m, C.E)
    if len(rs) != 1 or len(re_) != 1:
        return None
    gs, ge = _get(rs[0], "OWN", "P", "G"), _get(re_[0], "OWN", "P", "G")
    if not (_finite(gs) and _finite(ge)):
        return None
    return ge > gs


def report_only(c, m, meta):
    """Everything the criteria print and never test.  (dict, provisional item from the ledger or None)."""
    probe = m["probe"]
    early, _ = mean_over(probe, "rollout_mu14_mean", EARLY_STEPS)
    last_t = [s for s in probe["steps"][-PRIMARY_K:]]
    last_mu, _ = mean_over(probe, "rollout_mu14_mean", last_t)
    ss_last, _ = mean_over(probe, "tail3_statue_settled_mean", last_t)
    tr = probe.get("trailing8_mean") or []
    lm = (m.get("ledger") or {})
    d, m5 = lm.get("direct") or {}, lm.get("m5") or {}
    per = d.get("per_evaluation") if isinstance(d.get("per_evaluation"), dict) else {}
    cmd_per = {}
    for k, e in per.items():
        try:
            cmd_per[int(k)] = e.get("cmd_l") if isinstance(e, dict) else None
        except (TypeError, ValueError):
            continue
    ev = (meta or {}).get("arm_evidence") or {}
    aa = (meta or {}).get("arm_applied") or {}
    manip = {"n_rollouts": ev.get("n"), "buffer_gae_lambda": ev.get("buffer_gae_lambda"),
             "max_adv_err_vs_gae095": ev.get("max_adv_err_vs_gae095"), "min_dmu_actor": ev.get("min_dmu_actor"),
             "max_dv_predict_values": ev.get("max_dv_predict_values"), "max_dv_forward": ev.get("max_dv_forward"),
             "max_dv_evaluate_actions": ev.get("max_dv_evaluate_actions"), "max_dv_mirrored": ev.get("max_dv_mirrored"),
             "min_dv_predict_values": ev.get("min_dv_predict_values"),
             "min_buffer_values_vs_raw_value_mlp": ev.get("min_buffer_values_vs_raw_value_mlp"),
             "state_problems": ev.get("state_problems"), "mlp_extractor_class": aa.get("mlp_extractor_class"),
             "arm_state_end": (meta or {}).get("arm_state_end")}
    led, item = ledger_report(m)
    loop = loop_rows(m)
    end = [lr for lr in loop if type(lr["steps"]) is int and lr["steps"] == c["T"]]
    return {"early8_rollout_mu14": early, "last8_rollout_mu14_before_T": last_mu,
            "trajectory": trajectory(probe, c["T"]), "g_end_above_start": g_end_above_start(m),
            "statue_settled_last8_mean": ss_last, "statue_settled_trailing8_at_last_row": tr[-1] if tr else None,
            "statue_settled_trailing8_max": probe.get("trailing8_max"),
            "statue_settled_first_above": probe.get("trailing8_first_above"),
            "std14_last": (probe.get("std14") or [None])[-1], "obs6_mean_deg_last": (probe.get("obs6_mean_deg")
                                                                                     or [None])[-1],
            "cmd_l_last": d.get("cmd_l_last"), "cmd_l_max": d.get("cmd_l_max"), "cmd_l_per_evaluation": cmd_per,
            "ledger_firsts": d.get("firsts"),
            "m5_any_flagged": m5.get("any_flagged"), "m5_tail_flagged": m5.get("tail_flagged"),
            "m5_any_alarms": m5.get("any_alarms"), "m5_tail_alarms": m5.get("tail_alarms"),
            "m5_c6_counting_failures": m5.get("n_c6_counting_failures"), "loop": loop,
            "end_pair": end[0] if len(end) == 1 else None, "ledger": led,
            "manipulation": manip}, item


# ------------------------------------------------------------------------------------------------ PROVISIONAL
HEX64 = re.compile(r"[0-9a-f]{64}")
PINNED_TOOLS = (*C.TOOL_FILES, "score_bs.py")


def _hex(v):
    return v if isinstance(v, str) and HEX64.fullmatch(v) else None


def _load_pinned_m1_common():
    common = C.M1_COMMON
    try:
        if C.sha256_file(common) != C.PIN_SHA256[common]:
            return None, f"{common} is not at its pin"
    except OSError as exc:
        return None, f"{common}: {exc}"
    spec = importlib.util.spec_from_file_location("score_bs_r10m_common", common)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, None


def code_seal_items(code_dir):
    """The code copy measured again by M1's check_copy (never the seal file alone)."""
    if not Path(code_dir).is_dir():
        return [f"code copy {code_dir} missing: not measured"]
    rc, problem = _load_pinned_m1_common()
    if problem:
        return [f"code copy not measured: {problem}"]
    try:
        seal = rc.check_copy(str(code_dir), expect_commit=C.COMMIT)
    except rc.Refusal as exc:
        return [f"code copy: {getattr(exc, 'message', exc)}"]
    except (OSError, ValueError) as exc:
        return [f"code copy unreadable ({exc})"]
    if seal.get("tree_digest") != C.TREE_DIGEST or seal.get("files") != C.TREE_FILES:
        return [f"code copy tree digest {seal.get('tree_digest')} ({seal.get('files')} files) != pinned "
                f"{C.TREE_DIGEST} ({C.TREE_FILES} files)"]
    return []


def provisional_items(root, classes, checks, metas=None, markers=None):
    """Every reason the reading is PROVISIONAL (criteria WHEN THE READING IS MADE).  metas / markers None: the screen
    is incomplete and no run's run_meta.json or markers are opened (outcome-blind); only the pins and check.json."""
    root = Path(root)
    items = []
    sha = {}
    for name in ("criteria.txt", "SCREEN.json", "APPROVED"):
        try:
            sha[name] = C.sha256_file(root / name)
        except OSError:
            sha[name] = None
            items.append(f"no {root / name}")
    own = C.sha256_file(Path(__file__))
    tool_now = {}
    for t in PINNED_TOOLS:
        try:
            tool_now[t] = C.sha256_file(TOOLS / t)
        except OSError:
            tool_now[t] = None
    pins = {}
    if sha["APPROVED"] is not None:
        try:
            appr = json.loads((root / "APPROVED").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            appr = None
        if not isinstance(appr, dict):
            items.append("APPROVED is not a JSON object")
            appr = {}
        missing = [k for k in C.APPROVED_KEYS if k not in appr]
        if missing:
            items.append(f"APPROVED: missing keys {missing}")
        for key, name in (("criteria_sha256", "criteria.txt"), ("screen_sha256", "SCREEN.json")):
            if sha[name] is not None and appr.get(key) != sha[name]:
                items.append(f"{name} sha256 {sha[name]} != APPROVED {key} {appr.get(key)!r}")
        if appr.get("code_tree_digest") != C.TREE_DIGEST:
            items.append(f"APPROVED code_tree_digest {appr.get('code_tree_digest')!r} != {C.TREE_DIGEST}")
        tp = appr.get("tool_sha256") if isinstance(appr.get("tool_sha256"), dict) else {}
        for t in PINNED_TOOLS:
            pins[t] = _hex(tp.get(t))
            if pins[t] is None:
                items.append(f"APPROVED has no bare 64-hex tool_sha256[{t!r}]")
            elif pins[t] != tool_now[t]:
                items.append(f"tools/{t}: sha256 {tool_now[t]} != APPROVED's {pins[t]}")
    if sha["criteria.txt"] is not None:
        text = (root / "criteria.txt").read_text(encoding="utf-8", errors="replace")
        if own not in text:
            items.append(f"criteria.txt does not name this scorer's sha256 {own} (its PINS section)")
    items.extend(f"M5 pin: {x}" for x in C.check_pins())
    read_pin = pins.get("read_bs.py") or tool_now["read_bs.py"]
    for c in classes:
        name = c["name"]
        check = checks.get(name)
        if check is not None and check.get("tool_sha256") != read_pin:
            items.append(f"{name}: check.json made by read_bs.py {check.get('tool_sha256')}, not {read_pin}")
        if c["cls"] not in (COUNTED, STOPPED) or metas is None:
            continue
        meta, msha = metas.get(name, (None, None))
        if meta is None:
            items.append(f"{name}: run_meta.json missing")
            continue
        src = (check.get("sources") or {}).get("run_meta.json")
        if not isinstance(src, list) or len(src) != 2 or src[1] != msha:
            items.append(f"{name}: run_meta.json changed since CHECK")
        for key, name_ in (("criteria_sha256", "criteria.txt"), ("screen_sha256", "SCREEN.json"),
                           ("approved_sha256", "APPROVED")):
            if meta.get(key) != sha[name_] or sha[name_] is None:
                items.append(f"{name}: ran under {name_} {meta.get(key)!r}, not the current {sha[name_]!r}")
        rt = meta.get("tool_sha256") if isinstance(meta.get("tool_sha256"), dict) else {}
        for t in C.TOOL_FILES:
            want = pins.get(t) or tool_now[t]
            if rt.get(t) != want:
                items.append(f"{name}: ran with {t} {rt.get(t)!r}, not {want}")
        mk = (markers or {}).get(name)
        if mk is not None and mk.get("tool_sha256") != read_pin:
            items.append(f"{name}: markers.json made by read_bs.py {mk.get('tool_sha256')}, not {read_pin}")
        # criteria THE SECONDARIES / RUN HANDLING: a COUNTED continuation's markers must hold exactly one loop row at E
        # (the final pair, read_bs.saved_pairs); without it the final pair was not read (a tool or file defect), so
        # SEC-G and SEC-MARGIN read MISSING and the reading is held (review finding D7)
        if c["cls"] == COUNTED and mk is not None and len(loop_rows_at(mk, C.E)) != 1:
            items.append(f"{name}: COUNTED but its markers hold {len(loop_rows_at(mk, C.E))} loop rows at E = "
                         f"{C.E:,} (the final pair was not read: a tool or file defect; SEC-G and SEC-MARGIN MISSING)")
    # BS2 reuses BS's code copy read only: it is measured where bs_common.CODE names it (not under <root>)
    items.extend(code_seal_items(Path(C.CODE)))
    return items


# ------------------------------------------------------------------------------------------------ the reading
def median_rule(values):
    """THE MEDIAN RULE (criteria STAT-2): the values sorted; 4 -> the midpoint of the two middle ones, 3 -> the middle
    one, 2 -> the midpoint, 1 -> the value, 0 -> None; exact (statistics.median on Fractions)."""
    return statistics.median(values) if values else None


def decide(classes, primaries):
    """The reading from the classes and the primaries ({name: Fraction or None}).  The secondaries are not inputs."""
    vals = {arm: [] for arm in ARM_ORDER}
    counted = {arm: [] for arm in ARM_ORDER}
    missing = {arm: [] for arm in ARM_ORDER}
    for c in classes:
        v = primaries.get(c["name"])
        if v is None:
            vals[c["arm"]].append(INF_CONTROL if c["arm"] == CONTROL else INF_BLIND)
            missing[c["arm"]].append(c["name"])
        else:
            vals[c["arm"]].append((0, v))
            counted[c["arm"]].append(v)
    filled = perm_test(vals[CONTROL], vals[INTERVENTION])
    as_counted = perm_test([(0, v) for v in counted[CONTROL]], [(0, v) for v in counted[INTERVENTION]])
    n_missing = len(missing[CONTROL]) + len(missing[INTERVENTION])
    base = Fraction(BASELINE_S)
    med = {arm: median_rule(counted[arm]) for arm in ARM_ORDER}
    rise = med[CONTROL] - base if med[CONTROL] is not None else None
    informative = rise is not None and rise >= RISE_MIN
    if not informative:
        line = 1
    elif filled[0] <= ALPHA:
        line = 2
    else:
        line = 3
    additions = []
    if line == 3 and as_counted is not None and as_counted[0] <= ALPHA:
        additions.append(ADDITION_AS_COUNTED)
    if line == 1 and filled[0] <= ALPHA:
        additions.append(ADDITION_NOT_INFORMATIVE)
    pairs = []
    for seed in SEEDS:
        a, b = primaries.get(C.run_name(CONTROL, seed)), primaries.get(C.run_name(INTERVENTION, seed))
        pairs.append({"seed": seed, "control": a, "blind": b, "diff": (a - b) if a is not None and b is not None
                      else None})
    diffs = [p["diff"] for p in pairs if p["diff"] is not None]
    means = {arm: (sum(counted[arm], Fraction(0)) / len(counted[arm]) if counted[arm] else None) for arm in ARM_ORDER}
    # each arm's sample SD of the primaries read (report only, never a line: it places the power, criteria KNOWN LIMITS)
    sds = {arm: (statistics.stdev([float(v) for v in counted[arm]]) if len(counted[arm]) >= 2 else None)
           for arm in ARM_ORDER}
    return dict(line=line, key=LINE_KEYS[line], text=LINE_TEXT[line], additions=additions, filled=filled,
                as_counted=as_counted, n_missing=n_missing, missing=missing, counted=counted, medians=med,
                means=means, sds=sds, control_rise=rise, informative=informative, pairs=pairs,
                paired_test=sign_flip_test(diffs), filled_values=vals)


def secondary_test(spec, classes, values, causes, line):
    """One pre-registered secondary (criteria THE SECONDARIES): the exact one-sided test that spec["higher"] is higher,
    on the difference of arm means (the `higher` arm's sum), its fill (a MISSING value in the `higher` arm counts as
    -infinity, in the other arm as +infinity), SUPPORTED at filled p <= 0.05 unadjusted.  `line` only selects the
    printed note under line 1; nothing here feeds the reading."""
    higher = spec["higher"]
    other = INTERVENTION if higher == CONTROL else CONTROL
    vals = {arm: [] for arm in ARM_ORDER}
    read = {arm: [] for arm in ARM_ORDER}
    missing = {arm: [] for arm in ARM_ORDER}
    for c in classes:
        v = values.get(c["name"])
        if v is None:
            vals[c["arm"]].append(NEG_INF if c["arm"] == higher else POS_INF)
            missing[c["arm"]].append(c["name"])
        else:
            vals[c["arm"]].append((0, v))
            read[c["arm"]].append(v)
    filled = perm_test(vals[higher], vals[other])
    as_counted = perm_test([(0, v) for v in read[higher]], [(0, v) for v in read[other]])
    supported = filled[0] <= ALPHA
    notes = []
    if supported and filled[0] <= ALPHA / N_SECONDARIES:
        notes.append(SEC_BONFERRONI)
    if line == 1:
        notes.append(SEC_UNDER_LINE1)
    return dict(key=spec["key"], title=spec["title"], marker=spec["marker"], higher=higher, lower=other,
                verdict=SEC_SUPPORTED if supported else SEC_NOT_SUPPORTED, supported=supported, notes=notes,
                filled=filled, as_counted=as_counted, values=dict(values), causes=dict(causes), missing=missing,
                n_read={a: len(read[a]) for a in ARM_ORDER},
                means={a: (sum(read[a], Fraction(0)) / len(read[a]) if read[a] else None) for a in ARM_ORDER},
                medians={a: median_rule(read[a]) for a in ARM_ORDER})


def _f(x):
    return None if x is None else float(x)


def score(root):
    """(result dict, text lines, exit code)."""
    root = Path(root)
    screen_of(root)
    rows = gather(root)
    classes = [c for c, _ in rows]
    checks = {c["name"]: ch for c, ch in rows}
    L = [VERSION, f"tool sha256 {C.sha256_file(Path(__file__))}  utc {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
         f"root {root}"]
    undecided = [c for c in classes if c["cls"] == UNDECIDED]
    decided_read = [c for c in classes if c["cls"] in (COUNTED, STOPPED)]
    unread = [c for c in decided_read if not markers_path(root, c["name"]).is_file()]
    result = dict(version=VERSION, root=str(root))
    if undecided or unread:
        prov = provisional_items(root, classes, checks)
        result.update(status="incomplete", provisional=prov,
                      classes=[{k: c[k] for k in ("name", "arm", "seed")} | {"decided": c["cls"] != UNDECIDED}
                               for c in classes])
        L += ["", f"STATUS: incomplete ({len(classes) - len(undecided)} of {len(classes)} decided; {len(unread)} "
                  f"decided without the full read). No line is read; no outcome is printed."]
        for c in classes:
            if c["cls"] == UNDECIDED:
                kept = sorted(p.name for p in (root / "runs").glob(c["name"] + "_*") if p.is_dir()) \
                    if (root / "runs").is_dir() else []
                state = f"undecided: {c['cause']}" + (f"; kept folders {kept}" if kept else "")
            elif c in unread:
                state = "decided, markers not read yet"
            else:
                state = "decided"
            L.append(f"  {c['name']:<24} {state}")
        if prov:
            L.append("PROVISIONAL items (printed; nothing is acted on while any stands):")
            L.extend(f"  - {x}" for x in prov)
        return result, L, EXIT_NOT_FINAL
    # ---- complete: open the reads
    markers, metas, primaries, causes, reports, ledger_items = {}, {}, {}, {}, {}, []
    for c in classes:
        name = c["name"]
        if c["cls"] in (COUNTED, STOPPED):
            m, _msha = _read_json(markers_path(root, name))
            check_markers(m, c, checks[name])
            markers[name] = m
            metas[name] = _read_json(root / "runs" / name / C.RUN_META)
            reports[name], item = report_only(c, m, metas[name][0])
            if item:
                ledger_items.append(item)
        v, cause = primary_of(c, markers.get(name))
        primaries[name] = v
        causes[name] = cause
    prov = provisional_items(root, classes, checks, metas, markers) + ledger_items
    dec = decide(classes, primaries)
    secs = []
    for spec in SECONDARIES:
        sv, sc = {}, {}
        for c in classes:
            sv[c["name"]], sc[c["name"]] = secondary_value(spec["key"], c, markers.get(c["name"]), causes[c["name"]])
        secs.append(secondary_test(spec, classes, sv, sc, dec["line"]))
    status = "PROVISIONAL" if prov else "final"
    L += ["", f"STATUS: {status}" + (" (printed, not acted on)" if prov else "")]
    L.append(f"THE READING (line {dec['line']}, {dec['key']}):")
    L.append("  " + dec["text"])
    L.extend("  " + a for a in dec["additions"])
    L.append("")
    L.append("THE PRIMARY (on-policy tail_3 command: mean of rollout_mu14_mean over the probe rows "
             f"{PRIMARY_STEPS[0]:,} ... {PRIMARY_STEPS[-1]:,}); B_S = {BASELINE_S:+.5f} (the parent's 8 rows ending "
             f"at S = {C.S:,})")
    for c in classes:
        v = primaries[c["name"]]
        tail = (f"{float(v):+.5f}  (rise from B_S {float(v - Fraction(BASELINE_S)):+.5f})" if v is not None
                else f"MISSING, filled at {'-inf' if c['arm'] == CONTROL else '+inf'}: {causes[c['name']]}")
        L.append(f"  {c['name']:<24} {c['cls']:<8} {tail}")
    for arm in ARM_ORDER:
        L.append(f"  {arm:<24} n counted {len(dec['counted'][arm])}, missing {len(dec['missing'][arm])}; mean "
                 f"{fnum(dec['means'][arm], '+.5f')}, median {fnum(dec['medians'][arm], '+.5f')}, SD "
                 f"{fnum(dec['sds'][arm], '.5f')} (report only)")
    if dec["means"][CONTROL] is not None and dec["means"][INTERVENTION] is not None:
        L.append(f"  difference of arm means (control - critic_blind_tail3), as counted: "
                 f"{float(dec['means'][CONTROL] - dec['means'][INTERVENTION]):+.5f}")
    L.append("INFORMATIVENESS (line 1): the control arm's median primary must rise by >= 0.20 from B_S (inclusive, "
             "exact; the median rule: 4 read -> the midpoint of the two middle values, 3 -> the middle one, 2 -> their "
             "midpoint, 1 -> the value, none -> not informative)")
    L.append(f"  control median {fnum(dec['medians'][CONTROL], '+.5f')} over {len(dec['counted'][CONTROL])} read, "
             f"rise {fnum(dec['control_rise'], '+.5f')} -> {'informative' if dec['informative'] else 'NOT INFORMATIVE'}"
             f" (the parent's own rise over the window +{PARENT_TRAJECTORY[C.E] - BASELINE_S:.5f}, report only)")
    L.append("THE TEST: exact one-sided permutation test (control higher), difference of arm means, alpha 0.05; "
             "missing values filled against REDUCED (control -inf, critic_blind_tail3 +inf); size 3/70 when the 8 "
             "primaries are exchangeable under no effect, about 0.06 (up to about 0.09 given the gate) if blinding "
             "changes only the spread: read a REDUCED with each arm's SD")
    L.append(f"  p (filled, decides) = {fmt_p(dec['filled'])}  {'<=' if dec['filled'][0] <= ALPHA else '>'} 0.05; "
             f"minimum attainable 1/70 (0.0143)")
    L.append(f"  p as counted (report only) = {fmt_p(dec['as_counted'])}")
    L.append("PAIRED DIFFERENCES (control - critic_blind_tail3, same seeds; report only, not a test):")
    for p in dec["pairs"]:
        L.append(f"  seed {p['seed']}: control {fnum(p['control'], '+.5f')}  blind {fnum(p['blind'], '+.5f')}  "
                 f"diff {fnum(p['diff'], '+.5f')}")
    L.append(f"  exact one-sided sign-flip p (report only; 4 pairs cannot reach 0.05: minimum 1/16) = "
             f"{fmt_p(dec['paired_test'])}")
    L.append("THE SECONDARIES (pre-registered; each tested one-sided at 0.05, unadjusted; NONE CHANGES THE LINE; under "
             "the global null at most 9/70 = 0.129 that any reads SUPPORTED when each secondary's 8 values are "
             "exchangeable, about 0.18 if the arms' spreads differ; only p = 1/70 survives Bonferroni over the 3, "
             "0.05 / 3 = 0.0167):")
    for s in secs:
        L.append(f"  {s['key']}, {s['title']} [{s['marker']}]; direction: {s['higher']} HIGHER; fill: a MISSING "
                 f"{s['higher']} value -inf, a MISSING {s['lower']} value +inf")
        L.append("    " + " ".join([s["verdict"], *s["notes"]]))
        for c in classes:
            v = s["values"][c["name"]]
            L.append(f"    {c['name']:<24} " + (f"{float(v):+.5f}" if v is not None else
                                                f"MISSING, filled at {'-inf' if c['arm'] == s['higher'] else '+inf'}"
                                                f": {s['causes'][c['name']]}"))
        for arm in ARM_ORDER:
            L.append(f"    {arm:<24} n read {s['n_read'][arm]}, missing {len(s['missing'][arm])}; mean "
                     f"{fnum(s['means'][arm], '+.5f')}, median {fnum(s['medians'][arm], '+.5f')}")
        L.append(f"    p (filled, decides SUPPORTED) = {fmt_p(s['filled'])}; p as counted (report only) = "
                 f"{fmt_p(s['as_counted'])}")
    L.append("MANIPULATION (CHECK passed on every COUNTED / STOPPED one: the per-rollout evidence; summary from "
             "run_meta arm_evidence):")
    for c in classes:
        rp = reports.get(c["name"])
        if not rp:
            L.append(f"  {c['name']:<24} {c['cls']} (no run read)")
            continue
        mp = rp["manipulation"]
        if c["arm"] == INTERVENTION:
            s = (f"value change with obs 6 perturbed, max over {mp['n_rollouts']} rollouts: predict "
                 f"{mp['max_dv_predict_values']} forward {mp['max_dv_forward']} evaluate "
                 f"{mp['max_dv_evaluate_actions']} mirrored {mp['max_dv_mirrored']}; buffer values vs the "
                 f"unblinded value MLP, min {mp['min_buffer_values_vs_raw_value_mlp']}")
        else:
            s = (f"value change with obs 6 perturbed, min over {mp['n_rollouts']} rollouts: "
                 f"{mp['min_dv_predict_values']} (the critic sees it)")
        L.append(f"  {c['name']:<24} {mp['mlp_extractor_class']}, hook {mp['arm_state_end']}; {s}; actor dmu min "
                 f"{mp['min_dmu_actor']}; lambda {mp['buffer_gae_lambda']}; |adv - GAE(0.95)| max "
                 f"{mp['max_adv_err_vs_gae095']}")
    L.append("REPORT ONLY, PER CONTINUATION (never tested):")
    for c in classes:
        rp = reports.get(c["name"])
        if not rp:
            L.append(f"  {c['name']:<24} {c['cls']}: {c['cause']}")
            continue
        led = rp["ledger"]
        L.append(f"  {c['name']:<24} {c['cls']} T {c['T']:,}: on-policy first 8 rows "
                 f"{fnum(rp['early8_rollout_mu14'])}, last 8 before T {fnum(rp['last8_rollout_mu14_before_T'])}; "
                 f"statue_settled last 8 rows {fnum(rp['statue_settled_last8_mean'])}, 8-row trailing at the last row "
                 f"{fnum(rp['statue_settled_trailing8_at_last_row'])} (max {fnum(rp['statue_settled_trailing8_max'])}),"
                 f" first above {rp['statue_settled_first_above']}; std14 last {fnum(rp['std14_last'])}; tail_3 "
                 f"normaliser mean last {fnum(rp['obs6_mean_deg_last'], '+.3f')} deg")
        L.append(f"      ledger: CMD_L at T {fnum(rp['cmd_l_last'], '+.3f')} (max {fnum(rp['cmd_l_max'], '+.3f')}); "
                 f"firsts {rp['ledger_firsts']}; M5 ANY flagged {rp['m5_any_flagged']} TAIL flagged "
                 f"{rp['m5_tail_flagged']} ANY alarms {rp['m5_any_alarms']} TAIL alarms {rp['m5_tail_alarms']}; C6 "
                 f"counting failures {rp['m5_c6_counting_failures']}")
        L.append("      CMD_L per evaluation: " + ", ".join(f"{t:,} {fnum(v, '+.3f')}" for t, v in
                                                          sorted(rp["cmd_l_per_evaluation"].items())))
        if led.get("status") == "read":
            L.append(f"      returns: mean at the first / last evaluation {fnum(led['mean_return_first'], '.1f')} / "
                     f"{fnum(led['mean_return_at_T'], '.1f')}, window {fnum(led['mean_return_window'], '.1f')} over "
                     f"{led['n_episodes']} episodes in {led['n_evaluations']} evaluations; falls {led['n_falls']} "
                     + (str([(f['timesteps'], f['length'], f['cause']) for f in led['falls']]) if led["falls"] else ""))
            L.append("      mean return per evaluation: " + ", ".join(
                f"{t:,} {fnum(v, '.1f')}" for t, v in led["mean_return_per_evaluation"].items()))
        else:
            L.append(f"      returns and falls: {led.get('status')}")
        for lr in rp["loop"]:
            L.append(f"      pair {lr['steps']:>9,}: OWN G {fnum(lr['OWN_G'], '+.3f')} c0 {fnum(lr['OWN_c0'], '+.3f')} "
                     f"b* {fnum(lr['OWN_b_star'], '+.3f')} G_L {fnum(lr['OWN_G_L'], '+.3f')} b*_L "
                     f"{fnum(lr['OWN_b_star_L'], '+.3f')} own14 {fnum(lr['own14'], '+.3f')} falls {lr['OWN_falls']} | "
                     f"REF G {fnum(lr['REF_G'], '+.3f')} | critic dV/dq3 per deg eff/raw: on-policy "
                     f"{fnum(lr['slope_onpolicy_eff'], '+.5f')}/{fnum(lr['slope_onpolicy_raw'], '+.5f')} OWN "
                     f"{fnum(lr['slope_own_eff'], '+.5f')}/{fnum(lr['slope_own_raw'], '+.5f')} REF "
                     f"{fnum(lr['slope_ref_eff'], '+.5f')}/{fnum(lr['slope_ref_raw'], '+.5f')}")
    L.append("REPORT ONLY, THE ON-POLICY TRAJECTORY (the 8-row mean of rollout_mu14_mean ending at each evaluation "
             "step; at E the primary's rows), beside the parent's (F2 probe); untested:")
    names = [c["name"] for c in classes]
    L.append(f"  {'step':>9}  {'parent':>8}  " + "  ".join(f"{n[:4] + n.rsplit('_', 1)[1]:>9}" for n in names)
             + "  | median control / blind")
    traj = {n: dict(reports[n]["trajectory"]) if n in reports else {} for n in names}
    traj_medians = {}
    for t in EVAL_STEPS:
        meds = {}
        for arm in ARM_ORDER:
            v = [traj[c["name"]].get(t) for c in classes if c["arm"] == arm]
            v = [x for x in v if x is not None]
            meds[arm] = median_rule(v)
        traj_medians[t] = meds
        L.append(f"  {t:>9,}  {PARENT_TRAJECTORY[t]:>+8.4f}  " + "  ".join(f"{fnum(traj[n].get(t)):>9}" for n in names)
                 + f"  | {fnum(meds[CONTROL])} / {fnum(meds[INTERVENTION])}")
    gcount = {}
    for arm in ARM_ORDER:
        flags = [reports[c["name"]]["g_end_above_start"] for c in classes if c["arm"] == arm and c["name"] in reports]
        flags = [x for x in flags if x is not None]
        gcount[arm] = {"above": sum(1 for x in flags if x), "of": len(flags)}
    L.append("REPORT ONLY, G AT E ABOVE G AT S (OWN G at the final pair against the start pair's, the same start for "
             f"all; the parent's G(S) {PARENT_G_S:+.3f}): " + "; ".join(f"{arm} {gcount[arm]['above']} of "
                                                                         f"{gcount[arm]['of']}" for arm in ARM_ORDER))
    L.append("REPORT ONLY, ARM MEDIANS OVER THE READ CONTINUATIONS:")
    for arm in ARM_ORDER:
        rps = [reports[c["name"]] for c in classes if c["arm"] == arm and c["name"] in reports]

        def med(key, rps=rps):
            v = [r[key] for r in rps if _finite(r.get(key)) or isinstance(r.get(key), Fraction)]
            return statistics.median([float(x) for x in v]) if v else None

        def med_end(key, rps=rps):
            v = [r["end_pair"].get(key) for r in rps if r["end_pair"]]
            v = [x for x in v if _finite(x)]
            return statistics.median(v) if v else None

        def med_led(key, rps=rps):
            v = [r["ledger"].get(key) for r in rps if r["ledger"].get("status") == "read"]
            v = [x for x in v if _finite(x)]
            return statistics.median(v) if v else None
        L.append(f"  {arm:<24} on-policy first 8 rows {fnum(med('early8_rollout_mu14'))}; statue_settled last 8 rows "
                 f"{fnum(med('statue_settled_last8_mean'))}, trailing at last row "
                 f"{fnum(med('statue_settled_trailing8_at_last_row'))} (max "
                 f"{fnum(med('statue_settled_trailing8_max'))}); std14 last {fnum(med('std14_last'))}; tail_3 "
                 f"normaliser {fnum(med('obs6_mean_deg_last'), '+.3f')} deg; CMD_L at T "
                 f"{fnum(med('cmd_l_last'), '+.3f')} (max {fnum(med('cmd_l_max'), '+.3f')}); mean return window "
                 f"{fnum(med_led('mean_return_window'), '.1f')}, falls {fnum(med_led('n_falls'), '.1f')}")
        L.append(f"  {'':<24} at the end pair (the loop row at T) OWN G {fnum(med_end('OWN_G'), '+.3f')} c0 "
                 f"{fnum(med_end('OWN_c0'), '+.3f')} b* {fnum(med_end('OWN_b_star'), '+.3f')} G_L "
                 f"{fnum(med_end('OWN_G_L'), '+.3f')} b*_L {fnum(med_end('OWN_b_star_L'), '+.3f')} own14 "
                 f"{fnum(med_end('own14'), '+.3f')} REF G {fnum(med_end('REF_G'), '+.3f')}; critic slope on-policy eff "
                 f"{fnum(med_end('slope_onpolicy_eff'), '+.5f')} raw {fnum(med_end('slope_onpolicy_raw'), '+.5f')}, "
                 f"OWN eff {fnum(med_end('slope_own_eff'), '+.5f')} raw {fnum(med_end('slope_own_raw'), '+.5f')}, "
                 f"REF eff {fnum(med_end('slope_ref_eff'), '+.5f')} raw {fnum(med_end('slope_ref_raw'), '+.5f')}; "
                 f"the reader's own rollouts terminated {fnum(med_end('OWN_falls'), '.1f')}")
    odd = [c for c in classes if c["cls"] in (STOPPED, LOST) or primaries[c["name"]] is None]
    L.append("MISSING (STOPPED, LOST or primary unreadable; filled against REDUCED):")
    L.extend(f"  {c['name']:<24} {c['cls']:<8} {causes[c['name']]}" for c in odd)
    if not odd:
        L.append("  none")
    if prov:
        L.append("PROVISIONAL items (the reading is printed and not acted on while any stands):")
        L.extend(f"  - {x}" for x in prov)

    def frac(t):
        return None if t is None else {"p": str(t[0]), "p_float": float(t[0]), "count": t[1], "total": t[2]}
    result.update(
        status=status, provisional=prov, line=dec["line"], line_key=dec["key"], line_text=dec["text"],
        additions=dec["additions"], p_filled=frac(dec["filled"]), p_as_counted=frac(dec["as_counted"]),
        alpha=str(ALPHA), informative=dec["informative"], control_median=_f(dec["medians"][CONTROL]),
        control_rise=_f(dec["control_rise"]), rise_min=str(RISE_MIN), baseline_S=BASELINE_S,
        primary_steps=list(PRIMARY_STEPS), arm_means={a: _f(v) for a, v in dec["means"].items()},
        arm_medians={a: _f(v) for a, v in dec["medians"].items()}, arm_sd_report_only=dict(dec["sds"]),
        n_missing=dec["n_missing"],
        missing=dec["missing"], paired=[{k: (_f(v) if isinstance(v, Fraction) else v) for k, v in p.items()}
                                        for p in dec["pairs"]], paired_sign_flip=frac(dec["paired_test"]),
        classes=[{**{k: c[k] for k in ("name", "arm", "seed", "cls", "T", "cause")},
                  "primary": _f(primaries[c["name"]]), "primary_exact": str(primaries[c["name"]])
                  if primaries[c["name"]] is not None else None, "primary_cause": causes[c["name"]]}
                 for c in classes],
        secondaries=[{"key": s["key"], "title": s["title"], "marker": s["marker"], "higher": s["higher"],
                      "verdict": s["verdict"], "supported": s["supported"], "notes": s["notes"],
                      "p_filled": frac(s["filled"]), "p_as_counted": frac(s["as_counted"]),
                      "values": {n: _f(v) for n, v in s["values"].items()},
                      "values_exact": {n: (str(v) if v is not None else None) for n, v in s["values"].items()},
                      "causes": s["causes"], "n_read": s["n_read"],
                      "arm_means": {a: _f(v) for a, v in s["means"].items()},
                      "arm_medians": {a: _f(v) for a, v in s["medians"].items()}} for s in secs],
        parent_trajectory={str(t): v for t, v in PARENT_TRAJECTORY.items()},
        trajectory_arm_medians={str(t): {a: _f(v) for a, v in m.items()} for t, m in traj_medians.items()},
        g_end_above_start=gcount,
        report_only={n: rp for n, rp in reports.items()})
    return result, L, (EXIT_NOT_FINAL if prov else EXIT_FINAL)


def _jsonable(o):
    if isinstance(o, Fraction):
        return float(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=C.BS, help="the screen's folder (default <BS2>; another root for tests and the "
                                                  "dry run)")
    ap.add_argument("--out", default=None, help="default <root>/score")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.out).resolve() if a.out else root / "score"
    try:
        R.require_under(out, root)
        result, lines, code = score(root)
    except (InputError, R.InputError) as exc:
        print(f"{VERSION}\nINPUT ERROR: {exc}")
        return EXIT_INPUT
    text = "\n".join(lines)
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    R.atomic_write_text(out / "score.json", json.dumps(result, indent=1, default=_jsonable) + "\n")
    R.atomic_write_text(out / "score.txt", text + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
