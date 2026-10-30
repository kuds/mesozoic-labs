"""Bias screen (BS): the scorer.  Report only: reads files, never trains, never simulates.

    PY -I -B score_bs.py [--root <BS>] [--out <root>/score]

It applies <BS>/criteria.txt exactly (THE PRIMARY OUTCOME, THE TEST, THE READING, RUN HANDLING, WHEN THE READING IS
MADE); the constants below are the text's and are pinned with it (criteria.txt names this file's sha256; <BS>/APPROVED
pins it as tool_sha256["score_bs.py"]).

What it reads: <root>/SCREEN.json (it must be the criteria's queue); <root>/lost.json; for each queue entry
<root>/runs/<name>/check.json (read_bs.py --check).  Only once every entry is decided (COUNTED, STOPPED or LOST) and
every COUNTED or STOPPED one has its full read: <root>/reads/<name>/markers.json (read_bs.py's markers), the run's
run_meta.json (the per-rollout manipulation evidence summary and the pins it ran under) and the ledger the markers name
(returns and falls, report only).  For PROVISIONAL: <root>/APPROVED, <root>/criteria.txt, <root>/SCREEN.json, the five
BS tools and this file (in this file's folder) against APPROVED, M5's pins, and the code copy <root>/code measured by
M1's own r10m_common.check_copy (loaded only at its pin).
What it writes: <out>/score.json and <out>/score.txt (default <root>/score; refused outside <root>).

Classes: LOST (named in lost.json); COUNTED (check.json passed, stop reason "end" at E); STOPPED (check.json passed,
stop reason "collapse" at T < E); UNDECIDED otherwise (running, queued, crashed or VOID).  While any entry is UNDECIDED,
or a COUNTED / STOPPED one has no markers.json yet, the screen is "incomplete": no markers file is opened and no
outcome is printed (outcome-blind).

THE PRIMARY (one number per COUNTED continuation): the mean of the probe's rollout_mu14_mean (policy_probe.npz
rollout_mu_mean[:, 14], the actor's raw Gaussian mean for tail_3_pitch_act averaged over the rollout's own 8,192
states) over the 8 probe rows with update_timesteps E - 65,536 ... E - 8,192.  MISSING (filled): a LOST or STOPPED
continuation, or a COUNTED one whose 8 rows are not each present once and finite ("primary unreadable").
THE TEST: exact one-sided permutation test of control against critic_blind_tail3 on the difference of arm means, over
all C(8, 4) = 70 relabelings; p = (relabelings whose control-arm sum is >= the observed) / 70, exact rationals.
THE FILL: a missing control value counts as -infinity and a missing critic_blind_tail3 value as +infinity (the supremum
of p over every value the missing ones could have had: it never helps REDUCED).
THE READING, the first line that applies: 1 NOT INFORMATIVE (the median control primary rose by less than 0.10 from
B_S, or no control primary was read); 2 REDUCED (filled p <= 1/20); 3 NO REDUCTION DETECTED, with "(the as-counted p,
report only, is <= 0.05; the filled p decides)" when the as-counted p is <= 1/20 and the filled one is not.
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
VERSION = "score_bs 1.1 (2026-10-30; bias screen criteria v2)"


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
SEEDS = (2101, 2111, 2121, 2131)
QUEUE = tuple((arm, seed) for seed in SEEDS for arm in ARM_ORDER)        # interleaved, paired
PRIMARY_K = 8
PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K, 0, -1))   # 2,424,832 ... 2,482,176
EARLY_STEPS = tuple(C.S + C.ROLLOUT * k for k in range(1, PRIMARY_K + 1))     # 1,056,768 ... 1,114,112 (report only)
# B_S: the parent's own value at S in the primary's form: the mean of the parent F2 probe's rollout_mu_mean[:, 14] over
# its 8 rows with update_timesteps 991,232 ... 1,048,576 (rows 120-127; policy_probe.npz.copy1 sha256 4e0e7220...)
# [measured 2026-10-30, NOTES_criteria.txt; recomputed by tests/test_score_bs.py when the file is present].
BASELINE_S = -0.03157372563146055
BASELINE_STEPS = tuple(C.S - C.ROLLOUT * k for k in range(PRIMARY_K - 1, -1, -1))
RISE_MIN = Fraction(1, 10)
ALPHA = Fraction(1, 20)
STATUE_LINES = ("0.10", "0.14", "0.15", "0.16")

COUNTED, STOPPED, LOST, UNDECIDED = "COUNTED", "STOPPED", "LOST", "UNDECIDED"
LINE_KEYS = {1: "NOT INFORMATIVE", 2: "REDUCED", 3: "NO REDUCTION DETECTED"}
LINE_TEXT = {
    1: ("NOT INFORMATIVE: THE CONTROL ARM DID NOT REPRODUCE THE EARLY RISE OF THE ON-POLICY TAIL_3 COMMAND (its median "
        "primary rose by less than 0.10 from the value at S, or no control primary was read); no reading about the "
        "critic is made"),
    2: ("REDUCED: BLINDING THE CRITIC TO THE TAIL_3 ANGLE (observation 6) REDUCED THE ON-POLICY TAIL_3 COMMAND "
        "OVER THE LAST 8 UPDATES BEFORE E, FROM s42's 1,048,576 STATE (exact one-sided permutation test, p <= 0.05)"),
    3: ("NO REDUCTION DETECTED IN THE ON-POLICY TAIL_3 COMMAND OVER THE LAST 8 UPDATES BEFORE E FROM BLINDING THE "
        "CRITIC TO THE TAIL_3 ANGLE (not excluded: a reduction of about half the rise or smaller (power about 0.3-0.9 "
        "at half the rise, by the between-continuation spread; lower if paired seeds correlate), other inputs that "
        "carry the angle, the phase after E, the rail)"),
}
ADDITION_AS_COUNTED = "(the as-counted p, report only, is <= 0.05; the filled p decides)"
INF_CONTROL, INF_BLIND = (-1, Fraction(0)), (1, Fraction(0))          # THE FILL (lexicographic: infinity first)


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


def perm_test(control, blind):
    """The exact one-sided permutation test (control higher = the blind arm lower).  control, blind: lists of
    (infinite part, Fraction) pairs.  Returns (p, count, total): p = #{relabelings whose control sum >= observed} /
    total (a tie counts against the reading), as an exact Fraction.  None when an arm is empty."""
    if not control or not blind:
        return None
    values = list(control) + list(blind)
    k = len(control)
    obs = (sum(v[0] for v in control), sum((v[1] for v in control), Fraction(0)))
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


def check_markers(m, c, check):
    """markers.json against its class and check.json (raises InputError on a record that disagrees)."""
    name = c["name"]
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
            "n_falls": len(falls), "falls": falls[:20]}, None


def loop_rows(m):
    out = []
    for row in m.get("loop") or []:
        def g(*ks, row=row):
            d = row
            for k in ks:
                if not isinstance(d, dict) or k not in d:
                    return None
                d = d[k]
            return d
        out.append({"steps": row.get("steps"), "OWN_G": g("OWN", "P", "G"), "OWN_c0": g("OWN", "P", "c0"),
                    "OWN_b_star": g("OWN", "P", "b_star"), "OWN_G_L": g("OWN", "L", "G"),
                    "OWN_b_star_L": g("OWN", "b_star_L"), "own14": g("OWN", "own14_median"),
                    "OWN_falls": g("OWN", "n_terminated"), "REF_G": g("REF", "P", "G"), "REF_c0": g("REF", "P", "c0"),
                    "slope_onpolicy_eff": g("critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg"),
                    "slope_onpolicy_raw": g("critic_slope_obs6", "ONPOLICY", "raw", "dV_dq3_per_deg"),
                    "slope_own_eff": g("critic_slope_obs6", "OWN", "effective", "dV_dq3_per_deg")})
    return out


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
    return {"early8_rollout_mu14": early, "last8_rollout_mu14_before_T": last_mu,
            "statue_settled_last8_mean": ss_last, "statue_settled_trailing8_at_last_row": tr[-1] if tr else None,
            "statue_settled_trailing8_max": probe.get("trailing8_max"),
            "statue_settled_first_above": probe.get("trailing8_first_above"),
            "std14_last": (probe.get("std14") or [None])[-1], "obs6_mean_deg_last": (probe.get("obs6_mean_deg")
                                                                                     or [None])[-1],
            "cmd_l_last": d.get("cmd_l_last"), "cmd_l_max": d.get("cmd_l_max"), "ledger_firsts": d.get("firsts"),
            "m5_any_flagged": m5.get("any_flagged"), "m5_tail_flagged": m5.get("tail_flagged"),
            "m5_any_alarms": m5.get("any_alarms"), "m5_tail_alarms": m5.get("tail_alarms"),
            "m5_c6_counting_failures": m5.get("n_c6_counting_failures"), "loop": loop_rows(m), "ledger": led,
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
    items.extend(code_seal_items(root / "code"))
    return items


# ------------------------------------------------------------------------------------------------ the reading
def decide(classes, primaries):
    """The reading from the classes and the primaries ({name: Fraction or None})."""
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
    med = {arm: (statistics.median(counted[arm]) if counted[arm] else None) for arm in ARM_ORDER}
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
        additions.append("(the filled p is <= 0.05, but the control arm did not rise: no reading about the critic)")
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
            m, msha = _read_json(markers_path(root, name))
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
    status = "PROVISIONAL" if prov else "final"
    L += ["", f"STATUS: {status}" + (" (printed, not acted on)" if prov else "")]
    L.append(f"THE READING (line {dec['line']}, {dec['key']}):")
    L.append("  " + dec["text"])
    L.extend("  " + a for a in dec["additions"])
    L.append("")
    L.append("THE PRIMARY (on-policy tail_3 command: mean of rollout_mu14_mean over the probe rows "
             f"{PRIMARY_STEPS[0]:,} ... {PRIMARY_STEPS[-1]:,}); B_S = {BASELINE_S:+.5f} (the parent's 8 rows ending "
             f"at S)")
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
    L.append("INFORMATIVENESS (line 1): the control arm's median primary must rise by >= 0.10 from B_S")
    L.append(f"  control median {fnum(dec['medians'][CONTROL], '+.5f')}, rise {fnum(dec['control_rise'], '+.5f')} -> "
             f"{'informative' if dec['informative'] else 'NOT INFORMATIVE'}")
    L.append("THE TEST: exact one-sided permutation test (control higher), difference of arm means, alpha 0.05; "
             "missing values filled against REDUCED (control -inf, critic_blind_tail3 +inf)")
    L.append(f"  p (filled, decides) = {fmt_p(dec['filled'])}  {'<=' if dec['filled'][0] <= ALPHA else '>'} 0.05; "
             f"minimum attainable 1/70 (0.0143)")
    L.append(f"  p as counted (report only) = {fmt_p(dec['as_counted'])}")
    L.append("PAIRED DIFFERENCES (control - critic_blind_tail3, same seeds; report only, not a test):")
    for p in dec["pairs"]:
        L.append(f"  seed {p['seed']}: control {fnum(p['control'], '+.5f')}  blind {fnum(p['blind'], '+.5f')}  "
                 f"diff {fnum(p['diff'], '+.5f')}")
    L.append(f"  exact one-sided sign-flip p (report only; 4 pairs cannot reach 0.05: minimum 1/16) = "
             f"{fmt_p(dec['paired_test'])}")
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
        if led.get("status") == "read":
            L.append(f"      returns: mean at the first / last evaluation {fnum(led['mean_return_first'], '.1f')} / "
                     f"{fnum(led['mean_return_at_T'], '.1f')}, window {fnum(led['mean_return_window'], '.1f')} over "
                     f"{led['n_episodes']} episodes in {led['n_evaluations']} evaluations; falls {led['n_falls']} "
                     + (str([(f['timesteps'], f['length'], f['cause']) for f in led['falls']]) if led["falls"] else ""))
        else:
            L.append(f"      returns and falls: {led.get('status')}")
        for lr in rp["loop"]:
            L.append(f"      pair {lr['steps']:>9,}: OWN G {fnum(lr['OWN_G'], '+.3f')} c0 {fnum(lr['OWN_c0'], '+.3f')} "
                     f"b* {fnum(lr['OWN_b_star'], '+.3f')} G_L {fnum(lr['OWN_G_L'], '+.3f')} b*_L "
                     f"{fnum(lr['OWN_b_star_L'], '+.3f')} own14 {fnum(lr['own14'], '+.3f')} falls {lr['OWN_falls']} | "
                     f"REF G {fnum(lr['REF_G'], '+.3f')} | critic dV/dq3 on-policy eff "
                     f"{fnum(lr['slope_onpolicy_eff'], '+.5f')} raw {fnum(lr['slope_onpolicy_raw'], '+.5f')}")
    L.append("REPORT ONLY, ARM MEDIANS OVER THE READ CONTINUATIONS:")
    for arm in ARM_ORDER:
        rps = [reports[c["name"]] for c in classes if c["arm"] == arm and c["name"] in reports]

        def med(key, rps=rps):
            v = [r[key] for r in rps if _finite(r.get(key)) or isinstance(r.get(key), Fraction)]
            return statistics.median([float(x) for x in v]) if v else None

        def med_loop(key, rps=rps):
            v = [r["loop"][-1][key] for r in rps if r["loop"] and _finite(r["loop"][-1].get(key))]
            return statistics.median(v) if v else None
        L.append(f"  {arm:<24} statue_settled trailing at last row {fnum(med('statue_settled_trailing8_at_last_row'))}"
                 f"; CMD_L at T {fnum(med('cmd_l_last'), '+.3f')}; at the end pair OWN G "
                 f"{fnum(med_loop('OWN_G'), '+.3f')} c0 {fnum(med_loop('OWN_c0'), '+.3f')} b* "
                 f"{fnum(med_loop('OWN_b_star'), '+.3f')}; critic slope on-policy eff "
                 f"{fnum(med_loop('slope_onpolicy_eff'), '+.5f')} raw {fnum(med_loop('slope_onpolicy_raw'), '+.5f')}")
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
        report_only={n: rp for n, rp in reports.items()})
    return result, L, (EXIT_NOT_FINAL if prov else EXIT_FINAL)


def _jsonable(o):
    if isinstance(o, Fraction):
        return float(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=C.BS, help="the screen's folder (default <BS>; another root for tests and the "
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
