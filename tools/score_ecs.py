"""Entropy continuation screen (ECS): the scorer (build list v2 item 7).  Report only: reads files, never trains.

    PY -I -B score_ecs.py [--out <ECS>/score] [--root <ECS>] [--no-power]

What it reads (criteria_ecs_v3.txt THE ONE PRE-REGISTERED STATISTIC, RUN HANDLING, WHEN THE READING IS MADE):
  for each of the 24 continuations of ecs_common.QUEUE: <root>/runs/<name>/check.json (read_ecs.py --check),
  <root>/score/read/<name>/read.json (read_ecs.py's full read: R1, R2, R3) and <root>/lost.json; for PROVISIONAL,
  <root>/APPROVED, <root>/criteria.txt, <root>/SHA256SUMS.txt, the tool files and M1/M5 against their pins, and the
  code copy <root>/code, measured by M1's own r10m_common.check_copy (loaded only at its pin).
What it writes: <out>/score.json and <out>/score.txt (default <root>/score).

Classes (RUN HANDLING): LOST (in lost.json: three crashes, C6 or R1 unable to run on the ledger; or R1's C6 failing on
a grid, row, episode or ANY-input check up to T);
COUNTED (check.json passed, stop reason "end" at E); STOPPED (check.json passed, stop reason "collapse" at T < E):
POSITIVE when an ANY-flagged evaluation came in (S, T], otherwise CENSORED.  A continuation without a passing
check.json and not LOST is undecided; while any is undecided (or a decided one has no R1 read yet) the screen is
"incomplete" and no line is read, and nothing about any outcome is printed (outcome-blind).

The statistic: per start a arm-A and b arm-B continuations POSITIVE.  The exact stratified test (exact conditional
test of a common odds ratio, "exact Cochran-Mantel-Haenszel"; with equal arms a within-start permutation test):
X = a(s42) + a(s80), conditional on each start's total; P_UP = P(X >= x), P_DOWN = P(X <= x), exact rationals
(fractions), so a value on the 0.025 line is decided exactly (no p on a filled, 6-per-arm table equals 1/40, so this
matters only for the as-counted tables the additions read; review RR-8).  The code of <ECS>/v1/table_e_ecs.py
(stratum_null, null_T, pvals, binom, power) moved here unchanged in its arithmetic.
THE FILL: P_UP on the table with each LOST or censored continuation counted against B LOWER (arm A NEGATIVE, arm B
POSITIVE); P_DOWN on the table with each counted against B HIGHER (arm A POSITIVE, arm B NEGATIVE).
The reading, the first that applies: 1 B LOWER (P_UP <= 0.025); 2 B HIGHER (P_DOWN <= 0.025); 3 NO DIFFERENCE
DETECTED, or its NOT INFORMATIVE variant when both starts carry a LOW ARM-A RATE or NO ARM-A note, with the addition
"(B LOWER [or B HIGHER] holds only without the n LOST or censored continuations)" when that line holds on the table
as counted but not on the filled one.

Exit codes: 0 final; 3 provisional or incomplete; 2 input error.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402
from functools import lru_cache  # noqa: E402
from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
VERSION = "score_ecs 1.0 (ECS build list v2 item 7; lane R, 2026-10-28)"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = sys.modules.get("ecs_common") or _load("ecs_common")
R = sys.modules.get("read_ecs") or _load("read_ecs")

EXIT_FINAL, EXIT_NOT_FINAL, EXIT_INPUT = E.EXIT_OK, E.EXIT_PROVISIONAL, E.EXIT_REFUSED      # 0, 3, 2 (BL 7)
SIDE = Fraction(str(E.ALPHA_PER_SIDE))          # 0.025 per side (D6 (a)), as the exact rational 1/40
START_ORDER = tuple(E.STARTS)                   # ("s42", "s80")
K = E.SEEDS_PER_ARM                             # 6 per arm per start (D4 (a))
COUNTED, STOPPED, LOST = "COUNTED", "STOPPED", "LOST"
UNDECIDED = "UNDECIDED"                         # not a class of the text: no passing check.json and not LOST
POSITIVE, NEGATIVE, CENSORED = "POSITIVE", "NEGATIVE", "CENSORED"

LINE_LOWER = ("REMOVING THE REMAINING ENTROPY BONUS LOWERED THE SHARE OF CONTINUATIONS WITH AN ANY-FLAGGED EVALUATION "
              "BY E (pooled over both states; per-start tables beside)")
LINE_HIGHER = ("REMOVING THE REMAINING ENTROPY BONUS RAISED THE SHARE OF CONTINUATIONS WITH AN ANY-FLAGGED EVALUATION "
               "BY E (pooled over both states; per-start tables beside)")
LINE_NONE = ("NO DIFFERENCE DETECTED IN WHETHER AN ANY-FLAGGED EVALUATION CAME BY E FROM THESE STATES (not tested: a "
             "shift of the first flag within the window, a modest effect, the early phase)")
LINE_NOT_INFORMATIVE = ("NO DIFFERENCE DETECTED; NOT INFORMATIVE: arm A showed the outcome in fewer than half its "
                        "counted continuations, or none was counted, at both starts")       # criteria v3 (RR-4)
LINE_KEYS = {1: "B LOWER", 2: "B HIGHER", 3: "NO DIFFERENCE DETECTED"}


class InputError(Exception):
    """A record the scorer cannot read as the text requires (exit 2)."""


# ================================================================================================ the exact test
# Moved from <ECS>/v1/table_e_ecs.py (the arithmetic is unchanged; TE and CHK 8 are reproduced by the tests).
@lru_cache(maxsize=None)
def stratum_null(nA, nB, m):
    """Under the null, a of nA is hypergeometric (nA + nB, m, nA): ((a, P(a)), ...) as Fractions."""
    tot = comb(nA + nB, m)
    return tuple((a, Fraction(comb(nA, a) * comb(nB, m - a), tot))
                 for a in range(max(0, m - nB), min(nA, m) + 1))


def null_T(strata):
    """strata: tuple of (nA, nB, m).  Exact null distribution of X = sum of a as {x: Fraction}."""
    d = {0: Fraction(1)}
    for nA, nB, m in strata:
        nd = {}
        for t, p in d.items():
            for a, q in stratum_null(nA, nB, m):
                nd[t + a] = nd.get(t + a, Fraction(0)) + p * q
        d = nd
    return d


@lru_cache(maxsize=None)
def pvals(cells):
    """cells: tuple of (a, nA, b, nB) per start.  Returns (P_UP, P_DOWN) as Fractions."""
    strata = tuple((nA, nB, a + b) for a, nA, b, nB in cells)
    t = sum(a for a, _nA, _b, _nB in cells)
    d = null_T(strata)
    up = sum((p for v, p in d.items() if v >= t), Fraction(0))
    down = sum((p for v, p in d.items() if v <= t), Fraction(0))
    return up, down


def reading_of(cells):
    """("B LOWER" | "B HIGHER" | "NO DIFFERENCE DETECTED", P_UP, P_DOWN) on ONE table (table_e_ecs.reading)."""
    up, down = pvals(cells)
    if up <= SIDE:
        return "B LOWER", up, down
    if down <= SIDE:
        return "B HIGHER", up, down
    return "NO DIFFERENCE DETECTED", up, down


def fisher_one_sided(a, nA, b, nB):
    """One start's one-sided Fisher p both ways (report only, not a test): (A greater P(a' >= a), B greater
    P(a' <= a)) under the hypergeometric null given m = a + b."""
    dist = stratum_null(nA, nB, a + b)
    return (sum((p for v, p in dist if v >= a), Fraction(0)), sum((p for v, p in dist if v <= a), Fraction(0)))


def binom(n, k, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def power(k, rates):
    """P(B LOWER) and P(B HIGHER) with k per arm per start; rates = [(pA_s, pB_s)] per start (floats)."""
    lower = higher = 0.0

    def rec(i, cells, prob):
        nonlocal lower, higher
        if i == len(rates):
            r = reading_of(tuple(cells))[0]
            if r == "B LOWER":
                lower += prob
            elif r == "B HIGHER":
                higher += prob
            return
        pA, pB = rates[i]
        for a in range(k + 1):
            pa = binom(k, a, pA)
            if pa == 0:
                continue
            for b in range(k + 1):
                pb = binom(k, b, pB)
                if pb == 0:
                    continue
                rec(i + 1, cells + [(a, k, b, k)], prob * pa * pb)

    rec(0, [], 1.0)
    return lower, higher


@lru_cache(maxsize=1)
def power_lines():
    """TABLE E's power figures, recomputed (report only; criteria REFERENCE TABLE E, TE 4, CHK 2)."""
    out = ["P(B LOWER), k = 6, the same rates at both starts (exact enumeration; TE 4):",
           "      pA \\ pB    0.00   0.10   0.25   0.40   0.50"]
    for pA in (0.6, 0.75, 0.9):
        cells = ["%.3f" % power(K, [(pA, pB)] * 2)[0] for pB in (0.0, 0.1, 0.25, 0.4, 0.5)]
        out.append("      %.2f     %s" % (pA, "  ".join(cells)))
    hi = ["A %.2f %.3f" % (pA, power(K, [(pA, 1.0)] * 2)[1]) for pA in (0.9, 0.75, 0.6)]
    out.append("P(B HIGHER) against B 1.0 (CHK 2): " + "; ".join(hi))
    out.append("Size under the null (TE 5; RS 0.01 grid): at most 0.0116 per side, 0.0231 in all.")
    return tuple(out)


def fmt(p):
    """An exact p as 'num/den (0.xxxx)'."""
    p = Fraction(p)
    return f"{p.numerator}/{p.denominator} ({float(p):.4f})"


# ================================================================================================ classes
def classify(name, check, read, lost_entry):
    """One continuation's class and outcome from check.json, read.json (R1) and its lost.json entry.

    Returns a dict: name, start, arm, seed, cls (COUNTED / STOPPED / LOST / UNDECIDED), outcome (POSITIVE /
    NEGATIVE / CENSORED / None), T, cause, read_ok (R1 present for a COUNTED or STOPPED one), report (R1-R3's
    report-only values).  Raises InputError for records that disagree with each other or with ecs_common."""
    p = E.parse_run_name(name)
    if p is None or p["suffix"]:
        raise InputError(f"{name}: not a continuation name")
    out = dict(name=name, start=p["start"], arm=p["arm"], seed=p["seed"], cls=UNDECIDED, outcome=None, T=None,
               cause=None, read_ok=False, report=None, check_failures=None)
    S, Eend = E.window(p["start"])
    if lost_entry is not None:
        out.update(cls=LOST, cause=str(lost_entry.get("cause") or "listed in lost.json"))
        return out
    if check is None:
        out["cause"] = "no check.json (running, queued, crashed or VOID)"
        return out
    for key, want in (("name", name), ("start", p["start"]), ("arm", p["arm"]), ("seed", p["seed"]), ("S", S),
                      ("E", Eend)):
        if check.get(key) != want:
            raise InputError(f"{name}: check.json {key} = {check.get(key)!r}, expected {want!r}")
    if check.get("mode") != "continuation":
        raise InputError(f"{name}: check.json mode {check.get('mode')!r} is not a continuation's")
    if check.get("passed") is not True:
        out["cause"] = "CHECK failed (VOID until the cause is fixed and it reruns)"
        out["check_failures"] = list(check.get("failures") or [])
        return out
    T, stop = check.get("T"), check.get("stop_reason")
    if stop == "end" and T == Eend:
        cls = COUNTED
    elif stop == "collapse" and isinstance(T, int) and S < T < Eend:
        cls = STOPPED
    else:
        raise InputError(f"{name}: a passing check.json with stop reason {stop!r} at T = {T!r}")
    out.update(cls=cls, T=T)
    if read is None:
        out["cause"] = "R1 not read yet"
        return out
    r1 = read.get("r1") or {}
    for key, want in (("name", name), ("start", p["start"]), ("arm", p["arm"]), ("seed", p["seed"]), ("S", S),
                      ("T", T)):
        if read.get(key) != want:
            raise InputError(f"{name}: read.json {key} = {read.get(key)!r}, expected {want!r}")
    if read.get("check_sha256") != check.get("_sha256"):
        raise InputError(f"{name}: read.json was made from another check.json ({read.get('check_sha256')!r})")
    if not isinstance(r1.get("positive"), bool) or not isinstance(r1.get("c6_ok"), bool):
        raise InputError(f"{name}: read.json has no R1 result")
    out["read_ok"] = True
    out["report"] = read
    if not r1["c6_ok"]:
        fails = r1.get("c6_counting_failures") or []
        first = fails[0] if fails else {}
        first_s = (f"{first.get('failure')} [{first.get('category')}, {first.get('where')}]" if isinstance(first, dict)
                   else str(first or "?"))
        out.update(cls=LOST, outcome=None, cause=f"C6 failing up to T = {T} ({len(fails)} failures; first: {first_s})")
        return out
    if r1["positive"]:
        out["outcome"] = POSITIVE
    else:
        out["outcome"] = NEGATIVE if cls == COUNTED else CENSORED
    if cls == STOPPED:
        out["cause"] = f"the collapse stop at T = {T}"
    return out


def tables(classes):
    """Per start: the table as counted and the two filled tables (THE FILL).

    {start: {"counted": (a, nA, b, nB), "missing": {"A": n, "B": n}, "up": (a, 6, b, 6), "down": (a, 6, b, 6)}}."""
    t = {}
    for st in START_ORDER:
        a = nA = b = nB = 0
        miss = {"A": 0, "B": 0}
        for c in classes:
            if c["start"] != st:
                continue
            missing = c["cls"] == LOST or c["outcome"] == CENSORED
            if missing:
                miss[c["arm"]] += 1
            elif c["cls"] in (COUNTED, STOPPED) and c["outcome"] in (POSITIVE, NEGATIVE):
                pos = c["outcome"] == POSITIVE
                if c["arm"] == "A":
                    nA += 1
                    a += pos
                else:
                    nB += 1
                    b += pos
            else:
                raise InputError(f"{c['name']}: not decided ({c['cls']}, {c['outcome']})")
        up = (a, nA + miss["A"], b + miss["B"], nB + miss["B"])         # against B LOWER: A -> NEGATIVE, B -> POSITIVE
        down = (a + miss["A"], nA + miss["A"], b, nB + miss["B"])       # against B HIGHER: A -> POSITIVE, B -> NEGATIVE
        t[st] = {"counted": (a, nA, b, nB), "missing": miss, "up": up, "down": down}
    return t


def start_notes(counted):
    """The per-start notes on the table as counted (they never change a p or a line)."""
    a, nA, _b, _nB = counted
    notes = []
    if nA == 0:
        notes.append("NO ARM-A CONTINUATION COUNTED")
    elif 2 * a < nA:
        notes.append(f"LOW ARM-A RATE ({a} of {nA}): little power for B LOWER at this start")
    if nA > 0 and a == nA:
        notes.append(f"ARM A AT {a} OF {a}: no room for B HIGHER at this start")
    return notes


def low_note(notes):
    return any(n.startswith("LOW ARM-A RATE") or n == "NO ARM-A CONTINUATION COUNTED" for n in notes)


def decide(tabs):
    """The reading from the per-start tables: P_UP on the filled-against-B-LOWER table, P_DOWN on the
    filled-against-B-HIGHER one; the line, its variants and the as-counted report-only values."""
    up_cells = tuple(tabs[s]["up"] for s in START_ORDER)
    down_cells = tuple(tabs[s]["down"] for s in START_ORDER)
    counted_cells = tuple(tabs[s]["counted"] for s in START_ORDER)
    p_up = pvals(up_cells)[0]
    p_down = pvals(down_cells)[1]
    c_up, c_down = pvals(counted_cells)
    n_missing = sum(tabs[s]["missing"]["A"] + tabs[s]["missing"]["B"] for s in START_ORDER)
    notes = {s: start_notes(tabs[s]["counted"]) for s in START_ORDER}
    if p_up <= SIDE:
        line, text = 1, LINE_LOWER
    elif p_down <= SIDE:
        line, text = 2, LINE_HIGHER
    else:
        line = 3
        text = LINE_NOT_INFORMATIVE if all(low_note(notes[s]) for s in START_ORDER) else LINE_NONE
    additions = []
    if line == 3:
        if c_up <= SIDE:
            additions.append(f"(B LOWER holds only without the {n_missing} LOST or censored continuations)")
        if c_down <= SIDE:
            additions.append(f"(B HIGHER holds only without the {n_missing} LOST or censored continuations)")
    fisher = {s: {"counted": fisher_one_sided(*tabs[s]["counted"]), "up": fisher_one_sided(*tabs[s]["up"]),
                  "down": fisher_one_sided(*tabs[s]["down"])} for s in START_ORDER}
    return dict(line=line, key=LINE_KEYS[line], text=text, additions=additions, p_up=p_up, p_down=p_down,
                counted_p_up=c_up, counted_p_down=c_down, n_missing=n_missing, notes=notes, fisher=fisher,
                up_cells=up_cells, down_cells=down_cells, counted_cells=counted_cells)


def per_start_counts(tabs, st):
    """'arm A a of 6, arm B b of 6', with the missing named when there are any."""
    a, nA, b, nB = tabs[st]["counted"]
    m = tabs[st]["missing"]

    def one(arm, x, n, k):
        s = f"arm {arm} {x} of {n + k}" if k == 0 else f"arm {arm} {x} of {n} counted ({k} LOST or censored)"
        return s
    return f"{st}: {one('A', a, nA, m['A'])}, {one('B', b, nB, m['B'])}"


# ================================================================================================ records
def _read_json(path):
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


def gather(root):
    """[(name, check (with _sha256), read, lost entry, sources)] for the 24 continuations, in QUEUE order."""
    root = Path(root)
    try:
        lost = R.load_lost(root / "lost.json")
    except R.InputError as exc:
        raise InputError(str(exc)) from exc
    unknown = sorted(set(lost) - {E.run_name(*q) for q in E.QUEUE})
    if unknown:
        raise InputError(f"lost.json names continuations outside the queue: {unknown}")
    out = []
    for start, arm, seed in E.QUEUE:
        name = E.run_name(start, arm, seed)
        src = {}
        cpath = root / "runs" / name / R.CHECK_NAME
        check, csha = _read_json(cpath)
        if check is not None:
            check["_sha256"] = csha
            src["check.json"] = (str(cpath), csha)
        rpath = R.read_dir(root, name) / R.READ_NAME
        read, rsha = _read_json(rpath)
        if read is not None:
            src["read.json"] = (str(rpath), rsha)
        out.append((name, check, read, lost.get(name), src))
    return out


# ================================================================================================ PROVISIONAL
HEX64_FULL = re.compile(r"[0-9a-f]{64}")


def _hex(v):
    """A bare 64-hex sha256 (the only form APPROVED carries, as run_ecs.py compares it; ecs_common APPROVED_KEYS), or
    None for any other value (a "sha256:" prefix included; review RR-10)."""
    return v if isinstance(v, str) and HEX64_FULL.fullmatch(v) else None


def approved_pins(root, appr_text):
    """(criteria sha256, code tree digest, {tool path: sha256}, problems) from <root>/APPROVED: JSON with ecs_common
    APPROVED_KEYS, tool_sha256 a {file name: bare 64-hex sha256} object.  Anything else is a problem (PROVISIONAL):
    run_ecs.py refuses every continuation under such an APPROVED."""
    root = Path(root)
    probs, pins = [], {}
    try:
        d = json.loads(appr_text)
    except ValueError:
        d = None
    if not isinstance(d, dict):
        return None, None, pins, ["APPROVED is not a JSON object (run_ecs.py refuses it)"]
    crit, tree = _hex(d.get("criteria_sha256")), _hex(d.get("code_tree_digest"))
    if crit is None:
        probs.append(f"APPROVED criteria_sha256 {d.get('criteria_sha256')!r} is not a bare 64-hex sha256")
    if tree is None:
        probs.append(f"APPROVED code_tree_digest {d.get('code_tree_digest')!r} is not a bare 64-hex sha256")
    tools = d.get("tool_sha256")
    if not isinstance(tools, dict):
        probs.append("APPROVED has no tool_sha256 object")
        tools = {}
    for k, v in tools.items():
        path = str(root / "tools" / Path(str(k)).name)
        if _hex(v) is None:
            probs.append(f"APPROVED tool_sha256[{k!r}] {v!r} is not a bare 64-hex sha256")
        else:
            pins[path] = v
    return crit, tree, pins, probs


def _load_pinned_m1_common():
    """M1's r10m_common.py, loaded only at its pin (as run_ecs.py's preflight loads it).  (module, problem)."""
    common = E.M1_COMMON
    try:
        if E.sha256_file(common) != E.PIN_SHA256[common]:
            return None, f"{common} is not at its pin"
    except OSError as exc:
        return None, f"{common}: {exc}"
    spec = importlib.util.spec_from_file_location("score_ecs_r10m_common", common)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, None


def code_seal_items(code_dir):
    """The code copy measured again (review RR-2): its tree digest and file count by M1's check_copy against the pins;
    the seal file alone is never taken as the measurement."""
    rc, problem = _load_pinned_m1_common()
    if problem:
        return [f"code copy not measured: {problem}"]
    try:
        seal = rc.check_copy(str(code_dir), expect_commit=E.COMMIT)
    except rc.Refusal as exc:
        return [f"code copy: {getattr(exc, 'message', exc)}"]
    except (OSError, ValueError) as exc:
        return [f"code copy unreadable ({exc})"]
    if seal.get("tree_digest") != E.TREE_DIGEST or seal.get("files") != E.TREE_FILES:
        return [f"code copy tree digest {seal.get('tree_digest')} ({seal.get('files')} files) != pinned "
                f"{E.TREE_DIGEST} ({E.TREE_FILES} files)"]
    return []


def provisional_items(root, records):
    """Every reason the reading is PROVISIONAL (criteria WHEN THE READING IS MADE): a pinned file's sha256 differing
    from its pin, this text's sha256 differing from <ECS>/APPROVED's, or a record made with another tool or code.
    Pins: <ECS>/APPROVED's tool_sha256 (every ecs_common TOOL_FILES entry must have one), <ECS>/SHA256SUMS.txt
    (sha256sum lines, paths absolute or relative to <ECS>; a path both pin must agree), M1 / M5 from
    ecs_common.PIN_SHA256, and the code copy measured by M1's check_copy."""
    root = Path(root)
    items = []
    appr = root / "APPROVED"
    try:
        appr_text = appr.read_text(encoding="utf-8", errors="replace")
    except OSError:
        appr_text = None
        items.append(f"no {appr} (the screen is not approved)")
    crit = root / "criteria.txt"
    try:
        crit_sha = E.sha256_file(crit)
    except OSError:
        crit_sha = None
        items.append(f"no {crit}")
    pins = {}
    if appr_text is not None:
        a_crit, a_tree, pins, probs = approved_pins(root, appr_text)
        items.extend(probs)
        if crit_sha is not None and a_crit is not None and a_crit != crit_sha:
            items.append(f"criteria.txt sha256 {crit_sha} != APPROVED's {a_crit}")
        if a_tree is not None and a_tree != E.TREE_DIGEST:
            items.append(f"APPROVED code_tree_digest {a_tree} != {E.TREE_DIGEST}")
        for tool in E.TOOL_FILES:
            if str(root / "tools" / tool) not in pins:
                items.append(f"no pin for {tool} in APPROVED")
    sums = root / "SHA256SUMS.txt"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^([0-9a-f]{64})\s+\*?(\S.*)$", line.strip())
            if m:
                path = m.group(2)
                path = path if path.startswith("/") else str(root / path)
                if path in pins and pins[path] != m.group(1):
                    items.append(f"{path}: APPROVED pins {pins[path]}, SHA256SUMS.txt pins {m.group(1)}")
                pins.setdefault(path, m.group(1))
    else:
        items.append(f"no {sums}")
    for path, want in sorted(pins.items()):
        try:
            got = E.sha256_file(path)
        except OSError as exc:
            items.append(f"pinned file {path} unreadable ({exc})")
            continue
        if got != want:
            items.append(f"pinned file {path}: sha256 {got} != pin {want}")
    items.extend(f"M1/M5 pin: {x}" for x in E.check_pins())
    read_sha = E.sha256_file(TOOLS / "read_ecs.py")
    for name, check, read, _lost, _src in records:
        if check is not None and check.get("tool_sha256") != read_sha:
            items.append(f"{name}: check.json made by read_ecs.py {check.get('tool_sha256')}, not the current "
                         f"{read_sha}")
        if check is not None:
            run_tools = (check.get("meta") or {}).get("run_tool_sha256")
            if isinstance(run_tools, dict):
                for tool, sha in sorted(run_tools.items()):
                    want = pins.get(str(root / "tools" / Path(str(tool)).name))
                    if want is not None and sha != want:
                        items.append(f"{name}: ran with {tool} {sha}, not the pinned {want}")
            if crit_sha is not None and check.get("criteria_sha256") != crit_sha:
                items.append(f"{name}: run_meta criteria sha256 {check.get('criteria_sha256')!r} is not criteria.txt's "
                             f"{crit_sha}")
        if read is None:
            continue
        if read.get("tool_sha256") != read_sha:
            items.append(f"{name}: read.json made by read_ecs.py {read.get('tool_sha256')}, not the current {read_sha}")
        r1 = read.get("r1") or {}
        if r1.get("m5_sha256") != E.PIN_SHA256[E.M5] or r1.get("m5_common_sha256") != E.PIN_SHA256[E.M5_COMMON]:
            items.append(f"{name}: R1 ran M5 {r1.get('m5_sha256')} / {r1.get('m5_common_sha256')}, not the pinned")
    items.extend(code_seal_items(root / "code"))
    return items


# ================================================================================================ the score
def report_line(c):
    """One continuation's report-only values (never tested)."""
    rd = c.get("report") or {}
    r1 = rd.get("r1") or {}
    r2 = rd.get("r2") or {}
    r3 = c.get("r3_now") or rd.get("r3") or {}
    first = r1.get("first_any_flag")
    first_s = f"{first:,}" if isinstance(first, int) else f"none (censored at T = {c['T']:,})"
    s = (f"{c['name']:<12} {c['cls']:<8} {str(c['outcome']):<9} T {c['T']:,}  first ANY flag {first_s}; "
         f"flagged {r1.get('n_any_flagged')}; ANY ALARM {r1.get('any_alarms')}; TAIL-flagged {r1.get('tail_flagged')};"
         f" TAIL ALARM {r1.get('tail_alarms')}; TAIL-input missing (report only) "
         f"{len(r1.get('c6_tail_missing') or [])}")
    std = r2.get("std_mean_first_last")
    if std:
        s += f"; mean std first/last row {std[0]:.4f}/{std[1]:.4f}"
    ret = (rd.get("returns") or {}).get("mean_return_last")
    if ret is not None:
        s += f"; mean return at T {ret:.1f}"
    if r3.get("status") == "read":
        s += (f"; R3 CMD tail_3 {r3.get('cmd_tail_3')}, tail flags {r3.get('n_tail_flags')}, falls "
              f"{r3.get('n_fall_flags')} ({r3.get('file')} sha256 {r3.get('sha256')})")
    else:
        s += f"; R3 {r3.get('status', 'missing')}"
    return s


def score(root, with_power=True):
    """(result dict, text lines, exit code)."""
    root = Path(root)
    records = gather(root)
    classes = [classify(name, check, read, lost) for name, check, read, lost, _src in records]
    prov = provisional_items(root, records)
    L = [VERSION, f"tool sha256 {E.sha256_file(Path(__file__))}  utc {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
         f"root {root}"]
    for *_rec, src in records:
        for k, (path, sha) in src.items():
            L.append(f"  read {k:<10} {path}  sha256 {sha}")
    undecided = [c for c in classes if c["cls"] == UNDECIDED]
    unread = [c for c in classes if c["cls"] in (COUNTED, STOPPED) and not c["read_ok"]]
    result = dict(version=VERSION, root=str(root), provisional=prov)
    if undecided or unread:
        result["status"] = "incomplete"
        L.append("")
        L.append(f"STATUS: incomplete ({len(classes) - len(undecided)} of {len(classes)} decided; "
                 f"{len(unread)} decided without an R1 read). No line is read; no outcome is printed.")
        for c in classes:
            if c["cls"] == UNDECIDED:
                kept = sorted(p.name for p in (root / "runs").glob(c["name"] + "_*") if p.is_dir())
                void = any(E.parse_run_name(k) and E.parse_run_name(k)["suffix"] == "void" for k in kept)
                state = (f"undecided: {c['cause']}" + (" -- VOID until fixed" if void else "")
                         + (f"; kept folders {kept}" if kept else ""))
            elif not c["read_ok"] and c["cls"] in (COUNTED, STOPPED):
                state = "decided, R1 not read yet"
            else:
                state = "decided"
            L.append(f"  {c['name']:<12} {state}")
        result["classes"] = [{k: c[k] for k in ("name", "start", "arm", "seed")} | {
            "decided": c["cls"] != UNDECIDED} for c in classes]
        if prov:
            L.append("PROVISIONAL items (printed; the reading is not acted on while any stands):")
            L.extend(f"  - {x}" for x in prov)
        return result, L, EXIT_NOT_FINAL
    tabs = tables(classes)
    dec = decide(tabs)
    for c in classes:      # R3 (report only) is read here: the runner makes the end traces after the R1/R2 reads
        if c["cls"] in (COUNTED, STOPPED) and c["read_ok"]:
            stop = "end" if c["cls"] == COUNTED else "collapse"
            c["r3_now"] = R.trace_reading(root / "traces" / c["name"] / R.TRACE_NAME,
                                          R.end_zip(root / "runs" / c["name"], stop, c["T"]), c["T"])
    status = "PROVISIONAL" if prov else "final"
    result.update(status=status, classes=[{k: v for k, v in c.items() if k != "report"} for c in classes],
                  tables={s: {k: (list(v) if isinstance(v, tuple) else v) for k, v in t.items()}
                          for s, t in tabs.items()},
                  line=dec["line"], line_key=dec["key"], line_text=dec["text"], additions=dec["additions"],
                  p_up=str(dec["p_up"]), p_down=str(dec["p_down"]), p_up_float=float(dec["p_up"]),
                  p_down_float=float(dec["p_down"]), counted_p_up=str(dec["counted_p_up"]),
                  counted_p_down=str(dec["counted_p_down"]), n_missing=dec["n_missing"], notes=dec["notes"],
                  fisher={s: {k: [str(x) for x in v] for k, v in f.items()} for s, f in dec["fisher"].items()})
    L.append("")
    L.append(f"STATUS: {status}" + (" (printed, not acted on)" if prov else ""))
    L.append("THE READING (line %d, %s):" % (dec["line"], dec["key"]))
    L.append("  " + dec["text"])
    for a in dec["additions"]:
        L.append("  " + a)
    if dec["line"] == 3:
        for s in START_ORDER:
            L.append("    " + per_start_counts(tabs, s))
    L.append("")
    L.append("TABLES (a = arm-A POSITIVE, b = arm-B POSITIVE; LOST and censored continuations are missing)")
    for s in START_ORDER:
        t = tabs[s]
        a, nA, b, nB = t["counted"]
        L.append(f"  {s} as counted: arm A {a}/{nA}, arm B {b}/{nB}; missing A {t['missing']['A']}, "
                 f"B {t['missing']['B']}")
        L.append(f"      filled against B LOWER (P_UP's table):  arm A {t['up'][0]}/{t['up'][1]}, arm B "
                 f"{t['up'][2]}/{t['up'][3]}")
        L.append(f"      filled against B HIGHER (P_DOWN's table): arm A {t['down'][0]}/{t['down'][1]}, arm B "
                 f"{t['down'][2]}/{t['down'][3]}")
    L.append("THE TEST: exact stratified (exact Cochran-Mantel-Haenszel), two-sided, 0.025 per side; X = a(s42) + "
             "a(s80)")
    L.append(f"  P_UP   (filled against B LOWER)  = {fmt(dec['p_up'])}  {'<=' if dec['p_up'] <= SIDE else '>'} 1/40")
    L.append(f"  P_DOWN (filled against B HIGHER) = {fmt(dec['p_down'])}  "
             f"{'<=' if dec['p_down'] <= SIDE else '>'} 1/40")
    L.append(f"  as counted (report only): P_UP {fmt(dec['counted_p_up'])}, P_DOWN {fmt(dec['counted_p_down'])}")
    L.append("PER-START NOTES (they never change a p or a line):")
    for s in START_ORDER:
        L.append(f"  {s}: " + ("; ".join(dec["notes"][s]) if dec["notes"][s] else "none"))
    L.append("ONE-SIDED FISHER p PER START (report only, not a test): A greater / B greater")
    for s in START_ORDER:
        f = dec["fisher"][s]
        L.append(f"  {s}: as counted {float(f['counted'][0]):.4f} / {float(f['counted'][1]):.4f}; A greater on the "
                 f"B-LOWER fill {float(f['up'][0]):.4f}; B greater on the B-HIGHER fill {float(f['down'][1]):.4f}")
    if with_power:
        L.append("TABLE E POWER FIGURES (report only):")
        L.extend("  " + x for x in power_lines())
    L.append("STOPPED, CENSORED AND LOST CONTINUATIONS:")
    odd = [c for c in classes if c["cls"] in (STOPPED, LOST) or c["outcome"] == CENSORED]
    for c in odd:
        L.append(f"  {c['name']:<12} {c['cls']:<8} {str(c['outcome']):<9} {c['cause']}")
    if not odd:
        L.append("  none")
    L.append("REPORT ONLY, PER CONTINUATION (never tested; no line distinguishes prevention from delay):")
    for c in classes:
        if c["cls"] == LOST:
            L.append(f"  {c['name']:<12} LOST     {c['cause']}")
        else:
            L.append("  " + report_line(c))
    if prov:
        L.append("PROVISIONAL items (the reading is printed and not acted on while any stands):")
        L.extend(f"  - {x}" for x in prov)
    return result, L, (EXIT_NOT_FINAL if prov else EXIT_FINAL)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=E.ECS, help="the screen's folder (tests only; default <ECS>)")
    ap.add_argument("--out", default=None, help="default <root>/score")
    ap.add_argument("--no-power", action="store_true", help="leave out TABLE E's power figures (tests)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.out).resolve() if a.out else root / "score"
    try:
        R.require_under(out, root)
        result, lines, code = score(root, with_power=not a.no_power)
    except (InputError, R.InputError) as exc:
        print(f"{VERSION}\nINPUT ERROR: {exc}")
        return EXIT_INPUT
    text = "\n".join(lines)
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    R.atomic_write_text(out / "score.json", json.dumps(result, indent=1, default=str) + "\n")
    R.atomic_write_text(out / "score.txt", text + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
