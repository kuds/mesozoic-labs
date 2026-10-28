"""T. rex r10 round: the scorer (criteria lean v6: RUN HANDLING's LEAN-R, CLASSES, STATISTICAL ANALYSIS, ROUND
READING, WHEN THE READING IS MADE).  Report only; never trains; changes nothing anywhere.

    PY -I -B score_r10round.py [--round ROUND] [--runlog P] [--pins P] [--approved P] [--json OUT] [--panels-format F]

Reads RUNLOG's note lines only ("<date -u> STANCE6 <KIND> <RUN_ID> <fields>"; every other line naming a RUN_ID is
printed and read by no rule), the fetch stores <ROUND>/fetch/<F0|TRIGGER|F2|REFETCH>/<RUN_ID>/ with
<ROUND>/fetch/manifest.csv, the measurements <ROUND>/panels/<RUN_ID>/<label>/<m1|m3>/ (label = checkpoint step or
"final"), the pins file and <ROUND>/APPROVED (JSON).  Per seed: its run of the round by LEAN-R's relaunch rule; for
every launched RUN_ID C1-C7 (complete_r10round.py), PRESENCE p1-p3 on each fetch, D-16 when LOST; classes, kinds,
gate and outcome of COMPLETE runs; statistic A and "A WITH LOST RUNS"; the reading line; PROVISIONAL; pins.
Exit codes: 0 final; 10 incomplete (no line read); 11 provisional (read, printed, not acted on); 2 input error.

Fixer, 2026-10-26 (review findings):
  R-m6  relaunch (i) reads the TRIGGER LISTING only when RUNLOG's LISTING line equals the fetch store's complete
        listing of 01_stance/models/ (<ROUND>/fetch/TRIGGER/<RUN_ID>/listings/01_stance/models.json) and its fetch-
        manifest LISTING row (names and sizes); otherwise (i) is not decided and the reading is incomplete.
  R-m2  line 1's basis: "(end-only, drive-backed)" when any END-ONLY rail is DRIVE-BACKED (the text's else-chain).
  R-m4  a LOST run whose PRESENCE no fetch makes a rail and some fetch leaves undecided (a p-fact not established)
        keeps the reading incomplete.
  RA-8  lane A's own M1 reduction (m1_summary.json) is compared with this reader's and any difference printed
        (report only).
v5 (criteria lean v5, RUNLOG line 657; tool updater 2026-10-26):
  R-B2  a LISTING line folder=absent:<key> (the run folder, 01_stance/ or 01_stance/models/ absent from its parent's
        complete listing) is a TRIGGER LISTING that holds no file, read when the fetch store shows it: the parent's
        complete listing without that folder, made at or after the trigger note and not after the LISTING line (for
        the run folder one of the ppo folder's listings folders.json keeps, each with its fetch-manifest LISTING row),
        and the fetch manifest's LISTING row folder=absent:<key> (fetch_r10round.py runlog-listing writes it).
  O-6   CMD_a over the episodes that last beyond step 200 (the others printed); not formed: below 0.60 in either sign.
        (v5 held MEAN_a open with one CMD_a not formed and the other at or over 0.60, the reviser's V5-1; v6 below.)
  O-7   an exact sign tie counts a TAIL FLAG once in each sign (TRACE RAIL, RESOLVED BY TRACE, DRIVE-BACKED).
  O-8   a TRACE RAIL is DRIVE-BACKED when, at any checkpoint holding one of its flags, CMD_a >= 0.60 in its sign.
  F-R1  a refused M3 at R2 is GATE FAIL, its message the failing criterion (complete_r10round.py).
  O-13  a _b launched when the rule does not hold is no run of the round: no class, no PRESENCE, D-16 only.
  O-14  F2's delay after RUNLOG's last ending note is printed (at least 6 hours); report only.
  O-16  the re-fetch: a seed's run that passes C1 and fails C2 or C3 at F2, unless it fails C4 or C6 on files matching
        the manifest.
  O-17  the re-fetch's time: F2's last row in <ROUND>/fetch/manifest.csv and the REFETCH-START note, both printed.
  O-3   PROVISIONAL compares the branch HEAD with APPROVED's and the LAUNCH lines' head= (no pin holds it).
  O-2   the TRIGGER-a/FOUND-GONE and STOP/TRIGGER-d pairs are checked and printed (report only).
v6 (criteria lean v6, RUNLOG line 661: V5C-2 option A; amender 2026-10-27):
  MEAN_a = (CMD_a(R1) + CMD_a(R2)) / 2 with both formed, the formed one's CMD_a with one not formed, not formed (below
        0.60 in either sign) with neither (r10round_common.mean_a); TRACE RAIL's second clause, PARKED and DRIVEN read
        it so, and nothing about MEAN_a is held open (no "open" entry, no incomplete reading for it; V5C-3: a formed
        0.60 beside one not formed gives MEAN_a 0.60, which reaches 0.60).  Its basis is printed (mean_basis).
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("r10round_common")
LG = _load("ledger_r10round")
CP = _load("complete_r10round")

EXIT_FINAL, EXIT_INCOMPLETE, EXIT_PROVISIONAL, EXIT_INPUT = 0, 10, 11, 2
LINE_TEXT = {
    1: "TAIL RAIL FORMED ON r10 IN TRAINING",
    2: "INCOMPLETE",
    3: "NO RAIL, TAIL DRIVE PERSISTS",
    4: "NO RAIL IN 3 OF 3, NO PARKED TAIL, GATE CERTIFICATE PASSED IN G+ OF 3",
    5: "NO RAIL, GATE CERTIFICATE NOT PASSED",
}
WEAK = ("(end-only, not drive-backed: weak evidence; false-rail chance if r10 cured the habit 0.003-0.11 per round, "
        "up to 0.31 [inferred, EVID])")


# ------------------------------------------------------------------------------------------------- LEAN-R
def listing_names(note):
    files = note["fields"].get("files", "") if note else ""
    return [x.split(":", 1)[0] for x in files.split(";") if x.strip()]


ABSENT_PARENT = {"run_folder": ("ppo", None), "01_stance": ("root", "01_stance"), "01_stance/models": ("01_stance", "models")}


def _floor_minute(t):
    return t.replace(second=0, microsecond=0)


def _within_trigger(utc, trig_time, listing_time):
    """A listing made at or after the trigger note and not after the RUNLOG LISTING line that reports it (the line's
    time is a whole minute, so the listing's time is compared to the minute) [inferred: THE TRIGGER LISTING is made
    "at a trigger ... before the relaunch decision" and goes to RUNLOG]."""
    t = C.parse_utc(utc)
    return (t is not None and trig_time is not None and listing_time is not None and t >= trig_time
            and _floor_minute(t) <= listing_time)


def _store_listings(lp: Path):
    """Every version of a fetch store listing (<key>.json and the kept <key>.json.v<n>), parsed."""
    out = []
    for f in sorted(lp.parent.glob(lp.name + "*")) if lp.parent.is_dir() else []:
        if f.name == lp.name or (f.name.startswith(lp.name + ".v") and f.name[len(lp.name) + 2:].isdigit()):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return out


def absent_listing_state(key, round_dir, run_id, trig_time, listing_time=None):
    """v5 R-B2: "a run folder, 01_stance/ or 01_stance/models/ absent from its parent's complete listing (the ppo
    folder's (the run folder's parent), the run folder's, 01_stance/'s, each by folder id to its last page) is a
    TRIGGER LISTING that holds no file, noted folder=absent:<key>; the listing goes to RUNLOG and to the fetch manifest,
    and score_r10round.py reads it there for (i)".  ("ok", why) when the fetch manifest holds the LISTING row
    folder=absent:<key> (runlog-listing writes it) and the fetch store holds the parent's complete listing without that
    folder, made at or after the trigger note and not after the RUNLOG LISTING line (for the run folder: one of the ppo
    folder's listings kept in folders.json, with its LISTING row; for 01_stance/ and 01_stance/models/: a version of
    the TRIGGER store's listing of the run folder / 01_stance/, with its LISTING row); else ("undecided", why)."""
    if key not in ABSENT_PARENT:
        return "undecided", f"folder=absent:{key}: not the run folder, 01_stance/ or 01_stance/models/"
    if round_dir is None:
        return "undecided", "no round folder given: the fetch store cannot be compared"
    rd = Path(round_dir)
    fm = rd / "fetch" / "manifest.csv"
    parent, child = ABSENT_PARENT[key]
    when = (f"made at or after the trigger note ({trig_time and trig_time.isoformat()}) and not after the LISTING "
            f"line ({listing_time and listing_time.isoformat()})")
    if key == "run_folder":
        try:
            folders = json.loads((rd / "fetch" / "folders.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "undecided", "no ppo folder listing in the fetch store (folders.json)"
        kept = list(folders.get("_ppo_listings") or []) or ([folders["_ppo_listing"]] if folders.get("_ppo_listing")
                                                            else [])
        fit = [x for x in kept if x.get("complete") and run_id not in (x.get("folders") or [])
               and _within_trigger(x.get("utc"), trig_time, listing_time)]
        if not fit:
            return "undecided", (f"no complete ppo folder listing without the run folder {when} in folders.json "
                                 f"({len(kept)} kept)")
        rows = C.FetchStore(rd, fm, "FOLDERS", None).manifest_rows()
        if not any(r.get("status") == "LISTING" and r.get("utc") in [x.get("utc") for x in fit] for r in rows):
            return "undecided", "no LISTING row of that ppo folder listing in the fetch manifest"
    else:
        lp = rd / "fetch" / "TRIGGER" / run_id / "listings" / f"{parent}.json"
        kept = _store_listings(lp)
        if not kept:
            return "undecided", f"no TRIGGER listing of {parent} in the fetch store ({lp})"
        fit = [x for x in kept if x.get("complete") and child not in [f.get("name") for f in x.get("folders", [])]
               and _within_trigger(x.get("utc"), trig_time, listing_time)]
        if not fit:
            return "undecided", (f"no complete TRIGGER listing of {parent} without {child}/ {when} in the fetch store "
                                 f"({len(kept)} kept)")
        rows = C.FetchStore(rd / "fetch" / "TRIGGER" / run_id, fm, "TRIGGER", run_id).manifest_rows()
        if not any(r.get("status") == "LISTING" and r.get("path") == parent + "/"
                   and r.get("utc") in [x.get("utc") for x in fit] for r in rows):
            return "undecided", f"no LISTING row of that {parent}/ listing in the fetch manifest"
    rows = C.FetchStore(rd / "fetch" / "TRIGGER" / run_id, fm, "TRIGGER", run_id).manifest_rows()
    if not any(r.get("status") == "LISTING" and r.get("path") == "01_stance/models/"
               and (r.get("note") or "").startswith(f"folder=absent:{key} ") for r in rows):
        return "undecided", f"the fetch manifest holds no LISTING row folder=absent:{key} (runlog-listing writes it)"
    return "ok", (f"a TRIGGER LISTING that holds no file (v5 R-B2): {key} is absent from its parent's complete listing "
                  "in the fetch store and the fetch manifest")


def trigger_listing_state(note, round_dir, run_id, trig_time=None):
    """Fixer (R-m6) and v5 (R-B2): ("ok", why) when RUNLOG's LISTING line equals the fetch store's complete listing and
    the fetch manifest's LISTING row, or is a folder=absent:<key> line the fetch store shows (a TRIGGER LISTING that
    holds no file); else ("undecided", why).  Pin step (V5C-1): in both cases the store's listing must be made at or
    after the trigger note and not after the LISTING line."""
    if note is None:
        return "none", "no LISTING line"
    folder = note["fields"].get("folder", "")
    if folder.startswith("absent:"):
        if note["fields"].get("files", ""):
            return "undecided", f"{folder} with files listed"
        return absent_listing_state(folder[len("absent:"):], round_dir, run_id, trig_time, note["time"])
    if round_dir is None:
        return "undecided", "no round folder given: the fetch store's listing cannot be compared"
    rd = Path(round_dir)
    lp = rd / "fetch" / "TRIGGER" / run_id / "listings" / "01_stance" / "models.json"
    try:
        lst = json.loads(lp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "undecided", f"no TRIGGER listing of 01_stance/models/ in the fetch store ({lp})"
    if not lst.get("complete"):
        return "undecided", f"the fetch store's TRIGGER listing is not complete ({lp})"
    store = sorted(f"{f.get('name')}:{f.get('size')}" for f in lst.get("files", []))
    line = sorted(x for x in note["fields"].get("files", "").split(";") if x.strip())
    if store != line:
        return "undecided", f"RUNLOG's LISTING line {line} differs from the fetch store's listing {store}"
    # Pin step (fresh check V5C-1): the same time window absent_listing_state applies.  THE TRIGGER LISTING is made
    # "at a trigger noted within 2 hours of a launch, before the relaunch decision": the LISTING line at or after the
    # trigger note, and the store's listing made at or after the trigger note and not after the LISTING line.
    if trig_time is None or note["time"] < trig_time:
        return "undecided", (f"the LISTING line ({note['time'].isoformat()}) is not at or after the trigger note "
                             f"({trig_time and trig_time.isoformat()})")
    if not _within_trigger(lst.get("utc"), trig_time, note["time"]):
        return "undecided", (f"the fetch store's TRIGGER listing of 01_stance/models/ ({lst.get('utc')}) was not made at "
                             f"or after the trigger note ({trig_time.isoformat()}) and not after the LISTING line "
                             f"({note['time'].isoformat()}) ({lp})")
    rows = [r for r in C.FetchStore(rd / "fetch" / "TRIGGER" / run_id, rd / "fetch" / "manifest.csv", "TRIGGER",
                                    run_id).manifest_rows()
            if r.get("status") == "LISTING" and r.get("path") == "01_stance/models/"]
    if not rows:
        return "undecided", "no LISTING row of 01_stance/models/ in the fetch manifest"
    want = "; ".join(f"{f.get('name')}:{f.get('size')}" for f in lst.get("files", []))
    if not any(want in (r.get("note") or "") for r in rows):
        return "undecided", "the fetch manifest's LISTING row does not hold the store's listing"
    return "ok", f"the LISTING line equals the fetch store's listing ({len(store)} files) and its manifest row"


def seed_run(seed, notes, f0_checks, round_dir=None) -> dict:
    """The seed's run of the round under REPLACEMENT (LEAN-R): the _b stands in only when launched within 1 hour
    after RUNLOG's note of the first run's trigger and after its deletion (or "found gone" note), the trigger noted
    within 2 hours of the first run's launch, and (i) the first run ended by itself (trigger (a) or (b)) with no
    01_stance/models/stage1_<steps>_steps.zip in its TRIGGER LISTING, or (ii) the launch check stopped it for a fault
    (1)-(5) or (7) computed from F0's files.  No _c."""
    first = f"stance6_r10LTSB_s{seed}"
    b = first + "_b"
    tokens = sorted({n["run"] for n in notes if n["run"] == first or n["run"].startswith(first + "_")})
    launched = [t for t in tokens if CP.launch_note(notes, t)]
    d = dict(seed=seed, first=first, launched=launched, run=None, replaced=None, not_of_round=[], why=[], undecided=[])
    fl = CP.launch_note(notes, first)
    if fl is None:
        d["why"].append("no launch of the first RUN_ID: the seed's run was never launched")
        if CP.launch_note(notes, b):
            d["not_of_round"].append(b)
            d["why"].append(f"{b} launched without a first run: not of the round")
        return d
    d["run"] = first
    bl = CP.launch_note(notes, b)
    others = [t for t in launched if t not in (first, b)]
    d["not_of_round"] += others
    if others:
        d["why"].append(f"{others}: no _c; not of the round")
    if bl is None:
        return d
    trig = C.first_note(notes, first, ("TRIGGER-a", "TRIGGER-b", "TRIGGER-c", "TRIGGER-d"))
    end = C.first_note(notes, first, ("DELETED", "FOUND-GONE"))
    cond = {}
    cond["trigger note"] = trig["text"] if trig else None
    t_tr = C.minutes(fl["time"], trig["time"]) if trig else None
    cond["trigger within 2 h of the launch"] = (t_tr is not None and 0 <= t_tr <= C.TRIGGER_WINDOW_MIN, t_tr)
    t_b = C.minutes(trig["time"], bl["time"]) if trig else None
    cond["_b within 1 h after the trigger note"] = (t_b is not None and 0 <= t_b <= C.RELAUNCH_WINDOW_MIN, t_b)
    t_e = C.minutes(end["time"], bl["time"]) if end else None
    cond["_b after the deletion / found-gone note"] = (t_e is not None and t_e >= 0, t_e)
    lst = C.first_note(notes, first, ("LISTING",))
    zips = [n for n in listing_names(lst) if C.PERIODIC_ZIP_RE.match(n)]
    by_itself = bool(trig) and trig["kind"] in ("TRIGGER-a", "TRIGGER-b")
    lst_before = lst is not None and lst["time"] <= bl["time"]
    lstate, lwhy = trigger_listing_state(lst, round_dir, first, trig and trig["time"])
    i_holds = by_itself and lst_before and not zips
    if by_itself and lst_before and lstate != "ok":
        i_holds = None                                   # R-m6: the LISTING line is not the fetch store's
    cond["(i) ended by itself, trigger listing without a periodic zip"] = (
        i_holds, dict(trigger=trig and trig["kind"], listing=lst and lst["text"],
                      listing_before_relaunch=lst_before, periodic_zips=zips, listing_check=lwhy))
    stopped = bool(C.notes_for(notes, first, ("STOP", "TRIGGER-d")))
    f0 = f0_checks.get(first) or {}
    faults = [k for k in (f0.get("faults") or []) if k in (1, 2, 3, 4, 5, 7)]
    within = f0.get("f0_copies_within_60_min")
    cond["(ii) launch-check stop for a fault (1)-(5) or (7) computed from F0's files"] = (
        stopped and bool(faults) and within is not False, dict(stop_noted=stopped, computed_faults=f0.get("faults"),
                                                               f0_within_60_min=within))
    timing = all(cond[k][0] for k in ("trigger within 2 h of the launch", "_b within 1 h after the trigger note",
                                      "_b after the deletion / found-gone note"))
    r_i = cond["(i) ended by itself, trigger listing without a periodic zip"][0]
    r_ii = cond["(ii) launch-check stop for a fault (1)-(5) or (7) computed from F0's files"][0]
    rule = True if r_ii else (None if r_i is None else bool(r_i))
    d["relaunch_conditions"] = {k: (v if not isinstance(v, tuple) else dict(holds=v[0], value=v[1])) for k, v in cond.items()}
    if timing and rule is None:
        d["undecided"].append(f"{first}: relaunch (i) not decided: {lwhy}")
        d["why"].append(f"{b}: REPLACEMENT not decided ({lwhy})")
        return d
    if timing and rule:
        d["run"], d["replaced"] = b, first
        d["why"].append(f"{b} is the seed's run (REPLACEMENT); {first} is LOST (replaced)")
    else:
        d["not_of_round"].append(b)
        d["why"].append(f"{b} does not meet REPLACEMENT: not of the round (report only); {first} stays the seed's run")
    return d


# ------------------------------------------------------------------------------------------------- classes
def flags_full(tr):
    return [(a, sg, seed) for (a, sg, full, seed) in C.trace_flags(tr) if full]


def classes_of(comp) -> dict:
    """TAIL CLASS, kind, DRIVEN, GATE and OUTCOME of a COMPLETE run, read on its records as measured (criteria lean v6
    CLASSES: O-6 CMD_a over the episodes that last beyond step 200, not formed reading as below 0.60 in either sign;
    v6 MEAN_a, the formed one's CMD_a when one is not formed (r10round_common.mean_a); O-7 an exact sign tie counts
    once in each sign; O-8 DRIVE-BACKED at any checkpoint holding one of the rail's flags).  out["open"] lists what the
    text leaves without a reading where it would decide the class: nothing since v6 (it stays in the output, empty)."""
    res, traces = comp["m5"], comp["traces"]
    r1 = str(comp["r1"])
    out = dict(open=[])
    # RESOLVED BY TRACE: at least 2 full-horizon TAIL FLAGS on one actuator and sign at the RESOLUTION checkpoint
    resolved = []
    for s1, ck in sorted(comp["resolution"].items()):
        tr = traces[str(ck) if ck != "final" else "final"]
        g = {}
        for a, sg, _seed in flags_full(tr):
            for s in C.flag_signs(sg):
                g[(a, s)] = g.get((a, s), 0) + 1
        hit = {f"{C.ACTUATOR_NAMES[a]}{'+' if sg > 0 else '-'}": v for (a, sg), v in g.items() if v >= 2}
        out.setdefault("resolution", []).append(dict(alarm=s1, checkpoint=ck, flags=g and {f"{C.ACTUATOR_NAMES[a]}{sg:+d}": v for (a, sg), v in g.items()},
                                                     resolved_by_trace=bool(hit)))
        if hit:
            resolved.append(s1)
    tail_alarms = sorted(set(res["tail_alarms"]) | set(resolved))
    # R1, R2: flags, CMD (v5 O-6), MEAN
    t1, t2 = traces[r1], traces["final"]
    f1, f2 = flags_full(t1), flags_full(t2)
    cmd, cmd_n, cmd_others = {}, {}, {}
    for lab, tr in traces.items():
        cmd[lab], cmd_n[lab], cmd_others[lab] = {}, {}, {}
        for a in C.TAIL:
            cmd[lab][a], cmd_n[lab][a], cmd_others[lab][a] = C.trace_cmd(tr, a)
    mean, mean_basis = {}, {}
    for a in C.TAIL:                                       # v6: MEAN_a always has a reading (None: not formed)
        x, y = cmd[r1][a], cmd["final"][a]
        mean[a] = C.mean_a(x, y)
        mean_basis[a] = ("R1 and R2" if x is not None and y is not None else
                         "R2 only (CMD_a(R1) not formed)" if y is not None else
                         "R1 only (CMD_a(R2) not formed)" if x is not None else "not formed (neither formed)")
    groups = {}
    for lab, fl in ((r1, f1), ("final", f2)):
        for a, sg, seed in fl:
            for s in C.flag_signs(sg):                     # O-7: an exact tie counts once in each sign
                groups.setdefault((a, s), []).append((lab, seed))
    trace_rail, single = [], []
    for (a, sg), seeds in sorted(groups.items()):
        if len(seeds) >= 2:
            trace_rail.append((a, sg, "2+ flags"))
        elif C.mean_reaches(cmd[r1][a], cmd["final"][a], sg):
            trace_rail.append((a, sg, f"1 flag, MEAN {mean[a]:+.3f} ({mean_basis[a]})"))
        else:
            single.append((a, sg, seeds[0][1]))
    drive = []
    for a, sg, why in trace_rail:
        per = {}
        for lab, fl in ((r1, f1), ("final", f2)):
            if any(x == a and sg in C.flag_signs(y) for x, y, _s in fl):
                per[lab] = C.reaches(cmd[lab][a], sg)       # O-6: a CMD_a not formed reads as below 0.60
        drive.append(dict(actuator=C.ACTUATOR_NAMES[a], sign="+" if sg > 0 else "-", basis=why,
                          drive_backed=any(per.values()), drive_backed_by_checkpoint=per))
    any_flag_r1r2 = bool(f1 or f2)
    if tail_alarms:
        cls = "RAIL"
        kind = "RAIL-LEDGER-PERSISTENT" if any_flag_r1r2 else "RAIL-LEDGER-RESOLVED"
    elif trace_rail:
        cls, kind = "RAIL", "RAIL-END-ONLY"
    else:
        cls, kind = None, None
    # PARKED (tail pitch actuators, tail-up) on MEAN or J
    jl = [str(s) for s in comp["js"]] + ["final"]
    parked = []
    for a in C.PITCH:
        fired = []
        if C.mean_reaches(cmd[r1][a], cmd["final"][a], 1):
            fired.append(("END", round(mean[a] - C.PARK_LINE, 4)))
        jv = [(lab, cmd[lab][a]) for lab in jl]
        hits = [(lab, round(v - C.PARK_LINE, 4)) for lab, v in jv if C.reaches(v, 1)]
        if len(hits) >= 2:
            fired.append(("J", hits))
        if fired:
            parked.append(dict(actuator=C.ACTUATOR_NAMES[a], windows=fired))
    driven = []
    for a in C.TAIL:
        up, dn = C.mean_reaches(cmd[r1][a], cmd["final"][a], 1), C.mean_reaches(cmd[r1][a], cmd["final"][a], -1)
        if (up or dn) and not (a in C.PITCH and up):
            driven.append(dict(actuator=C.ACTUATOR_NAMES[a], mean=round(mean[a], 4)))
    stopped = comp["stopped"]
    if cls is None:
        to_stop = "PARKED" if parked else "CLEAR"
        cls = "UNREAD" if stopped else to_stop
        if stopped:
            out["class_to_stop"] = dict(cls=to_stop, E=comp["E"])
    # GATE
    if stopped:
        gate, gate_why = "FAIL", "STOPPED (no stage1_final at 12,001,280)"
    else:
        cert = comp["certificate"]
        gate = "PASS" if cert["passed"] else "FAIL"
        gate_why = None if cert["passed"] else (cert.get("failures") or "failed")
    reasons = []
    if cls == "RAIL":
        reasons.append(f"RAIL ({kind})")
    if cls == "UNREAD":
        reasons.append("UNREAD (STOPPED without RAIL)")
    if cls == "PARKED":
        reasons.append("PARKED")
    if gate == "FAIL":
        reasons.append(f"GATE FAIL ({gate_why})")
    outcome = "POSITIVE" if (cls == "CLEAR" and gate == "PASS") else "NEGATIVE"
    alarm_detail = [C.tail_alarm_detail(res, s) for s in res["tail_alarms"]]
    any_detail = []
    for s in res["any_alarms"]:
        e0, e1 = res["evals"][s - C.GRID], res["evals"][s]
        names, reas = {}, {}
        for e in (e0, e1):
            for k, v in e["any_by"].items():
                names[k] = names.get(k, 0) + v
            for k, v in e["reasons"].items():
                reas[k] = reas.get(k, 0) + v
        tail_scope = "tail" if any(n in C.ACTUATOR_NAMES[11:] for n in names) else "non-tail or unnamed"
        any_detail.append(dict(step=s, actuators=names, tail_scope=tail_scope, reasons=reas))
    out.update(cls=cls, kind=kind, tail_alarms=tail_alarms, resolved_by_trace=resolved, tail_alarm_detail=alarm_detail,
               flagged_evaluations=len(res["tail_flagged"]), trace_rail=drive, single_tail_flags=[
                   dict(actuator=C.ACTUATOR_NAMES[a], sign="+" if sg > 0 else "-", seed=s) for a, sg, s in single],
               parked=parked, driven=driven, mean={C.ACTUATOR_NAMES[a]: mean[a] for a in C.TAIL},
               mean_basis={C.ACTUATOR_NAMES[a]: mean_basis[a] for a in C.TAIL},
               cmd={lab: {C.ACTUATOR_NAMES[a]: v for a, v in d.items()} for lab, d in cmd.items()},
               cmd_episodes={lab: {C.ACTUATOR_NAMES[a]: v for a, v in d.items()} for lab, d in cmd_n.items()},
               cmd_others_ended_by_200={lab: sorted({s for v in d.values() for s in v}) for lab, d in cmd_others.items()},
               gate=gate, gate_why=gate_why, outcome=outcome, reasons=reasons, any_alarm=bool(res["any_alarms"]),
               any_alarms=any_detail, stopped=stopped, E=comp["E"],
               fall_flags=[(lab, C.ACTUATOR_NAMES[a], sg, seed) for lab, tr in traces.items()
                           for (a, sg, full, seed) in C.trace_flags(tr) if not full])
    return out


# ------------------------------------------------------------------------------------------------- the reading
def reading(per_seed, presence_rails, lost_any) -> dict:
    """ROUND READING over the seeds' runs: R, U, P, G+; statistic A with TABLE E; "A WITH LOST RUNS"."""
    R = U = P = G = 0
    A = nA = 0
    basis = []
    for d in per_seed:
        c = d.get("classes")
        if d.get("verdict") != "COMPLETE":
            U += 1
            continue
        if c["cls"] == "RAIL":
            R += 1
            basis.append("ledger" if c["tail_alarms"] else "end")
        elif c["cls"] == "UNREAD":
            U += 1
        elif c["cls"] == "PARKED":
            P += 1
        G += c["outcome"] == "POSITIVE"
        if c["any_alarm"]:
            A += 1
        if not c["stopped"] or c["any_alarm"]:
            nA += 1
    R_total = R + len(presence_rails)
    if R_total >= 1:
        line = 1
        if "ledger" in basis:
            b = "(ledger)"
        elif presence_rails:
            b = "(ledger, a LOST run)"
        else:
            ends = [x for d in per_seed if d.get("classes") and d["classes"]["kind"] == "RAIL-END-ONLY"
                    for x in d["classes"]["trace_rail"]]
            if any(x["drive_backed"] for x in ends):     # the text's else-chain (R-m2); O-8: any flagged checkpoint
                b = "(end-only, drive-backed)"
            else:
                b = WEAK
        text = (f"{LINE_TEXT[1]} {b}. An existence result for r10 as shipped (weak for the last); not a rate. To the "
                "maintainer with the DIAGNOSTIC READOUTS; nothing is proposed by this text.")
    elif U >= 1:
        line, text = 2, (f"INCOMPLETE. No tail rail in the {3 - U} run(s) read to 12M; {U} seed(s) LOST or STOPPED "
                         "without a rail: not a no-rail reading. No parent, confirmation or certification route follows.")
    elif P >= 1:
        line, text = 3, (f"NO RAIL, TAIL DRIVE PERSISTS (PARKED in {P} of 3). To the posture decision (RUNLOG line 509 "
                         "(3)); the best gate-passing final by stance3's parent rule is NAMED with its posture (no "
                         "automatic offer, D8 (a)).")
    elif G >= 2:
        line, text = 4, (f"NO RAIL IN 3 OF 3, NO PARKED TAIL, GATE CERTIFICATE PASSED IN {G} OF 3. Consistent with r10 "
                         "removing the habit, not confirmatory (if nothing had changed, 0 of 3 has probability 0.24). "
                         "To the maintainer: the confirmation or certification step and the best POSITIVE final by "
                         "stance3's parent rule as the candidate stance parent for a new walking pilot.")
    else:
        line, text = 5, ("NO RAIL, GATE CERTIFICATE NOT PASSED; each GATE FAIL's failing criteria and K1 failure modes "
                         "named; the maintainer chooses the next round.")
    if nA == 0:
        stat = dict(A=A, n_A=nA, text="no run counted", fisher_p=None, table_e=None)
    else:
        p = C.fisher_lower(A, nA)
        te = C.TABLE_E.get((A, nA))
        stat = dict(A=A, n_A=nA, text=f"{A} of {nA} against LTSB r9's 2 of 5", fisher_p=round(p, 4),
                    fisher_p_table_e=te[0] if te else None,
                    table_e=dict(p_r10_below_r9=te[1], p_r10_below_half=te[2], upper_95=te[3]) if te else None)
    wl_a = A + len(lost_any)
    wl_n = nA + len(lost_any)
    return dict(line=line, text=text, R=R_total, R_complete=R, presence_rails=len(presence_rails), U=U, P=P, G=G,
                stat=stat, with_lost=dict(A=wl_a, n_A=wl_n, runs=lost_any, text=f"{wl_a} of {wl_n} (report only)"),
                beside=dict(family="4 of 10", same_seed_ltsb="2 of 3"))


def parent_named(per_seed, round_dir, line, panels_format="{label}/{kind}") -> dict:
    """Lines 3 and 4: the best gate-passing final (line 3) or the best POSITIVE final (line 4) by stance3's parent rule
    (most clean panel episodes at final; ties to fewer joint-space plus own-sigma noise falls; ties named), from M4's
    judge record at final (<panels>/<RUN_ID>/final/m4/*.json: panel n_clean, noise_matched_own and
    noise_matched_control_joint falls).  Report only: it names, it never offers; absent inputs are printed."""
    if line not in (3, 4):
        return {}
    cands = []
    for x in per_seed:
        c = x.get("classes")
        if not c or c["gate"] != "PASS" or (line == 4 and c["outcome"] != "POSITIVE"):
            continue
        d = Path(round_dir) / "panels" / x["run"] / panels_format.format(label="final", kind="m4")
        rec = None
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                j = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(j, dict) and "noise_matched_own" in j and "noise_matched_control_joint" in j:
                rec = j
                break
        if rec is None:
            cands.append(dict(run=x["run"], missing=f"no M4 judge record at {d}"))
            continue
        panel = rec.get("panel") or {}
        cands.append(dict(run=x["run"], clean=panel.get("n_clean"),
                          falls=(rec["noise_matched_own"].get("falls") or 0) + (rec["noise_matched_control_joint"].get("falls") or 0)))
    ok = [c for c in cands if "missing" not in c and c["clean"] is not None]
    if not ok or len(ok) != len(cands):
        return dict(named=None, candidates=cands, note="not named: an input is absent")
    best = max(c["clean"] for c in ok)
    top = [c for c in ok if c["clean"] == best]
    least = min(c["falls"] for c in top)
    top = [c for c in top if c["falls"] == least]
    return dict(named=[c["run"] for c in top], tie=len(top) > 1, candidates=cands)


# ------------------------------------------------------------------------------------------------- PROVISIONAL
def provisional(pins, approved, notes, runs_detail, round_dir) -> list[str]:
    out = []
    for k in C.unfilled_pins(pins):
        out.append(f"pin unfilled: {k}")
    th = C.pin(pins, "tool_sha256") or {}
    for n, h in C.tool_hashes().items():
        if th.get(n) != h:
            out.append(f"tool {n} sha256 {h[:12]}... is not its pin ({str(th.get(n))[:12]})")
    r = Path(round_dir)
    if not approved:
        out.append("<ROUND>/APPROVED absent or unreadable")
        approved = {}
    for key, fname in (("criteria_sha256", "criteria.txt"), ("this12_sha256", "criteria_v12_cited.txt"),
                       ("evid_sha256", "evidence.txt")):
        f = r / fname
        got = C.sha256_file(f) if f.is_file() else None
        if got is None or got != approved.get(key):
            out.append(f"{fname} sha256 {got} is not APPROVED's {approved.get(key)}")
    head = approved.get("branch_head")                   # v5 O-3: APPROVED's, and the LAUNCH lines' head= (no pin)
    heads = {n["fields"].get("head") for n in notes if n["kind"] in ("LAUNCH", "RELAUNCH")}
    if head is None or any(h != head for h in heads):
        out.append(f"branch HEAD: APPROVED {head}, RUNLOG LAUNCH lines {sorted(map(str, heads))}")
    val = approved.get("validations") or {}
    for v in ("V0", "V1", "V2", "V3", "V4", "V5", "V8 (i)", "V8 (ii)", "V9"):
        if val.get(v) != "PASS":
            out.append(f"validation {v}: {val.get(v)!r}, not PASS")
    for rid, d in runs_detail.items():
        comp = d.get("completeness") or {}
        if comp.get("verdict") != "COMPLETE":
            continue
        head_run = (comp.get("stage_git_commit") or None)
        for lab, tr in (comp.get("traces") or {}).items():
            if tr is None:
                continue
            if tr.get("physics_revision") != 10:
                out.append(f"{rid} M1 {lab}: physics_revision {tr.get('physics_revision')!r}, not 10")
            if head_run and tr.get("commit") != head_run:
                out.append(f"{rid} M1 {lab}: measured on {tr.get('commit')!r}, not the run's commit {head_run}")
            if list(tr.get("seeds") or []) != list(C.TRACE_SEEDS) or tr.get("deterministic") is not True or tr.get("holds"):
                out.append(f"{rid} M1 {lab}: seeds/deterministic/holds not M1's")
        cert = comp.get("certificate")
        if cert is not None:
            if cert.get("physics_revision") != 10:
                out.append(f"{rid} M3: physics_revision {cert.get('physics_revision')!r}, not 10")
            if head_run and cert.get("commit") != head_run:
                out.append(f"{rid} M3: measured on {cert.get('commit')!r}, not the run's commit {head_run}")
            if not cert.get("refused") and list(cert.get("seeds") or []) != list(C.CERT_SEEDS):
                out.append(f"{rid} M3: seeds not 3042-3081")
        for store_name, files in (d.get("files_read") or {}).items():
            for p, h in files.items():
                if not d["recorded"].get((store_name, p, h)):
                    out.append(f"{rid} {store_name} {p}: sha256 {h[:12]}... not a COPY in the fetch manifest")
    return [x for x in out if x]


# ------------------------------------------------------------------------------------------------- main
def score(round_dir=C.ROUND, runlog=C.RUNLOG, pins_path=None, approved_path=None, panels_format="{label}/{kind}"):
    pins = C.load_pins(pins_path)
    notes, others = C.parse_runlog(runlog, C.RUN_IDS)
    r = Path(round_dir)
    try:
        approved = json.loads(Path(approved_path or r / "APPROVED").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        approved = None
    out = dict(version=C.VERSION, round=str(r), others=others, seeds=[], runs={}, incomplete=[])
    fm = r / "fetch" / "manifest.csv"
    f0_checks = {}
    for t in sorted({n["run"] for n in notes if CP.base_seed(n["run"]) is not None}):
        f0 = r / "fetch" / "F0" / t
        if f0.is_dir():
            f0_checks[t] = CP.launch_check(t, C.FetchStore(f0, fm, "F0", t), notes, pins)
    seeds = [seed_run(s, notes, f0_checks, r) for s in C.SEEDS]
    out["seeds"] = seeds
    for d in seeds:
        out["incomplete"] += d.get("undecided", [])
    launched = [t for d in seeds for t in d["launched"]]
    # ENDED and F2's time
    ends = {}
    for t in launched:
        e = C.first_note(notes, t, ("DELETED", "FOUND-GONE"))
        ends[t] = e
        if e is None:
            out["incomplete"].append(f"{t} has not ENDED (no DELETED or FOUND-GONE note)")
    last_end = max((e["time"] for e in ends.values() if e), default=None)
    f2s = [n for n in notes if n["kind"] == "F2-START"]
    out["f2_timing"] = dict(last_ending_note=last_end and last_end.isoformat(),
                            planned_start=last_end and C.add_minutes(last_end, 360).isoformat(),
                            f2_start_notes=[n["text"] for n in f2s],
                            # v5 O-14: "starting at least 6 hours after RUNLOG's last ending note, the delay printed"
                            f2_start_hours_after_last_ending=[round(C.minutes(last_end, n["time"]) / 60.0, 2)
                                                              for n in f2s] if last_end else [],
                            at_least_6_hours=[C.minutes(last_end, n["time"]) >= 360.0 for n in f2s] if last_end else [])
    # v5 O-17: the re-fetch "6 hours after F2's last row in <ROUND>/fetch/manifest.csv (the scorer prints both times)"
    f2_rows = [C.parse_utc(x.get("utc")) for x in C.FetchStore(r, fm, "F2", None).manifest_rows()]
    f2_last = max((x for x in f2_rows if x is not None), default=None)
    rfs = [n for n in notes if n["kind"] == "REFETCH-START"]
    out["refetch_timing"] = dict(f2_last_fetch_manifest_row=f2_last and f2_last.isoformat(),
                                 planned_start=f2_last and C.add_minutes(f2_last, 360).isoformat(),
                                 refetch_start_notes=[n["text"] for n in rfs],
                                 hours_after_f2_last_row=[round(C.minutes(f2_last, n["time"]) / 60.0, 2)
                                                          for n in rfs] if f2_last else [])
    out["records"] = C.records_pairing(notes)
    if not f2s:
        out["incomplete"].append("no F2-START note")
    of_round = {d["run"] for d in seeds if d["run"]}
    lost_any, presence_rails = [], []
    for t in launched:
        f2 = r / "fetch" / "F2" / t
        if not f2.is_dir():
            out["incomplete"].append(f"{t}: no F2 store")
            out["runs"][t] = dict(verdict="NOT FETCHED")
            continue
        stores = [C.FetchStore(f2, fm, "F2", t)]
        panels = CP.PanelLocator(r / "panels" / t, panels_format)
        first = CP.completeness(t, stores[0], notes, pins, panels)
        det = dict(f2=dict(verdict=first["verdict"], C=_brief(first)))
        comp = first
        refetch_due = (t in of_round and first["C1"]["ok"] is True and
                       (first["C2"]["ok"] is not True or first["C3"]["ok"] is not True)
                       # v5 O-16: "unless it fails C4 or C6 on files matching the manifest"
                       and not (first["C4"]["ok"] is False and first.get("C4_on_manifest_files"))
                       and not (first["C6"]["ok"] is False and first.get("C6_on_manifest_files")))
        rf = r / "fetch" / "REFETCH" / t
        if refetch_due:
            if rf.is_dir():
                stores.append(C.FetchStore(rf, fm, "REFETCH", t))
                comp = CP.completeness(t, stores[1], notes, pins, panels)
                det["refetch"] = dict(verdict=comp["verdict"], C=_brief(comp))
            else:
                out["incomplete"].append(f"{t}: passes C1 and fails C2 or C3 at F2; the re-fetch is not held")
        elif rf.is_dir():
            det["refetch_note"] = "a REFETCH store exists but the rule does not call for one: not read"

        det["completeness"] = dict(comp)
        det["completeness"]["stage_git_commit"] = None
        if comp["verdict"] == "COMPLETE":
            scj = C.load_json_bytes(_bytes_for(stores[-1], comp, "01_stance/stage_config.json"))
            det["completeness"]["stage_git_commit"] = scj.get("git_commit") if isinstance(scj, dict) else None
        det["verdict"] = "REPLACED (LOST)" if any(d["replaced"] == t for d in seeds) else comp["verdict"]
        if comp["verdict"] in ("PENDING (C7: a measurement absent)",) or comp["verdict"].startswith("UNDECIDED"):
            out["incomplete"].append(f"{t}: {comp['verdict']}")
        replaced = any(d["replaced"] == t for d in seeds)
        outside = t not in of_round and not replaced
        is_lost = comp["verdict"] == "LOST" or replaced or outside
        if outside:
            # v5 O-13: "A _b launched when the rule does not hold is no run of the round: no class, no PRESENCE,
            # printed with D-16 only." (a _c likewise: no _c)
            det["verdict"] = f"NOT OF THE ROUND ({comp['verdict']}; report only)"
        det["presence"] = [CP.presence(t, s, notes, pins) for s in stores] if (is_lost and not outside) else []
        if is_lost and any(p["rail"] for p in det["presence"]):
            pr = next(p for p in det["presence"] if p["rail"])
            presence_rails.append(dict(run=t, detail=pr["detail"], fetch=pr["fetch"]))
        elif is_lost and any(p.get("rail_decided") is None for p in det["presence"]):
            und = [f"{p['fetch']}: " + ", ".join(f"{k} {CP.status_of(p[k])}" for k in ("p1", "p2", "p3"))
                   for p in det["presence"] if p.get("rail_decided") is None]
            out["incomplete"].append(f"{t}: PRESENCE undecided (a p-fact not established): {und}")
        if is_lost:
            f0c = f0_checks.get(t)
            det["d16"] = CP.d16(t, notes, stores, pins, r / "outputs" / f"{t}.ipynb", f0c)
            if det["d16"]["any_alarm_shown"]:
                lost_any.append(t)
        # files read and the fetch manifest (PROVISIONAL)
        det["files_read"] = {f"{s.fetch_name}": comp["files_read"] for s in stores[-1:]}
        det["recorded"] = {}
        for s in stores[-1:]:
            for p, h in comp["files_read"].items():
                det["recorded"][(s.fetch_name, p, h)] = any(
                    x.get("path") == p and x.get("status") == "COPY" and x.get("sha256") == h for x in s.manifest_rows())
        out["runs"][t] = det
    per_seed = []
    for d in seeds:
        t = d["run"]
        det = out["runs"].get(t) if t else None
        x = dict(seed=d["seed"], run=t, verdict=(det or {}).get("verdict", "NEVER LAUNCHED" if t is None else "NOT FETCHED"))
        if det and det["verdict"] == "COMPLETE":
            comp = det["completeness"]
            x["classes"] = classes_of(comp)
            if x["classes"]["open"]:
                out["incomplete"].append(f"{t}: class not decided: {x['classes']['open']}")
        per_seed.append(x)
    out["per_seed"] = per_seed
    out["presence_rails"] = presence_rails
    rd = reading(per_seed, presence_rails, lost_any) if not out["incomplete"] else None
    if rd is not None:
        rd["parent"] = parent_named(per_seed, r, rd["line"], panels_format)
    out["reading"] = rd
    prov = provisional(pins, approved, notes, out["runs"], r)
    out["provisional"] = prov
    if out["incomplete"]:
        status = EXIT_INCOMPLETE
    elif prov:
        status = EXIT_PROVISIONAL
    else:
        status = EXIT_FINAL
    out["status"] = {EXIT_FINAL: "FINAL", EXIT_INCOMPLETE: "INCOMPLETE (no line read)",
                     EXIT_PROVISIONAL: "PROVISIONAL (read, printed, not acted on)"}[status]
    return out, status


def _brief(comp):
    return {k: CP.status_of(comp[k]) for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")}


def _bytes_for(store, comp, path):
    want = comp["files_read"].get(path)
    for _k, b, _f in store.copies(path):
        if C.sha256_bytes(b) == want:
            return b
    return b""


def render(out) -> str:
    L = [C.VERSION, f"ROUND {out['round']}"]
    for i, line in out["others"]:
        if line.startswith(C.LATE_PREFIX):
            L.append(f"RUNLOG line {i}: {line[:len(C.LATE_PREFIX) + 160]}")
            continue
        L.append(f"RUNLOG line {i} names a RUN_ID and is not a STANCE6 note of a round RUN_ID (printed; read by no "
                 f"rule; for the coordinator to restate): {line[:160]}")
    L.append(f"F2 timing (O-14, report only): {json.dumps(out['f2_timing'])}")
    L.append(f"re-fetch timing (O-17, both times printed): {json.dumps(out['refetch_timing'])}")
    for x in out.get("records") or []:
        L.append(f"RECORDS (report only): {x}")
    for d in out["seeds"]:
        L.append(f"SEED {d['seed']}: run of the round {d['run']}; launched {d['launched']}; {'; '.join(d['why'])}")
        if d.get("relaunch_conditions"):
            for k, v in d["relaunch_conditions"].items():
                L.append(f"    {k}: {json.dumps(v, default=str)[:300]}")
    for t, det in out["runs"].items():
        L.append(f"RUN {t}: {det.get('verdict')}  F2 {json.dumps(det.get('f2', {}).get('C'))}"
                 + (f"  re-fetch {json.dumps(det['refetch']['C'])}" if det.get("refetch") else ""))
        comp = det.get("completeness")
        if comp:
            for ln in CP.render_facts(f"  C1-C7 of {t}", comp)[1:]:
                L.append("  " + ln)
            for x in comp.get("measured_on_another_file") or []:
                L.append(f"    NOT COUNTED (R-B4): {x}")
            for lab, tr in sorted((comp.get("traces") or {}).items()):
                cc = (tr or {}).get("cross_check")
                if cc is not None:
                    L.append(f"    M1 cross-check {lab} (report only, RA-8): "
                             + ("lane A's reduction equals this reader's" if cc["equal"] else
                                f"DIFFERS {json.dumps(cc['differences'], default=str)[:400]}"))
        for p in det.get("presence") or []:
            for ln in CP.render_presence(p):
                L.append("  " + ln)
        if det.get("d16"):
            dd = det["d16"]
            L.append(f"    D-16 LOST (report only; not in a class or A; in the reading only as a PRESENCE RAIL): "
                     f"triggers {dd['triggers']}; checks {json.dumps(dd['trigger_checks'], default=str)[:400]}; "
                     f"notebook last error {dd['notebook_last_error']!r}; shown {json.dumps(dd['shown'], default=str)[:400]}")
    for x in out["per_seed"]:
        c = x.get("classes")
        if c:
            L.append(f"CLASS s{x['seed']} {x['run']}: {c['cls']}{' (' + c['kind'] + ')' if c['kind'] else ''}; GATE {c['gate']}"
                     f" {c['gate_why'] or ''}; OUTCOME {c['outcome']} {c['reasons']}; ANY ALARM {c['any_alarms']}")
            L.append(f"    tail alarms {c['tail_alarm_detail']}; resolved by trace {c['resolved_by_trace']}; trace rail "
                     f"{c['trace_rail']}; single tail flags {c['single_tail_flags']}; PARKED {c['parked']}; DRIVEN "
                     f"{c['driven']}; MEAN {json.dumps(c['mean'])}" + (f"; class to its stop {c['class_to_stop']}" if c.get('class_to_stop') else ""))
            L.append(f"    CMD_a episodes beyond step 200 {json.dumps(c['cmd_episodes'])}; episodes ended by step 200 "
                     f"(printed, not averaged) {json.dumps(c['cmd_others_ended_by_200'])}; MEAN_a from "
                     f"{json.dumps(c['mean_basis'])}")
        else:
            L.append(f"CLASS s{x['seed']} {x['run']}: {x['verdict']} (no class)")
    for p in out["presence_rails"]:
        L.append(f"PRESENCE RAIL {p['run']} on {p['fetch']}: {json.dumps(p['detail'])}")
    if out["incomplete"]:
        L.append("READING: incomplete; no line is read. Open:")
        for x in out["incomplete"]:
            L.append(f"    {x}")
    else:
        rd = out["reading"]
        L.append(f"READING: line {rd['line']}: {rd['text']}")
        L.append(f"    R {rd['R']} (COMPLETE {rd['R_complete']}, PRESENCE {rd['presence_rails']}), U {rd['U']}, P {rd['P']}, G+ {rd['G']}")
        st = rd["stat"]
        L.append(f"    Statistic A: {st['text']}; one-sided Fisher p {st['fisher_p']} (TABLE E {st.get('fisher_p_table_e')}); "
                 f"TABLE E {st['table_e']}; nothing is significant")
        L.append(f"    beside A: family {rd['beside']['family']}; same-seed LTSB {rd['beside']['same_seed_ltsb']}; "
                 f"A WITH LOST RUNS {rd['with_lost']['text']} {rd['with_lost']['runs']}")
        if rd.get("parent"):
            L.append(f"    stance3's parent rule (report only, named, never offered): {json.dumps(rd['parent'])}")
        for d in out["runs"].values():
            if d.get("d16"):
                L.append(f"    LOST (report only; not in a class or A; in the reading only as a PRESENCE RAIL): "
                         f"{d['d16']['run_id']}: {json.dumps(d['d16']['shown'], default=str)[:300]}")
    L.append(f"STATUS: {out['status']}")
    for p in out["provisional"]:
        L.append(f"    PROVISIONAL: {p}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", default=C.ROUND)
    ap.add_argument("--runlog", default=C.RUNLOG)
    ap.add_argument("--pins", default=None)
    ap.add_argument("--approved", default=None)
    ap.add_argument("--panels-format", default="{label}/{kind}")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        out, status = score(a.round, a.runlog, a.pins, a.approved, a.panels_format)
    except C.InputError as exc:
        print(f"{C.VERSION}\nINPUT ERROR: {exc}")
        return EXIT_INPUT
    print(render(out))
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(out), indent=1, default=str) + "\n", encoding="utf-8")
    return status


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k not in ("m5", "c6", "manifest", "rows")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


if __name__ == "__main__":
    sys.exit(main())
