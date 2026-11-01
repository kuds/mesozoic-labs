"""PC study (phase-critic Colab round): the scorer (ST/criteria.txt: COMPLETE RUN C1-C8, CLASSES PER RUN, HABIT
ELSEWHERE, THE LEDGER'S EARLY SIGNS, TAIL DRIVE, THE READING, THE ONE TEST, REPORT ONLY, WHEN THE READING IS MADE).
Report only; never trains; changes nothing anywhere.

    PY -I -B score_pc.py [--round PCR] [--runlog P] [--pins P] [--approved P] [--json OUT] [--no-comparators]

Reads RUNLOG's STANCE7 note lines only (every other line naming a PC RUN_ID is printed and read by no rule), the fetch
stores <PCR>/fetch/<F0|TRIGGER|F2|REFETCH>/<RUN_ID>/ with <PCR>/fetch/manifest.csv, the measurements
<PCR>/panels/<RUN_ID>/<label>/<m1|m3|c8>/, the PC pins file and <PCR>/APPROVED (JSON); the comparators (STANCE6's three
runs) from their FMAN-checked F2 copies, score.json and the round's own M1 records at their pinned sha256.
Reused unchanged from the round's pinned tools: M5 (r10round_common.m5), the class function
(score_r10round.classes_of), LEAN-R's listing check and the parent rule (score_r10round).  Adapted: seed_run (the PC
RUN_ID prefix; launch-check faults (1)-(5), (7) and (8)); C1-C8 (complete_r10round, PC); THE READING (lines 1-5);
THE ONE TEST; PROVISIONAL (the PC APPROVED's keys).
Exit codes: 0 final; 10 incomplete (no line read); 11 provisional (read, printed, not acted on); 2 input error.

PC AMENDMENT (2026-11-01, ST/amend/AMEND_NOTES.txt; the maintainer's CONTROL (b), "make them the test's control"): six
runs.  Each arm's seeds are run-handled alike (seed_run, arm "pc" or "off": the round's all-or-nothing rule and one _b per
run); THE READING's line is the phase runs' (unchanged); the concurrent switch-off runs' classes are printed beside;
THE ONE TEST compares the phase TDs with the switch-off TDs (a missing phase run at +infinity, a missing switch-off run at
-infinity; a switch-off run launched after a phase run's results were opened -- its arm's first F2-START,
REFETCH-START or SNAPSHOT line of a phase RUN_ID -- counts as missing: outcome blindness); the phase-off start gaps
are printed (report only); STANCE6 is report only (printed beside and in the OFF-vs-STANCE6 replication block; a
comparator input that cannot be read is printed and holds nothing open).
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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
SR = _load("score_r10round")
P = _load("pc_common")

VERSION = ("score_pc pc-3 (2026-11-01 amendment: six runs, THE ONE TEST phase vs the concurrent switch-off runs, "
           "outcome blindness, STANCE6 report only; pc-2 fixer: printed wording; PC criteria THE READING lines 1-5, "
           "THE ONE TEST, C1-C8)")
EXIT_FINAL, EXIT_INCOMPLETE, EXIT_PROVISIONAL, EXIT_INPUT = 0, 10, 11, 2
VALIDATIONS = ("VP0", "VP1", "VP2", "VP3", "VP4")
#: VP0, VP2, VP3 are before the launch (<PCR>/APPROVED "validations"); VP1, VP4 and the F2 tools' sha256 are pinned
#: before F2 (CHOICE TOOLS) in <PCR>/F2_PINS.json (written once, its sha256 in a RUNLOG line before the F2-START lines).
PRE_LAUNCH_VALIDATIONS = ("VP0", "VP2", "VP3")
PRE_F2_VALIDATIONS = ("VP1", "VP4")
F2_PINS_NAME = "F2_PINS.json"
C_KEYS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")


# ---------------------------------------------------------------------------------------------------- LEAN-R (PC)
def seed_run(seed, notes, f0_checks, round_dir=None, arm="pc") -> dict:
    """score_r10round.seed_run with the PC RUN_ID of the arm (C.ARM_PREFIX: "pc" the phase runs, "off" the concurrent
    switch-off runs, PC AMENDMENT) and the PC stop faults (1)-(5), (7), (8) (C.STOP_FAULTS); every other clause the
    round's, unchanged: per run, one _b relaunch within 2 h of that run's own launch."""
    first = f"{C.ARM_PREFIX[arm]}{seed}"
    b = first + "_b"
    tokens = sorted({n["run"] for n in notes if n["run"] == first or n["run"].startswith(first + "_")})
    launched = [t for t in tokens if CP.launch_note(notes, t)]
    d = dict(arm=arm, seed=seed, first=first, launched=launched, run=None, replaced=None, not_of_round=[], why=[],
             undecided=[])
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
    cond = {"trigger note": trig["text"] if trig else None}
    t_tr = C.minutes(fl["time"], trig["time"]) if trig else None
    cond["trigger within 2 h of the launch"] = (t_tr is not None and 0 <= t_tr <= C.TRIGGER_WINDOW_MIN, t_tr)
    t_b = C.minutes(trig["time"], bl["time"]) if trig else None
    cond["_b within 1 h after the trigger note"] = (t_b is not None and 0 <= t_b <= C.RELAUNCH_WINDOW_MIN, t_b)
    t_e = C.minutes(end["time"], bl["time"]) if end else None
    cond["_b after the deletion / found-gone note"] = (t_e is not None and t_e >= 0, t_e)
    lst = C.first_note(notes, first, ("LISTING",))
    zips = [n for n in SR.listing_names(lst) if C.PERIODIC_ZIP_RE.match(n)]
    by_itself = bool(trig) and trig["kind"] in ("TRIGGER-a", "TRIGGER-b")
    lst_before = lst is not None and lst["time"] <= bl["time"]
    lstate, lwhy = SR.trigger_listing_state(lst, round_dir, first, trig and trig["time"])
    i_holds = by_itself and lst_before and not zips
    if by_itself and lst_before and lstate != "ok":
        i_holds = None
    cond["(i) ended by itself, trigger listing without a periodic zip"] = (
        i_holds, dict(trigger=trig and trig["kind"], listing=lst and lst["text"], listing_before_relaunch=lst_before,
                      periodic_zips=zips, listing_check=lwhy))
    stopped = bool(C.notes_for(notes, first, ("STOP", "TRIGGER-d")))
    f0 = f0_checks.get(first) or {}
    faults = [k for k in (f0.get("faults") or []) if k in C.STOP_FAULTS]
    within = f0.get("f0_copies_within_60_min")
    key_ii = "(ii) launch-check stop for a fault (1)-(5), (7) or (8) computed from F0's files"
    cond[key_ii] = (stopped and bool(faults) and within is not False,
                    dict(stop_noted=stopped, computed_faults=f0.get("faults"), f0_within_60_min=within))
    timing = all(cond[k][0] for k in ("trigger within 2 h of the launch", "_b within 1 h after the trigger note",
                                      "_b after the deletion / found-gone note"))
    r_i = cond["(i) ended by itself, trigger listing without a periodic zip"][0]
    r_ii = cond[key_ii][0]
    rule = True if r_ii else (None if r_i is None else bool(r_i))
    d["relaunch_conditions"] = {k: (v if not isinstance(v, tuple) else dict(holds=v[0], value=v[1]))
                                for k, v in cond.items()}
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


def _brief(comp):
    return {k: CP.status_of(comp[k]) for k in C_KEYS if k in comp}


# ---------------------------------------------------------------------------------------------------- PROVISIONAL
def provisional(pins, approved, notes, runs_detail, round_dir, comparators_err=None, runlog_path=None) -> list[str]:
    """WHEN THE READING IS MADE (ROUND l.490-497 with SUBST): a pin unfilled or a tool's sha256 other than its pin; the
    PC text's sha256 or the branch HEAD other than APPROVED's (and the LAUNCH lines' head=); a class input measured
    on a copy of another commit, with other seeds or physics_revision not 10; a read file not a COPY in the fetch
    manifest; VP0, VP2, VP3 not PASS in APPROVED or VP1, VP4 not PASS in F2_PINS.json; APPROVED's or F2_PINS.json's
    sha256 in no RUNLOG line; the tools' sha256 other than F2_PINS.json's (else the pins file's); the comparators'
    inputs other than their pins."""
    out = [f"pin unfilled: {k}" for k in C.unfilled_pins(pins)]
    r = Path(round_dir)
    f2p, f2p_sha = None, None
    f2f = r / F2_PINS_NAME
    if f2f.is_file():
        f2p_sha = C.sha256_file(f2f)
        try:
            f2p = json.loads(f2f.read_text(encoding="utf-8"))
        except ValueError:
            f2p = None
    try:
        runlog_text = Path(runlog_path).read_text(encoding="utf-8", errors="replace") if runlog_path else ""
    except OSError:
        runlog_text = ""
    if f2p is None:
        out.append(f"{F2_PINS_NAME} absent or unreadable (the F2 tools' sha256, VP1 and VP4 are pinned there before F2)")
    elif f2p_sha not in runlog_text:
        out.append(f"{F2_PINS_NAME} sha256 {f2p_sha} is in no RUNLOG line")
    th = (f2p or {}).get("tool_sha256") or C.pin(pins, "tool_sha256") or {}
    for n, h in C.tool_hashes().items():
        if th.get(n) != h:
            out.append(f"tool {n} sha256 {h[:12]}... is not its pin ({str(th.get(n))[:12]})")
    if not approved:
        out.append("<PCR>/APPROVED absent or unreadable")
        approved = {}
    f = r / "criteria.txt"
    got = C.sha256_file(f) if f.is_file() else None
    if got is None or got != approved.get("criteria_sha256"):
        out.append(f"criteria.txt sha256 {got} is not APPROVED's {approved.get('criteria_sha256')}")
    head = approved.get("branch_head")
    heads = {n["fields"].get("head") for n in notes if n["kind"] in ("LAUNCH", "RELAUNCH")}
    if head is None or any(h != head for h in heads):
        out.append(f"branch HEAD: APPROVED {head}, RUNLOG LAUNCH lines {sorted(map(str, heads))}")
    ap_file = r / "APPROVED"
    if ap_file.is_file() and C.sha256_file(ap_file) not in runlog_text:
        out.append(f"APPROVED sha256 {C.sha256_file(ap_file)} is in no RUNLOG line")
    val = approved.get("validations") or {}
    for v in PRE_LAUNCH_VALIDATIONS:
        if val.get(v) != "PASS":
            out.append(f"validation {v} (APPROVED): {val.get(v)!r}, not PASS")
    val2 = (f2p or {}).get("validations") or {}
    for v in PRE_F2_VALIDATIONS:
        if val2.get(v) != "PASS":
            out.append(f"validation {v} ({F2_PINS_NAME}): {val2.get(v)!r}, not PASS")
    # PC AMENDMENT: STANCE6 is report only; a comparator input that cannot be read as pinned is printed with the
    # report-only block (score's comparators_error) and makes nothing PROVISIONAL (comparators_err is not read here)
    for rid, d in runs_detail.items():
        comp = d.get("completeness") or {}
        if comp.get("verdict") != "COMPLETE":
            continue
        head_run = comp.get("stage_git_commit") or None
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
    return out


# ---------------------------------------------------------------------------------------------------- per-run extras
def run_extras(store, comp) -> dict:
    """For a COMPLETE run: the ledger's early signs, TD (and the J window), Q4 per evaluation, HABIT ELSEWHERE, the
    guard (tail_1 / tail_2 pitch commands), the on-policy tail_3 command (report only)."""
    raw = SR._bytes_for(store, comp, "01_stance/eval_episodes.csv")
    header, rows = P.ledger_rows(raw)
    per = P.per_evaluation(rows, C.ACTUATOR_NAMES)
    out = dict(early=P.early_signs(per, comp["m5"]), td=P.tail_drive(per))
    m1 = {lab: (tr or {}).get("record") or {} for lab, tr in (comp.get("traces") or {}).items() if tr}
    res_labels = [str(v) for v in (comp.get("resolution") or {}).values() if v != "final"]
    out["habit"] = P.habit_elsewhere(C, header, rows, SR.classes_of(comp), m1, comp["js"], comp["r1"], res_labels)
    rets = [e["ret"] for e in per.values() if e["ret"] is not None]
    out["q4"] = dict(mean_return=sum(rets) / len(rets) if rets else None, falls=sum(e["falls"] for e in per.values()),
                     rows=sum(e["n"] for e in per.values()), clean=sum(e["clean"] for e in per.values()))
    out["guard"] = {a: {str(s): e["acts"].get(a) for s, e in per.items() if s in (*C.J_STEPS, max(per))}
                    for a in ("tail_1_pitch_act", "tail_2_pitch_act")}
    probe = None
    ent = (comp.get("entries") or {}).get("01_stance/policy_probe.npz")
    if ent and ent[1]:
        b, how = store.matching("01_stance/policy_probe.npz", ent[1], ent[0] if isinstance(ent[0], int) else None)
        if b is not None:
            try:
                import io

                import numpy as np

                with np.load(io.BytesIO(b), allow_pickle=False) as z:
                    mu = np.asarray(z["rollout_mu_mean"], dtype=np.float64)
                    ut = np.asarray(z["update_timesteps"])
                probe = dict(last8_mean=float(np.mean(mu[-8:, 14])), rows=[int(ut[-8]), int(ut[-1])])
            except Exception as exc:  # noqa: BLE001 - report only
                probe = dict(error=f"{type(exc).__name__}: {exc}")
        else:
            probe = dict(absent=how)
    out["probe_tail3"] = probe or dict(absent="policy_probe.npz not listed or not fetched")
    return out


# ---------------------------------------------------------------------------------------------------- PC AMENDMENT
OPENING_KINDS = ("F2-START", "REFETCH-START", "SNAPSHOT")


def test_value(x, missing):
    """THE ONE TEST's value of an arm's seed: the run's TD when it is COMPLETE and FINISHED with a TD, else the arm's
    MISSING value (+infinity for a phase run, -infinity for a switch-off run: missing never helps H1)."""
    return (x["td"]["td"] if (x.get("verdict") == "COMPLETE" and x.get("finished") and x.get("td")
                              and x["td"]["td"] is not None) else missing)


def blindness(notes, seeds) -> dict:
    """PC AMENDMENT (RUN HANDLING, outcome blindness): every switch-off run starts before any phase run's results are
    opened.  The opening is the earliest F2-START, REFETCH-START or SNAPSHOT note of a phase RUN_ID (by its time, then
    line).  Per seed: the launch line of the switch-off seed's run (LAUNCH, or its _b's RELAUNCH) must be earlier;
    a later one is NOT BLIND and that run enters THE ONE TEST as missing (-infinity).  Report only: each seed's
    phase-off start gap (the switch-off run's launch minus the phase run's, minutes) and every launch time."""
    opens = sorted((n for n in notes if n["kind"] in OPENING_KINDS and C.arm_of(n["run"]) == "pc"),
                   key=lambda n: (n["time"], n["line"]))
    first = opens[0] if opens else None
    by = {(d["arm"], d["seed"]): d for d in seeds}
    rows, not_blind = {}, []
    for s in C.SEEDS:
        dp, do = by.get(("pc", s)), by.get(("off", s))
        lp = CP.launch_note(notes, dp["run"]) if dp and dp["run"] else None
        lo = CP.launch_note(notes, do["run"]) if do and do["run"] else None
        blind = None if lo is None else (first is None or lo["time"] < first["time"])
        if blind is False:
            not_blind.append(do["run"])
        rows[s] = dict(phase_run=dp and dp["run"], phase_launch=lp and lp["time"].isoformat(),
                       off_run=do and do["run"], off_launch=lo and lo["time"].isoformat(),
                       gap_minutes_report_only=(round(C.minutes(lp["time"], lo["time"]), 1) if (lp and lo) else None),
                       blind=blind)
    launches = sorted(({"run": n["run"], "kind": n["kind"], "time": n["time"].isoformat(), "line": n["line"]}
                       for n in notes if n["kind"] in ("LAUNCH", "RELAUNCH")), key=lambda x: (x["time"], x["line"]))
    return dict(first_opening=first and dict(kind=first["kind"], run=first["run"], time=first["time"].isoformat(),
                                             line=first["line"]),
                seeds=rows, not_blind=not_blind, launches=launches)


# ---------------------------------------------------------------------------------------------------- main
def score(round_dir=C.ROUND, runlog=C.RUNLOG, pins_path=None, approved_path=None, panels_format="{label}/{kind}",
          comparators=None):
    pins = C.load_pins(pins_path)
    notes, others = C.parse_runlog(runlog, C.RUN_IDS)
    r = Path(round_dir)
    try:
        approved = json.loads(Path(approved_path or r / "APPROVED").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        approved = None
    out = dict(version=VERSION, tools=C.VERSION, round=str(r), others=others, seeds=[], runs={}, incomplete=[])
    fm = r / "fetch" / "manifest.csv"
    f0_checks = {}
    for t in sorted({n["run"] for n in notes if CP.base_seed(n["run"]) is not None}):
        f0 = r / "fetch" / "F0" / t
        if f0.is_dir():
            f0_checks[t] = CP.launch_check(t, C.FetchStore(f0, fm, "F0", t), notes, pins)
    out["launch_checks"] = {t: dict(faults=v["faults"], within_60=v["f0_copies_within_60_min"])
                            for t, v in f0_checks.items()}
    # PC AMENDMENT: both arms' seeds, each run handled by the round's rules on its own launch
    seeds = [seed_run(s, notes, f0_checks, r, arm) for arm in C.ARMS for s in C.SEEDS]
    out["seeds"] = seeds
    for d in seeds:
        out["incomplete"] += d.get("undecided", [])
    launched = [t for d in seeds for t in d["launched"]]
    ends = {}
    for t in launched:
        e = C.first_note(notes, t, ("DELETED", "FOUND-GONE"))
        ends[t] = e
        if e is None:
            out["incomplete"].append(f"{t} has not ENDED (no DELETED or FOUND-GONE note)")
    last_end = max((e["time"] for e in ends.values() if e), default=None)
    f2s = [n for n in notes if n["kind"] == "F2-START"]
    out["f2_timing"] = dict(last_ending_note=last_end and last_end.isoformat(),
                            f2_start_hours_after_last_ending=[round(C.minutes(last_end, n["time"]) / 60.0, 2)
                                                              for n in f2s] if last_end else [],
                            at_least_6_hours=[C.minutes(last_end, n["time"]) >= 360.0 for n in f2s] if last_end else [])
    out["records"] = C.records_pairing(notes)
    if not f2s:
        out["incomplete"].append("no F2-START note")
    of_round = {d["run"] for d in seeds if d["run"]}
    lost_any = {a: [] for a in C.ARMS}
    presence_rails = {a: [] for a in C.ARMS}
    stores_of = {}
    for t in launched:
        f2 = r / "fetch" / "F2" / t
        if not f2.is_dir():
            out["incomplete"].append(f"{t}: no F2 store")
            out["runs"][t] = dict(verdict="NOT FETCHED")
            continue
        stores = [C.FetchStore(f2, fm, "F2", t)]
        panels = CP.PanelLocator(r / "panels" / t, panels_format)
        first = CP.completeness(t, stores[0], notes, pins, panels)
        det = dict(arm=C.arm_of(t), f2=dict(verdict=first["verdict"], C=_brief(first)))
        comp = first
        refetch_due = (t in of_round and first["C1"]["ok"] is True and
                       (first["C2"]["ok"] is not True or first["C3"]["ok"] is not True)
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
        stores_of[t] = stores[-1]
        det["completeness"] = dict(comp)
        det["completeness"]["stage_git_commit"] = None
        if comp["verdict"] == "COMPLETE":
            scj = C.load_json_bytes(SR._bytes_for(stores[-1], comp, "01_stance/stage_config.json"))
            det["completeness"]["stage_git_commit"] = scj.get("git_commit") if isinstance(scj, dict) else None
        det["verdict"] = "REPLACED (LOST)" if any(d["replaced"] == t for d in seeds) else comp["verdict"]
        if comp["verdict"].startswith("PENDING") or comp["verdict"].startswith("UNDECIDED"):
            out["incomplete"].append(f"{t}: {comp['verdict']}")
        replaced = any(d["replaced"] == t for d in seeds)
        outside = t not in of_round and not replaced
        is_lost = comp["verdict"] == "LOST" or replaced or outside
        if outside:
            det["verdict"] = f"NOT OF THE ROUND ({comp['verdict']}; report only)"
        det["presence"] = [CP.presence(t, s, notes, pins) for s in stores] if (is_lost and not outside) else []
        if is_lost and any(p["rail"] for p in det["presence"]):
            pr = next(p for p in det["presence"] if p["rail"])
            presence_rails[C.arm_of(t)].append(dict(run=t, seed=CP.base_seed(t), detail=pr["detail"],
                                                    fetch=pr["fetch"]))
        elif is_lost and any(p.get("rail_decided") is None for p in det["presence"]):
            und = [f"{p['fetch']}: " + ", ".join(f"{k} {CP.status_of(p[k])}" for k in ("p1", "p2", "p3"))
                   for p in det["presence"] if p.get("rail_decided") is None]
            out["incomplete"].append(f"{t}: PRESENCE undecided (a p-fact not established): {und}")
        if is_lost:
            f0c = f0_checks.get(t)
            det["d16"] = CP.d16(t, notes, stores, pins, r / "outputs" / f"{t}.ipynb", f0c)
            if det["d16"]["any_alarm_shown"]:
                lost_any[C.arm_of(t)].append(t)
        det["files_read"] = {f"{s.fetch_name}": comp["files_read"] for s in stores[-1:]}
        det["recorded"] = {}
        for s in stores[-1:]:
            for p, h in comp["files_read"].items():
                det["recorded"][(s.fetch_name, p, h)] = any(
                    x.get("path") == p and x.get("status") == "COPY" and x.get("sha256") == h for x in s.manifest_rows())
        out["runs"][t] = det
    per_arm = {a: [] for a in C.ARMS}
    for d in seeds:
        t = d["run"]
        det = out["runs"].get(t) if t else None
        x = dict(arm=d["arm"], seed=d["seed"], run=t,
                 verdict=(det or {}).get("verdict", "NEVER LAUNCHED" if t is None else "NOT FETCHED"))
        if det and det["verdict"] == "COMPLETE":
            comp = det["completeness"]
            x["classes"] = SR.classes_of(comp)
            x.update(run_extras(stores_of[t], comp))
            x["finished"] = bool(comp.get("finished"))
        per_arm[d["arm"]].append(x)
    out["per_seed"] = per_arm["pc"]
    out["per_seed_off"] = per_arm["off"]                      # PC AMENDMENT: the concurrent switch-off runs
    out["presence_rails"] = presence_rails["pc"]
    out["presence_rails_off"] = presence_rails["off"]
    out["blindness"] = blindness(notes, seeds)                # PC AMENDMENT: outcome blindness and start gaps
    # STANCE6 (recomputed from its pinned inputs): REPORT ONLY (PC AMENDMENT): printed beside, never deciding
    comp_err = None
    if comparators is None:
        try:
            comparators = P.comparators(C, pins)
        except (OSError, ValueError, KeyError) as exc:
            comp_err = f"{type(exc).__name__}: {exc}"
            comparators = None
    out["comparators"] = comparators
    out["comparators_error"] = comp_err
    out["replication"] = P.replication(per_arm["off"], comparators)
    rd = None
    if not out["incomplete"]:
        rd = P.reading(per_arm["pc"], presence_rails["pc"], per_arm["off"], presence_rails["off"])
        if rd["line"] == 4:
            rd["parent"] = SR.parent_named([dict(x, classes=x.get("classes")) for x in per_arm["pc"]], r, 4, panels_format)
        phase_td = [test_value(x, math.inf) for x in per_arm["pc"]]
        nb = set(out["blindness"]["not_blind"])
        off_td = [(-math.inf if x["run"] in nb else test_value(x, -math.inf)) for x in per_arm["off"]]
        rd["one_test"] = P.one_test(phase_td, off_td)
        rd["one_test"]["not_blind_entered_as_missing"] = sorted(nb)
        rd["one_test"]["td_j_report_only"] = dict(
            phase=[(x.get("td") or {}).get("td_j") for x in per_arm["pc"]],
            off=[(x.get("td") or {}).get("td_j") for x in per_arm["off"]])
        if comparators is not None:
            rd["one_test"]["stance6_report_only"] = dict(
                td=[comparators["runs"][s]["td"]["td"] for s in C.SEEDS],
                td_j=[comparators["runs"][s]["td"]["td_j"] for s in C.SEEDS])
        A = sum(1 for x in per_arm["pc"] if x.get("classes") and x["classes"]["any_alarm"])
        nA = sum(1 for x in per_arm["pc"] if x.get("classes") and (not x["classes"]["stopped"] or x["classes"]["any_alarm"]))
        Ao = sum(1 for x in per_arm["off"] if x.get("classes") and x["classes"]["any_alarm"])
        nAo = sum(1 for x in per_arm["off"] if x.get("classes")
                  and (not x["classes"]["stopped"] or x["classes"]["any_alarm"]))
        rd["statistic_a"] = dict(A=A, n_A=nA, text=f"{A} of {nA}" if nA else "no run counted",
                                 off=dict(A=Ao, n_A=nAo, text=f"{Ao} of {nAo}" if nAo else "no run counted",
                                          with_lost=dict(A=Ao + len(lost_any["off"]), n_A=nAo + len(lost_any["off"]),
                                                         runs=lost_any["off"])),
                                 fisher_vs_off=round(P.fisher_lower(A, nA, Ao, nAo), 4) if (nA and nAo) else None,
                                 fisher_vs_stance6_2_of_3=round(P.fisher_lower(A, nA, 2, 3), 4) if nA else None,
                                 fisher_vs_ltsb_r9_2_of_5=round(P.fisher_lower(A, nA, 2, 5), 4) if nA else None,
                                 with_lost=dict(A=A + len(lost_any["pc"]), n_A=nA + len(lost_any["pc"]),
                                                runs=lost_any["pc"]),
                                 note="report only; never tested")
    out["reading"] = rd
    prov = provisional(pins, approved, notes, out["runs"], r, None, runlog)
    out["provisional"] = prov
    status = EXIT_INCOMPLETE if out["incomplete"] else (EXIT_PROVISIONAL if prov else EXIT_FINAL)
    out["status"] = {EXIT_FINAL: "FINAL", EXIT_INCOMPLETE: "INCOMPLETE (no line read)",
                     EXIT_PROVISIONAL: "PROVISIONAL (read, printed, not acted on)"}[status]
    return out, status


def _fmt_step(s):
    return "none" if s is None else f"{s:,}"


def _beside_s6(cs):
    """STANCE6's same seed, REPORT ONLY (PC AMENDMENT)."""
    if not cs:
        return ""
    c = cs.get("classes", {})
    return (f"[STANCE6 s{cs.get('seed', '')} REPORT ONLY: {c.get('cls')} {c.get('kind') or ''}, GATE {c.get('gate')}, "
            f"{c.get('outcome')}, TD {(cs.get('td') or {}).get('td')}, habit {(cs.get('habit') or {}).get('status')} "
            f"{(cs.get('habit') or {}).get('saturating')}]")


def _run_lines(x, beside="") -> list:
    """One seed's run of an arm: its class line, early signs, MEAN_a, habit sites and the report-only extras."""
    c = x.get("classes")
    tag = f"  {x['arm']} s{x['seed']} {x['run']}"
    if not c:
        miss = "+infinity" if x["arm"] == "pc" else "-infinity"
        return [f"{tag}: {x['verdict']} (no class; THE ONE TEST: missing, {miss}) {beside}"]
    e = x["early"]
    L = [f"{tag}: {x['verdict']}; TAIL CLASS {c['cls']}{' (' + c['kind'] + ')' if c['kind'] else ''}"
         f"; GATE {c['gate']} {P.GATE_LABEL} {c['gate_why'] or ''}; OUTCOME {c['outcome']} {c['reasons']}; HABIT ELSEWHERE "
         f"{x['habit']['status']} saturating {x['habit']['saturating']} drive-only {x['habit']['drive_only']}; "
         f"TD {x['td']['td']} (J window {x['td']['td_j']}; missing {x['td']['missing']}) {beside}",
         f"      early signs: {json.dumps(e['firsts'])}; CMD_L >= 0.60 in {e['n_cmd_l_ge_060']} evaluations; max "
         f"{e['cmd_l_max']}; last {e['cmd_l_last']}; M5 TAIL-flagged {len(e.get('m5_tail_flagged') or [])}, TAIL "
         f"ALARMs {len(e.get('m5_tail_alarms') or [])} (first {_fmt_step((e.get('m5_tail_alarms') or [None])[0])})",
         f"      MEAN_a {json.dumps(c['mean'])}; PARKED {c['parked']}; DRIVEN {c['driven']}; tail alarms "
         f"{c['tail_alarm_detail']}; trace rail {c['trace_rail']}; ANY ALARMs {len(c['any_alarms'])}"]
    for n, v in x["habit"]["per_actuator"].items():
        if v["sites"]:
            L.append(f"      habit {n}: {v['status']} sites {v['sites']} own alarms {v['own_alarms']} ANY alarms "
                     f"{v['any_alarms_naming']} traced {v['trace_sites']} drive {v['drive']}")
    L.append(f"      report only: Q4 {json.dumps(x['q4'])}; on-policy tail_3 {json.dumps(x['probe_tail3'])}; guard "
             f"{json.dumps(x['guard'])}")
    return L


def _brief_class(x):
    c = x.get("classes")
    if not c:
        return f"{x['run']} {x['verdict']}"
    return (f"{x['run']}: {c['cls']}{' (' + c['kind'] + ')' if c['kind'] else ''}, GATE {c['gate']}, {c['outcome']}, "
            f"habit {(x.get('habit') or {}).get('status')}, TD {(x.get('td') or {}).get('td')}")


def render(out) -> str:
    L = [VERSION, out["tools"], f"<PCR> {out['round']}"]
    for i, line in out["others"]:
        L.append(f"RUNLOG line {i} names a PC RUN_ID and is not a {C.NOTE_KIND} note of a round RUN_ID (printed; read by "
                 f"no rule): {line[:160]}")
    L.append(f"F2 timing (report only): {json.dumps(out['f2_timing'])}")
    for x in out.get("records") or []:
        L.append(f"RECORDS (report only): {x}")
    for t, v in out.get("launch_checks", {}).items():
        L.append(f"LAUNCH CHECK {t} (from F0's files): faults {v['faults'] or 'none'}; copies within 60 min {v['within_60']}")
    for d in out["seeds"]:
        L.append(f"SEED {d['seed']} ({d['arm']}): run of the round {d['run']}; launched {d['launched']}; {'; '.join(d['why'])}")
    bl = out.get("blindness") or {}
    L.append(f"OUTCOME BLINDNESS (every switch-off run starts before any phase run's results are opened): first opening "
             f"{json.dumps(bl.get('first_opening'))}; not blind (enter THE ONE TEST as missing) {bl.get('not_blind')}")
    for s, v in (bl.get("seeds") or {}).items():
        L.append(f"    s{s}: phase {v['phase_run']} launched {v['phase_launch']}; switch-off {v['off_run']} launched "
                 f"{v['off_launch']}; phase-off start gap (report only) {v['gap_minutes_report_only']} min; blind "
                 f"{v['blind']}")
    for t, det in out["runs"].items():
        L.append(f"RUN {t} ({det.get('arm')}): {det.get('verdict')}  F2 {json.dumps(det.get('f2', {}).get('C'))}"
                 + (f"  re-fetch {json.dumps(det['refetch']['C'])}" if det.get("refetch") else ""))
        comp = det.get("completeness")
        if comp:
            for ln in CP.render_facts(f"  C1-C8 of {t}", comp)[1:]:
                L.append("  " + ln)
        for p in det.get("presence") or []:
            for ln in CP.render_presence(p):
                L.append("  " + ln)
        if det.get("d16"):
            dd = det["d16"]
            L.append(f"    D-16 LOST (report only; not in a class; in the reading only as a PRESENCE RAIL): triggers "
                     f"{dd['triggers']}; shown {json.dumps(dd['shown'], default=str)[:300]}")
    comps = (out.get("comparators") or {}).get("runs") or {}
    offs = {x["seed"]: x for x in out.get("per_seed_off") or []}
    L.append("PER-RUN TABLE (the phase runs; the same-seed concurrent switch-off run beside; STANCE6 beside, report only):")
    for x in out["per_seed"]:
        cs = dict(comps.get(x["seed"]) or {}, seed=x["seed"]) if comps.get(x["seed"]) else None
        xo = offs.get(x["seed"])
        L += _run_lines(x, (f"[switch-off s{x['seed']}: {_brief_class(xo)}] " if xo else "") + _beside_s6(cs))
    L.append("THE CONCURRENT SWITCH-OFF RUNS (the one test's control; classes printed beside the phase runs', deciding "
             "no line):")
    for x in out.get("per_seed_off") or []:
        L += _run_lines(x)
    for p in out["presence_rails"]:
        L.append(f"PRESENCE RAIL (phase) {p['run']} on {p['fetch']}: {json.dumps(p['detail'])}")
    for p in out.get("presence_rails_off") or []:
        L.append(f"PRESENCE RAIL (switch-off, beside) {p['run']} on {p['fetch']}: {json.dumps(p['detail'])}")
    rep = out.get("replication") or {}
    L.append(f"OFF-vs-STANCE6 REPLICATION ({rep.get('note')}):")
    for x in rep.get("rows") or []:
        L.append(f"    s{x['seed']}: switch-off {x['run']} {x.get('off') or x['verdict']}; STANCE6 {x.get('stance6')}")
    if rep.get("totals"):
        L.append(f"    totals: {json.dumps(rep['totals'])}")
    if comps:
        L.append("STANCE6 (historical, REPORT ONLY: decides nothing; recomputed; score.json "
                 + out["comparators"]["score_json_sha256"][:16] + "..., fetch manifest "
                 + out["comparators"]["fetch_manifest_sha256"][:16] + "...):")
        for s, cs in comps.items():
            L.append(f"  s{s} {cs['run']}: {cs['classes']['cls']} {cs['classes']['kind'] or ''}; first TAIL ALARM "
                     f"{_fmt_step(cs['classes']['first_tail_alarm'])} ({cs['classes']['tail_alarms']}); GATE "
                     f"{cs['classes']['gate']}; {cs['classes']['outcome']}; ANY ALARMs {cs['classes']['any_alarms']}; "
                     f"habit {cs['habit']['status']} {cs['habit']['sites']}; TD {cs['td']['td']:.6f} (J {cs['td']['td_j']:.6f});"
                     f" early {json.dumps(cs['early']['firsts'])}; CMD_L >= 0.60 in {cs['early']['n_cmd_l_ge_060']}; "
                     f"on-policy tail_3 {cs['probe_tail3']}; mean return {cs['mean_return']:.1f}; falls {cs['falls']} of "
                     f"{cs['rows']}")
    else:
        L.append(f"STANCE6 (REPORT ONLY): not read ({out.get('comparators_error')}); it decides nothing")
    if out["incomplete"]:
        L.append("READING: incomplete; no line is read. Open:")
        for x in out["incomplete"]:
            L.append(f"    {x}")
    else:
        rd = out["reading"]
        L.append(f"READING: line {rd['line']}: {rd['text']}")
        L.append(f"    {rd['off_text']}")
        if rd.get("no_rail_chance"):
            L.append(f"    {rd['no_rail_chance']}")
        L.append(f"    r {rd['r']} (COMPLETE {rd['r_complete']}, PRESENCE {rd['presence_rails']}), u {rd['u']}, p {rd['p']}, "
                 f"g {rd['g']}, hs {rd['hs']}, hd {rd['hd']}")
        for f in rd["first_tail_alarms"]:
            L.append(f"    first TAIL ALARM {f}")
        for nm in rd["named"]:
            L.append(f"    {nm}")
        if rd.get("parent"):
            L.append(f"    stance3's parent rule (named, never offered): {json.dumps(rd['parent'])}")
        ot = rd["one_test"]
        L.append(f"    THE ONE TEST (phase vs the concurrent switch-off runs): {ot['text']} ({ot['note']}); phase TD "
                 f"{ot['phase']} (missing, +infinity: {ot['phase_missing']}); switch-off TD {ot['off']} (missing, "
                 f"-infinity: {ot['off_missing']}; not blind {ot['not_blind_entered_as_missing']}); J-window TD (report "
                 f"only) {json.dumps(ot['td_j_report_only'])}; STANCE6 (report only) "
                 f"{json.dumps(ot.get('stance6_report_only'))}; {ot['caveat']}")
        st = rd["statistic_a"]
        L.append(f"    Statistic A (report only): phase {st['text']}; switch-off {st['off']['text']} (one-sided Fisher p "
                 f"{st['fisher_vs_off']}); beside: STANCE6's 2 of 3 {st['fisher_vs_stance6_2_of_3']}, LTSB r9's 2 of 5 "
                 f"{st['fisher_vs_ltsb_r9_2_of_5']}; with LOST runs phase {st['with_lost']}, switch-off "
                 f"{st['off']['with_lost']}")
        L.append("    Q1-Q3 counts (phase): RAIL r = %d, PARKED p = %d, GATE PASS g = %d, SATURATING habit elsewhere hs = %d, "
                 "DRIVE ONLY hd = %d" % (rd["r"], rd["p"], rd["g"], rd["hs"], rd["hd"]))
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
        print(f"{VERSION}\nINPUT ERROR: {exc}")
        return EXIT_INPUT
    print(render(out))
    if a.json:
        Path(a.json).write_text(json.dumps(SR._jsonable(out), indent=1, default=str) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
