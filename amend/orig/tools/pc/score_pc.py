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

VERSION = "score_pc pc-2 (2026-11-01 fixer: printed wording; PC criteria THE READING lines 1-5, THE ONE TEST, C1-C8)"
EXIT_FINAL, EXIT_INCOMPLETE, EXIT_PROVISIONAL, EXIT_INPUT = 0, 10, 11, 2
VALIDATIONS = ("VP0", "VP1", "VP2", "VP3", "VP4")
#: VP0, VP2, VP3 are before the launch (<PCR>/APPROVED "validations"); VP1, VP4 and the F2 tools' sha256 are pinned
#: before F2 (CHOICE TOOLS) in <PCR>/F2_PINS.json (written once, its sha256 in a RUNLOG line before the F2-START lines).
PRE_LAUNCH_VALIDATIONS = ("VP0", "VP2", "VP3")
PRE_F2_VALIDATIONS = ("VP1", "VP4")
F2_PINS_NAME = "F2_PINS.json"
C_KEYS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")


# ---------------------------------------------------------------------------------------------------- LEAN-R (PC)
def seed_run(seed, notes, f0_checks, round_dir=None) -> dict:
    """score_r10round.seed_run with the PC RUN_ID (C.RUN_ID_PREFIX) and the PC stop faults (1)-(5), (7), (8)
    (C.STOP_FAULTS); every other clause the round's, unchanged."""
    first = f"{C.RUN_ID_PREFIX}{seed}"
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
    if comparators_err:
        out.append(f"comparators not read as pinned: {comparators_err}")
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
    seeds = [seed_run(s, notes, f0_checks, r) for s in C.SEEDS]
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
    lost_any, presence_rails = [], []
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
        det = dict(f2=dict(verdict=first["verdict"], C=_brief(first)))
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
            x["classes"] = SR.classes_of(comp)
            x.update(run_extras(stores_of[t], comp))
            x["finished"] = bool(comp.get("finished"))
        per_seed.append(x)
    out["per_seed"] = per_seed
    out["presence_rails"] = presence_rails
    # the comparators (recomputed from their pinned inputs; report beside every line, and THE ONE TEST's values)
    comp_err = None
    if comparators is None:
        try:
            comparators = P.comparators(C, pins)
        except (OSError, ValueError, KeyError) as exc:
            comp_err = f"{type(exc).__name__}: {exc}"
            comparators = None
    out["comparators"] = comparators
    if comparators is None:
        out["incomplete"].append(f"the comparators cannot be read: {comp_err}")
    rd = None
    if not out["incomplete"]:
        rd = P.reading(per_seed, presence_rails)
        if rd["line"] == 4:
            rd["parent"] = SR.parent_named([dict(x, classes=x.get("classes")) for x in per_seed], r, 4, panels_format)
        pc_td = [(x["td"]["td"] if (x.get("verdict") == "COMPLETE" and x.get("finished") and x.get("td")
                                    and x["td"]["td"] is not None) else math.inf) for x in per_seed]
        comp_td = [comparators["runs"][s]["td"]["td"] for s in C.SEEDS]
        rd["one_test"] = P.one_test(pc_td, comp_td)
        rd["one_test"]["td_j_report_only"] = dict(
            pc=[(x.get("td") or {}).get("td_j") for x in per_seed],
            comparators=[comparators["runs"][s]["td"]["td_j"] for s in C.SEEDS])
        A = sum(1 for x in per_seed if x.get("classes") and x["classes"]["any_alarm"])
        nA = sum(1 for x in per_seed if x.get("classes") and (not x["classes"]["stopped"] or x["classes"]["any_alarm"]))
        rd["statistic_a"] = dict(A=A, n_A=nA, text=f"{A} of {nA}" if nA else "no run counted",
                                 fisher_vs_stance6_2_of_3=round(P.fisher_lower(A, nA, 2, 3), 4) if nA else None,
                                 fisher_vs_ltsb_r9_2_of_5=round(P.fisher_lower(A, nA, 2, 5), 4) if nA else None,
                                 with_lost=dict(A=A + len(lost_any), n_A=nA + len(lost_any), runs=lost_any),
                                 note="report only; never tested")
    out["reading"] = rd
    prov = provisional(pins, approved, notes, out["runs"], r, comp_err, runlog)
    out["provisional"] = prov
    status = EXIT_INCOMPLETE if out["incomplete"] else (EXIT_PROVISIONAL if prov else EXIT_FINAL)
    out["status"] = {EXIT_FINAL: "FINAL", EXIT_INCOMPLETE: "INCOMPLETE (no line read)",
                     EXIT_PROVISIONAL: "PROVISIONAL (read, printed, not acted on)"}[status]
    return out, status


def _fmt_step(s):
    return "none" if s is None else f"{s:,}"


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
        L.append(f"SEED {d['seed']}: run of the round {d['run']}; launched {d['launched']}; {'; '.join(d['why'])}")
    for t, det in out["runs"].items():
        L.append(f"RUN {t}: {det.get('verdict')}  F2 {json.dumps(det.get('f2', {}).get('C'))}"
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
    L.append("PER-RUN TABLE (PC runs; STANCE6 same seed beside):")
    comps = (out.get("comparators") or {}).get("runs") or {}
    for x in out["per_seed"]:
        cs = comps.get(x["seed"]) or {}
        c = x.get("classes")
        beside = (f"[STANCE6 s{x['seed']}: {cs.get('classes', {}).get('cls')} {cs.get('classes', {}).get('kind') or ''}, "
                  f"GATE {cs.get('classes', {}).get('gate')}, {cs.get('classes', {}).get('outcome')}, TD "
                  f"{(cs.get('td') or {}).get('td')}, habit {(cs.get('habit') or {}).get('status')} "
                  f"{(cs.get('habit') or {}).get('saturating')}]") if cs else ""
        if not c:
            L.append(f"  s{x['seed']} {x['run']}: {x['verdict']} (no class; TD +infinity in THE ONE TEST) {beside}")
            continue
        e = x["early"]
        L.append(f"  s{x['seed']} {x['run']}: {x['verdict']}; TAIL CLASS {c['cls']}{' (' + c['kind'] + ')' if c['kind'] else ''}"
                 f"; GATE {c['gate']} {P.GATE_LABEL} {c['gate_why'] or ''}; OUTCOME {c['outcome']} {c['reasons']}; HABIT ELSEWHERE "
                 f"{x['habit']['status']} saturating {x['habit']['saturating']} drive-only {x['habit']['drive_only']}; "
                 f"TD {x['td']['td']} (J window {x['td']['td_j']}; missing {x['td']['missing']}) {beside}")
        L.append(f"      early signs: {json.dumps(e['firsts'])}; CMD_L >= 0.60 in {e['n_cmd_l_ge_060']} evaluations; max "
                 f"{e['cmd_l_max']}; last {e['cmd_l_last']}; M5 TAIL-flagged {len(e.get('m5_tail_flagged') or [])}, TAIL "
                 f"ALARMs {len(e.get('m5_tail_alarms') or [])} (first {_fmt_step((e.get('m5_tail_alarms') or [None])[0])})")
        L.append(f"      MEAN_a {json.dumps(c['mean'])}; PARKED {c['parked']}; DRIVEN {c['driven']}; tail alarms "
                 f"{c['tail_alarm_detail']}; trace rail {c['trace_rail']}; ANY ALARMs {len(c['any_alarms'])}")
        for n, v in x["habit"]["per_actuator"].items():
            if v["sites"]:
                L.append(f"      habit {n}: {v['status']} sites {v['sites']} own alarms {v['own_alarms']} ANY alarms "
                         f"{v['any_alarms_naming']} traced {v['trace_sites']} drive {v['drive']}")
        L.append(f"      report only: Q4 {json.dumps(x['q4'])}; on-policy tail_3 {json.dumps(x['probe_tail3'])}; guard "
                 f"{json.dumps(x['guard'])}")
    for p in out["presence_rails"]:
        L.append(f"PRESENCE RAIL {p['run']} on {p['fetch']}: {json.dumps(p['detail'])}")
    if comps:
        L.append("THE COMPARATORS (STANCE6, recomputed; score.json " + out["comparators"]["score_json_sha256"][:16]
                 + "..., fetch manifest " + out["comparators"]["fetch_manifest_sha256"][:16] + "...):")
        for s, cs in comps.items():
            L.append(f"  s{s} {cs['run']}: {cs['classes']['cls']} {cs['classes']['kind'] or ''}; first TAIL ALARM "
                     f"{_fmt_step(cs['classes']['first_tail_alarm'])} ({cs['classes']['tail_alarms']}); GATE "
                     f"{cs['classes']['gate']}; {cs['classes']['outcome']}; ANY ALARMs {cs['classes']['any_alarms']}; "
                     f"habit {cs['habit']['status']} {cs['habit']['sites']}; TD {cs['td']['td']:.6f} (J {cs['td']['td_j']:.6f});"
                     f" early {json.dumps(cs['early']['firsts'])}; CMD_L >= 0.60 in {cs['early']['n_cmd_l_ge_060']}; "
                     f"on-policy tail_3 {cs['probe_tail3']}; mean return {cs['mean_return']:.1f}; falls {cs['falls']} of "
                     f"{cs['rows']}")
    if out["incomplete"]:
        L.append("READING: incomplete; no line is read. Open:")
        for x in out["incomplete"]:
            L.append(f"    {x}")
    else:
        rd = out["reading"]
        L.append(f"READING: line {rd['line']}: {rd['text']}")
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
        L.append(f"    THE ONE TEST: {ot['text']} ({ot['note']}); PC TD {ot['pc']} (+infinity: {ot['pc_infinite']}); STANCE6 TD "
                 f"{ot['comparators']}; J-window TD (report only) {json.dumps(ot['td_j_report_only'])}; {ot['caveat']}")
        st = rd["statistic_a"]
        L.append(f"    Statistic A (report only): {st['text']}; one-sided Fisher p vs STANCE6's 2 of 3 "
                 f"{st['fisher_vs_stance6_2_of_3']}, vs LTSB r9's 2 of 5 {st['fisher_vs_ltsb_r9_2_of_5']}; with LOST "
                 f"runs {st['with_lost']}")
        L.append("    Q1-Q3 counts: RAIL r = %d, PARKED p = %d, GATE PASS g = %d, SATURATING habit elsewhere hs = %d, DRIVE "
                 "ONLY hd = %d" % (rd["r"], rd["p"], rd["g"], rd["hs"], rd["hd"]))
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
