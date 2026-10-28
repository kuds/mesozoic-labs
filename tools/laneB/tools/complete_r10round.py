"""T. rex r10 round: the completeness tool (criteria lean v5: launch check, C1-C7, PRESENCE p1-p3, D-16).  Report only.

    PY -I -B complete_r10round.py launch-check --run RUN_ID [--round ROUND] [--f0 DIR] [--runlog P] [--pins P] [--json O]
    PY -I -B complete_r10round.py run --run RUN_ID [--round ROUND] [--f2 DIR] [--refetch DIR] [--panels DIR]
                                      [--notebook P] [--runlog P] [--pins P] [--json O]

Every fact is computed from the files a fetch holds (fetch_r10round.py's store: <ROUND>/fetch/<FETCH>/<RUN_ID>/) and
from RUNLOG's note lines ("<date -u> STANCE6 <KIND> <RUN_ID> <fields>"), never transcribed; each is printed with the
value read.  A fact whose file the fetch does not hold as the rule requires is "not established" (C-facts) or "not
computed" (launch check), never assumed.

Launch check (RUN HANDLING), from F0's files (each read only when two copies agree; a third on a mismatch):
  (1) provenance.json repository_commit and 01_stance/stage_config.json git_commit = the branch HEAD RUNLOG's LAUNCH
      line records (head=), repository_dirty false; (2) run.seed the RUN_ID's; (3) run.hyperparameters_sha256,
      task_fingerprint.task_sha256 and the plant_identity block the pins (as the file writes them); (4)
      zero_action_baseline.json results.tyrannosaurus_rex.anatomical.reward_mean not within 50 of 3766.6; (5) the first
      ledger rows carry screen_v2_reasons and #640's saturated_actuator columns; (6) one provenance "sessions" entry;
      (7) provenance run_id the RUN_ID.  A fact whose file F0 does not hold is "not computed" (no stop).
C1-C7 (CLASSES, COMPLETE RUN) on F2, or on the re-fetch when one is made for a seed's run of the round; PRESENCE
p1-p3 on each fetch (two agreeing copies of provenance.json, stage_config.json and the ledger); D-16 for a LOST run.
Exit codes: 0 printed; 2 input error (launch-check: also a pin it compares unfilled, fixer R-m3).

Fixer, 2026-10-26 (review findings):
  R-i1  launch-check (5) on a ledger with a header and no row is "not computed" (no first rows to read).
  R-i5  recorded only: the pinned TEL columns missing from F0's ledger header, and BEHAVIOR as the notebook printed it
        when a CHECK line carries behavior="..." (the LAUNCH line's BEHAVIOR is what was set).
  R-m3  launch-check refuses (exit 2) when hyperparameters_sha256, task_sha256 or plant_identity is unfilled.
  R-B4  a label's M1/M3 output (or the measuring tool's refusal record) counts for C7 only when its record measured
        that label's C3-verified pair (checkpoint and vecnorm sha256), on the checkpoint's own num_timesteps (E for
        final), not as a validation input, and (when the runner filed it) under this RUN_ID; otherwise that
        measurement is absent (C7 PENDING), the mismatch printed.
  R-m4  PRESENCE per fetch is three-valued: rail_decided True / False / None (a p-fact not established: a pin unfilled).
v5 (criteria lean v5, RUNLOG line 657; tool updater 2026-10-26):
  R-m1  R1 of a STOPPED run that ended before 10,485,760 is the pair at the last multiple of 262,144 at or before E,
        and each RESOLUTION checkpoint the grid's (CLASSES): C3 fails when that pair is not in the manifest.
  F-R1  a refused M1 (m1_refused.json) at a label C7 names fails C7: the run is LOST; a refused M3 at R2
        (m3_refused.json) is the certificate "refused" (GATE FAIL, its message the failing criterion) and counts as
        R2's certificate for C7.  A runner preflight refusal writes nothing: the measurement is absent (PENDING) until
        it is made again with the right inputs.
  O-2   C1 reads every ACTION line of the RUN_ID ("ACTION <RUN_ID> what=<resume|restart|relaunch|cell>") as an action.
  O-5   P-K3's launch settings are recorded from the LAUNCH line (no F0 file holds them).
  O-16  the facts the re-fetch rule reads ("fails C4 or C6 on files matching the manifest") are exposed:
        C4_on_manifest_files, C6_on_manifest_files.
  O-18  C4's started_at within 60 minutes before or after RUNLOG's launch time.
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


def fact(name, ok, value=None, note=None):
    """ok: True (holds), False (fails), None (not established / not computed)."""
    return dict(name=name, ok=ok, value=value, note=note)


def status_of(f):
    return "PASS" if f["ok"] is True else ("FAIL" if f["ok"] is False else "NOT ESTABLISHED")


def base_seed(run_id):
    for rid, s in C.RUN_IDS.items():
        if run_id == rid or run_id.startswith(rid + "_"):
            return s
    return None


def launch_note(notes, run_id):
    """The run's launch line (LAUNCH, or RELAUNCH for a <RUN_ID>_b): the first one."""
    return C.first_note(notes, run_id, ("LAUNCH", "RELAUNCH"))


def _json(b):
    return None if b is None else C.load_json_bytes(b)


# ------------------------------------------------------------------------------------------------- launch check
def launch_check(run_id, store, notes, pins) -> dict:
    """The launch-check facts (1)-(7) from F0's files, P-K3's launch settings as RUNLOG records them, and what is
    recorded only.  faults: the computed facts that fail."""
    ln = launch_note(notes, run_id)
    head = ln["fields"].get("head") if ln else None
    seed = base_seed(run_id)
    pb, ph = store.agreed("provenance.json")
    sb, sh = store.agreed("01_stance/stage_config.json")
    zb, zh = store.agreed("zero_action_baseline.json")
    lb, lh = store.agreed("01_stance/eval_episodes.csv")
    prov, sc, zab = _json(pb), _json(sb), _json(zb)
    F = {}
    # (1)
    if prov is None or sc is None:
        F[1] = fact("(1) commits = branch HEAD, not dirty", None, None,
                    f"not computed: provenance.json {ph if pb is None else 'held'}; stage_config.json "
                    f"{sh if sb is None else 'held'}")
    elif head is None:
        F[1] = fact("(1) commits = branch HEAD, not dirty", None, None, "not computed: RUNLOG's LAUNCH line records no head=")
    else:
        v = dict(repository_commit=prov.get("repository_commit"), git_commit=sc.get("git_commit"),
                 repository_dirty=prov.get("repository_dirty"), head=head)
        F[1] = fact("(1) commits = branch HEAD, not dirty", v["repository_commit"] == head and v["git_commit"] == head
                    and v["repository_dirty"] is False, v)
    # (2)
    if sc is None:
        F[2] = fact("(2) run.seed", None, None, f"not computed: stage_config.json {sh}")
    else:
        F[2] = fact("(2) run.seed", C.dig(sc, "run.seed") == seed, dict(run_seed=C.dig(sc, "run.seed"), want=seed))
    # (3)
    if sc is None:
        F[3] = fact("(3) hyperparameters, task, plant = pins", None, None, f"not computed: stage_config.json {sh}")
    else:
        want = dict(hyperparameters_sha256=C.pin(pins, "hyperparameters_sha256"), task_sha256=C.pin(pins, "task_sha256"),
                    plant_identity=C.pin(pins, "plant_identity"))
        got = dict(hyperparameters_sha256=C.dig(sc, "run.hyperparameters_sha256"),
                   task_sha256=C.dig(sc, "task_fingerprint.task_sha256"), plant_identity=sc.get("plant_identity"))
        if any(v is None for v in want.values()):
            F[3] = fact("(3) hyperparameters, task, plant = pins", None, got, "not computed: a pin is unfilled")
        else:
            F[3] = fact("(3) hyperparameters, task, plant = pins", all(got[k] == want[k] for k in want),
                        dict(got={k: (v if k != "plant_identity" else "(block)") for k, v in got.items()},
                             equal={k: got[k] == want[k] for k in want}))
    # (4)
    rm = C.dig(zab, "results.tyrannosaurus_rex.anatomical.reward_mean") if isinstance(zab, dict) else None
    if zab is None:
        F[4] = fact("(4) zero-action reward_mean not within 50 of 3766.6", None, None,
                    f"not computed: zero_action_baseline.json {zh}")
    elif C.num(rm) is None:
        F[4] = fact("(4) zero-action reward_mean not within 50 of 3766.6", None, rm,
                    "not computed: no results.tyrannosaurus_rex.anatomical.reward_mean")
    else:
        F[4] = fact("(4) zero-action reward_mean not within 50 of 3766.6", abs(float(rm) - C.STATUE_MAIN) > C.STATUE_BAND,
                    float(rm))
    # (5)
    if lb is None:
        F[5] = fact("(5) ledger carries screen_v2_reasons and #640's columns", None, None,
                    f"not computed: eval_episodes.csv {lh}")
    else:
        try:
            header, rows = C.read_ledger_bytes(lb)
        except C.InputError as exc:
            header, rows = [], []
            F[5] = fact("(5) ledger carries screen_v2_reasons and #640's columns", None, None, f"not computed: {exc}")
        if 5 not in F and not rows:
            F[5] = fact("(5) ledger carries screen_v2_reasons and #640's columns", None, dict(rows=0),
                        "not computed: the ledger holds no row yet (no first rows to read; fixer R-i1)")
        if 5 not in F:
            need = ("screen_v2_reasons",) + C.SAT640_COLUMNS
            missing = [c for c in need if c not in header]
            F[5] = fact("(5) ledger carries screen_v2_reasons and #640's columns", not missing,
                        dict(missing=missing, rows=len(rows)))
    # (6), (7)
    if prov is None:
        F[6] = fact("(6) one provenance sessions entry", None, None, f"not computed: provenance.json {ph}")
        F[7] = fact("(7) provenance run_id = RUN_ID", None, None, f"not computed: provenance.json {ph}")
    else:
        ss = prov.get("sessions")
        F[6] = fact("(6) one provenance sessions entry", isinstance(ss, list) and len(ss) == 1,
                    len(ss) if isinstance(ss, list) else ss)
        F[7] = fact("(7) provenance run_id = RUN_ID", prov.get("run_id") == run_id, prov.get("run_id"))
    faults = sorted(k for k, f in F.items() if f["ok"] is False)
    # when F0's copies were made against the launch (a fault first seen after 60 minutes stops nothing)
    times = [C.parse_utc(r.get("utc")) for r in store.manifest_rows() if r.get("status") == "COPY"]
    times = [t for t in times if t is not None]
    within = None
    if ln and times:
        within = C.minutes(ln["time"], max(times)) <= C.LAUNCH_WINDOW_MIN
    rec = {}
    if ln:
        fl = ln["fields"]
        rec = dict(runtime=fl.get("runtime"), BEHAVIOR=fl.get("BEHAVIOR"), K3_PANEL=fl.get("K3_PANEL"),
                   K3_PANEL_SIGMA_FILE=fl.get("K3_PANEL_SIGMA_FILE"), K3_PANEL_SIGMA_KEY=fl.get("K3_PANEL_SIGMA_KEY"))
        rec["P-K3 as RUNLOG records it"] = dict(
            K3_PANEL_ok=fl.get("K3_PANEL") == "True", sigma_file_ok=fl.get("K3_PANEL_SIGMA_FILE") == C.SIGMA_FILE_COLAB,
            sigma_key_ok=fl.get("K3_PANEL_SIGMA_KEY") == C.SIGMA_KEYS.get(seed))
        rec["zero_action reward_mean"] = rm
    tel = C.pin(pins, "tel_columns")
    if lb is None:
        rec["TEL columns missing from F0's ledger header"] = f"not computed: eval_episodes.csv {lh}"
    elif tel is None:
        rec["TEL columns missing from F0's ledger header"] = "not computed: pin tel_columns unfilled"
    else:
        try:
            hdr = C.read_ledger_bytes(lb)[0]
            rec["TEL columns missing from F0's ledger header"] = [c for c in tel if c not in hdr]
        except C.InputError as exc:
            rec["TEL columns missing from F0's ledger header"] = f"not computed: {exc}"
    chk = [n for n in C.notes_for(notes, run_id, ("CHECK",)) if "behavior" in n["fields"]]
    rec["BEHAVIOR as the notebook printed it"] = (chk[-1]["fields"]["behavior"] if chk else
                                                  "not recorded (no CHECK line with behavior=\"...\")")
    p_k3 = (rec.get("P-K3 as RUNLOG records it") if ln else "no LAUNCH line: P-K3's launch settings not recorded")
    return dict(run_id=run_id, launch=ln and ln["text"], facts=F, faults=faults,
                f0_copies_within_60_min=within,
                p_k3_launch_settings=dict(source="the LAUNCH line (no F0 file holds them; FETCHES F0, v5 O-5)",
                                          settings=p_k3),
                recorded_only=rec)


# ------------------------------------------------------------------------------------------------- C1
def c1(run_id, notes) -> dict:
    """C1: exactly one launch (a LAUNCH or, for a _b, RELAUNCH line of the RUN_ID), no launch-check stop (STOP or
    TRIGGER-d) and no action (v5: every ACTION line of the RUN_ID, "read by C1 as an action").  THE ENDING
    PROCEDURE's own steps are noted as TRIGGER-*/DELETED/FOUND-GONE lines, never as ACTION."""
    launches = C.notes_for(notes, run_id, ("LAUNCH", "RELAUNCH"))
    stops = C.notes_for(notes, run_id, ("STOP", "TRIGGER-d"))
    actions = C.notes_for(notes, run_id, ("ACTION",))
    v = dict(launches=[n["line"] for n in launches], launch_check_stops=[n["line"] for n in stops],
             actions=[dict(line=n["line"], what=n["fields"].get("what"),
                           what_not_a_listed_kind=n["fields"].get("what") not in C.ACTION_WHATS) for n in actions],
             lines={n["line"]: n["kind"] for n in C.notes_for(notes, run_id)})
    return fact("C1 one launch, no launch-check stop, no other action",
                len(launches) == 1 and not stops and not actions, v,
                "C1 reads the RUN_ID's LAUNCH/RELAUNCH lines, its launch-check stops (STOP, TRIGGER-d) and its ACTION "
                "lines (what=resume|restart|relaunch|cell)")


# ------------------------------------------------------------------------------------------------- C2
def c2(run_id, store):
    b, how = store.agreed("artifact_manifest.json")
    if b is None:
        return fact("C2 artifact_manifest.json", False, None, f"no manifest read: {how}"), None
    m = C.load_json_bytes(b)
    if m is None:
        return fact("C2 artifact_manifest.json", False, None, f"{how}; not JSON"), None
    probs = C.manifest_schema_problems(m)
    v = dict(copies=how, sha256=C.sha256_bytes(b), schema_problems=probs, run_id=m.get("run_id"),
             status=m.get("status"), entries=len(m.get("files") or []) if isinstance(m.get("files"), list) else None)
    ok = not probs and m.get("run_id") == run_id
    return fact("C2 artifact_manifest.json", ok, v), (m if not probs else None)


# ------------------------------------------------------------------------------------------------- C3-C6 helpers
def held_c3(store, entries, path):
    """(bytes or None, note): a copy of path equal to its manifest entry (size and the hex after "sha256:")."""
    if path not in entries:
        return None, "not listed in the manifest"
    size, hx = entries[path]
    if hx is None:
        return None, "manifest entry has no sha256 hex"
    return store.matching(path, hx, size if isinstance(size, int) else None)


def c4_facts(run_id, prov, sc, notes, pins, manifest_entries=None, include_manifest_clause=True) -> list:
    """C4's session and configuration facts (p2 = these without the manifest clause)."""
    out = []
    ln = launch_note(notes, run_id)
    head = ln["fields"].get("head") if ln else None
    seed = base_seed(run_id)
    if prov is None:
        out.append(fact("C4 provenance.json read", False, None, "not established: no provenance.json read"))
    else:
        ss = prov.get("sessions")
        one = isinstance(ss, list) and len(ss) == 1 and isinstance(ss[0], dict)
        started = C.parse_utc(ss[0].get("started_at")) if one else None
        resumed = one and ("resumed_at" in ss[0])
        dmin = C.minutes(ln["time"], started) if (ln and started) else None
        out.append(fact("C4 exactly one sessions entry with started_at, no resumed_at",
                        one and started is not None and not resumed,
                        dict(sessions=len(ss) if isinstance(ss, list) else ss, started_at=ss[0].get("started_at") if one else None,
                             resumed_at=resumed)))
        out.append(fact("C4 started_at within 60 minutes before or after RUNLOG's launch time",
                        False if dmin is None else abs(dmin) <= C.LAUNCH_WINDOW_MIN,
                        dict(minutes_from_launch=None if dmin is None else round(dmin, 1),
                             launch=ln["time"].isoformat() if ln else None),
                        None if dmin is not None else "not established: launch line or started_at absent"))
        out.append(fact("C4 provenance repository_commit = branch HEAD, repository_dirty false",
                        False if head is None else (prov.get("repository_commit") == head and prov.get("repository_dirty") is False),
                        dict(repository_commit=prov.get("repository_commit"), repository_dirty=prov.get("repository_dirty"),
                             head=head), None if head else "not established: RUNLOG's LAUNCH line records no head="))
    if sc is None:
        out.append(fact("C4 stage_config.json read", False, None, "not established: no stage_config.json read"))
    else:
        lk = C.keys_at_any_depth(sc, C.LOAD_KEYS)
        out.append(fact("C4 no load key at any depth", not lk, lk))
        out.append(fact("C4 run.seed", C.dig(sc, "run.seed") == seed, dict(got=C.dig(sc, "run.seed"), want=seed)))
        out.append(fact("C4 run.n_envs 4", C.dig(sc, "run.n_envs") == 4, C.dig(sc, "run.n_envs")))
        for key, path in (("hyperparameters_sha256", "run.hyperparameters_sha256"),
                          ("task_sha256", "task_fingerprint.task_sha256")):
            want = C.pin(pins, key)
            got = C.dig(sc, path)
            out.append(fact(f"C4 {path} = pin", None if want is None else got == want, got,
                            None if want is not None else "pin unfilled"))
        want = C.pin(pins, "plant_identity")
        got = sc.get("plant_identity")
        out.append(fact("C4 plant_identity block = pin", None if want is None else got == want,
                        "(block)" if got is not None else None, None if want is not None else "pin unfilled"))
        vw = C.view_sha256(sc)
        want = C.pin(pins, "stage_config_view_sha256")
        out.append(fact("C4 stage_config_view_sha256 = pin", None if want is None else vw == want, vw,
                        None if want is not None else "pin unfilled"))
        out.append(fact("C4 git_commit = branch HEAD", False if head is None else sc.get("git_commit") == head,
                        dict(git_commit=sc.get("git_commit"), head=head),
                        None if head else "not established: RUNLOG's LAUNCH line records no head="))
    if include_manifest_clause:
        failed = C.pin(pins, "v7_failed_on_head")
        listed = None if manifest_entries is None else ("01_stance/k3_panel.csv" in manifest_entries)
        if failed is None:
            out.append(fact("C4 k3_panel.csv absent from the manifest if V7 failed on HEAD", None,
                            dict(k3_panel_csv_listed=listed), "pin v7_failed_on_head unfilled"))
        else:
            out.append(fact("C4 k3_panel.csv absent from the manifest if V7 failed on HEAD",
                            True if failed is False else (listed is False), dict(v7_failed_on_head=failed,
                                                                                  k3_panel_csv_listed=listed)))
    return out


def all_ok(fs):
    if any(f["ok"] is False for f in fs):
        return False
    if any(f["ok"] is None for f in fs):
        return None
    return True


def c3_pairs(entries, E, res):
    """The pairs C3 names for a run ending at E (v5, R-m1: from the 262,144 grid, whatever the manifest lists): R1
    (10,485,760; for a STOPPED run that ended before it, the pair at the last multiple of 262,144 at or before E), the
    J checkpoints up to E, every RESOLUTION checkpoint (from M5 on the ledger: the pair at the second evaluation's step
    when it is a multiple of 262,144, else at the next multiple at or before E, else stage1_final) and the final pair.
    A pair C3 names that the manifest does not list fails C3.  Returns (pairs, r1, js, resolution, notes); entries is
    kept for the callers (the grid does not read it); notes is always empty (v4's R-m1 open points are decided)."""
    if E is None:
        return [], None, [], {}, []
    js = [s for s in C.J_STEPS if s <= E]
    r1 = C.r1_of(E)
    resolution = {s1: C.resolution_checkpoint(s1, None, E) for s1 in (res["needs_resolution"] if res else [])}
    pairs = sorted({x for x in js + ([r1] if r1 else []) + [v for v in resolution.values() if v != "final"]})
    return pairs + ["final"], r1, js, resolution, []


def c3_small(entries):
    small = ["provenance.json", "01_stance/stage_config.json", "01_stance/progress.json", "01_stance/eval_episodes.csv",
             "01_stance/gate_verdict.json"]
    if "01_stance/hop_watch.json" in entries:
        small.append("01_stance/hop_watch.json")
    return small


# ------------------------------------------------------------------------------------------------- one fetch: C1-C7
class PanelLocator:
    """Where M1/M3 outputs live: <panels>/<label>/<kind>/ with label the checkpoint step or "final" and kind m1/m3
    (lane A's measure_r10round.py --run-id RUN_ID --label LABEL)."""

    def __init__(self, panels_dir, fmt="{label}/{kind}"):
        self.dir = Path(panels_dir) if panels_dir else None
        self.fmt = fmt

    def path(self, label, kind):
        return None if self.dir is None else self.dir / self.fmt.format(label=label, kind=kind)

    def refusal(self, label, kind):
        """The measuring tool's own refusal record at label/kind (m1_refused.json / m3_refused.json) when there is no
        output beside it, else None (v5, F-R1)."""
        p = self.path(label, kind)
        out = "episode_trace.json" if kind == "m1" else "stance_heading_report.json"
        if p is None or (p / out).is_file():
            return None
        return C.read_refusal(p, kind)

    def trace(self, label):
        p = self.path(label, "m1")
        if p is not None and not (p / "episode_trace.json").is_file() and C.read_refusal(p, "m1"):
            return None, f"REFUSED by the trace tool at {p}: {C.read_refusal(p, 'm1')['message']}"
        if p is None or not (p / "episode_trace.json").is_file():
            return None, f"no M1 output at {p}"
        try:
            return C.read_trace(p), None
        except (C.InputError, OSError, ValueError) as exc:
            return None, f"M1 output unreadable at {p}: {exc}"

    def certificate(self, label="final"):
        p = self.path(label, "m3")
        if p is not None and not (p / "stance_heading_report.json").is_file() and C.read_refusal(p, "m3"):
            return None, f"REFUSED by the certificate tool at {p}: {C.read_refusal(p, 'm3')['message']}"
        if p is None or not (p / "stance_heading_report.json").is_file():
            return None, f"no M3 output at {p}"
        try:
            return C.read_certificate(p), None
        except (C.InputError, OSError, ValueError) as exc:
            return None, f"M3 output unreadable at {p}: {exc}"


def completeness(run_id, store, notes, pins, panels: PanelLocator) -> dict:
    """C1-C7 on one fetch's files.  Returns the facts, E, FINISHED/STOPPED, the verified files and the checkpoint
    lists the classes read."""
    out = dict(run_id=run_id, fetch_dir=str(store.dir))
    out["C1"] = c1(run_id, notes)
    out["C2"], man = c2(run_id, store)
    entries = C.manifest_entries(man) if man else {}
    files = {}
    # the non-pair C3 files, verified against the manifest
    small = c3_small(entries)
    c3 = {}
    for p in small:
        b, how = held_c3(store, entries, p) if man else (None, "no manifest")
        c3[p] = how
        files[p] = b
    prog = _json(files["01_stance/progress.json"])
    E = None
    if isinstance(prog, dict) and prog.get("status") == "ended" and isinstance(prog.get("timesteps"), int):
        E = prog["timesteps"]
    # the ledger (C3-verified) and M5
    led = None
    if files["01_stance/eval_episodes.csv"] is not None:
        try:
            led = C.read_ledger_bytes(files["01_stance/eval_episodes.csv"])
        except C.InputError as exc:
            c3["01_stance/eval_episodes.csv"] = f"unreadable: {exc}"
    res = C.m5(*led) if led else None
    pairs, r1, js, resolution, _notes = c3_pairs(entries, E, res)      # v5: the grid's (R-m1)
    for st in pairs:
        for p in C.pair_paths(st):
            b, how = held_c3(store, entries, p) if man else (None, "no manifest")
            c3[p] = how
            files[p] = b
    c3_ok = (man is not None and E is not None and (res is not None) and r1 is not None
             and all(files.get(p) is not None for p in c3))
    note = None
    if man is None:
        note = "no valid manifest"
    elif E is None:
        note = "progress.json not held equal to the manifest or not 'ended': E unknown, the pairs C3 names unknown"
    elif r1 is None:
        note = "no multiple of 262,144 at or before E: R1 does not exist (v5 R-m1)"
    c3_val = {p: ("OK" if files.get(p) is not None else h) for p, h in c3.items()}
    out["C3"] = fact("C3 every file the reading uses equals its manifest entry", True if c3_ok else False, c3_val, note)
    prov = _json(files["provenance.json"])
    sc = _json(files["01_stance/stage_config.json"])
    src = "manifest-verified copies"
    if prov is None or sc is None:
        # read on two agreeing copies when the manifest-verified copy is absent (the run fails C3 then anyway)
        pb, _ = store.agreed("provenance.json")
        sb, _ = store.agreed("01_stance/stage_config.json")
        prov = prov if prov is not None else _json(pb)
        sc = sc if sc is not None else _json(sb)
        src = "two agreeing copies (not manifest-verified; C3 decides)"
    c4 = c4_facts(run_id, prov, sc, notes, pins, entries if man else None, True)
    out["C4"] = fact("C4 one session from step 0 in the round's configuration", all_ok(c4), [dict(f) for f in c4],
                     f"read on {src}")
    out["C4_on_manifest_files"] = src == "manifest-verified copies"     # v5 O-16: the re-fetch rule reads it
    # C5
    fz = files.get("01_stance/models/stage1_final.zip")
    nts, zprob = (C.zip_num_timesteps(fz) if fz is not None else (None, "stage1_final.zip not held equal to the manifest"))
    gv = _json(files["01_stance/gate_verdict.json"])
    ja = C.parse_utc(gv.get("judged_at")) if isinstance(gv, dict) else None
    wa = C.parse_utc(prog.get("written_at")) if isinstance(prog, dict) else None
    gap = C.minutes(wa, ja) if (ja and wa) else None
    c5v = dict(progress_status=prog.get("status") if isinstance(prog, dict) else None,
               E=E, final_zip_num_timesteps=nts, zip_problem=zprob,
               judged_at=gv.get("judged_at") if isinstance(gv, dict) else None,
               written_at=prog.get("written_at") if isinstance(prog, dict) else None,
               judged_minus_written_min=None if gap is None else round(gap, 1))
    c5ok = (E is not None and nts == E and gap is not None and gap <= C.JUDGE_GAP_MIN and E <= C.FINAL)
    finished = c5ok and E == C.FINAL
    stopped = c5ok and E < C.FINAL
    c5v["ending"] = "FINISHED" if finished else ("STOPPED" if stopped else None)
    out["C5"] = fact("C5 the end", True if c5ok else False, c5v)
    # C6
    if led is None or E is None:
        out["C6"] = fact("C6 ledger whole to E, every ANY and TAIL input measured", False, None,
                         "ledger not held equal to the manifest" if led is None else "E unknown")
        c6 = None
    else:
        c6 = LG.c6_check(led[0], led[1], E)
        out["C6"] = fact("C6 ledger whole to E, every ANY and TAIL input measured", c6["passed"],
                         dict(grid_evaluations=c6["n_grid"], failures=c6["failures"][:20],
                              n_failures=len(c6["failures"])))
    out["C6_on_manifest_files"] = c6 is not None          # the ledger and progress.json equal to the manifest (O-16)
    # C7 (a measurement absent is PENDING, never a failure)
    labels = []
    if E is not None:
        labels = sorted({*(str(s) for s in js), *( [str(r1)] if r1 else []),
                         *(str(v) for v in resolution.values() if v != "final"), "final"},
                        key=lambda x: (x == "final", int(x) if x != "final" else 0))
    traces, c7_missing, c7_refused, c7_other_file = {}, [], [], []
    for lab in labels:
        tr, why = panels.trace(lab)
        if tr is not None:
            bad = binding_problems(tr.get("binding") or {}, lab, files, E, run_id)
            if bad:
                c7_other_file.append(f"M1 {lab}: measured on another file or as a validation input: {bad}")
                tr, why = None, f"M1 {lab}: not a measurement of this run's C3-verified pair ({'; '.join(bad)})"
        elif why.startswith("REFUSED"):
            # v5 F-R1: the trace tool's own refusal of this label's C3-verified pair fails C7 (the run is LOST); a
            # refusal record of another file (R-B4) is no measurement of this label: absent, printed
            ref = panels.refusal(lab, "m1") or {}
            bad = binding_problems(ref.get("binding") or {}, lab, files, E, run_id)
            if bad:
                c7_other_file.append(f"M1 {lab}: a refusal recorded on another file or a validation input: {bad}")
                why = f"M1 {lab}: the refusal is not of this run's C3-verified pair ({'; '.join(bad)})"
        traces[lab] = tr
        if tr is None:
            (c7_refused if why.startswith("REFUSED") else c7_missing).append(why)
    cert = None
    if finished:
        cert, why = panels.certificate("final")
        if cert is not None:
            bad = binding_problems(cert.get("binding") or {}, "final", files, E, run_id)
            if bad:
                c7_other_file.append(f"M3 final: measured on another file or as a validation input: {bad}")
                cert, why = None, f"M3 final: not a measurement of this run's C3-verified pair ({'; '.join(bad)})"
        elif why.startswith("REFUSED"):
            # v5 F-R1: "a refused M3 at R2 is the certificate 'refused' of GATE CLASS (GATE FAIL, its message the
            # failing criterion)": R2's certificate is measured (C7), refused
            ref = panels.refusal("final", "m3") or {}
            bad = binding_problems(ref.get("binding") or {}, "final", files, E, run_id)
            if bad:
                c7_other_file.append(f"M3 final: a refusal recorded on another file or a validation input: {bad}")
                why = f"M3 final: the refusal is not of this run's C3-verified pair ({'; '.join(bad)})"
            else:
                rec = ref.get("record") or {}
                cert = dict(passed=False, refused=True, failures=[f"refused: {ref.get('message')}"], seeds=None,
                            physics_revision=rec.get("physics_revision"), commit=C.record_commit(rec), record=rec,
                            label_dir=str(panels.path("final", "m3")), binding=ref.get("binding"))
        if cert is None:
            c7_missing.append(why)
    if E is None:
        out["C7"] = fact("C7 traces (and the final certificate)", None, None, "E unknown")
    elif c7_refused:
        out["C7"] = fact("C7 traces (and the final certificate)", False,
                         dict(labels=labels, refused=c7_refused, missing=c7_missing),
                         "FAILS: the trace tool refused a C7 label's pair (v5 F-R1: a refused M1 at a label C7 names "
                         "fails C7; the run is LOST)")
    else:
        out["C7"] = fact("C7 traces (and the final certificate)", True if not c7_missing else None,
                         dict(labels=labels, missing=c7_missing),
                         None if not c7_missing else "PENDING: a measurement absent (run M1/M3; never a failure)")
    out["measured_on_another_file"] = c7_other_file
    out.update(E=E, finished=finished, stopped=stopped, r1=r1, js=js, resolution=resolution, m5=res, c6=c6,
               traces=traces, certificate=cert, manifest=man, entries=entries, files_read={
                   p: C.sha256_bytes(b) for p, b in files.items() if b is not None})
    cs = [out[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")]
    if all(f["ok"] is True for f in cs):
        out["verdict"] = "COMPLETE"
    elif any(f["ok"] is False for f in cs):
        out["verdict"] = "LOST"
    elif out["C4"]["ok"] is None:
        out["verdict"] = "UNDECIDED (a pin is unfilled: C4 cannot be compared)"
    else:
        out["verdict"] = "PENDING (C7: a measurement absent)"
    return out


def binding_problems(b: dict, label, files: dict, E, run_id) -> list[str]:
    """Fixer (R-B4): why a runner's record is not a measurement of this label's C3-verified pair ([] when it is)."""
    zp, pp = C.pair_paths("final" if label == "final" else int(label))
    zb, pb = files.get(zp), files.get(pp)
    out = []
    if zb is None or pb is None:
        return [f"the pair {zp} / {pp} is not held equal to the manifest"]
    if b.get("checkpoint_sha256") != C.sha256_bytes(zb):
        out.append(f"checkpoint sha256 {b.get('checkpoint_sha256')} != the verified {C.sha256_bytes(zb)}")
    if b.get("vecnorm_sha256") != C.sha256_bytes(pb):
        out.append(f"vecnorm sha256 {b.get('vecnorm_sha256')} != the verified {C.sha256_bytes(pb)}")
    want_n = E if label == "final" else int(label)
    if b.get("num_timesteps") != want_n:
        out.append(f"num_timesteps {b.get('num_timesteps')!r} != {want_n}")
    if b.get("validation_no_run_records") is not False:
        out.append(f"validation_no_run_records {b.get('validation_no_run_records')!r} (a validation input)")
    if b.get("filed_run_id") not in (None, run_id):
        out.append(f"filed under {b.get('filed_run_id')}, not {run_id}")
    return out


# ------------------------------------------------------------------------------------------------- PRESENCE
def presence(run_id, store, notes, pins) -> dict:
    """p1-p3 on one fetch's files: provenance.json, 01_stance/stage_config.json and the ledger, each read only when
    two copies agree (a third on a mismatch)."""
    p1 = c1(run_id, notes)
    pb, ph = store.agreed("provenance.json")
    sb, sh = store.agreed("01_stance/stage_config.json")
    lb, lh = store.agreed("01_stance/eval_episodes.csv")
    if pb is None or sb is None:
        p2 = fact("p2 C4's session and configuration facts", False, None,
                  f"fails this fetch: provenance.json {ph}; stage_config.json {sh}")
        p2facts = []
    else:
        p2facts = c4_facts(run_id, _json(pb), _json(sb), notes, pins, None, include_manifest_clause=False)
        p2 = fact("p2 C4's session and configuration facts", all_ok(p2facts), [dict(f) for f in p2facts],
                  f"provenance.json {ph}; stage_config.json {sh}")
    if lb is None:
        p3 = fact("p3 TAIL ALARM by M5 (b) on the ledger as it stands", False, None, f"fails this fetch: ledger {lh}")
        detail = None
    else:
        try:
            header, rows = C.read_ledger_bytes(lb)
            r = C.m5(header, rows)
        except C.InputError as exc:
            r, header = None, []
            p3 = fact("p3 TAIL ALARM by M5 (b) on the ledger as it stands", False, None, f"ledger unreadable: {exc}")
        detail = None
        if r is not None:
            detail = C.tail_alarm_detail(r, r["tail_alarms"][0]) if r["tail_alarms"] else None
            p3 = fact("p3 TAIL ALARM by M5 (b) on the ledger as it stands", bool(r["tail_alarms"]),
                      dict(tail_alarms=r["tail_alarms"], tail_flagged=r["tail_flagged"], first=detail,
                           ledger=lh))
    rail = p1["ok"] is True and p2["ok"] is True and p3["ok"] is True
    oks = (p1["ok"], p2["ok"], p3["ok"])
    decided = False if any(o is False for o in oks) else (None if any(o is None for o in oks) else True)
    return dict(fetch=str(store.dir), p1=p1, p2=p2, p3=p3, rail=rail, rail_decided=decided,
                detail=detail if rail else None)


# ------------------------------------------------------------------------------------------------- D-16
def notebook_last(path):
    """(supports_no_cell_running, last_error_line, note) from a downloaded .ipynb: the last output an error, or
    the notebook's last code cell executed (the end of Run all)."""
    if not path or not Path(path).is_file():
        return None, None, f"no downloaded notebook at {path}"
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, None, f"notebook unreadable: {exc}"
    cells = nb.get("cells") or []
    last_err, last_out_err = None, False
    with_out = [c for c in cells if c.get("cell_type") == "code" and c.get("outputs")]
    for c in cells:
        for o in c.get("outputs") or []:
            if o.get("output_type") == "error":
                tb = o.get("traceback") or []
                last_err = f"{o.get('ename')}: {o.get('evalue')}" if (o.get("ename") or o.get("evalue")) else (tb[-1] if tb else "error")
    if with_out:
        last_out_err = any(o.get("output_type") == "error" for o in with_out[-1]["outputs"])
    code = [c for c in cells if c.get("cell_type") == "code"]
    ran_to_end = bool(code) and code[-1].get("execution_count") is not None
    return (last_out_err or ran_to_end), last_err, f"last output an error: {last_out_err}; last code cell executed: {ran_to_end}"


def d16(run_id, notes, stores, pins, notebook=None, f0_check=None) -> dict:
    """D-16 for a LOST run: RUNLOG's trigger and times, the trigger check, the notebook's last error line, PRESENCE
    p1-p3 per fetch, and what the last fetch's files show as they stand (two agreeing copies, no manifest check)."""
    trig = sorted(C.notes_for(notes, run_id, ("TRIGGER-a", "TRIGGER-b", "TRIGGER-c", "TRIGGER-d")),
                  key=lambda n: (n["time"], n["line"]))
    ends = sorted(C.notes_for(notes, run_id, ("DELETED", "FOUND-GONE")), key=lambda n: (n["time"], n["line"]))
    ln = launch_note(notes, run_id)
    checks = []
    last = stores[-1] if stores else None
    for t in trig:
        k = t["kind"]
        if k == "TRIGGER-c":
            pb, how = last.agreed("01_stance/progress.json") if last else (None, "no fetch")
            p = _json(pb)
            wa = C.parse_utc(p.get("written_at")) if isinstance(p, dict) else None
            ok = wa is not None and C.minutes(wa, t["time"]) >= 120.0
            checks.append(dict(trigger=k, line=t["line"], verdict="SUPPORTED" if ok else "NOT SUPPORTED BY ITS FILES",
                               value=dict(progress_written_at=p.get("written_at") if isinstance(p, dict) else None,
                                          minutes_before_note=None if wa is None else round(C.minutes(wa, t["time"]), 1),
                                          progress=how)))
        elif k == "TRIGGER-b":
            sup, _err, note = notebook_last(notebook)
            checks.append(dict(trigger=k, line=t["line"], verdict="SUPPORTED" if sup else "NOT SUPPORTED BY ITS FILES",
                               value=note))
        elif k == "TRIGGER-d":
            faults = (f0_check or {}).get("faults") or []
            checks.append(dict(trigger=k, line=t["line"], verdict="SUPPORTED" if faults else "NOT SUPPORTED BY ITS FILES",
                               value=dict(computed_faults=faults)))
        else:
            checks.append(dict(trigger=k, line=t["line"], verdict="SUPPORTED (found gone, as noted)", value=None))
    _s, last_err, nbnote = notebook_last(notebook)
    pres = [presence(run_id, s, notes, pins) for s in stores]
    shown = None
    if last is not None:
        lb, lh = last.agreed("01_stance/eval_episodes.csv")
        if lb is not None:
            try:
                h, rows = C.read_ledger_bytes(lb)
                r = C.m5(h, rows)
                shown = dict(ledger=lh, last_evaluation=max(r["evals"]) if r["evals"] else None,
                             any_flagged=r["any_flagged"], tail_flagged=r["tail_flagged"], any_alarms=r["any_alarms"],
                             tail_alarms=r["tail_alarms"], needs_resolution=r["needs_resolution"])
            except C.InputError as exc:
                shown = dict(ledger=f"unreadable: {exc}")
        else:
            shown = dict(ledger=lh)
        lst = last.listing("01_stance/models")
        names = [f.get("name") for f in (lst or {}).get("files", [])]
        steps = sorted(int(m.group(1)) for m in (C.PERIODIC_ZIP_RE.match(n or "") for n in names) if m)
        shown["last_checkpoint_listed"] = steps[-1] if steps else ("stage1_final.zip" if "stage1_final.zip" in names else None)
        if lst is None:
            shown["last_checkpoint_listed"] = "no listing of 01_stance/models/ in the last fetch"
    return dict(run_id=run_id, launch=ln and ln["text"], triggers=[t["text"] for t in trig],
                ends=[e["text"] for e in ends], trigger_checks=checks, notebook_last_error=last_err,
                notebook=nbnote, presence=pres, shown=shown,
                any_alarm_shown=bool(shown and shown.get("any_alarms")))


# ------------------------------------------------------------------------------------------------- printing
def render_facts(title, d) -> list[str]:
    L = [title]
    for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        if k not in d:
            continue
        f = d[k]
        L.append(f"  {k}: {status_of(f)}  {f['name']}" + (f"  [{f['note']}]" if f.get("note") else ""))
        if k == "C4" and isinstance(f["value"], list):
            for g in f["value"]:
                L.append(f"      {status_of(g):16} {g['name']}: {json.dumps(g['value'], default=str)}"
                         + (f" [{g['note']}]" if g.get("note") else ""))
        elif k == "C3" and isinstance(f["value"], dict):
            for p, h in f["value"].items():
                L.append(f"      {p}: {h}")
        else:
            L.append(f"      value: {json.dumps(f['value'], default=str)[:600]}")
    L.append(f"  VERDICT: {d.get('verdict')}")
    return L


def render_presence(p) -> list[str]:
    L = [f"  PRESENCE on {p['fetch']}: {'PRESENCE RAIL' if p['rail'] else 'no PRESENCE RAIL'}"]
    for k in ("p1", "p2", "p3"):
        f = p[k]
        val = f["value"]
        if k == "p2" and isinstance(val, list):
            bad = [g["name"] + ": " + status_of(g) for g in val if g["ok"] is not True]
            val = "all hold" if not bad else bad
        L.append(f"    {k}: {status_of(f)}  {json.dumps(val, default=str)[:500]}" + (f" [{f['note']}]" if f.get("note") else ""))
    if p["rail"]:
        L.append(f"    PRESENCE RAIL: {json.dumps(p['detail'])}")
    return L


def default_dirs(round_dir, run_id):
    r = Path(round_dir)
    return dict(F0=r / "fetch" / "F0" / run_id, F2=r / "fetch" / "F2" / run_id, REFETCH=r / "fetch" / "REFETCH" / run_id,
                TRIGGER=r / "fetch" / "TRIGGER" / run_id, manifest=r / "fetch" / "manifest.csv",
                panels=r / "panels" / run_id, notebook=r / "outputs" / f"{run_id}.ipynb")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("launch-check", "run"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--round", default=C.ROUND)
    ap.add_argument("--f0")
    ap.add_argument("--f2")
    ap.add_argument("--refetch")
    ap.add_argument("--panels")
    ap.add_argument("--notebook")
    ap.add_argument("--runlog", default=C.RUNLOG)
    ap.add_argument("--pins", default=None)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    print(C.VERSION)
    try:
        pins = C.load_pins(a.pins)
        notes, others = C.parse_runlog(a.runlog, C.RUN_IDS)
    except C.InputError as exc:
        print(f"INPUT ERROR: {exc}")
        return 2
    dd = default_dirs(a.round, a.run)
    if a.mode == "launch-check":
        unfilled = [k for k in ("hyperparameters_sha256", "task_sha256", "plant_identity") if C.pin(pins, k) is None]
        if unfilled:
            print(f"INPUT ERROR: pins unfilled: {unfilled} (pass --pins with the approved pins file; fixer R-m3)")
            return 2
    for i, line in others:
        if line.startswith(C.LATE_PREFIX):
            print(f"RUNLOG line {i}: {line[:len(C.LATE_PREFIX) + 200]}")
            continue
        print(f"RUNLOG line {i} names a RUN_ID but is not a STANCE6 note (printed, read by no rule): {line[:200]}")
    if a.mode == "launch-check":
        st = C.FetchStore(a.f0 or dd["F0"], dd["manifest"], "F0", a.run)
        r = launch_check(a.run, st, notes, pins)
        print(f"LAUNCH CHECK {a.run} (F0 {st.dir}); launch line: {r['launch']}")
        for k in sorted(r["facts"]):
            f = r["facts"][k]
            print(f"  {status_of(f) if f['ok'] is not None else 'NOT COMPUTED':16} {f['name']}: "
                  f"{json.dumps(f['value'], default=str)[:400]}" + (f" [{f['note']}]" if f.get("note") else ""))
        print(f"  FAULTS (computed): {r['faults'] or 'none'}; F0 copies within 60 minutes of the launch: "
              f"{r['f0_copies_within_60_min']}")
        print(f"  P-K3's launch settings (from {r['p_k3_launch_settings']['source']}): "
              f"{json.dumps(r['p_k3_launch_settings']['settings'], default=str)}")
        print(f"  recorded only: {json.dumps(r['recorded_only'], default=str)}")
        out = r
    else:
        f2 = C.FetchStore(a.f2 or dd["F2"], dd["manifest"], "F2", a.run)
        stores = [f2]
        rf = Path(a.refetch or dd["REFETCH"])
        if rf.is_dir():
            stores.append(C.FetchStore(rf, dd["manifest"], "REFETCH", a.run))
        panels = PanelLocator(a.panels or dd["panels"])
        res = completeness(a.run, stores[-1], notes, pins, panels)
        for L in render_facts(f"C1-C7 {a.run} on {stores[-1].dir}", res):
            print(L)
        pres = [presence(a.run, s, notes, pins) for s in stores]
        for p in pres:
            for L in render_presence(p):
                print(L)
        out = dict(completeness={k: v for k, v in res.items() if k not in ("m5", "c6", "traces", "manifest")},
                   presence=pres)
        if res["verdict"] == "LOST":
            f0 = C.FetchStore(dd["F0"], dd["manifest"], "F0", a.run)
            f0c = launch_check(a.run, f0, notes, pins) if f0.dir.is_dir() else None
            d = d16(a.run, notes, stores, pins, a.notebook or dd["notebook"], f0c)
            print(f"  D-16: triggers {d['triggers']}; checks {json.dumps(d['trigger_checks'], default=str)}")
            print(f"  D-16: notebook last error: {d['notebook_last_error']}; shown: {json.dumps(d['shown'], default=str)}")
            out["d16"] = d
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
