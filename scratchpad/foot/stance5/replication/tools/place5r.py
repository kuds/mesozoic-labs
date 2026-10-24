"""place5r.py: place fetched stance5 REPLICATION files into R/runs/<RUN_ID>/01_stance/ (R = SP/foot/stance5/replication;
the notebook layout judge_stance5r.py reads).  SP/foot/stance5/tools/place5.py with the replication runs
(stance5_LTSB_s70, stance5_LTSB_s80) and root; every check is place5.py's (the same commit, task, band keys and
collapse reference), plus one report-only note: "config" notes a stage_config.json run.label that is not the run's
RUN_LABEL (it never refuses on it).  Three additions carry out the run-handling rules (criteria RUN HANDLING):
  --source-folder <RUN_ID>_b (or _c), with config, STEPS, csv or extra: the files come from that Drive folder, the
    restart of a run cut off before its first periodic checkpoint, and are placed under the RUN_ID.  The first such
    call switches the run (R/runs/<RUN_ID>/source_folder.json).  The switch is refused once a periodic checkpoint is
    placed from the earlier folder; what the earlier folder placed is moved aside.  After it, every call names the
    folder (an earlier one is refused), and progress.json must carry the folder as its run_id.
  --supersede, with STEPS and --resumed-from N (STEPS after N, or final, best, robust_best): a placed pair that the
    resumed session saved again is moved aside with its judge files and refusal record (every mode), its panels
    output, its saturation roll and, at final, its certificate folder.  The new pair is placed, and the queue
    judges the slot again.  A ledger replaced under --resumed-from or --replace-csv is moved aside too.  With
    --resumed-from N, STEPS and csv also print "placed_after_N", the periodic pairs placed after N: one placed
    before the resume is the old session's; once the resumed session has saved it again, fetch it again and place it
    with --resumed-from N --supersede.
  --run-records-fail, with config RUN_ID DIR DIR2: the run's own records fail the checks below, or are absent on
    Drive, and DIR and DIR2 agree (byte-equal copies, or neither holds the file).  They are placed as they are (never
    over a different placed copy) beside R/runs/<RUN_ID>/01_stance/records_check_failed.json (the problems, the
    files absent on Drive, the sha256s).  While it stands, checkpoint pairs are placed without the absent records,
    the judge refuses on the records, and judge_one5r.sh writes refused_<steps>.json: an invalid J file.
  Moved aside means R/superseded/<RUN_ID>/<UTC time>/<the path under R>, listed in SUPERSEDED.json there; nothing
  is deleted.
place5.py's text follows (with these additions), with place5.py read as place5r.py, S5 as R and satwatch5.py as
satwatch5r.py:
SP/foot/stance4/tools/place4.py with the stance5 identity (arm LTSB: the config commit, the band-on task, the band key
and the LTSB collapse reference) and one addition, "extra" (report-only progress.json and gate_verdict.json).

    python3 -I place5r.py config RUN_ID DIR [DIR2]       the three run records, once (DIR DIR2 --run-records-fail: above)
    python3 -I place5r.py STEPS RUN_ID FETCH_DIR [--no-csv] [--resumed-from N [--supersede] | --replace-csv]
    python3 -I place5r.py csv RUN_ID FETCH_DIR [--resumed-from N | --replace-csv]   the two ledgers only (FETCH_DIR/csv/)
    python3 -I place5r.py extra RUN_ID FETCH_DIR          report-only extras (FETCH_DIR/extra/): progress.json (replaced
                                                         by each newer copy), gate_verdict.json (once; a different
                                                         copy is refused)
    python3 -I place5r.py status [RUN_ID ...]            what is placed, with sha256 (writes nothing)

RUN_ID: stance5_LTSB_s70 or stance5_LTSB_s80 (the seed is read from the name; a restart under
"<RUN_ID>_b" or "_c" is placed under its RUN_ID with --source-folder, above).
STEPS: a multiple of 524,288, or final, best, robust_best.  The local names equal the Drive titles:
  periodic  models/stage1_<N>_steps.zip + models/stage1_vecnormalize_<N>_steps.pkl
  final     models/stage1_final.zip + models/stage1_final_vecnorm.pkl
  best      models/best_model.zip + models/best_model_vecnorm.pkl (robust_best likewise)

config: copies plant_identity.json, stage_config.json and task_fingerprint.json from DIR (and requires DIR2's copies to
  be byte-equal when DIR2 is given).  Each file is checked first: stage_config git_commit CONFIG_COMMIT (the stance5
  config commit), run.seed = the run's seed, run.n_envs 4, run.timesteps 12,000,000, stance_stagger 0.30 / 0.045,
  stance_cop_band_weight 0.30 with the other seven band keys at their defaults, curriculum
  collapse_peak_floor_reference 4457.4, task_sha256 TASK_SHA256 (the band-on task, stage_config and
  task_fingerprint.json), plant physics r9.  While CONFIG_COMMIT is a PLACEHOLDER (None) nothing is placed.  A placed copy is never overwritten.  A differing copy
  is refused, except a stage_config.json that differs from the placed one only by run.duration_seconds (the trainer
  adds it when a run ends): the placed copy is kept and the difference is reported.

STEPS: FETCH_DIR is a fetch workflow's output for one run (fetch/r<STEPS>/<RUN_ID> or fetch/<TAG>/<RUN_ID>), holding
  A/ and B/ (and C/ after a tie-break) with the transcribed pkl, zip/ with the zip, and csv/ with the ledgers.
  - the pkl: A and B must agree byte for byte; otherwise C must equal one of them (two of three agree).  Else refused.
  - the zip: zip/<title>; zipfile.testzip must pass.  It must be there (the judge needs the pair).
  - the records: if A/ and B/ also hold the three JSON files (a configs:true fetch), they must agree and match the
    placed copies (or are placed now, after the config checks).  Otherwise the placed copies must already exist.
  - the ledgers (report only; skipped with --no-csv): csv/eval_episodes.csv and csv/train_episodes.csv replace the
    placed copies when they extend them (the old file is a byte prefix of the new one).  A shorter or non-prefix copy
    is refused, and the refusal is left beside the ledger as <name>.refused.json (satwatch5r.py reports it as a ledger
    problem until a later placement succeeds and removes it); the checkpoint pair itself is still placed, and the
    output lists the refused ledgers under "ledger_refused".  After a Colab resume from checkpoint N (the notebook
    keeps every ledger row up to step N byte for byte and appends after them), --resumed-from N accepts a fetched
    ledger whose header and rows with timesteps <= N equal the placed copy's byte for byte, whose rows are in step
    order, and which continues past N; the old session's rows past N are dropped and counted.  --replace-csv accepts
    any fetched copy (last resort; it prints a warning).
  A placed checkpoint file is never overwritten: an identical copy is a no-op, a different one is refused (with
  --supersede it is moved aside first).
Prints one JSON object (what was placed, sha256 of every file).  Never imports, unpickles or runs the payload.
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile

S5 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication"  # R
S5 = os.environ.get("PLACE5R_TEST_ROOT") or S5  # tests only: place under a scratch root instead of R
RUNS = {"stance5_LTSB_s70": 70, "stance5_LTSB_s80": 80}
RUN_LABELS = {"stance5_LTSB_s70": "stance screen LTSB s70", "stance5_LTSB_s80": "stance screen LTSB s80"}  # notes only
RESTARTS = ("_b", "_c")  # criteria RUN HANDLING: a restart before the first periodic checkpoint is that run
JUDGED = ("judge_{s}.json", "judge_quick_{s}.json", "judge_fresh_{s}.json", "judge_tail_{s}.json",
          "judge_common_noise_{s}.json", "refused_{s}.json", "refused_{s}.review.json")  # judge_one5r.sh's, per checkpoint
RECORDS_FAILED = "records_check_failed.json"  # criteria RUN HANDLING: config --run-records-fail (the run's records)
CONFIG_COMMIT = "cf80ecb14d1630ce3b4dcdea28a05835d143470b"  # claude/screen-trex-r9-armlts-copband's config commit (judge_stance5.py ARMS["LTSB"])
TASK_SHA256 = "sha256:8c6998d260b850ae054c9c2d49a8e5bef2bf59d4770ad77d3674b911721918cf"  # criteria: the band-on task
BAND = {"stance_cop_band_weight": 0.3, "stance_cop_band_lower": -0.6, "stance_cop_band_upper": 0.4,
        "stance_cop_band_pad_lower": -0.65, "stance_cop_band_pad_upper": 0.65, "stance_cop_band_tolerance": 0.357,
        "stance_cop_band_full_price_load_share": 0.25, "stance_cop_band_min_load_share": 0.35}
COLLAPSE_REFERENCE = 4457.4
R9_PHYSICS = "sha256:3d68aec73fbc4642681ed601bedaf3524459d693d6a7afce036bb14bac872d67"
RECORDS = ("plant_identity.json", "stage_config.json", "task_fingerprint.json")
CADENCE = 524288
NAMED = ("final", "best", "robust_best")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def die(msg):
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(1)


def utc_stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def move_aside(run, paths, why):
    """criteria RUN HANDLING: move each existing path (under the root) to <root>/superseded/<RUN_ID>/<UTC time>/<its
    path under the root>, never deleting anything, and list the moves in SUPERSEDED.json there."""
    base = os.path.join(S5, "superseded", run, utc_stamp())
    moves = []
    for p in paths:
        if not os.path.lexists(p):
            continue
        rel = os.path.relpath(p, S5)
        if rel == os.curdir or rel.split(os.sep)[0] == os.pardir:
            die(f"{p} is not under {S5}")
        dst = os.path.join(base, rel)
        if os.path.lexists(dst):
            die(f"{dst} exists; refusing to overwrite it")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(p, dst)
        moves.append({"from": p, "to": dst})
    if moves:
        rec = os.path.join(base, "SUPERSEDED.json")
        old = []
        if os.path.exists(rec):
            with open(rec) as fh:
                old = json.load(fh)
        with open(rec, "w") as fh:
            json.dump(old + [{"why": why, "moved": moves}], fh, indent=1)
    return moves


def source_marker(run):
    return os.path.join(S5, "runs", run, "source_folder.json")


def current_source(run):
    """The Drive folder this run's files come from: its RUN_ID, or its restart folder once switched."""
    m = source_marker(run)
    if not os.path.exists(m):
        return run
    with open(m) as fh:
        return json.load(fh)["source_folder"]


def use_source(run, folder):
    """criteria RUN HANDLING: a run cut off before its first periodic checkpoint (524,288) and restarted under
    <RUN_ID>_b (then _c) IS that run, placed under its RUN_ID.  The first call naming a later folder switches the run to
    it: refused once a periodic checkpoint is placed from the earlier folder; what the earlier folder placed (records,
    ledgers, extras) is moved aside.  After the switch every call names that folder; an earlier one is refused."""
    stage = stage_dir(run)
    order = [run] + [run + s for s in RESTARTS]
    folder = folder or run
    if folder not in order:
        die(f"--source-folder {folder} is not {run}'s restart folder ({' or '.join(order[1:])})")
    cur = current_source(run)
    if folder == cur:
        return cur
    if order.index(folder) < order.index(cur):
        die(f"{run}'s files come from the Drive folder {cur} (source_folder.json); {folder} is an earlier folder, left "
            f"alone and never judged: pass --source-folder {cur}")
    models = os.path.join(stage, "models")
    placed = sorted(f for f in (os.listdir(models) if os.path.isdir(models) else []) if f.endswith(".zip"))
    if placed:
        die(f"{run} holds checkpoints placed from {cur} ({', '.join(placed[:3])}): a restart under {folder} is that run "
            "only when the earlier folder was cut off before its first periodic checkpoint (524,288). Stop and ask.")
    moved = move_aside(run, [os.path.join(stage, n) for n in sorted(os.listdir(stage))] if os.path.isdir(stage) else [],
                       f"the run's source folder {cur} -> {folder} (a restart before the first checkpoint)")
    os.makedirs(os.path.dirname(source_marker(run)), exist_ok=True)
    with open(source_marker(run) + ".part", "w") as fh:
        json.dump({"run_id": run, "source_folder": folder, "previous": cur, "set_at": utc_stamp(),
                   "moved_aside": moved}, fh, indent=1)
    os.replace(source_marker(run) + ".part", source_marker(run))
    return folder


def superseded_paths(run, steps, zname, pname):
    """What a superseded checkpoint takes with it: the placed pair, its judge files and refusal record (every mode), its
    panels output, its saturation roll and, at final, the certificate folder."""
    out = [os.path.join(stage_dir(run), "models", n) for n in (zname, pname)]
    out += [os.path.join(S5, "judged", run, n.format(s=steps)) for n in JUDGED]
    out += [os.path.join(S5, "panels", "out", run, f"panels_{steps}.json"),
            os.path.join(S5, "tools", "satroll", f"{run}_{steps}.json")]
    if steps == "final":
        out.append(os.path.join(S5, "cert", run))
    return out


def after_resume(run, n, exclude=None):
    """criteria RUN HANDLING (resumes): the periodic pairs placed after N (less *exclude*, a pair this call placed).  One
    placed before the resume is the old session's: the session resumed from N saves it again, and it is fetched again
    then and placed with --resumed-from N --supersede.  Nothing is moved here."""
    models = os.path.join(stage_dir(run), "models")
    after = []
    for f in (os.listdir(models) if os.path.isdir(models) else []):
        g = re.fullmatch(r"stage1_(\d+)_steps\.zip", f)
        if g and int(g.group(1)) > n and f != exclude:
            after.append((int(g.group(1)), f))
    out = {"placed_after_N": [f for _s, f in sorted(after)]}
    if after:
        out["placed_after_N_note"] = (
            f"periodic pairs placed after {n:,}: each one placed before the resume is the old session's; once the resumed "
            f"session has saved it again, fetch it again and place it with --resumed-from {n} --supersede (criteria RUN "
            "HANDLING)")
    return out


def failed_marker(run):
    return os.path.join(stage_dir(run), RECORDS_FAILED)


def read_failed_marker(run):
    """criteria RUN HANDLING (judge refusals): the --run-records-fail marker of the run, or None."""
    m = failed_marker(run)
    if not os.path.exists(m):
        return None
    with open(m) as fh:
        return json.load(fh)


def place_failed_records(run, src, src2):
    """criteria RUN HANDLING (judge refusals): config --run-records-fail.  The run's own records fail check_record, or
    are absent on Drive, and A and B agree (src and src2 byte-equal, or neither holds the file).  They are placed as
    they are, never over a different placed copy, beside records_check_failed.json, so that the run's checkpoints can be
    placed and judged: the judge then refuses on the records, and judge_one5r.sh records refused_<steps>.json (an
    invalid J file).  A marker already placed must be this one (the same problems, absent files and sha256s)."""
    if CONFIG_COMMIT is None:
        die("CONFIG_COMMIT is a PLACEHOLDER: fill it with the config commit before placing anything")
    dst, seed = stage_dir(run), RUNS[run]
    raws, absent, problems = {}, [], []
    for name in RECORDS:  # check all three before writing anything
        s, s2 = os.path.join(src, name), os.path.join(src2, name)
        if not os.path.exists(s) and not os.path.exists(s2):
            absent.append(name)
            problems.append(f"{name}: absent on Drive (neither transcription holds it)")
            continue
        if not (os.path.exists(s) and os.path.exists(s2)) or open(s, "rb").read() != open(s2, "rb").read():
            die(f"{name}: {src} and {src2} differ or one is missing (tie-break before placing)")
        raws[name] = open(s, "rb").read()
        problems += check_record(name, raws[name], seed)
    for name in RECORDS:
        d = os.path.join(dst, name)
        if name in raws and os.path.exists(d) and open(d, "rb").read() != raws[name]:
            die(f"{name}: the placed copy differs from {os.path.join(src, name)}")
        if name in absent and os.path.exists(d):
            die(f"{name} is placed, but neither {src} nor {src2} holds it: stop and ask")
    key = {"problems": problems, "absent_on_drive": absent,
           "sha256": {n: hashlib.sha256(raw).hexdigest() for n, raw in raws.items()}}
    old = read_failed_marker(run)
    if old is not None:
        if {k: old.get(k) for k in key} != key:
            die(f"{failed_marker(run)} records other problems, absent files or copies than {src} and {src2} hold: "
                "stop and ask")
        return ({n: sha(os.path.join(dst, n)) for n in raws},
                [f"the run's records failed the checks ({RECORDS_FAILED} unchanged): {'; '.join(problems)}"])
    if not problems:
        die("the run's records pass every check and none is absent on Drive: place them with config, without "
            "--run-records-fail")
    os.makedirs(dst, exist_ok=True)
    for name, raw in raws.items():
        d = os.path.join(dst, name)
        if not os.path.exists(d):
            with open(d + ".part", "wb") as fh:
                fh.write(raw)
            os.replace(d + ".part", d)
    rec = {"schema": "stance5r-records-check-failed/v1", "run_id": run, "source_folder": current_source(run), **key,
           "from": [os.path.abspath(src), os.path.abspath(src2)], "set_at": utc_stamp(),
           "rule": "criteria RUN HANDLING: a judge that refuses on the run's own records counts as an INVALID J file; "
                   "the records are placed as they are so that the judge refuses on them"}
    with open(failed_marker(run) + ".part", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(failed_marker(run) + ".part", failed_marker(run))
    return ({n: sha(os.path.join(dst, n)) for n in raws},
            [f"placed with --run-records-fail ({RECORDS_FAILED}): {'; '.join(problems)}"])


def stage_dir(run):
    if run not in RUNS:
        die(f"unknown RUN_ID {run}; expected one of {sorted(RUNS)}")
    return os.path.join(S5, "runs", run, "01_stance")


def pair_names(steps):
    if steps == "final":
        return "stage1_final.zip", "stage1_final_vecnorm.pkl"
    if steps in ("best", "robust_best"):
        return f"{steps}_model.zip", f"{steps}_model_vecnorm.pkl"
    n = int(steps)
    if n <= 0 or n % CADENCE:
        die(f"STEPS {steps} is not a multiple of {CADENCE}")
    return f"stage1_{n}_steps.zip", f"stage1_vecnormalize_{n}_steps.pkl"


def check_record(name, raw, seed):
    """The record's own content checks (json parse only).  Returns a list of problems."""
    bad = []
    try:
        d = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"{name}: not JSON ({exc})"]
    if name == "stage_config.json":
        run = d.get("run") or {}
        rw = d.get("reward_weights") or {}
        if d.get("git_commit") != CONFIG_COMMIT:
            bad.append(f"git_commit {d.get('git_commit')}")
        if run.get("seed") != seed:
            bad.append(f"run.seed {run.get('seed')} != {seed}")
        if run.get("n_envs") != 4:
            bad.append(f"run.n_envs {run.get('n_envs')}")
        if run.get("timesteps") != 12000000:
            bad.append(f"run.timesteps {run.get('timesteps')}")
        if (rw.get("stance_stagger_weight"), rw.get("stance_stagger_tolerance_m")) != (0.3, 0.045):
            bad.append("stagger keys")
        if {k: rw.get(k) for k in BAND} != BAND:
            bad.append(f"band keys {({k: rw.get(k) for k in BAND})}")
        if (d.get("curriculum") or {}).get("collapse_peak_floor_reference") != COLLAPSE_REFERENCE:
            bad.append(f"collapse_peak_floor_reference {(d.get('curriculum') or {}).get('collapse_peak_floor_reference')}")
        if (d.get("task_fingerprint") or {}).get("task_sha256") != TASK_SHA256:
            bad.append("stage_config task_sha256")
    elif name == "task_fingerprint.json":
        if d.get("task_sha256") != TASK_SHA256:
            bad.append(f"task_sha256 {d.get('task_sha256')}")
    elif name == "plant_identity.json":
        if d.get("physics_sha256") != R9_PHYSICS or d.get("physics_revision") != 9:
            bad.append("plant not r9")
    return [f"{name}: {b}" for b in bad]


def strip_duration(raw):
    d = json.loads(raw.decode("utf-8"))
    (d.get("run") or {}).pop("duration_seconds", None)
    return d


def place_records(run, src, src2=None):
    if CONFIG_COMMIT is None:
        die("CONFIG_COMMIT is a PLACEHOLDER: fill it with the config commit before placing anything")
    dst, seed = stage_dir(run), RUNS[run]
    raws = {}
    for name in RECORDS:  # check all three before writing anything
        s = os.path.join(src, name)
        if not os.path.exists(s):
            die(f"{s} missing")
        raw = open(s, "rb").read()
        if src2 is not None:
            s2 = os.path.join(src2, name)
            if not os.path.exists(s2) or open(s2, "rb").read() != raw:
                die(f"{name}: {src} and {src2} differ or one is missing (tie-break before placing)")
        bad = check_record(name, raw, seed)
        if bad:
            die("; ".join(bad))
        raws[name] = raw
    os.makedirs(dst, exist_ok=True)
    out, notes = {}, []
    label = (json.loads(raws["stage_config.json"].decode("utf-8")).get("run") or {}).get("label")
    if label != RUN_LABELS[run]:  # report only: the RUN_ID (this folder), the seed and the commit are what bind
        notes.append(f"stage_config.json run.label {label!r} is not {RUN_LABELS[run]!r} (report only)")
    for name in RECORDS:
        s, raw = os.path.join(src, name), raws[name]
        d = os.path.join(dst, name)
        if os.path.exists(d):
            old = open(d, "rb").read()
            if old != raw:
                if name == "stage_config.json" and strip_duration(old) == strip_duration(raw):
                    notes.append(f"{name}: the fetched copy differs only by run.duration_seconds; kept the placed copy")
                else:
                    die(f"{name}: the placed copy differs from {s}")
        else:
            tmp = d + ".part"
            with open(tmp, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, d)
        out[name] = sha(d)
    return out, notes


def agreed_pkl(fetch, title):
    shas = {}
    for role in ("A", "B", "C"):
        p = os.path.join(fetch, role, title)
        if os.path.exists(p):
            shas[role] = sha(p)
    a, b, c = shas.get("A"), shas.get("B"), shas.get("C")
    if a and b and a == b:
        return os.path.join(fetch, "A", title), shas
    if c and c in (a, b):
        return os.path.join(fetch, "C", title), shas
    die(f"{title}: no two transcriptions agree ({shas})")


def copy_once(src, dst):
    if os.path.exists(dst):
        if sha(dst) != sha(src):
            die(f"{dst} exists and differs from {src}; refusing to overwrite (a checkpoint the resumed session saved again: "
            "--resumed-from N --supersede)")
        return "kept (identical)"
    tmp = dst + ".part"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return "placed"


def resumed_check(old, new, n):
    """--resumed-from N: (ok, detail).  ok when the header and every row with timesteps <= N are byte-identical in the
    placed (old) and fetched (new) ledgers, both list those rows first (step order), and the fetched copy continues
    past N."""
    ol, nl = old.splitlines(keepends=True), new.splitlines(keepends=True)
    if not ol or not nl or ol[0] != nl[0]:
        return False, "the headers differ"
    cols = next(csv.reader(io.StringIO(ol[0].decode("utf-8"))))
    if "timesteps" not in cols:
        return False, "no timesteps column"
    k = cols.index("timesteps")

    def steps(line):
        return int(float(next(csv.reader(io.StringIO(line.decode("utf-8"))))[k]))

    try:
        os_, ns_ = [steps(x) for x in ol[1:]], [steps(x) for x in nl[1:]]
    except (ValueError, IndexError, StopIteration) as exc:
        return False, f"unreadable timesteps ({exc})"
    ko, kn = sum(v <= n for v in os_), sum(v <= n for v in ns_)
    if any(v > n for v in os_[:ko]) or any(v > n for v in ns_[:kn]) or any(v <= n for v in ns_[kn:]) \
            or any(v <= n for v in os_[ko:]):
        return False, "the rows up to N are not listed first"
    if ol[1:1 + ko] != nl[1:1 + kn]:
        return False, f"the rows up to {n:,} differ ({ko} placed, {kn} fetched)"
    if len(ns_) == kn:
        return False, f"the fetched copy does not continue past {n:,}"
    return True, {"rows_kept_to_N": kn, "old_rows_past_N_dropped": len(os_) - ko, "new_rows_past_N": len(ns_) - kn,
                  "first_new_step_past_N": ns_[kn]}


def place_csv(fetch, dst, replace, resumed_from=None):
    out = {}
    for name in ("eval_episodes.csv", "train_episodes.csv"):
        s = os.path.join(fetch, "csv", name)
        if not os.path.exists(s):
            out[name] = "not in fetch"
            continue
        d = os.path.join(dst, name)
        marker = d + ".refused.json"
        new = open(s, "rb").read()
        note = None
        if os.path.exists(d):
            old = open(d, "rb").read()
            if old == new:
                if os.path.exists(marker):
                    os.remove(marker)
                out[name] = {"action": "kept (identical)", "bytes": len(new), "sha256": sha(d)}
                continue
            if not (len(new) >= len(old) and new[: len(old)] == old):
                if resumed_from is not None:
                    ok, detail = resumed_check(old, new, resumed_from)
                    if not ok:
                        out[name] = {"action": f"REFUSED: --resumed-from {resumed_from}: {detail}",
                                     "placed_bytes": len(old), "fetched_bytes": len(new)}
                        with open(marker, "w") as fh:
                            json.dump({"fetch": os.path.abspath(fetch), "placed_bytes": len(old), "fetched_bytes": len(new),
                                       "fetched_sha256": sha(s), "why": out[name]["action"]}, fh, indent=1)
                        continue
                    note = {"resumed_from": resumed_from, **detail}
                elif replace:
                    note = "WARNING: --replace-csv replaced a placed ledger the fetched copy does not extend"
                else:
                    out[name] = {"action": "REFUSED: the fetched copy does not extend the placed one (after a resume "
                                           "from N: --resumed-from N)", "placed_bytes": len(old), "fetched_bytes": len(new)}
                    with open(marker, "w") as fh:
                        json.dump({"fetch": os.path.abspath(fetch), "placed_bytes": len(old), "fetched_bytes": len(new),
                                   "fetched_sha256": sha(s), "why": out[name]["action"]}, fh, indent=1)
                    continue
        moved = None
        if note is not None and os.path.exists(d):  # criteria RUN HANDLING: the superseded copy is moved aside
            moved = move_aside(os.path.basename(os.path.dirname(dst)), [d],
                               f"{name}: superseded by {os.path.abspath(s)} ("
                               + (f"--resumed-from {resumed_from}" if resumed_from is not None else "--replace-csv") + ")")
        tmp = d + ".part"
        with open(tmp, "wb") as fh:
            fh.write(new)
        os.replace(tmp, d)
        if os.path.exists(marker):
            os.remove(marker)
        out[name] = {"action": "placed", "bytes": len(new), "sha256": sha(d)}
        if note:
            out[name]["note"] = note
        if moved:
            out[name]["moved_aside"] = moved
    return out


def refused(res):
    return sorted(k for k, v in res.items() if isinstance(v, dict) and str(v.get("action", "")).startswith("REFUSED"))


def resumed_arg(flags):
    """--resumed-from=N (flags) or None; N a positive multiple of 524,288."""
    vals = [f.split("=", 1)[1] for f in flags if f.startswith("--resumed-from=")]
    if not vals:
        return None
    try:
        n = int(vals[0])
    except ValueError:
        die(f"--resumed-from {vals[0]!r} is not a step count")
    if n <= 0 or n % CADENCE:
        die(f"--resumed-from {n} is not a positive multiple of {CADENCE}")
    return n


def place_checkpoint(steps, run, fetch, no_csv, replace_csv, resumed_from=None, supersede=False):
    dst = stage_dir(run)
    models = os.path.join(dst, "models")
    os.makedirs(models, exist_ok=True)
    zname, pname = pair_names(steps)
    res = {"ok": True, "run": run, "steps": steps, "fetch": os.path.abspath(fetch), "notes": []}
    # records: from this fetch when both A and B carry them, else they must be placed already
    failed = read_failed_marker(run)  # criteria RUN HANDLING: the run's records placed with --run-records-fail
    if all(os.path.exists(os.path.join(fetch, r, n)) for r in ("A", "B") for n in RECORDS):
        rec, notes = (place_failed_records if failed is not None else place_records)(
            run, os.path.join(fetch, "A"), os.path.join(fetch, "B"))
        res["records"], res["notes"] = rec, res["notes"] + notes
    else:
        absent = (failed or {}).get("absent_on_drive") or []
        missing = [n for n in RECORDS if not os.path.exists(os.path.join(dst, n)) and n not in absent]
        if missing:
            die(f"records not placed yet ({missing}); run 'place5r.py config {run} DIR' or fetch with configs: true "
                f"(records that fail the checks or are absent on Drive, A and B agreeing: 'place5r.py config {run} "
                "DIR DIR2 --run-records-fail')")
        res["records"] = {n: sha(os.path.join(dst, n)) for n in RECORDS if os.path.exists(os.path.join(dst, n))}
    if failed is not None:
        res["notes"].append(f"the run's records failed the checks ({RECORDS_FAILED}): the judge refuses on them, and "
                            "judge_one5r.sh records refused_<steps>.json, an invalid J file (criteria RUN HANDLING)")
    pkl_src, shas = agreed_pkl(fetch, pname)
    zsrc = os.path.join(fetch, "zip", zname)
    if not os.path.exists(zsrc):
        die(f"{zsrc} missing (the judge needs the zip and the pkl together)")
    with zipfile.ZipFile(zsrc) as z:
        badm = z.testzip()
    if badm is not None:
        die(f"{zsrc}: testzip failed at {badm}")
    if supersede:  # criteria RUN HANDLING: the resumed session saved this checkpoint again
        placed = ((os.path.join(models, zname), zsrc), (os.path.join(models, pname), pkl_src))
        if any(os.path.exists(d) and sha(d) != sha(s) for d, s in placed):
            res["superseded"] = move_aside(run, superseded_paths(run, steps, zname, pname),
                                           f"{steps}: saved again by the session resumed from {resumed_from:,}")
        else:
            res["notes"].append("--supersede: no different pair is placed; nothing was moved aside")
    res["pkl"] = {"title": pname, "transcriptions": shas, "from": pkl_src,
                  "action": copy_once(pkl_src, os.path.join(models, pname)), "sha256": sha(os.path.join(models, pname))}
    res["zip"] = {"title": zname, "testzip": "OK", "action": copy_once(zsrc, os.path.join(models, zname)),
                  "bytes": os.path.getsize(os.path.join(models, zname)), "sha256": sha(os.path.join(models, zname))}
    res["csv"] = "skipped (--no-csv)" if no_csv else place_csv(fetch, dst, replace_csv, resumed_from)
    if not no_csv and refused(res["csv"]):
        res["ledger_refused"] = refused(res["csv"])
        res["notes"].append(f"the checkpoint pair is placed; the ledgers {res['ledger_refused']} were refused (left as "
                            "<name>.refused.json; satwatch5r.py reports them until a placement succeeds)")
    if resumed_from is not None:  # criteria RUN HANDLING (resumes): the old session's pairs after N, if any
        res.update(after_resume(run, resumed_from, zname if res["zip"]["action"] == "placed" else None))
    return res


def place_extra(run, fetch):
    """Report-only extras: progress.json replaces the placed copy (a snapshot; the newer written_at wins), gate_verdict.json
    is placed once (a different copy is refused)."""
    dst = stage_dir(run)
    os.makedirs(dst, exist_ok=True)
    out = {}
    for name in ("progress.json", "gate_verdict.json"):
        s = os.path.join(fetch, "extra", name)
        if not os.path.exists(s):
            out[name] = "not in fetch"
            continue
        raw = open(s, "rb").read()
        try:
            new = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            out[name] = f"REFUSED: not JSON ({exc})"
            continue
        d = os.path.join(dst, name)
        if name == "progress.json":
            if new.get("run_id") != current_source(run):  # a restart's progress.json names its own folder
                out[name] = f"REFUSED: run_id {new.get('run_id')} is not {current_source(run)}"
                continue
            if os.path.exists(d):
                old = json.loads(open(d, "rb").read().decode("utf-8"))
                if str(old.get("written_at")) > str(new.get("written_at")):
                    out[name] = {"action": "kept (the placed copy is newer)", "placed_written_at": old.get("written_at")}
                    continue
        elif os.path.exists(d):
            out[name] = {"action": "kept (identical)" if open(d, "rb").read() == raw else
                         "REFUSED: a different gate_verdict.json is placed", "sha256": sha(d)}
            continue
        tmp = d + ".part"
        with open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, d)
        out[name] = {"action": "placed", "sha256": sha(d), "written_at": new.get("written_at"),
                     "timesteps": new.get("timesteps"), "steps_per_s": new.get("steps_per_s")}
    return out


def status(runs):
    out = {}
    for run in runs or sorted(RUNS):
        dst = stage_dir(run)
        if not os.path.isdir(dst):
            out[run] = "not placed"
            continue
        ent = {n: sha(os.path.join(dst, n)) for n in RECORDS if os.path.exists(os.path.join(dst, n))}
        for n in ("eval_episodes.csv", "train_episodes.csv"):
            p = os.path.join(dst, n)
            if os.path.exists(p):
                ent[n] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
        m = os.path.join(dst, "models")
        pairs = {}
        if os.path.isdir(m):
            names = set(os.listdir(m))
            for f in sorted(names):
                g = re.fullmatch(r"stage1_(\d+)_steps\.zip", f)
                key = g.group(1) if g else {"stage1_final.zip": "final", "best_model.zip": "best",
                                            "robust_best_model.zip": "robust_best"}.get(f)
                if key:
                    z, p = pair_names(key)
                    pairs[key] = {"zip": sha(os.path.join(m, z)), "pkl": sha(os.path.join(m, p)) if p in names else None}
        ent["checkpoints"] = pairs
        ent["source_folder"] = current_source(run)
        if read_failed_marker(run) is not None:  # criteria RUN HANDLING: config --run-records-fail
            ent["records_check_failed"] = read_failed_marker(run)
        out[run] = ent
    return out


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    source = None  # criteria RUN HANDLING: --source-folder <RUN_ID>_b (or _c), a restart placed under its RUN_ID
    if "--source-folder" in argv:
        i = argv.index("--source-folder")
        if argv.count("--source-folder") > 1 or i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            die("--source-folder takes one Drive folder name (<RUN_ID>_b or <RUN_ID>_c)")
        source, argv = argv[i + 1], argv[:i] + argv[i + 2:]
        if not argv or argv[0] in ("status", "-h", "--help"):
            die("--source-folder goes with config, STEPS, csv or extra")
    fail = "--run-records-fail" in argv  # criteria RUN HANDLING: the run's own records fail the checks
    if fail:
        argv = [a for a in argv if a != "--run-records-fail"]
        if not argv or argv[0] != "config" or len(argv) != 4:
            die("--run-records-fail goes with config RUN_ID DIR DIR2 (A and B must agree)")
    if argv[0] == "config":
        if len(argv) not in (3, 4):
            die("usage: place5r.py config RUN_ID DIR [DIR2] [--source-folder F] [--run-records-fail]")
        use_source(argv[1], source)
        if fail:
            rec, notes = place_failed_records(argv[1], argv[2], argv[3])
            print(json.dumps({"ok": True, "run": argv[1], "records": rec, "notes": notes,
                              "records_check_failed": read_failed_marker(argv[1])}, indent=1))
            return 0
        rec, notes = place_records(argv[1], argv[2], argv[3] if len(argv) == 4 else None)
        print(json.dumps({"ok": True, "run": argv[1], "records": rec, "notes": notes}, indent=1))
        return 0
    argv = [f"--resumed-from={argv[i + 1]}" if a == "--resumed-from" and i + 1 < len(argv) else a
            for i, a in enumerate(argv) if not (i > 0 and argv[i - 1] == "--resumed-from")]
    if argv[0] == "csv":
        pos = [a for a in argv[1:] if not a.startswith("--")]
        flags = [a for a in argv[1:] if a.startswith("--")]
        unknown = [f for f in flags if f != "--replace-csv" and not f.startswith("--resumed-from=")]
        if len(pos) != 2 or unknown or ("--replace-csv" in flags and resumed_arg(flags) is not None):
            die("usage: place5r.py csv RUN_ID FETCH_DIR [--resumed-from N | --replace-csv] [--source-folder F]")
        use_source(pos[0], source)
        dst = stage_dir(pos[0])
        os.makedirs(dst, exist_ok=True)
        res = place_csv(pos[1], dst, "--replace-csv" in flags, resumed_arg(flags))
        ok = not refused(res)
        out = {"ok": ok, "run": pos[0], "csv": res}
        if resumed_arg(flags) is not None:  # criteria RUN HANDLING (resumes): the old session's pairs after N, if any
            out.update(after_resume(pos[0], resumed_arg(flags)))
        print(json.dumps(out, indent=1))
        return 0 if ok else 1
    if argv[0] == "extra":
        if len(argv) != 3:
            die("usage: place5r.py extra RUN_ID FETCH_DIR [--source-folder F]")
        use_source(argv[1], source)
        res = place_extra(argv[1], argv[2])
        ok = not any(str(v if not isinstance(v, dict) else v.get("action", "")).startswith("REFUSED") for v in res.values())
        print(json.dumps({"ok": ok, "run": argv[1], "extra": res}, indent=1))
        return 0 if ok else 1
    if argv[0] == "status":
        print(json.dumps(status(argv[1:]), indent=1))
        return 0
    flags = [a for a in argv if a.startswith("--")]
    pos = [a for a in argv if not a.startswith("--")]
    unknown = [f for f in flags if f not in ("--no-csv", "--replace-csv", "--supersede")
               and not f.startswith("--resumed-from=")]
    if len(pos) != 3 or unknown or ("--replace-csv" in flags and resumed_arg(flags) is not None):
        die("usage: place5r.py STEPS RUN_ID FETCH_DIR [--no-csv] [--resumed-from N [--supersede] | --replace-csv] "
            "[--source-folder F]")
    steps = pos[0] if pos[0] in NAMED else str(int(pos[0]))
    supersede = "--supersede" in flags
    if supersede and (resumed_arg(flags) is None or (steps not in NAMED and int(steps) <= resumed_arg(flags))):
        die("--supersede needs --resumed-from N and a checkpoint the resumed session saved again: STEPS after N, "
            "or final, best, robust_best")
    use_source(pos[1], source)
    print(json.dumps(place_checkpoint(steps, pos[1], pos[2], "--no-csv" in flags, "--replace-csv" in flags,
                                      resumed_arg(flags), supersede), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
