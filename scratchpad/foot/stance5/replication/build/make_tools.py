"""Build SP/foot/stance5/replication/tools/ from SP/foot/stance5/tools/ (read-only): each tool copied with the
replication identity and the run-handling additions (criteria RUN HANDLING; the *_RH and JS_SOURCE lists), every change
an exact (old, new, count) replacement asserted to match, and the whole set of diffs written to tools/CHANGES.diff.

    python3 -I make_tools.py            writes the tools (refuses to overwrite a different existing file unless --force)
    python3 -I make_tools.py --check    rebuilds in memory and compares with the files on disk (writes nothing)
    python3 -I make_tools.py --help     this text; any other argument is refused, and a file whose content (and mode) is
                                        already the built one is not rewritten
"""
from __future__ import annotations

import difflib
import os
import stat
import sys

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
SRC = SCRATCH + "/foot/stance5/tools"
DST = SCRATCH + "/foot/stance5/replication/tools"

RUNS_JS_OLD = "const SEEDS = { stance5_LTSB_s52: 52, stance5_LTSB_s60: 60, stance5_LTSB_s42: 42 }"
RUNS_JS_NEW = "const SEEDS = { stance5_LTSB_s70: 70, stance5_LTSB_s80: 80 }"
DRIVE_OLD = """const DRIVE_IDS = {
  stance5_LTSB_s52: { folder: '1Va3QrqNOpQfXS0z7mCVdF50KqUpn1lkv', stance: '1fLQ_sTkaP8AVvjFrISb5Fsf8Lgg4L2Ve', models: '1yLd5AZ5nw9W_M06886B1xaWRHLKyHtx4' },
  stance5_LTSB_s60: { folder: '1ep6i1eLa4S7yjG3vHwEnjg2a5S4ATsDC', stance: '1PmPXkiZSbCQW4JP2mFpYYtqPCcXGnMVl', models: '154V5GHWnryuO4RQbt7POLIzg6YH2iA-F' },
  stance5_LTSB_s42: { folder: '14zl7lwcSnpB7dxr5gfuhDDVzo937uXq3', stance: '1neBe5o8wvIgdQnidnZepYjQWia0f_AcI', models: '1KFu_ONME7_lzH2k-iL57TvoBdwYy3fCc' },
}"""
DRIVE_NEW = """const DRIVE_IDS = {
  stance5_LTSB_s70: { folder: null, stance: null, models: null },
  stance5_LTSB_s80: { folder: null, stance: null, models: null },
}"""
DRIVE_COMMENT_OLD = """// Drive folder ids (read-only): PLACEHOLDERS (null) until the three runs exist on Drive.  Fill them here once (from
// stance5-find-folders.js, or search_files title = RUN_ID), or pass them per call as"""
DRIVE_COMMENT_NEW = """// Drive folder ids (read-only): PLACEHOLDERS (null) until the two replication runs exist on Drive.  Fill them here once
// (from stance5r-find-folders.js, or search_files title = RUN_ID), or pass them per call as"""
JS_COMMON = [
    ("const S5 = S + '/foot/stance5'\nconst F = S5 + '/fetch'\nconst TOOLS = S5 + '/tools'",
     "const R = S + '/foot/stance5/replication'\nconst F = R + '/fetch'\nconst TOOLS = R + '/tools'", 1),
    ("const PLACE = TOOLS + '/place5.py'", "const PLACE = TOOLS + '/place5r.py'", 1),
    (DRIVE_COMMENT_OLD, DRIVE_COMMENT_NEW, 1),
    (DRIVE_OLD, DRIVE_NEW, 1),
    (RUNS_JS_OLD, RUNS_JS_NEW, 1),
    ("args.drive names no stance5 run", "args.drive names no stance5 replication run", 1),
    ("(stance5, arm LTSB, seed ${run.seed})", "(stance5 replication, arm LTSB, seed ${run.seed})", 1),
]

QUEUE_HEADER_OLD = "# queue5.sh judge|panels [--dry-run | --once | --watch | --stop] : the stance5 judging and panels queues.\n"
QUEUE_HEADER_NEW = """# queue5r.sh judge|panels [--dry-run | --once | --watch | --stop] : the stance5 replication judging and panels queues.
# HOW TO RUN IT IN THIS CONTAINER (the stance5 lesson, run_log 2026-10-21 23:48 UTC): a detached setsid worker is not
# tracked by the harness, and the idle container is reclaimed about 10 minutes after a turn ends, killing it.  Run each
# worker as a harness-tracked background command, by its ABSOLUTE path, under a time limit, and relaunch it when it exits:
#     timeout 7000 /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication/tools/queue5r.sh judge --workerwatch
#     timeout 7000 /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication/tools/queue5r.sh panels --workerwatch
# (Bash run_in_background, one command each).  A job cut by the limit leaves no output and is rerun by the next worker.
# --watch below (setsid nohup) is stance5's detached form, kept unchanged; do not rely on it here.  Never launch it as
# "bash queue5r.sh": the worker relaunches "$0" under nohup and needs a path (stance5, 2026-10-21 20:32 UTC).
# Everything below is SP/foot/stance5/tools/queue5.sh with the replication runs (s70, s80), judge_one5r.sh, panels5r_one.sh
# and the replication root (S5 below names SP/foot/stance5/replication), plus one status: "refused", a full-mode slot
# whose judge refused on the run's own records (judge_one5r.sh wrote refused_<steps>.json: an invalid J file, criteria
# RUN HANDLING).  It is not rerun, its panels and final extras are skipped, and it counts as settled.  And "review": a
# refusal on the checkpoint's own attributes only (refused_<steps>.review.json), held until the coordinator confirms it
# (renamed to refused_<steps>.json) or moves it aside; it is not rerun and not settled.
"""

# ------------------------------------------------------------------------------------------------ run handling
# criteria RUN HANDLING (the maintainer's rules of 2026-10-22 16:32 UTC, run_log): a restart under <RUN_ID>_b (then
# _c) before the first periodic checkpoint IS that run (the fetch workflows' args.source, place5r.py --source-folder); a
# judge refused on the run's own records is an invalid J file (judge_one5r.sh's refused_<steps>.json, the queue's
# "refused"); a file the resumed session saved again is moved aside, never deleted (place5r.py --supersede, and the
# ledgers it replaces).
JS_SOURCE = [
    ("const ALL_RUNS = Object.keys(SEEDS).map(id => ({ id, seed: SEEDS[id], ...DRIVE_IDS[id], ...(OVERRIDE[id] || {}) }))",
     r"""// criteria RUN HANDLING: a run cut off before its first periodic checkpoint (524,288) and restarted under <RUN_ID>_b
// (then _c) IS that run.  args.source = {RUN_ID: '<RUN_ID>_b'} fetches it from that Drive folder, whose ids come in
// args.drive in the same call, and places it under the RUN_ID (place5r.py --source-folder).
const SOURCE = (args && args.source && typeof args.source === 'object') ? args.source : {}
for (const [k, v] of Object.entries(SOURCE)) {
  if (!(k in SEEDS)) throw new Error(`args.source names no stance5 replication run: ${k}`)
  if (v !== `${k}_b` && v !== `${k}_c`) throw new Error(`args.source.${k} must be ${k}_b or ${k}_c (got ${v})`)
  const o = OVERRIDE[k] || {}
  if (!o.folder || !o.stance || !o.models) throw new Error(`args.source.${k} = ${v} needs that folder's ids in args.drive.${k} (folder, stance, models) in the same call`)
}
const ALL_RUNS = Object.keys(SEEDS).map(id => ({ id, seed: SEEDS[id], ...DRIVE_IDS[id], ...(OVERRIDE[id] || {}), src: SOURCE[id] || id }))
const SRC_FLAG = (run) => (run.src !== run.id ? ` --source-folder ${run.src}` : '')
const SRC_CHECK = (run) => (run.src === run.id ? '' : `This run was restarted under the Drive folder ${run.src} before its first checkpoint; under the criteria (RUN HANDLING) it IS ${run.id}, and the earlier folder is left alone. First run get_file_metadata on ${run.folder}: its title must be exactly ${run.src}. If it is not, fetch nothing, return every expected file with method "missing" and ok false, and put the title you found in notes.\n`)""", 1),
    ("for the Drive run ${run.id}: a T. rex", "for the Drive run ${run.src}: a T. rex", 1),
]
FETCH_CKPT_RH = [
    ("  whenToUse: 'args {steps: <multiple of 524288>, runs?: [RUN_ID...], configs?: bool, csv?: bool, place?: bool}',",
     "  whenToUse: 'args {steps: <multiple of 524288>, runs?: [RUN_ID...], configs?: bool, csv?: bool, place?: bool, drive?: {RUN_ID: {folder, stance, models}}, source?: {RUN_ID: \"<RUN_ID>_b\"}}',", 1),
    ("${RULES}\n\nList the models folder", "${RULES}\n${SRC_CHECK(run)}\nList the models folder", 1),
    ("cd ${TOOLS} && python3 -I ${PLACE} ${STEPS} ${run.id} ${F}/r${STEPS}/${run.id}\n",
     "cd ${TOOLS} && python3 -I ${PLACE} ${STEPS} ${run.id} ${F}/r${STEPS}/${run.id}${SRC_FLAG(run)}\n", 1),
    # the tie-break transcriber C names the Drive folder the files come from, as A and B do (args.source)
    ("You are transcriber C (tie-break) for the Drive run ${run.id}.", "You are transcriber C (tie-break) for the Drive run ${run.src}.", 1),
]
FETCH_NAMED_RH = [
    ("  whenToUse: 'args {tag, items: [[zipTitle, pklTitle], ...], csv?: bool, extras?: [title, ...], runs?: [RUN_ID...], place?: bool}',",
     "  whenToUse: 'args {tag, items: [[zipTitle, pklTitle], ...], csv?: bool, extras?: [title, ...], runs?: [RUN_ID...], place?: bool, drive?: {RUN_ID: {folder, stance, models}}, source?: {RUN_ID: \"<RUN_ID>_b\"}}',", 1),
    ("${RULES}\n\n${listing}", "${RULES}\n${SRC_CHECK(run)}\n${listing}", 1),
    ("${(CSV && i === 0) ? '' : ' --no-csv'}`)", "${(CSV && i === 0) ? '' : ' --no-csv'}${SRC_FLAG(run)}`)", 1),
    ("python3 -I ${PLACE} csv ${run.id} ${F}/${TAG}/${run.id}`]", "python3 -I ${PLACE} csv ${run.id} ${F}/${TAG}/${run.id}${SRC_FLAG(run)}`]", 1),
    ("python3 -I ${PLACE} extra ${run.id} ${F}/${TAG}/${run.id}`)", "python3 -I ${PLACE} extra ${run.id} ${F}/${TAG}/${run.id}${SRC_FLAG(run)}`)", 1),
    ("You are transcriber C (tie-break) for the Drive run ${run.id}.", "You are transcriber C (tie-break) for the Drive run ${run.src}.", 1),
]
PLACE_RH = [
    ('RUN_LABEL (it never refuses on it).  place5.py\'s text follows, with place5.py read as place5r.py, S5 as R and\n'
     'satwatch5.py as satwatch5r.py:\n',
     'RUN_LABEL (it never refuses on it).  Three additions carry out the run-handling rules (criteria RUN HANDLING):\n'
     '  --source-folder <RUN_ID>_b (or _c), with config, STEPS, csv or extra: the files come from that Drive folder, the\n'
     '    restart of a run cut off before its first periodic checkpoint, and are placed under the RUN_ID.  The first such\n'
     '    call switches the run (R/runs/<RUN_ID>/source_folder.json).  The switch is refused once a periodic checkpoint is\n'
     '    placed from the earlier folder; what the earlier folder placed is moved aside.  After it, every call names the\n'
     '    folder (an earlier one is refused), and progress.json must carry the folder as its run_id.\n'
     '  --supersede, with STEPS and --resumed-from N (STEPS after N, or final, best, robust_best): a placed pair that the\n'
     '    resumed session saved again is moved aside with its judge files and refusal record (every mode), its panels\n'
     '    output, its saturation roll and, at final, its certificate folder.  The new pair is placed, and the queue\n'
     '    judges the slot again.  A ledger replaced under --resumed-from or --replace-csv is moved aside too.  With\n'
     '    --resumed-from N, STEPS and csv also print "placed_after_N", the periodic pairs placed after N: one placed\n'
     '    before the resume is the old session\'s; once the resumed session has saved it again, fetch it again and place it\n'
     '    with --resumed-from N --supersede.\n'
     '  --run-records-fail, with config RUN_ID DIR DIR2: the run\'s own records fail the checks below, or are absent on\n'
     '    Drive, and DIR and DIR2 agree (byte-equal copies, or neither holds the file).  They are placed as they are (never\n'
     '    over a different placed copy) beside R/runs/<RUN_ID>/01_stance/records_check_failed.json (the problems, the\n'
     '    files absent on Drive, the sha256s).  While it stands, checkpoint pairs are placed without the absent records,\n'
     '    the judge refuses on the records, and judge_one5r.sh writes refused_<steps>.json: an invalid J file.\n'
     '  Moved aside means R/superseded/<RUN_ID>/<UTC time>/<the path under R>, listed in SUPERSEDED.json there; nothing\n'
     '  is deleted.\n'
     'place5.py\'s text follows (with these additions), with place5.py read as place5r.py, S5 as R and satwatch5.py as\n'
     'satwatch5r.py:\n', 1),
    ('    python3 -I place5r.py config RUN_ID DIR [DIR2]       the three run records, once\n',
     '    python3 -I place5r.py config RUN_ID DIR [DIR2]       the three run records, once (DIR DIR2 --run-records-fail: above)\n', 1),
    ('    python3 -I place5r.py STEPS RUN_ID FETCH_DIR [--no-csv] [--resumed-from N | --replace-csv]\n',
     '    python3 -I place5r.py STEPS RUN_ID FETCH_DIR [--no-csv] [--resumed-from N [--supersede] | --replace-csv]\n', 1),
    ('a restarted run\'s\n"<RUN_ID>_b" is not known here: stop and ask, as the Colab steps say).\n',
     'a restart under\n"<RUN_ID>_b" or "_c" is placed under its RUN_ID with --source-folder, above).\n', 1),
    ('  A placed checkpoint file is never overwritten: an identical copy is a no-op, a different one is refused.\n',
     '  A placed checkpoint file is never overwritten: an identical copy is a no-op, a different one is refused (with\n'
     '  --supersede it is moved aside first).\n', 1),
    ('import sys\nimport zipfile\n', 'import sys\nimport time\nimport zipfile\n', 1),
    ('RUN_LABELS = {"stance5_LTSB_s70": "stance screen LTSB s70", "stance5_LTSB_s80": "stance screen LTSB s80"}  # notes only\n',
     'RUN_LABELS = {"stance5_LTSB_s70": "stance screen LTSB s70", "stance5_LTSB_s80": "stance screen LTSB s80"}  # notes only\n'
     'RESTARTS = ("_b", "_c")  # criteria RUN HANDLING: a restart before the first periodic checkpoint is that run\n'
     'JUDGED = ("judge_{s}.json", "judge_quick_{s}.json", "judge_fresh_{s}.json", "judge_tail_{s}.json",\n'
     '          "judge_common_noise_{s}.json", "refused_{s}.json", "refused_{s}.review.json")  # judge_one5r.sh\'s, per checkpoint\n'
     'RECORDS_FAILED = "records_check_failed.json"  # criteria RUN HANDLING: config --run-records-fail (the run\'s records)\n', 1),
    ('    sys.exit(1)\n\n\ndef stage_dir(run):\n',
     r'''    sys.exit(1)


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
''', 1),
    ('            if new.get("run_id") != run:\n                out[name] = f"REFUSED: run_id {new.get(\'run_id\')} is not {run}"\n',
     '            if new.get("run_id") != current_source(run):  # a restart\'s progress.json names its own folder\n'
     '                out[name] = f"REFUSED: run_id {new.get(\'run_id\')} is not {current_source(run)}"\n', 1),
    ('        tmp = d + ".part"\n        with open(tmp, "wb") as fh:\n            fh.write(new)\n        os.replace(tmp, d)\n'
     '        if os.path.exists(marker):\n            os.remove(marker)\n'
     '        out[name] = {"action": "placed", "bytes": len(new), "sha256": sha(d)}\n        if note:\n'
     '            out[name]["note"] = note\n',
     '        moved = None\n'
     '        if note is not None and os.path.exists(d):  # criteria RUN HANDLING: the superseded copy is moved aside\n'
     '            moved = move_aside(os.path.basename(os.path.dirname(dst)), [d],\n'
     '                               f"{name}: superseded by {os.path.abspath(s)} ("\n'
     '                               + (f"--resumed-from {resumed_from}" if resumed_from is not None else "--replace-csv") + ")")\n'
     '        tmp = d + ".part"\n        with open(tmp, "wb") as fh:\n            fh.write(new)\n        os.replace(tmp, d)\n'
     '        if os.path.exists(marker):\n            os.remove(marker)\n'
     '        out[name] = {"action": "placed", "bytes": len(new), "sha256": sha(d)}\n        if note:\n'
     '            out[name]["note"] = note\n'
     '        if moved:\n            out[name]["moved_aside"] = moved\n', 1),
    ('def place_checkpoint(steps, run, fetch, no_csv, replace_csv, resumed_from=None):\n',
     'def place_checkpoint(steps, run, fetch, no_csv, replace_csv, resumed_from=None, supersede=False):\n', 1),
    ('        die(f"{zsrc}: testzip failed at {badm}")\n    res["pkl"] = {',
     '        die(f"{zsrc}: testzip failed at {badm}")\n'
     '    if supersede:  # criteria RUN HANDLING: the resumed session saved this checkpoint again\n'
     '        placed = ((os.path.join(models, zname), zsrc), (os.path.join(models, pname), pkl_src))\n'
     '        if any(os.path.exists(d) and sha(d) != sha(s) for d, s in placed):\n'
     '            res["superseded"] = move_aside(run, superseded_paths(run, steps, zname, pname),\n'
     '                                           f"{steps}: saved again by the session resumed from {resumed_from:,}")\n'
     '        else:\n'
     '            res["notes"].append("--supersede: no different pair is placed; nothing was moved aside")\n'
     '    res["pkl"] = {', 1),
    ('die(f"{dst} exists and differs from {src}; refusing to overwrite")',
     'die(f"{dst} exists and differs from {src}; refusing to overwrite (a checkpoint the resumed session saved again: "\n'
     '            "--resumed-from N --supersede)")', 1),
    ('        ent["checkpoints"] = pairs\n',
     '        ent["checkpoints"] = pairs\n        ent["source_folder"] = current_source(run)\n'
     '        if read_failed_marker(run) is not None:  # criteria RUN HANDLING: config --run-records-fail\n'
     '            ent["records_check_failed"] = read_failed_marker(run)\n', 1),
    # criteria RUN HANDLING (judge refusals): while records_check_failed.json stands, a pair is placed without the
    # records absent on Drive, and a configs:true fetch must carry the records the marker holds
    ('    if all(os.path.exists(os.path.join(fetch, r, n)) for r in ("A", "B") for n in RECORDS):\n'
     '        rec, notes = place_records(run, os.path.join(fetch, "A"), os.path.join(fetch, "B"))\n'
     '        res["records"], res["notes"] = rec, res["notes"] + notes\n'
     '    else:\n'
     '        missing = [n for n in RECORDS if not os.path.exists(os.path.join(dst, n))]\n'
     '        if missing:\n'
     '            die(f"records not placed yet ({missing}); run \'place5r.py config {run} DIR\' or fetch with configs: true")\n'
     '        res["records"] = {n: sha(os.path.join(dst, n)) for n in RECORDS}\n',
     '    failed = read_failed_marker(run)  # criteria RUN HANDLING: the run\'s records placed with --run-records-fail\n'
     '    if all(os.path.exists(os.path.join(fetch, r, n)) for r in ("A", "B") for n in RECORDS):\n'
     '        rec, notes = (place_failed_records if failed is not None else place_records)(\n'
     '            run, os.path.join(fetch, "A"), os.path.join(fetch, "B"))\n'
     '        res["records"], res["notes"] = rec, res["notes"] + notes\n'
     '    else:\n'
     '        absent = (failed or {}).get("absent_on_drive") or []\n'
     '        missing = [n for n in RECORDS if not os.path.exists(os.path.join(dst, n)) and n not in absent]\n'
     '        if missing:\n'
     '            die(f"records not placed yet ({missing}); run \'place5r.py config {run} DIR\' or fetch with configs: true "\n'
     '                f"(records that fail the checks or are absent on Drive, A and B agreeing: \'place5r.py config {run} "\n'
     '                "DIR DIR2 --run-records-fail\')")\n'
     '        res["records"] = {n: sha(os.path.join(dst, n)) for n in RECORDS if os.path.exists(os.path.join(dst, n))}\n'
     '    if failed is not None:\n'
     '        res["notes"].append(f"the run\'s records failed the checks ({RECORDS_FAILED}): the judge refuses on them, and "\n'
     '                            "judge_one5r.sh records refused_<steps>.json, an invalid J file (criteria RUN HANDLING)")\n', 1),
    ('"<name>.refused.json; satwatch5r.py reports them until a placement succeeds)")\n    return res\n',
     '"<name>.refused.json; satwatch5r.py reports them until a placement succeeds)")\n'
     '    if resumed_from is not None:  # criteria RUN HANDLING (resumes): the old session\'s pairs after N, if any\n'
     '        res.update(after_resume(run, resumed_from, zname if res["zip"]["action"] == "placed" else None))\n'
     '    return res\n', 1),
    ('        ok = not refused(res)\n        print(json.dumps({"ok": ok, "run": pos[0], "csv": res}, indent=1))\n',
     '        ok = not refused(res)\n        out = {"ok": ok, "run": pos[0], "csv": res}\n'
     '        if resumed_arg(flags) is not None:  # criteria RUN HANDLING (resumes): the old session\'s pairs after N, if any\n'
     '            out.update(after_resume(pos[0], resumed_arg(flags)))\n'
     '        print(json.dumps(out, indent=1))\n', 1),
    ('        print(__doc__)\n        return 0\n    if argv[0] == "config":\n        if len(argv) not in (3, 4):\n'
     '            die("usage: place5r.py config RUN_ID DIR [DIR2]")\n',
     '        print(__doc__)\n        return 0\n'
     '    source = None  # criteria RUN HANDLING: --source-folder <RUN_ID>_b (or _c), a restart placed under its RUN_ID\n'
     '    if "--source-folder" in argv:\n'
     '        i = argv.index("--source-folder")\n'
     '        if argv.count("--source-folder") > 1 or i + 1 >= len(argv) or argv[i + 1].startswith("--"):\n'
     '            die("--source-folder takes one Drive folder name (<RUN_ID>_b or <RUN_ID>_c)")\n'
     '        source, argv = argv[i + 1], argv[:i] + argv[i + 2:]\n'
     '        if not argv or argv[0] in ("status", "-h", "--help"):\n'
     '            die("--source-folder goes with config, STEPS, csv or extra")\n'
     '    fail = "--run-records-fail" in argv  # criteria RUN HANDLING: the run\'s own records fail the checks\n'
     '    if fail:\n'
     '        argv = [a for a in argv if a != "--run-records-fail"]\n'
     '        if not argv or argv[0] != "config" or len(argv) != 4:\n'
     '            die("--run-records-fail goes with config RUN_ID DIR DIR2 (A and B must agree)")\n'
     '    if argv[0] == "config":\n        if len(argv) not in (3, 4):\n'
     '            die("usage: place5r.py config RUN_ID DIR [DIR2] [--source-folder F] [--run-records-fail]")\n'
     '        use_source(argv[1], source)\n'
     '        if fail:\n'
     '            rec, notes = place_failed_records(argv[1], argv[2], argv[3])\n'
     '            print(json.dumps({"ok": True, "run": argv[1], "records": rec, "notes": notes,\n'
     '                              "records_check_failed": read_failed_marker(argv[1])}, indent=1))\n'
     '            return 0\n', 1),
    ('            die("usage: place5r.py csv RUN_ID FETCH_DIR [--resumed-from N | --replace-csv]")\n        dst = stage_dir(pos[0])\n',
     '            die("usage: place5r.py csv RUN_ID FETCH_DIR [--resumed-from N | --replace-csv] [--source-folder F]")\n'
     '        use_source(pos[0], source)\n        dst = stage_dir(pos[0])\n', 1),
    ('            die("usage: place5r.py extra RUN_ID FETCH_DIR")\n        res = place_extra(argv[1], argv[2])\n',
     '            die("usage: place5r.py extra RUN_ID FETCH_DIR [--source-folder F]")\n'
     '        use_source(argv[1], source)\n        res = place_extra(argv[1], argv[2])\n', 1),
    ('    unknown = [f for f in flags if f not in ("--no-csv", "--replace-csv") and not f.startswith("--resumed-from=")]\n'
     '    if len(pos) != 3 or unknown or ("--replace-csv" in flags and resumed_arg(flags) is not None):\n'
     '        die("usage: place5r.py STEPS RUN_ID FETCH_DIR [--no-csv] [--resumed-from N | --replace-csv]")\n'
     '    steps = pos[0] if pos[0] in NAMED else str(int(pos[0]))\n'
     '    print(json.dumps(place_checkpoint(steps, pos[1], pos[2], "--no-csv" in flags, "--replace-csv" in flags,\n'
     '                                      resumed_arg(flags)), indent=1))\n',
     '    unknown = [f for f in flags if f not in ("--no-csv", "--replace-csv", "--supersede")\n'
     '               and not f.startswith("--resumed-from=")]\n'
     '    if len(pos) != 3 or unknown or ("--replace-csv" in flags and resumed_arg(flags) is not None):\n'
     '        die("usage: place5r.py STEPS RUN_ID FETCH_DIR [--no-csv] [--resumed-from N [--supersede] | --replace-csv] "\n'
     '            "[--source-folder F]")\n'
     '    steps = pos[0] if pos[0] in NAMED else str(int(pos[0]))\n'
     '    supersede = "--supersede" in flags\n'
     '    if supersede and (resumed_arg(flags) is None or (steps not in NAMED and int(steps) <= resumed_arg(flags))):\n'
     '        die("--supersede needs --resumed-from N and a checkpoint the resumed session saved again: STEPS after N, "\n'
     '            "or final, best, robust_best")\n'
     '    use_source(pos[1], source)\n'
     '    print(json.dumps(place_checkpoint(steps, pos[1], pos[2], "--no-csv" in flags, "--replace-csv" in flags,\n'
     '                                      resumed_arg(flags), supersede), indent=1))\n', 1),
]
JUDGE_RH = [
    ("# --check exits 0 when the output is complete, 1 when it is not, 3 when the checkpoint is not placed.  Writes nothing.\n",
     "# --check exits 0 when the output is complete, 1 when it is not, 3 when the checkpoint is not placed, 5 (full mode)\n"
     "# when the slot holds a refusal record, and 6 (full mode) when it holds a refusal held for review.  Writes nothing.\n"
     "# REFUSALS (criteria RUN HANDLING, the maintainer's rule of 2026-10-22 16:32 UTC): a full-mode judge that refuses on\n"
     "# the run's own records (seed, commit, n_envs, hyperparameters, task, env kwargs, gate bars, plant, a record absent on\n"
     "# Drive: the judge's \"refusing (nothing written): ...\" line naming stage_config.json, plant_identity.json or\n"
     "# task_fingerprint.json) leaves R/judged/RUN_ID/refused_STEPS.json with the refusal text and exits 5.\n"
     "# score_stance5r.py counts that record as an INVALID J file.  While it stands, a later call exits 5 and judges\n"
     "# nothing.  A record is absent on Drive only when place5r.py config --run-records-fail listed it so\n"
     "# (01_stance/records_check_failed.json); any other missing record is a local file not placed.  An operator-side\n"
     "# refusal (a wrong path or clone, a missing local file: no such line, a record not placed, or a symmetry check the\n"
     "# judge could not run) writes no record: fix it and judge again.  A refusal that names only the checkpoint's own\n"
     "# attributes (its seed, plant, task or symmetry record) while the run's records pass is most likely a zip fetched\n"
     "# from another run's folder (the operator's side): it is held for review as refused_STEPS.review.json (exit 6; the\n"
     "# scorer does not read it), and nothing is judged while it stands.  Confirmed as the run's (the fetched zip's parent\n"
     "# is the run's models folder): rename it to refused_STEPS.json.  Otherwise move it and the placed pair aside by hand\n"
     "# and fetch the slot again.  A record found to be the operator's side is moved aside by hand (never deleted), and\n"
     "# the slot is judged again.\n"
     "# JUDGE5R_TEST_ROOT (tests only): runs/ and judged/ under a scratch root; the judge stays R's.\n", 1),
    ("JUDGE=$S5/judge/judge_stance5r.py\n",
     "JUDGE=$S5/judge/judge_stance5r.py\n"
     "S5=${JUDGE5R_TEST_ROOT:-$S5}   # tests only: runs/ and judged/ under a scratch root (the judge above stays R's)\n", 1),
    ("JL=$LOGD/${RUN}_$NAME.log\n", "JL=$LOGD/${RUN}_$NAME.log\nREFUSED=$OUTD/refused_$STEPS.json   # criteria RUN HANDLING (full mode)\n"
     "REVIEW=$OUTD/refused_$STEPS.review.json; MARKER=$RD/01_stance/records_check_failed.json   # (full mode)\n", 1),
    ("EOF\n}\n\nif [ \"$HEAD_EXPECT\" = PLACEHOLDER_HEAD_LTSB ]",
     r'''EOF
}

# refusal RC : 0 after writing REFUSED from the judge log when the judge refused on the run's own records; 2 after
# writing REVIEW when it refused on the checkpoint's own attributes only (held for review); else 1 (nothing written).
refusal() {
  python3 -I - "$JL" "$REFUSED" "$RUN" "$SEED" "$STEPS" "$JUDGE" "$Z" "$P" "$HEAD_EXPECT" "$1" "$REVIEW" "$MARKER" <<'EOF'
import datetime, hashlib, json, os, re, sys
log, out, run, seed, steps, judge, z, p, head, rc, review, marker = sys.argv[1:13]
PREFIX = "[judge_stance5r] refusing (nothing written): "
RECS = r"(plant_identity|stage_config|task_fingerprint)\.json"
hits = []
if os.path.exists(log):
    with open(log, errors="replace") as fh:
        hits = [x.rstrip("\n")[len(PREFIX):] for x in fh if x.startswith(PREFIX)]
if not hits:
    print("no refusal on the run's own records in the log (the operator's side, or a crash): nothing recorded")
    sys.exit(1)
items = [x.strip() for x in hits[-1].split(" | ") if x.strip()]
# A record the judge found missing is the run's own only when place5r.py config --run-records-fail listed it as absent
# on Drive (records_check_failed.json); otherwise it is a local file not placed: the operator's side.
absent = []
if os.path.exists(marker):
    with open(marker) as fh:
        absent = json.load(fh).get("absent_on_drive") or []


def missing(x):
    m = re.search(r"/" + RECS + r" missing \(required\)$", x)
    return m.group(1) + ".json" if m else None


operator = [x for x in items if (missing(x) and missing(x) not in absent) or "(not checked; clone mirror map" in x]
if operator:
    print("the operator's side (fix it and judge again; nothing recorded): " + " | ".join(operator))
    sys.exit(1)
run_records = [x for x in items if re.search(r"(^|/)" + RECS, x)]


def sha(path):
    with open(path, "rb") as fh:
        return "sha256:" + hashlib.sha256(fh.read()).hexdigest()


rec = {"schema": "stance5r-refusal/v1", "run": run, "seed": int(seed), "steps": int(steps) if steps.isdigit() else steps,
       "mode": "full", "refusal": hits[-1], "items": items, "judge": judge, "judge_sha256": sha(judge),
       "clone_head": head, "model": z, "model_sha256": sha(z), "vecnorm": p, "vecnorm_sha256": sha(p), "log": log,
       "judge_rc": int(rc), "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "rule": "criteria RUN HANDLING: a judge refused on the run's own records counts as an INVALID J file"}
if not run_records:
    # Only the checkpoint's own attributes (its seed, plant, task or symmetry record) while the run's placed records
    # pass: most likely a zip fetched from another run's models folder, the operator's side.  Held for review; the
    # scorer does not read this file.
    rec["held_for_review"] = (
        "the refusal names only the checkpoint's own attributes while the run's placed records pass: confirm with "
        "get_file_metadata that the fetched zip's parent is this run's models folder.  If it is, rename this file to "
        f"{os.path.basename(out)} (an invalid J file); if not, move it and the placed pair aside by hand and fetch the "
        "slot again (tools/README.md, Run handling rule 2)")
    with open(review, "x") as fh:
        json.dump(rec, fh, indent=1)
    print("held for review " + review + " (the checkpoint's own attributes only: " + " | ".join(items) + ")")
    sys.exit(2)
with open(out, "x") as fh:
    json.dump(rec, fh, indent=1)
print("recorded " + out)
EOF
}

if [ "$HEAD_EXPECT" = PLACEHOLDER_HEAD_LTSB ]''', 1),
    ("  [ -e $OUT ] && valid $OUT >/dev/null 2>&1 && exit 0\n  for f in $Z $P; do [ -f $f ] || exit 3; done\n  exit 1\n",
     "  [ -e $OUT ] && valid $OUT >/dev/null 2>&1 && exit 0\n  [ \"$MODE\" = full ] && [ -e $REFUSED ] && exit 5\n"
     "  [ \"$MODE\" = full ] && [ -e $REVIEW ] && exit 6\n"
     "  for f in $Z $P; do [ -f $f ] || exit 3; done\n  exit 1\n", 1),
    ("  echo \"free KB on scratch: $(df -Pk $S | awk 'NR==2{print $4}')\"\n",
     "  [ \"$MODE\" = full ] && [ -e $REFUSED ] && echo \"REFUSAL RECORD $REFUSED: the judge refused on the run's own records (an invalid J file); nothing is judged while it stands\"\n"
     "  [ \"$MODE\" = full ] && [ -e $REVIEW ] && echo \"REFUSAL HELD FOR REVIEW $REVIEW: the checkpoint's own attributes only (confirm the zip's Drive parent; tools/README.md rule 2); nothing is judged while it stands\"\n"
     "  echo \"free KB on scratch: $(df -Pk $S | awk 'NR==2{print $4}')\"\n", 1),
    ("  mv $OUT $OUT.bad.$(date +%Y%m%dT%H%M%S); echo \"moved-aside $(date -Is) $RUN $NAME (incomplete)\" >> $LOGD/queue.log\nfi\n",
     "  mv $OUT $OUT.bad.$(date +%Y%m%dT%H%M%S); echo \"moved-aside $(date -Is) $RUN $NAME (incomplete)\" >> $LOGD/queue.log\nfi\n"
     "if [ \"$MODE\" = full ] && [ -e $REFUSED ]; then echo \"refused-recorded $(date -Is) $RUN $NAME ($REFUSED)\" >> $LOGD/queue.log; exit 5; fi\n"
     "if [ \"$MODE\" = full ] && [ -e $REVIEW ]; then echo \"held-for-review $(date -Is) $RUN $NAME ($REVIEW)\" >> $LOGD/queue.log; exit 6; fi\n", 1),
    ("if [ $rc -eq 0 ] && ! valid $OUT >/dev/null 2>&1; then rc=4; fi\n",
     "if [ $rc -eq 0 ] && ! valid $OUT >/dev/null 2>&1; then rc=4; fi\n"
     "if [ $rc -ne 0 ] && [ \"$MODE\" = full ] && [ ! -e $OUT ]; then  # criteria RUN HANDLING: a refusal on the run's records\n"
     "  msg=$(refusal $rc); case $? in 0) rc=5;; 2) rc=6;; esac\n"
     "  echo \"$msg\"; echo \"refusal $(date -Is) $RUN $NAME: $msg\" >> $LOGD/queue.log\n"
     "fi\n", 1),
]
QUEUE_RH = [
    ("status_of() {  # status_of RUN SEED STEPS MODE -> done | ready | missing | waiting-judge | failed\n",
     "status_of() {  # status_of RUN SEED STEPS MODE -> done | ready | missing | waiting-judge | refused | review | failed\n", 1),
    ("      $T/judge_one5r.sh --check final $run $seed full || { echo waiting-judge; return; }\n",
     "      $T/judge_one5r.sh --check final $run $seed full; case $? in 0) ;; 5) echo refused; return;; 6) echo review; return;; *) echo waiting-judge; return;; esac\n", 1),
    ("    if [ $rc -eq 1 ]; then $T/judge_one5r.sh --check $steps $run $seed full || { echo waiting-judge; return; }; fi\n",
     "    if [ $rc -eq 1 ]; then $T/judge_one5r.sh --check $steps $run $seed full; case $? in 0) ;; 5) echo refused; return;; 6) echo review; return;; *) echo waiting-judge; return;; esac; fi\n", 1),
    ("  case $rc in 0) echo done;; 3) echo missing;;\n", "  case $rc in 0) echo done;; 3) echo missing;; 5) echo refused;; 6) echo review;;\n", 1),
    ("  if [ $rc -eq 0 ]; then say \"ok $run $steps $mode $(( $(date +%s) - t0 ))s\"\n",
     "  if [ $rc -eq 0 ]; then say \"ok $run $steps $mode $(( $(date +%s) - t0 ))s\"\n"
     "  elif [ $rc -eq 5 ]; then say \"refused $run $steps $mode: the judge refused on the run's own records (refused_$steps.json, an invalid J file; criteria RUN HANDLING)\"\n"
     "  elif [ $rc -eq 6 ]; then say \"review $run $steps $mode: the judge refused on the checkpoint's own attributes only (refused_$steps.review.json, held for review; tools/README.md rule 2)\"\n", 1),
    ("all_done() { local line; while read -r line; do set -- $line; [ \"$(status_of $1 $2 $3 $4)\" = done ] || return 1; done < <(joblist); return 0; }\n",
     "all_done() { local line; while read -r line; do set -- $line; case \"$(status_of $1 $2 $3 $4)\" in done|refused) ;; *) return 1;; esac; done < <(joblist); return 0; }\n", 1),
]

FILES = {
    "stance5r-find-folders.js": ("stance5-find-folders.js", [
        ("name: 'stance5-find-folders'", "name: 'stance5r-find-folders'", 1),
        ("of the three stance5 LTSB runs by RUN_ID, read-only, to fill DRIVE_IDS in the stance5 fetch workflows",
         "of the two stance5 replication LTSB runs (s70, s80) by RUN_ID, read-only, to fill DRIVE_IDS in the stance5r fetch workflows", 1),
        ("const TOOLS = S + '/foot/stance5/tools'", "const TOOLS = S + '/foot/stance5/replication/tools'", 1),
        (RUNS_JS_OLD, RUNS_JS_NEW + "\nconst LABELS = { stance5_LTSB_s70: 'stance screen LTSB s70', stance5_LTSB_s80: 'stance screen LTSB s80' }", 1),
        ("args.runs names no stance5 run", "args.runs names no stance5 replication run", 1),
        ("(a T. rex stance5 screen run, seed ${SEEDS[run]})", "(a T. rex stance5 replication run, seed ${SEEDS[run]})", 1),
        ("A folder titled \"${run}_b\" is a different (restarted) run: do not use it, but mention it in notes.",
         "A folder titled \"${run}_b\" (or \"${run}_c\") is this run's restart: under the criteria (RUN HANDLING) it IS "
         "this run when the earlier folder was cut off before its first periodic checkpoint (524,288). Do not put its ids in "
         "folder, stance or models; report in notes its id, its 01_stance and models ids, and whether the earlier folder's "
         "models folder lists any stage1_*_steps.zip (search_files query \"parentId = '<models id>' and title contains "
         "'_steps.zip'\"). The fetch workflows take its ids as args.drive with args.source. Also search once "
         "for a folder titled '${LABELS[run]}' (the RUN_LABEL typed into RUN_ID by mistake, as happened once on stance5): if "
         "one exists, do not use it; report its id and modifiedTime in notes.", 1),
        ("DRIVE_IDS for ${TOOLS}/stance5-fetch-*.js", "DRIVE_IDS for ${TOOLS}/stance5r-fetch-*.js", 1),
    ]),
    "stance5r-fetch-checkpoint.js": ("stance5-fetch-checkpoint.js", [
        ("name: 'stance5-fetch-checkpoint'", "name: 'stance5r-fetch-checkpoint'", 1),
        ("'Fetch one stance5 checkpoint step for the three LTSB runs from Drive",
         "'Fetch one checkpoint step for the two stance5 replication LTSB runs (s70, s80) from Drive", 1),
        ("'place5.py copies the agreed pkl and the zip into S5/runs/<RUN>/01_stance/models'",
         "'place5r.py copies the agreed pkl and the zip into R/runs/<RUN>/01_stance/models (R = S5/replication)'", 1),
        ("// run place5.py when the pkl transcriptions agree", "// run place5r.py when the pkl transcriptions agree", 1),
        ("log(`${run.id}: place5.py refused:", "log(`${run.id}: place5r.py refused:", 1),
    ] + JS_COMMON + JS_SOURCE + FETCH_CKPT_RH),
    "stance5r-fetch-named.js": ("stance5-fetch-named.js", [
        ("name: 'stance5-fetch-named'", "name: 'stance5r-fetch-named'", 1),
        ("progress.json) for the three LTSB runs from Drive", "progress.json) for the two stance5 replication LTSB runs (s70, s80) from Drive", 1),
        ("'place5.py copies each agreed pair (and the ledgers) into S5/runs/<RUN>/01_stance'",
         "'place5r.py copies each agreed pair (and the ledgers) into R/runs/<RUN>/01_stance (R = S5/replication)'", 1),
        ("// The checkpoint names place5.py understands", "// The checkpoint names place5r.py understands", 1),
        ("log(`${run.id}: place5.py did not place everything", "log(`${run.id}: place5r.py did not place everything", 1),
    ] + JS_COMMON + JS_SOURCE + FETCH_NAMED_RH),
    "place5r.py": ("place5.py", [
        ('"""place5.py: place fetched stance5 files into S5/runs/<RUN_ID>/01_stance/ (the notebook layout judge_stance5.py reads).\n',
         '"""place5r.py: place fetched stance5 REPLICATION files into R/runs/<RUN_ID>/01_stance/ (R = SP/foot/stance5/replication;\n'
         'the notebook layout judge_stance5r.py reads).  SP/foot/stance5/tools/place5.py with the replication runs\n'
         '(stance5_LTSB_s70, stance5_LTSB_s80) and root; every check is place5.py\'s (the same commit, task, band keys and\n'
         'collapse reference), plus one report-only note: "config" notes a stage_config.json run.label that is not the run\'s\n'
         'RUN_LABEL (it never refuses on it).  place5.py\'s text follows, with place5.py read as place5r.py, S5 as R and\n'
         'satwatch5.py as satwatch5r.py:\n', 1),
        ('RUN_ID: stance5_LTSB_s52, stance5_LTSB_s60 or stance5_LTSB_s42 (the seed',
         'RUN_ID: stance5_LTSB_s70 or stance5_LTSB_s80 (the seed', 1),
        ('S5 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5"\n'
         'S5 = os.environ.get("PLACE5_TEST_ROOT") or S5  # tests only: place under a scratch root instead of S5\n'
         'RUNS = {"stance5_LTSB_s52": 52, "stance5_LTSB_s60": 60, "stance5_LTSB_s42": 42}\n',
         'S5 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication"  # R\n'
         'S5 = os.environ.get("PLACE5R_TEST_ROOT") or S5  # tests only: place under a scratch root instead of R\n'
         'RUNS = {"stance5_LTSB_s70": 70, "stance5_LTSB_s80": 80}\n'
         'RUN_LABELS = {"stance5_LTSB_s70": "stance screen LTSB s70", "stance5_LTSB_s80": "stance screen LTSB s80"}  # notes only\n', 1),
        ('    os.makedirs(dst, exist_ok=True)\n    out, notes = {}, []\n',
         '    os.makedirs(dst, exist_ok=True)\n    out, notes = {}, []\n'
         '    label = (json.loads(raws["stage_config.json"].decode("utf-8")).get("run") or {}).get("label")\n'
         '    if label != RUN_LABELS[run]:  # report only: the RUN_ID (this folder), the seed and the commit are what bind\n'
         '        notes.append(f"stage_config.json run.label {label!r} is not {RUN_LABELS[run]!r} (report only)")\n', 1),
        ('"<name>.refused.json; satwatch5.py reports them until a placement succeeds)"',
         '"<name>.refused.json; satwatch5r.py reports them until a placement succeeds)"', 1),
        ("    python3 -I place5.py ", "    python3 -I place5r.py ", 5),
        ("(satwatch5.py reports it as a ledger", "(satwatch5r.py reports it as a ledger", 1),
        ("run 'place5.py config {run} DIR'", "run 'place5r.py config {run} DIR'", 1),
        ('die("usage: place5.py ', 'die("usage: place5r.py ', 4),
    ] + PLACE_RH),
    "judge_one5r.sh": ("judge_one5.sh", [
        ("# judge_one5.sh [--dry-run] STEPS RUN_ID SEED [MODE] : judge one stance5 checkpoint placed under S5/runs/RUN_ID/01_stance.",
         "# judge_one5r.sh [--dry-run] STEPS RUN_ID SEED [MODE] : judge one stance5 replication checkpoint placed under\n"
         "# R/runs/RUN_ID/01_stance (R = SP/foot/stance5/replication; the variable S5 below names R).", 1),
        ("#   RUN_ID stance5_LTSB_s52 | stance5_LTSB_s60 | stance5_LTSB_s42      SEED 52 | 60 | 42 (must match RUN_ID)",
         "#   RUN_ID stance5_LTSB_s70 | stance5_LTSB_s80      SEED 70 | 80 (must match RUN_ID)", 1),
        ("# Runs judge_stance5.py --arm LTSB", "# Runs judge_stance5r.py --arm LTSB", 1),
        ("# Output: S5/judged/RUN_ID/judge_STEPS.json (full; what score_stance5.py reads)",
         "# Output: R/judged/RUN_ID/judge_STEPS.json (full; what score_stance5r.py reads)", 1),
        ("# Logs: S5/judged/logs/queue.log (start/end lines) and S5/judged/logs/RUN_ID_<name>.log",
         "# Logs: R/judged/logs/queue.log (start/end lines) and R/judged/logs/RUN_ID_<name>.log", 1),
        ("# SP/foot/stance4/tools/judge_one4.sh with the stance5 identity.",
         "# SP/foot/stance5/tools/judge_one5.sh with the replication identity (runs s70/s80, judge_stance5r.py, R); before it,\n"
         "# SP/foot/stance4/tools/judge_one4.sh with the stance5 identity.", 1),
        ("S3=$S/foot/stance3; S5=$S/foot/stance5\n", "S3=$S/foot/stance3; S5=$S/foot/stance5/replication   # R\n", 1),
        ("JUDGE=$S5/judge/judge_stance5.py", "JUDGE=$S5/judge/judge_stance5r.py", 1),
        ('case "$RUN:$SEED" in stance5_LTSB_s52:52|stance5_LTSB_s60:60|stance5_LTSB_s42:42) ;;',
         'case "$RUN:$SEED" in stance5_LTSB_s70:70|stance5_LTSB_s80:80) ;;', 1),
        ('d.get("schema") == "stance5-judge/v1"', 'd.get("schema") == "stance5r-judge/v1"', 1),
        ("(place5.py config)", "(place5r.py config)", 1),
        ("(fetch, then place5.py)", "(fetch, then place5r.py)", 1),
    ] + JUDGE_RH),
    "panels5r_one.sh": ("panels5_one.sh", [
        ("# panels5_one.sh [--dry-run] [--wait] [--any-slot] STEPS RUN_ID : report-only diagnostic panels for one stance5 checkpoint.",
         "# panels5r_one.sh [--dry-run] [--wait] [--any-slot] STEPS RUN_ID : report-only diagnostic panels for one stance5\n"
         "# replication checkpoint (SP/foot/stance5/tools/panels5_one.sh with the runs s70/s80 and R = SP/foot/stance5/replication:\n"
         "# RUN_DIR=R/runs/RUN_ID, PANELS_OUT=R/panels/out; panels_one5.sh and panels5.py are stance5's, unchanged, read-only).", 1),
        ("#   RUN_ID stance5_LTSB_s52 | stance5_LTSB_s60 | stance5_LTSB_s42", "#   RUN_ID stance5_LTSB_s70 | stance5_LTSB_s80", 1),
        ("# Output: S5/panels/out/RUN_ID/panels_STEPS.json; logs in S5/panels/out/logs/ (queue.log, RUN_ID_STEPS.log) and one line\n"
         "# per job in S5/panels/out/logs/panels5_stance5.log.",
         "# Output: R/panels/out/RUN_ID/panels_STEPS.json; logs in R/panels/out/logs/ (queue.log, RUN_ID_STEPS.log) and one line\n"
         "# per job in R/panels/out/logs/panels5_replication.log.", 1),
        ("S5=$S/foot/stance5; P=$S5/panels\n", "S5=$S/foot/stance5; P=$S5/panels; R=$S5/replication\n", 1),
        ('case "$RUN" in stance5_LTSB_s52|stance5_LTSB_s60|stance5_LTSB_s42) ;;', 'case "$RUN" in stance5_LTSB_s70|stance5_LTSB_s80) ;;', 1),
        ("--expect-head <the config commit> with RUN_DIR=S5/runs/RUN_ID, nice -n 19,", "--expect-head <the config commit> with RUN_DIR=R/runs/RUN_ID, nice -n 19,", 1),
        ("RD=$S5/runs/$RUN; M=$RD/01_stance/models; OUT=$P/out/$RUN/panels_$STEPS.json; LOGD=$P/out/logs; LOG=$LOGD/panels5_stance5.log",
         "RD=$R/runs/$RUN; M=$RD/01_stance/models; OUT=$R/panels/out/$RUN/panels_$STEPS.json; LOGD=$R/panels/out/logs; LOG=$LOGD/panels5_replication.log", 1),
        ('CMD="RUN_DIR=$RD nice -n 19 $P/panels_one5.sh', 'CMD="RUN_DIR=$RD PANELS_OUT=$R/panels/out nice -n 19 $P/panels_one5.sh', 1),
        ("\nRUN_DIR=$RD nice -n 19 $P/panels_one5.sh $STEPS $RUN $CL", "\nRUN_DIR=$RD PANELS_OUT=$R/panels/out nice -n 19 $P/panels_one5.sh $STEPS $RUN $CL", 1),
    ]),
    "queue5r.sh": ("queue5.sh", [
        (QUEUE_HEADER_OLD, QUEUE_HEADER_NEW, 1),
        ("#   queue5.sh judge --watch    detach a worker (setsid nohup) that judges every placed, unjudged checkpoint of the three\n"
         "#                              stance5 runs,",
         "#   queue5.sh judge --watch    detach a worker (setsid nohup) that judges every placed, unjudged checkpoint of the two\n"
         "#                              replication runs,", 1),
        ("# Logs: S5/judged/logs/queue5_judge.log, S5/panels/out/logs/queue5_panels.log; per job: see judge_one5.sh, panels5_one.sh.",
         "# Logs: R/judged/logs/queue5_judge.log, R/panels/out/logs/queue5_panels.log; per job: see judge_one5r.sh, panels5r_one.sh.", 1),
        ("S5=$S/foot/stance5; T=$S5/tools\n", "S5=$S/foot/stance5/replication; T=$S5/tools   # R\n", 1),
        ('echo "usage: queue5.sh judge|panels [--dry-run|--once|--watch|--stop]"', 'echo "usage: queue5r.sh judge|panels [--dry-run|--once|--watch|--stop]"', 2),
        ('RUNS="stance5_LTSB_s52:52 stance5_LTSB_s60:60 stance5_LTSB_s42:42"', 'RUNS="stance5_LTSB_s70:70 stance5_LTSB_s80:80"', 1),
        ("$T/judge_one5.sh", "$T/judge_one5r.sh", 4),
        ("$T/panels5_one.sh", "$T/panels5r_one.sh", 2),
        ("#   queue5.sh ", "#   queue5r.sh ", 5),
        ("(judge_one5.sh / panels5_one.sh do that)", "(judge_one5r.sh / panels5r_one.sh do that)", 1),
        ("four variants, panels5_one.sh --wait.", "four variants, panels5r_one.sh --wait.", 1),
    ] + QUEUE_RH),
    "satwatch5r.py": ("satwatch5.py", [
        ('"""Saturation watch for the stance5 band runs (report only;',
         '"""Saturation watch for the stance5 replication band runs, s70 and s80 (SP/foot/stance5/tools/satwatch5.py with the\n'
         'replication runs, root R = SP/foot/stance5/replication, satroll5r.py, place5r.py and its own state file; the alarm rule\n'
         'and every report line unchanged) (report only;', 1),
        ("    python3 -I satwatch5.py [--runs RUN_ID ...] [--runs-root S5/runs]", "    python3 -I satwatch5r.py [--runs RUN_ID ...] [--runs-root R/runs]", 1),
        ("    python3 -I satwatch5.py --validate-on-stance4", "    python3 -I satwatch5r.py --validate-on-stance4", 1),
        ('S5 = SCRATCH + "/foot/stance5"\nS4 = SCRATCH + "/foot/stance4"\nTOOLS = S5 + "/tools"\n',
         'S5 = SCRATCH + "/foot/stance5"\nS4 = SCRATCH + "/foot/stance4"\nR = S5 + "/replication"\nTOOLS = R + "/tools"\n', 1),
        ('RUNS = {"stance5_LTSB_s52": 52, "stance5_LTSB_s60": 60, "stance5_LTSB_s42": 42}',
         'RUNS = {"stance5_LTSB_s70": 70, "stance5_LTSB_s80": 80}', 1),
        ('help="default: the three band runs"', 'help="default: the two replication runs"', 1),
        ('help="default: S5/runs (S4/runs with --validate-on-stance4)"', 'help="default: R/runs (S4/runs with --validate-on-stance4)"', 1),
        ('help="run satroll5.py on the latest placed checkpoint', 'help="run satroll5r.py on the latest placed checkpoint', 1),
        ('ap.add_argument("--state", default=TOOLS + "/state/satwatch5_state.json")', 'ap.add_argument("--state", default=TOOLS + "/state/satwatch5r_state.json")', 1),
        ('root = a.runs_root or (S4 + "/runs" if a.validate_on_stance4 else S5 + "/runs")',
         'root = a.runs_root or (S4 + "/runs" if a.validate_on_stance4 else R + "/runs")', 1),
        ("python3 -I place5.py csv {run}", "python3 -I place5r.py csv {run}", 1),
        ("satroll5.py {run_dir} --arm {arm}", "satroll5r.py {run_dir} --arm {arm}", 1),
        ("(stance5-fetch-checkpoint.js ", "(stance5r-fetch-checkpoint.js ", 1),
        ('why = f"satroll5 FAILED (rc {roll_rc})', 'why = f"satroll5r FAILED (rc {roll_rc})', 1),
        ("The actuator (criteria): satroll5.py's 20-episode rollout", "The actuator (criteria): satroll5r.py's 20-episode rollout", 1),
        ("a ledger that place5.py refused at the last fetch", "a ledger that place5r.py refused at the last fetch", 1),
        ('mine.append(f"place5.py REFUSED the ledger', 'mine.append(f"place5r.py REFUSED the ledger', 1),
    ]),
    "satroll5r.py": ("satroll5.py", [
        ('"""Name the saturating actuator of a placed stance checkpoint (report only;',
         '"""Name the saturating actuator of a placed stance checkpoint (stance5 replication copy of SP/foot/stance5/tools/\n'
         'satroll5.py: judge_stance5r.py in place of judge_stance5.py, and --seed also takes 70 and 80) (report only;', 1),
        ("iso.py satroll5.py RUN_DIR --arm {LTSB,LTS} --seed N", "iso.py satroll5r.py RUN_DIR --arm {LTSB,LTS} --seed N", 1),
        ('JUDGE = SCRATCH + "/foot/stance5/judge/judge_stance5.py"', 'JUDGE = SCRATCH + "/foot/stance5/replication/judge/judge_stance5r.py"', 1),
        ('spec = importlib.util.spec_from_file_location("judge_stance5", JUDGE)', 'spec = importlib.util.spec_from_file_location("judge_stance5r", JUDGE)', 1),
        ('sys.modules["judge_stance5"] = J', 'sys.modules["judge_stance5r"] = J', 1),
        ('choices=(42, 52, 60))', 'choices=(42, 52, 60, 70, 80))', 1),
        ('help="default: the arm\'s pinned HEAD in judge_stance5.py ARMS"', 'help="default: the arm\'s pinned HEAD in judge_stance5r.py ARMS"', 1),
        ('HEAD is a PLACEHOLDER in judge_stance5.py; pass --expect-head', 'HEAD is a PLACEHOLDER in judge_stance5r.py; pass --expect-head', 1),
        ('print(f"[satroll5] {exc.code}"', 'print(f"[satroll5r] {exc.code}"', 1),
    ]),
    "cert5r_one.sh": ("cert5_one.sh", [
        ("# cert5_one.sh RUN_ID: six-heading certificate (stance_quality_heading/v1) of a stance5 run's final checkpoint, report only.",
         "# cert5r_one.sh RUN_ID: six-heading certificate (stance_quality_heading/v1) of a stance5 replication run's final checkpoint,\n"
         "# report only (SP/foot/stance5/tools/cert5_one.sh with the runs s70/s80 and R = SP/foot/stance5/replication; S5 below\n"
         "# names R).", 1),
        ("S5=$S/foot/stance5; RUN=$1; CL=$S/fix/scratch_copband\n",
         "S5=$S/foot/stance5/replication; RUN=$1; CL=$S/fix/scratch_copband   # R\n"
         'case "$RUN" in stance5_LTSB_s70|stance5_LTSB_s80) ;; *) echo "RUN_ID $RUN is not a replication run"; exit 2;; esac\n', 1),
        # R/cert/<RUN> is created only once the final pair is found placed (a refused call leaves no empty directory)
        ("M=$S5/runs/$RUN/01_stance/models; O=$S5/cert/$RUN; mkdir -p $O\n"
         '[ -f $M/stage1_final.zip ] && [ -f $M/stage1_final_vecnorm.pkl ] || { echo "final pair not placed for $RUN"; exit 3; }\n',
         "M=$S5/runs/$RUN/01_stance/models; O=$S5/cert/$RUN\n"
         '[ -f $M/stage1_final.zip ] && [ -f $M/stage1_final_vecnorm.pkl ] || { echo "final pair not placed for $RUN"; exit 3; }\n'
         "mkdir -p $O\n", 1),
    ]),
}
#: The stance5 workflow dry-run harness, with the replication runs in its scenarios.
HARNESS = ("wfcheck/harness.mjs", "wfcheck/harness.mjs", [
    ("const run = (/stance5_LTSB_s\\d\\d/.exec(opts.label) || [''])[0]", "const run = (/stance5_LTSB_s\\d\\d/.exec(opts.label) || [''])[0]  // replication: s70, s80", 1),
    ("const notyet = scen === 'notyet' && run === 'stance5_LTSB_s42'", "const notyet = scen === 'notyet' && run === 'stance5_LTSB_s80'", 1),
    ("run === 'stance5_LTSB_s60') ? 'bad'", "run === 'stance5_LTSB_s70') ? 'bad'", 1),
])


def build(name, src_name, reps):
    with open(os.path.join(SRC, src_name)) as fh:
        old = fh.read()
    new = old
    for a, b, n in reps:
        got = new.count(a)
        if got != n:
            raise SystemExit(f"{name}: {a[:80]!r} occurs {got} times, expected {n}")
        new = new.replace(a, b)
    mode = os.stat(os.path.join(SRC, src_name)).st_mode & 0o777
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"stance5/tools/{src_name}",
                                        f"replication/tools/{name}", n=1))
    return new, mode, diff


def main(argv):
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    unknown = [a for a in argv if a not in ("--check", "--force")]
    if unknown or ("--check" in argv and "--force" in argv):
        raise SystemExit(f"usage: make_tools.py [--force | --check | --help] (refused: {argv})")
    check, force = "--check" in argv, "--force" in argv
    items = [(n, s, r) for n, (s, r) in FILES.items()] + [HARNESS]
    diffs, bad, written = [], [], []
    for name, src_name, reps in items:
        new, mode, diff = build(name, src_name, reps)
        diffs.append(diff)
        path = os.path.join(DST, name)
        if check:
            if not os.path.exists(path) or open(path).read() != new:
                bad.append(name)
            continue
        if os.path.exists(path) and open(path).read() != new and not force:
            raise SystemExit(f"{path} exists and differs (--force to replace)")
        want_mode = mode | (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH if name.endswith(".sh") else 0)
        if os.path.exists(path) and open(path).read() == new and (os.stat(path).st_mode & 0o777) == want_mode:
            continue   # identical: not rewritten (the mtime stays)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(new)
        os.chmod(path, want_mode)
        written.append(name)
    text = ("# Every change of SP/foot/stance5/replication/tools against SP/foot/stance5/tools (written by\n"
            "# SP/foot/stance5/replication/build/make_tools.py; README.md is new and not listed).\n\n" + "\n".join(diffs))
    cpath = os.path.join(DST, "CHANGES.diff")
    if check:
        if not os.path.exists(cpath) or open(cpath).read() != text:
            bad.append("CHANGES.diff")
        print({"check": "ok" if not bad else "DIFFERS", "files": bad})
        return 1 if bad else 0
    if not (os.path.exists(cpath) and open(cpath).read() == text):
        with open(cpath, "w") as fh:
            fh.write(text)
        written.append("CHANGES.diff")
    print({"wrote": written, "unchanged": len(items) + 1 - len(written), "dst": DST})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
