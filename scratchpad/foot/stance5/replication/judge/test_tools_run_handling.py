"""The tools' side of the run-handling rules (criteria RUN HANDLING), on scratch roots under pytest's tmp_path.

    cd R/judge && python3 -I -B -m pytest -p no:cacheprovider -q test_tools_run_handling.py --basetemp=<work>/tmp

Covers place5r.py under PLACE5R_TEST_ROOT: a restart under <RUN_ID>_b (rule 1: the switch writes source_folder.json and
moves the earlier folder's records aside; the earlier folder is refused after it; a _c switch is refused once a pair is
placed; a progress.json naming another folder is refused); a resume (rule 3: --supersede refused without N and with N
at or after STEPS; with N it moves the pair, its judge files, refusal record, review hold, panels and roll aside, never
deleting, and writes SUPERSEDED.json; a ledger replaced under --resumed-from is moved aside; "placed_after_N" lists the
old session's pairs after N); --run-records-fail (rule 2: the run's own records that fail place5r.py's checks or are
absent on Drive, A and B agreeing, are placed beside records_check_failed.json so that the judge can refuse on them; the
refusals around it).  judge_one5r.sh's refusal classification, its refusal() python run as judge_one5r.sh runs it, on
hand-made judge logs: a refusal naming the run's records writes refused_<steps>.json; a missing record is the operator's
side unless records_check_failed.json lists it as absent on Drive; a refusal on the checkpoint's own attributes only is
held for review (refused_<steps>.review.json); no refusal line, or a symmetry check not run, writes nothing.
judge_one5r.sh --check (5 refused, 6 review) and the queue's status under JUDGE5R_TEST_ROOT.  make_tools.py --check and
make_colab_steps.py --check (the files on disk are the built ones).

The run records are stance5 s52's (read only) with run.seed set to the run's; checkpoints are tiny zip files and byte
strings (place5r.py never opens a payload beyond zipfile.testzip).  The judge is never run.  Nothing is written outside
tmp_path.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile

import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
T = os.path.join(R, "tools")
PLACE = os.path.join(T, "place5r.py")
JUDGE_ONE = os.path.join(T, "judge_one5r.sh")
QUEUE = os.path.join(T, "queue5r.sh")
S52 = os.path.join(os.path.dirname(R), "runs", "stance5_LTSB_s52", "01_stance")
R70, R80 = "stance5_LTSB_s70", "stance5_LTSB_s80"
RECORDS = ("plant_identity.json", "stage_config.json", "task_fingerprint.json")


# ------------------------------------------------------------------------------------------------ helpers
def records(d, seed, drop=(), note=None):
    """stance5 s52's three run records with run.seed = seed (and a gpu note when given), less *drop*."""
    os.makedirs(d, exist_ok=True)
    for n in ("plant_identity.json", "task_fingerprint.json"):
        if n not in drop:
            with open(os.path.join(S52, n), "rb") as src, open(os.path.join(d, n), "wb") as dst:
                dst.write(src.read())
    if "stage_config.json" not in drop:
        with open(os.path.join(S52, "stage_config.json")) as fh:
            sc = json.load(fh)
        sc["run"]["seed"] = seed
        if note:
            sc["gpu"] = {**(sc.get("gpu") or {}), "note": note}
        with open(os.path.join(d, "stage_config.json"), "w") as fh:
            json.dump(sc, fh, indent=1)
    return str(d)


def names(steps):
    if steps == "final":
        return "stage1_final.zip", "stage1_final_vecnorm.pkl"
    if steps in ("best", "robust_best"):
        return f"{steps}_model.zip", f"{steps}_model_vecnorm.pkl"
    return f"stage1_{steps}_steps.zip", f"stage1_vecnormalize_{steps}_steps.pkl"


def fetch(d, steps, tag=b"old", recs=None, ledger_to=None, resumed=None):
    """A fetch directory: A/ and B/ with the pkl (and the records when *recs* = (seed, drop)), zip/ with a tiny valid
    zip, csv/ with ledgers up to *ledger_to* (resumed = N: rows up to N as before, one new row past N)."""
    z, p = names(steps)
    for role in ("A", "B"):
        os.makedirs(os.path.join(d, role), exist_ok=True)
        with open(os.path.join(d, role, p), "wb") as fh:
            fh.write(b"pkl-" + tag)
        if recs is not None:
            records(os.path.join(d, role), recs[0], drop=recs[1] if len(recs) > 1 else ())
    os.makedirs(os.path.join(d, "zip"), exist_ok=True)
    with zipfile.ZipFile(os.path.join(d, "zip", z), "w") as zf:
        zf.writestr("data", b"zip-" + tag)
    if ledger_to is not None:
        ledgers(d, ledger_to, resumed)
    return str(d)


def ledgers(d, to, resumed=None):
    os.makedirs(os.path.join(d, "csv"), exist_ok=True)
    for n in ("eval_episodes.csv", "train_episodes.csv"):
        rows = [f"{s},{s % 7}\n" for s in range(131072, to + 1, 131072)]
        if resumed is not None:
            rows = [f"{s},{s % 7}\n" for s in range(131072, resumed + 1, 131072)] + [f"{resumed + 131072},new\n"]
        with open(os.path.join(d, "csv", n), "w") as fh:
            fh.write("timesteps,value\n" + "".join(rows))


def progress(d, run_id, at):
    os.makedirs(os.path.join(d, "extra"), exist_ok=True)
    with open(os.path.join(d, "extra", "progress.json"), "w") as fh:
        json.dump({"schema": "mesozoic.training-progress/v1", "run_id": run_id, "written_at": at, "timesteps": 1}, fh)
    return str(d)


def place(root, *args):
    """place5r.py under PLACE5R_TEST_ROOT=root: (exit code, the printed JSON)."""
    r = subprocess.run([sys.executable, "-I", "-B", PLACE, *map(str, args)], capture_output=True, text=True, check=False,
                       env={**os.environ, "PLACE5R_TEST_ROOT": str(root)}, cwd=str(root.parent))
    try:
        out = json.loads(r.stdout)
    except ValueError:
        out = {"raw": r.stdout + r.stderr}
    return r.returncode, out


def files(root):
    return sorted(os.path.relpath(os.path.join(a, f), root) for a, _d, fs in os.walk(root) for f in fs)


def stage(root, run):
    return root / "runs" / run / "01_stance"


@pytest.fixture
def root(tmp_path):
    rt = tmp_path / "root"
    os.makedirs(rt)
    return rt


# ------------------------------------------------------------------------------------------------ rule 1: _b / _c
def test_a_restart_folder_switches_the_run(root, tmp_path):
    first = tmp_path / "first"
    records(first / "A", 70), records(first / "B", 70), progress(first, R70, "2026-10-23T00:10:00Z")
    rb = tmp_path / "b"
    records(rb / "A", 70, note="restart"), records(rb / "B", 70, note="restart"), progress(rb, f"{R70}_b", "2026-10-23T01:10:00Z")
    assert place(root, "config", R70, first / "A", first / "B")[0] == 0
    assert place(root, "extra", R70, first)[0] == 0
    # the restart's records differ: without the folder they are refused as a differing copy
    rc, out = place(root, "config", R70, rb / "A", rb / "B")
    assert rc == 1 and "differs" in out["error"]
    rc, out = place(root, "config", R70, rb / "A", rb / "B", "--source-folder", f"{R70}_b")
    assert rc == 0 and out["ok"]
    marker = json.load(open(root / "runs" / R70 / "source_folder.json"))
    assert marker["source_folder"] == f"{R70}_b" and marker["previous"] == R70
    moved = {os.path.basename(m["from"]) for m in marker["moved_aside"]}
    assert moved == set(RECORDS) | {"progress.json"}
    sup = [f for f in files(root) if f.startswith(f"superseded/{R70}/") and f.endswith("SUPERSEDED.json")]
    assert len(sup) == 1
    assert {os.path.basename(f) for f in files(root) if f.startswith(f"superseded/{R70}/")} == set(RECORDS) | {
        "progress.json", "SUPERSEDED.json"}
    # after the switch: the earlier folder is refused, and a progress.json naming it too
    rc, out = place(root, "config", R70, first / "A", first / "B")
    assert rc == 1 and "earlier folder" in out["error"]
    rc, out = place(root, "extra", R70, first, "--source-folder", f"{R70}_b")
    assert rc == 1 and out["extra"]["progress.json"].startswith("REFUSED: run_id")
    assert place(root, "extra", R70, rb, "--source-folder", f"{R70}_b")[0] == 0
    # a pair from the _b folder is placed under the RUN_ID; then a _c switch is refused
    rc, out = place(root, 4194304, R70, fetch(tmp_path / "r4", 4194304), "--source-folder", f"{R70}_b")
    assert rc == 0 and out["zip"]["action"] == "placed"
    assert (stage(root, R70) / "models" / "stage1_4194304_steps.zip").exists()
    rc, out = place(root, 4194304, R70, tmp_path / "r4", "--source-folder", f"{R70}_c")
    assert rc == 1 and "holds checkpoints placed from" in out["error"]
    rc, out = place(root, 4194304, R70, tmp_path / "r4", "--source-folder", f"{R80}_b")
    assert rc == 1 and "is not" in out["error"]
    rc, out = place(root, "status", R70)
    assert out[R70]["source_folder"] == f"{R70}_b"


# ------------------------------------------------------------------------------------------------ rule 3: resumes
def test_supersede_moves_the_slot_aside_never_deleting(root, tmp_path):
    g = tmp_path / "g"
    records(g / "A", 80), records(g / "B", 80)
    assert place(root, "config", R80, g / "A", g / "B")[0] == 0
    assert place(root, 4194304, R80, fetch(tmp_path / "old", 4194304, ledger_to=4194304))[0] == 0
    stand_ins = [f"judged/{R80}/judge_4194304.json", f"judged/{R80}/judge_quick_4194304.json",
                 f"judged/{R80}/refused_4194304.json", f"judged/{R80}/refused_4194304.review.json",
                 f"panels/out/{R80}/panels_4194304.json", f"tools/satroll/{R80}_4194304.json",
                 f"judged/{R80}/judge_5242880.json"]
    for f in stand_ins:
        os.makedirs(os.path.dirname(root / f), exist_ok=True)
        (root / f).write_text('{"stand_in": true}')
    before = files(root)
    new = fetch(tmp_path / "new", 4194304, tag=b"new", ledger_to=3670016, resumed=3670016)
    rc, out = place(root, 4194304, R80, new)
    assert rc == 1 and "--resumed-from N --supersede" in out["error"]
    rc, out = place(root, 4194304, R80, new, "--supersede")
    assert rc == 1 and "--supersede needs --resumed-from N" in out["error"]
    rc, out = place(root, 4194304, R80, new, "--resumed-from", 4194304, "--supersede")
    assert rc == 1 and "STEPS after N" in out["error"]
    assert files(root) == before
    rc, out = place(root, 4194304, R80, new, "--resumed-from", 3670016, "--supersede")
    assert rc == 0 and out["ok"], out
    moved = {os.path.relpath(m["from"], root) for m in out["superseded"]}
    assert moved == {f"runs/{R80}/01_stance/models/stage1_4194304_steps.zip",
                     f"runs/{R80}/01_stance/models/stage1_vecnormalize_4194304_steps.pkl"} | set(stand_ins[:-1])
    assert (root / stand_ins[-1]).exists()   # another slot's judge file stays
    with zipfile.ZipFile(stage(root, R80) / "models" / "stage1_4194304_steps.zip") as zf:
        assert zf.read("data") == b"zip-new"
    # the replaced ledgers are moved aside too
    assert {os.path.basename(m["from"]) for k in ("eval_episodes.csv", "train_episodes.csv")
            for m in out["csv"][k]["moved_aside"]} == {"eval_episodes.csv", "train_episodes.csv"}
    after = files(root)
    for f in before:   # nothing deleted: every earlier file is in place or under superseded/
        assert f in after or any(a.startswith(f"superseded/{R80}/") and a.endswith("/" + f) for a in after), f
    assert sum(a.endswith("SUPERSEDED.json") for a in after) >= 1
    # the same pair again: nothing to move
    rc, out = place(root, 4194304, R80, new, "--resumed-from", 3670016, "--supersede", "--no-csv")
    assert rc == 0 and any("nothing was moved aside" in n for n in out["notes"])


def test_placed_after_n_lists_the_old_sessions_pairs(root, tmp_path):
    g = tmp_path / "g"
    records(g / "A", 80), records(g / "B", 80)
    assert place(root, "config", R80, g / "A", g / "B")[0] == 0
    for s in (3670016, 4194304, 5242880):
        assert place(root, s, R80, fetch(tmp_path / f"o{s}", s, ledger_to=s))[0] == 0
    rc, out = place(root, "csv", R80, fetch(tmp_path / "led", 3145728, ledger_to=3670016, resumed=3670016),
                    "--resumed-from", 3670016)
    assert rc == 0 and out["placed_after_N"] == ["stage1_4194304_steps.zip", "stage1_5242880_steps.zip"]
    assert "--resumed-from 3670016 --supersede" in out["placed_after_N_note"]
    rc, out = place(root, 4194304, R80, fetch(tmp_path / "n4", 4194304, tag=b"new"), "--resumed-from", 3670016,
                    "--supersede", "--no-csv")
    assert rc == 0 and out["placed_after_N"] == ["stage1_5242880_steps.zip"]   # the pair just placed is the new one
    rc, out = place(root, "csv", R80, tmp_path / "led")
    assert rc == 0 and "placed_after_N" not in out   # only with --resumed-from


# ------------------------------------------------------------------------------------------------ rule 2: records
def test_run_records_that_fail_the_checks_are_placed_for_the_judge(root, tmp_path):
    bad = tmp_path / "bad"
    records(bad / "A", 52), records(bad / "B", 52)   # s70's folder holding seed 52's records
    rc, out = place(root, "config", R70, bad / "A", bad / "B")
    assert rc == 1 and "run.seed 52 != 70" in out["error"]
    rc, out = place(root, 4194304, R70, fetch(tmp_path / "r4", 4194304))
    assert rc == 1 and "records not placed yet" in out["error"] and "--run-records-fail" in out["error"]
    assert place(root, "config", R70, bad / "A", "--run-records-fail")[0] == 1          # DIR2 is required
    assert place(root, "csv", R70, bad, "--run-records-fail")[0] == 1                   # config only
    other = tmp_path / "other"
    records(other / "A", 52), records(other / "B", 53)
    rc, out = place(root, "config", R70, other / "A", other / "B", "--run-records-fail")
    assert rc == 1 and "differ" in out["error"]
    rc, out = place(root, "config", R70, bad / "A", bad / "B", "--run-records-fail")
    assert rc == 0 and out["ok"]
    mk = out["records_check_failed"]
    assert mk["problems"] == ["stage_config.json: run.seed 52 != 70"] and mk["absent_on_drive"] == []
    assert set(mk["sha256"]) == set(RECORDS) and all((stage(root, R70) / n).exists() for n in RECORDS)
    rc, out = place(root, "config", R70, bad / "A", bad / "B", "--run-records-fail")
    assert rc == 0 and "unchanged" in out["notes"][0]   # idempotent
    rc, out = place(root, 4194304, R70, tmp_path / "r4")
    assert rc == 0 and any("records_check_failed.json" in n for n in out["notes"])
    # a configs:true fetch carrying the same records is accepted; other records are refused
    rc, out = place(root, 5242880, R70, fetch(tmp_path / "r5", 5242880, recs=(52,)))
    assert rc == 0, out
    rc, out = place(root, 5767168, R70, fetch(tmp_path / "r55", 5767168, recs=(70,)))
    assert rc == 1 and "the placed copy differs" in out["error"]
    assert place(root, "status", R70)[1][R70]["records_check_failed"]["problems"] == mk["problems"]


def test_a_record_absent_on_drive(root, tmp_path):
    gone = tmp_path / "gone"
    records(gone / "A", 80, drop=("stage_config.json",)), records(gone / "B", 80, drop=("stage_config.json",))
    rc, out = place(root, "config", R80, gone / "A", gone / "B")
    assert rc == 1 and "missing" in out["error"]
    half = tmp_path / "half"
    records(half / "A", 80, drop=("stage_config.json",)), records(half / "B", 80)
    assert place(root, "config", R80, half / "A", half / "B", "--run-records-fail")[0] == 1   # A and B disagree
    rc, out = place(root, "config", R80, gone / "A", gone / "B", "--run-records-fail")
    assert rc == 0 and out["records_check_failed"]["absent_on_drive"] == ["stage_config.json"]
    assert not (stage(root, R80) / "stage_config.json").exists()
    rc, out = place(root, 4194304, R80, fetch(tmp_path / "r4", 4194304))
    assert rc == 0 and set(out["records"]) == {"plant_identity.json", "task_fingerprint.json"}
    # the record turning up later contradicts the marker: stop and ask
    rc, out = place(root, 5242880, R80, fetch(tmp_path / "r5", 5242880, recs=(80,)))
    assert rc == 1 and "stop and ask" in out["error"]


def test_records_that_pass_are_not_placed_under_the_flag(root, tmp_path):
    good = tmp_path / "good"
    records(good / "A", 70), records(good / "B", 70)
    rc, out = place(root, "config", R70, good / "A", good / "B", "--run-records-fail")
    assert rc == 1 and "pass every check" in out["error"]
    assert not (stage(root, R70) / "records_check_failed.json").exists()


# ------------------------------------------------------------------------------------------------ judge_one5r.sh
def refusal_script():
    text = open(JUDGE_ONE).read()
    m = re.search(r"^refusal\(\) \{\n  python3 -I - (.*?) <<'EOF'\n(.*?)\nEOF\n\}", text, re.S | re.M)
    assert m, "refusal() not found in judge_one5r.sh"
    assert m.group(1).split() == ['"$JL"', '"$REFUSED"', '"$RUN"', '"$SEED"', '"$STEPS"', '"$JUDGE"', '"$Z"', '"$P"',
                                  '"$HEAD_EXPECT"', '"$1"', '"$REVIEW"', '"$MARKER"']
    return m.group(2) + "\n"


PREFIX = "[judge_stance5r] refusing (nothing written): "
ST = "/x/runs/stance5_LTSB_s70/01_stance"
CASES = {  # name: (refusal items or None for no refusal line, marker absent list or None, expected rc)
    "run_side": ([f"{ST}/stage_config.json: run seed 52 != --seed 70", "the checkpoint records seed 52, the judge was told 70"], None, 0),
    "commit": ([f"{ST}/stage_config.json: git_commit 99e205bb != clone HEAD cf80ecb1 (--allow-commit-mismatch to judge anyway)"], None, 0),
    "plant_record": ([f"{ST}/plant_identity.json physics_sha256 sha256:aa != the clone's sha256:bb"], None, 0),
    "plant_not_placed": ([f"{ST}/plant_identity.json missing (required)"], None, 1),
    "plant_absent_on_drive": ([f"{ST}/plant_identity.json missing (required)"], ["plant_identity.json"], 0),
    "stage_config_not_placed": ([f"{ST}/stage_config.json missing (required)", "the checkpoint records seed 52, the judge was told 70"], None, 1),
    "stage_config_absent_on_drive": ([f"{ST}/stage_config.json missing (required)"], ["stage_config.json"], 0),
    "absent_but_other_missing": ([f"{ST}/task_fingerprint.json missing (required)"], ["stage_config.json"], 1),
    "checkpoint_only": (["the checkpoint records seed 52, the judge was told 70",
                         "checkpoint physics_sha256 sha256:aa != the clone's"], None, 2),
    "symmetry_only": (["mesozoic_symmetry {'mirror_map_sha256': 'sha256:cc'} does not match the clone (mirror map differs; clone mirror map sha256:dd)"], None, 2),
    "symmetry_not_checked": (["mesozoic_symmetry None does not match the clone (not checked; clone mirror map None)",
                              f"{ST}/stage_config.json: run seed 52 != --seed 70"], None, 1),
    "no_refusal_line": (None, None, 1),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_refusal_classification(tmp_path, case):
    items, absent, want = CASES[case]
    log, out, review, marker = (tmp_path / n for n in ("judge.log", "refused_final.json", "refused_final.review.json",
                                                     "records_check_failed.json"))
    log.write_text("[judge_stance5r] start\n" + (PREFIX + " | ".join(items) + "\n" if items else "Traceback: boom\n"))
    if absent is not None:
        marker.write_text(json.dumps({"absent_on_drive": absent}))
    for n in ("z.zip", "p.pkl"):
        (tmp_path / n).write_bytes(b"x")
    r = subprocess.run([sys.executable, "-I", "-", str(log), str(out), R70, "70", "final", os.path.join(HERE, "judge_stance5r.py"),
                        str(tmp_path / "z.zip"), str(tmp_path / "p.pkl"), "cf80", "1", str(review), str(marker)],
                       input=refusal_script(), capture_output=True, text=True, check=False, cwd=str(tmp_path))
    assert r.returncode == want, (case, r.stdout, r.stderr)
    assert out.exists() == (want == 0) and review.exists() == (want == 2), case
    if want in (0, 2):
        d = json.load(open(out if want == 0 else review))
        assert d["schema"] == "stance5r-refusal/v1" and d["run"] == R70 and d["seed"] == 70 and d["steps"] == "final"
        assert d["mode"] == "full" and d["items"] == items and d["judge_sha256"].startswith("sha256:")
        assert ("held_for_review" in d) == (want == 2)


def test_check_and_queue_status_of_a_refused_and_a_held_slot(tmp_path):
    env = {**os.environ, "JUDGE5R_TEST_ROOT": str(tmp_path)}
    m = tmp_path / "runs" / R70 / "01_stance" / "models"
    os.makedirs(m)
    for s in (4194304, 5242880):
        for n in names(s):
            (m / n).write_bytes(b"x")
    j = tmp_path / "judged" / R70
    os.makedirs(j)
    (j / "refused_4194304.json").write_text("{}")
    (j / "refused_5242880.review.json").write_text("{}")

    def check(steps, mode="full"):
        return subprocess.run([JUDGE_ONE, "--check", str(steps), R70, "70", mode], env=env, check=False).returncode

    assert (check(4194304), check(5242880), check(6291456)) == (5, 6, 3)
    assert check(5242880, "quick") == 1   # report-only modes ignore both
    r = subprocess.run([JUDGE_ONE, "5242880", R70, "70"], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 6   # nothing is judged while the review hold stands
    assert "held-for-review" in open(tmp_path / "judged" / "logs" / "queue.log").read()
    r = subprocess.run([JUDGE_ONE, "--dry-run", "5242880", R70, "70"], env=env, capture_output=True, text=True, check=False)
    assert "REFUSAL HELD FOR REVIEW" in r.stdout
    r = subprocess.run([QUEUE, "judge", "--dry-run"], env=env, capture_output=True, text=True, check=False)
    lines = r.stdout.splitlines()
    assert f"refused {R70} 4194304 full" in lines and f"review {R70} 5242880 full" in lines
    assert f"missing {R70} 6291456 full" in lines


# ------------------------------------------------------------------------------------------------ the built files
@pytest.mark.parametrize("script", ["make_tools.py", "make_colab_steps.py"])
def test_the_files_on_disk_are_the_built_ones(script):
    r = subprocess.run([sys.executable, "-I", "-B", os.path.join(R, "build", script), "--check"], capture_output=True,
                       text=True, check=False)
    assert r.returncode == 0 and "'check': 'ok'" in r.stdout, r.stdout + r.stderr
