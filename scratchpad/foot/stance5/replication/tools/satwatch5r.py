"""Saturation watch for the stance5 replication band runs, s70 and s80 (SP/foot/stance5/tools/satwatch5.py with the
replication runs, root R = SP/foot/stance5/replication, satroll5r.py, place5r.py and its own state file; the alarm rule
and every report line unchanged) (report only; criteria PRE-REGISTERED READOUTS "Saturation watch",
D-D40 5a): read each run's evaluation ledger and report the alarm with run, step and actuator.

    python3 -I satwatch5r.py [--runs RUN_ID ...] [--runs-root R/runs] [--ledgers RUN_ID=DIR ...] [--roll]
                            [--roll-dir DIR] [--state FILE | --no-state] [--json OUT.json]
    python3 -I satwatch5r.py --validate-on-stance4 --no-state [--runs-root ROOT] [--roll] [--roll-dir DIR]

The alarm rule is score_stance5.py's saturation_alarm (identical to score_stance4.py's; loaded from the scorer's own
file): per evaluation of eval_episodes.csv, the episodes listing max_actuator_saturation_fraction in screen_v2_reasons;
an evaluation with >= 5 of its 10 is flagged; two flagged evaluations in a row are an alarm, reported at the evaluation
that completes the first flagged pair (first_alarm_at).

The actuator (criteria): satroll5r.py's 20-episode rollout of the latest placed periodic checkpoint AT OR AFTER THE FIRST
FLAGGED EVALUATION F (the actuator with the largest fraction of window steps at |command| >= 0.9 and the sign of its
mean command).  A checkpoint before F is never rolled: it predates the habit.  When none at or after F is placed, the
line says "actuator NOT NAMED YET" and names the checkpoint to fetch for that run alone (the first multiple of 524,288
at or after F); with --roll the placed one is rolled here (about 1-2 minutes, one CPU, nice 19) and the result kept in
--roll-dir/<RUN_ID>_<steps>.json (never re-rolled).  The rolled checkpoint's own count of episodes over the gate's
saturation bar is printed beside the name.

What to report.  --state (default tools/state/satwatch5_state.json) keeps, per run, the alarm step and the actuator last
reported.  A line is tagged
  NEW ALARM          an alarm not reported before (with the actuator, or NOT NAMED YET);
  ACTUATOR NOW NAMED the actuator of an alarm reported earlier without one;
  ACTUATOR CHANGED   a later checkpoint names another actuator or sign than the one reported;
  ALARM              nothing new.
Ledger problems (the run's alarm state is unknown, so it is NOT "no alarm"): no eval_episodes.csv placed (after the run
has evaluated), an empty one, a ledger that place5r.py refused at the last fetch (<stage>/eval_episodes.csv.refused.json:
after a Colab resume the fetched ledger does not extend the placed copy), no progress.json placed, or a ledger whose last
evaluation is more than one evaluation (131,072 steps) behind or ahead of progress.json's.  Under
--validate-on-stance4 the progress.json checks are skipped (the stance4 runs have none placed).

Exit code: 11 something new to report (a NEW ALARM, ACTUATOR NOW NAMED or ACTUATOR CHANGED line: tell the maintainer
at once), 10 an alarm, nothing new, 3 a ledger problem and no alarm, 0 every ledger placed, current and clear, 2 a usage
error.  The problem lines are printed whatever the code.  Fails stay fails; no run is stopped or restarted on an alarm
(D-D40 5a).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S5 = SCRATCH + "/foot/stance5"
S4 = SCRATCH + "/foot/stance4"
R = S5 + "/replication"
TOOLS = R + "/tools"
SCORER = S5 + "/judge/score_stance5.py"
RUNS = {"stance5_LTSB_s70": 70, "stance5_LTSB_s80": 80}
RUNS_S4 = {"stance4_LTS_s52": 52, "stance4_LTS_s60": 60, "stance4_LTS_s42": 42}
CADENCE = 524288
EVAL = 131072


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_stance5_for_satwatch", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def placed_steps(run_dir: str) -> list[int]:
    out = []
    for p in glob.glob(os.path.join(run_dir, "01_stance", "models", "stage1_*_steps.zip")):
        m = re.search(r"stage1_(\d+)_steps\.zip$", p)
        n = int(m.group(1))
        if os.path.exists(os.path.join(run_dir, "01_stance", "models", f"stage1_vecnormalize_{n}_steps.pkl")):
            out.append(n)
    return sorted(out)


def progress_eval(SC, root: str, run: str, ledger_dir: str | None):
    """(path, the last evaluation's timesteps or None, progress.json's timesteps) from the placed progress.json."""
    path = SC.SC3._find(root, run, "progress.json", ledger_dir)
    if not path:
        return None, None, None
    try:
        with open(path) as fh:
            prog = json.load(fh)
    except (OSError, ValueError):
        return path, None, None
    ev = prog.get("evaluation") if isinstance(prog.get("evaluation"), dict) else {}
    return path, ev.get("timesteps"), prog.get("timesteps")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", default=None, help="default: the two replication runs")
    ap.add_argument("--runs-root", default=None, help="default: R/runs (S4/runs with --validate-on-stance4)")
    ap.add_argument("--ledgers", action="append", default=[], metavar="RUN_ID=DIR")
    ap.add_argument("--roll", action="store_true",
                    help="run satroll5r.py on the latest placed checkpoint at or after the first flagged evaluation")
    ap.add_argument("--roll-dir", default=TOOLS + "/satroll", help="where satroll5 outputs are kept (default tools/satroll)")
    ap.add_argument("--state", default=TOOLS + "/state/satwatch5r_state.json")
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--validate-on-stance4", action="store_true",
                    help="read the stance4 runs' ledgers (S4/runs): s42's alarm must read 2,228,224, s52 and s60 none")
    a = ap.parse_args()
    if a.validate_on_stance4 and a.runs:
        print("--validate-on-stance4 reads all three stance4 runs; it does not take --runs", file=sys.stderr)
        return 2
    if a.json and os.path.exists(a.json):
        print(f"refusing to overwrite {a.json}", file=sys.stderr)
        return 2
    SC = load_scorer()
    runs = dict(RUNS_S4 if a.validate_on_stance4 else RUNS)
    if a.runs:
        unknown = [r for r in a.runs if r not in runs]
        if unknown:
            print(f"unknown run(s) {unknown}; known: {sorted(runs)}", file=sys.stderr)
            return 2
        runs = {r: runs[r] for r in a.runs}
    root = a.runs_root or (S4 + "/runs" if a.validate_on_stance4 else R + "/runs")
    ledgers = {}
    for item in a.ledgers:
        rid, sep, path = item.partition("=")
        if not sep:
            print(f"--ledgers {item!r} is not RUN_ID=DIR", file=sys.stderr)
            return 2
        ledgers[rid] = os.path.abspath(path)
    state = {}
    if not a.no_state and os.path.exists(a.state):
        with open(a.state) as fh:
            state = json.load(fh)
    report, any_alarm, any_new, problems = {}, False, False, []
    for run, seed in runs.items():
        run_dir = os.path.join(root, run)
        stage = os.path.join(run_dir, "01_stance")
        path = SC.SC3._find(root, run, "eval_episodes.csv", ledgers.get(run))
        ppath, p_eval, p_steps = (None, None, None) if a.validate_on_stance4 else progress_eval(SC, root, run, ledgers.get(run))
        refused = os.path.join(stage, "eval_episodes.csv.refused.json")
        entry = {"ledger": path, "progress": ppath, "progress_evaluation_timesteps": p_eval, "progress_timesteps": p_steps,
                 "problems": []}
        mine = entry["problems"]
        if os.path.exists(refused) and not ledgers.get(run):
            with open(refused) as fh:
                rf = json.load(fh)
            mine.append(f"place5r.py REFUSED the ledger fetched at the last check-in ({rf.get('fetch')}): it does not extend "
                        "the placed copy, so the watch reads an old ledger. After a Colab resume the maintainer reported "
                        f"(RUN_ID and N): cd {TOOLS} && python3 -I place5r.py csv {run} {rf.get('fetch')} --resumed-from N; "
                        "otherwise fetch the ledgers again")
        if not path:
            if not a.validate_on_stance4 and p_eval is None and (p_steps or 0) < EVAL:
                entry["status"] = "no evaluation yet"
                print(f"{run}: no eval_episodes.csv placed; progress.json "
                      + (f"at {p_steps:,} steps, before the first evaluation" if p_steps is not None else "not placed either")
                      + " (place the ledgers once the run has evaluated)")
                if p_steps is None:
                    mine.append("neither eval_episodes.csv nor progress.json is placed")
            else:
                mine.append("no eval_episodes.csv placed (place the ledgers; a ledger that came back inline early in a "
                            "run is skipped by the fetch)")
                entry["status"] = "no ledger"
                print(f"{run}: no eval_episodes.csv under {run_dir}")
            for m in mine:
                print(f"  LEDGER PROBLEM: {run}: {m}. Not 'no alarm': this run's alarm state is unknown until it is fixed.")
            problems += [f"{run}: {m}" for m in mine]
            report[run] = entry
            continue
        rows = SC.SC3.read_csv(path)
        sa = SC.saturation_alarm(rows)
        last = max((int(float(r["timesteps"])) for r in rows), default=None)
        entry.update({"rows": len(rows), "last_evaluation": last, "alarm": sa})
        if last is None:
            mine.append("eval_episodes.csv has no rows")
        if not a.validate_on_stance4 and not ledgers.get(run):
            if ppath is None:
                mine.append("no progress.json placed: the ledger's freshness is unchecked (fetch it with the ledgers)")
            elif last is not None and p_eval is not None:
                if p_eval - last > EVAL:
                    mine.append(f"STALE: the ledger ends at {last:,} but progress.json's last evaluation is {p_eval:,}")
                elif last - p_eval > EVAL:
                    mine.append(f"STALE: the ledger ends at {last:,}, past progress.json's last evaluation {p_eval:,} "
                                "(a session resumed from an earlier checkpoint?)")
        line = (f"{run}: {sa.get('evaluations', 0)} evaluations to {last:,}; flagged "
                f"{len(sa.get('flagged', []))} (last {sa.get('flagged', [])[-3:]})"
                + ("" if p_eval is None else f"; progress.json's last evaluation {p_eval:,}")
                if sa.get("computed") and last is not None else f"{run}: not computed ({sa.get('why')})")
        if sa.get("alarm"):
            any_alarm = True
            first_pair = sa["runs"][0]
            F = first_pair["from"]
            prev = state.get(run, {}) if state.get(run, {}).get("first_alarm_at") == sa["first_alarm_at"] else {}
            new_alarm = not prev
            reported = prev.get("reported")
            after = [n for n in placed_steps(run_dir) if n >= F]
            latest = after[-1] if after else None
            fetch_at = -(-F // CADENCE) * CADENCE
            roll_out = None if latest is None else os.path.join(a.roll_dir, f"{run}_{latest}.json")
            arm = "LTS" if a.validate_on_stance4 else "LTSB"
            clone = SCRATCH + ("/fix/screen_s4_LTS" if a.validate_on_stance4 else "/fix/scratch_copband")
            cmd = None if latest is None else (
                f"cd {TOOLS} && env ISO_WT={clone} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                f"PYTHONPYCACHEPREFIX={SCRATCH}/caches/pyc MPLCONFIGDIR={SCRATCH}/caches/mpl nice -n 19 python3 "
                f"{SCRATCH}/foot/stance3/tools/iso.py satroll5r.py {run_dir} --arm {arm} --seed {seed} --steps {latest} "
                f"--out {roll_out}")
            named, roll_rc, rolled = None, None, None
            if roll_out and not os.path.exists(roll_out) and a.roll:
                os.makedirs(os.path.dirname(roll_out), exist_ok=True)
                roll_rc = subprocess.run(cmd, shell=True, check=False).returncode
            if roll_out and os.path.exists(roll_out):
                with open(roll_out) as fh:
                    rolled = json.load(fh)["checkpoints"].get(str(latest)) or {}
                named = rolled.get("named")
            if latest is None:
                why = (f"no checkpoint at or after the first flagged evaluation {F:,} is placed for {run}: fetch "
                       f"{fetch_at:,} for this run alone now (stance5r-fetch-checkpoint.js "
                       f"{{\"steps\": {fetch_at}, \"runs\": [\"{run}\"]}}), then re-run with --roll")
            elif roll_rc not in (None, 0):
                why = f"satroll5r FAILED (rc {roll_rc}) on checkpoint {latest:,}: {cmd}"
            elif rolled is not None:
                why = (f"checkpoint {latest:,} rolled: no actuator at |command| >= 0.9 in the window "
                       f"({rolled.get('flagged_over_bar')} of {rolled.get('episodes')} episodes over the gate's bar); "
                       "the next placed checkpoint is rolled")
            else:
                why = f"checkpoint {latest:,} is placed but not rolled yet: re-run with --roll ({cmd})"
            if named:
                sign = "+" if named["sign"] > 0 else "-"
                act = (f"actuator {named['actuator']} at {sign}1 (|command| >= 0.9 on {100 * named['frac90']:.0f}% of window "
                       f"steps, mean {named['mean_command']:+.2f}, checkpoint {latest:,}, "
                       f"{rolled.get('flagged_over_bar')} of {rolled.get('episodes')} rolled episodes over the gate's bar"
                       + (f"; {named['note']}" if named.get("note") else "") + ")")
            else:
                act = "actuator NOT NAMED YET (" + why + ")"
            now = None if not named else {"actuator": named["actuator"], "sign": named["sign"], "checkpoint": latest}
            if new_alarm:
                tag = "NEW ALARM"
            elif now and reported is None:
                tag = "ACTUATOR NOW NAMED"
            elif now and (reported["actuator"], reported["sign"]) != (now["actuator"], now["sign"]):
                tag = "ACTUATOR CHANGED"
            else:
                tag = "ALARM"
            any_new = any_new or tag != "ALARM"
            line += (f"\n  {tag}: run {run}, step {sa['first_alarm_at']:,} (first flagged evaluation {F:,}), {act}. "
                     "Report only: the run is not stopped or changed.")
            entry.update({"tag": tag, "new": new_alarm, "roll_rc": roll_rc, "rolled_checkpoint": latest,
                          "fetch_for_naming": None if latest is not None else fetch_at, "satroll5_command": cmd,
                          "satroll5_output": roll_out, "named": named})
            if not a.no_state:
                keep = reported if (now is None or (tag == "ALARM" and reported)) else now
                state[run] = {"first_alarm_at": sa["first_alarm_at"], "reported": keep}
        else:
            line += "; no alarm"
        print(line)
        for m in mine:
            print(f"  LEDGER PROBLEM: {run}: {m}. Not 'no alarm': this run's alarm state is unknown until it is fixed.")
        problems += [f"{run}: {m}" for m in mine]
        report[run] = entry
    if a.validate_on_stance4:
        r42, r52, r60 = (report.get(f"stance4_LTS_s{s}", {}).get("alarm") or {} for s in (42, 52, 60))
        ok = (r42.get("first_alarm_at") == 2228224 and (r42.get("runs") or [{}])[0].get("from") == 2097152
              and r52.get("computed") and not r52.get("alarm") and r60.get("computed") and not r60.get("alarm"))
        print(f"VALIDATION ON STANCE4: s42 alarm at 2,228,224 (first flagged 2,097,152), s52 and s60 none: {'yes' if ok else 'NO'}")
        report["validation_ok"] = ok
    report["ledger_problems"] = problems
    if not a.no_state:
        os.makedirs(os.path.dirname(a.state), exist_ok=True)
        with open(a.state, "w") as fh:
            json.dump(state, fh, indent=1)
    if a.json:
        with open(a.json, "x") as fh:
            json.dump(report, fh, indent=1, default=str)
    return 11 if any_new else 10 if any_alarm else 3 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
