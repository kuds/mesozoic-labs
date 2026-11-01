"""PC VP3 dry run (report only): the adapted launch check (pc/complete_r10round.py launch-check) on
  POS: the switch-ON smoke bundle (dry/smoke_bundle/stance7_r10LTSBpc_s42: provenance.json, stage_config.json with the
       launch run block, the smoke ledger), two byte copies each in an F0-shaped store; expected: no fault;
  NEG: STANCE6 s42's F0 files (S1/foot/stance6/fetch/F0/stance6_r10LTSB_s42, read only), copied into an F0-shaped store
       under the PC RUN_ID; expected: facts (1) and (8) FAULT (and (3), (7)).
Each store gets a synthetic RUNLOG (one STANCE7 LAUNCH line, head = the measuring copy's commit) and fetch manifest
COPY rows (10 minutes after the launch).  Writes only under --out.
PC AMENDMENT (2026-11-01): every case is run under the RUN_ID of the arm named, with that arm's LAUNCH fields
(CRITIC_PHASE, RUN_LABEL), and the computed faults are compared with the expected ones (exit 1 on a difference):
  POS-pc   the switch-ON bundle under stance7_r10LTSBpc_s42 ................................. none
  POS-off  a switch-OFF bundle (dry/smoke_bundle_off/stance7_r10LTSBoff_s42: smoke_pc.py bundle --switch off,
           the same writers with CRITIC_PHASE False) under stance7_r10LTSBoff_s42 ............... none
  X-on-off the switch-ON bundle under the switch-off RUN_ID ................................... (3), (7), (8)
  X-off-pc the switch-OFF bundle under the phase RUN_ID ....................................... (3), (7), (8)
  NEG-pc   STANCE6 s42's F0 files under the phase RUN_ID ...................................... (1), (3), (7), (8)
  NEG-off  STANCE6 s42's F0 files under the switch-off RUN_ID (8737cd3d's files: only the commit and the
           run_id differ from a switch-off run's) ............................................ (1), (7)
  SC-*     the stage_config.json files of dry/pins_sc (on, off: written by HEAD's own save_stage_config as
           train_base calls it, no training) alone, under each arm's RUN_ID: on-pc and off-off none; on-off and
           off-pc (3), (8) (facts (1) and (4)-(7) not computed: no other file)
A bundle that is absent is SKIPPED (printed): the bundles are deleted after recording; the pin step makes them again.

    PY -I -B vp3_dryrun.py --out DIR --head SHA [--pins P] [--bundle-pc DIR] [--bundle-off DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
RUN = "stance7_r10LTSBpc_s42"
RUNS = {"pc": "stance7_r10LTSBpc_s42", "off": "stance7_r10LTSBoff_s42"}             # PC AMENDMENT
LABELS = {"pc": "r10 LTSB phase-critic s42", "off": "r10 LTSB switch-off s42"}
PHASE = {"pc": "True", "off": "False"}
FILES = ("provenance.json", "01_stance/stage_config.json", "01_stance/eval_episodes.csv", "zero_action_baseline.json",
         "01_stance/progress.json")
COLS = ("utc", "fetch", "run_id", "path", "drive_id", "listing_size", "transfer", "decoded_size", "status", "copy",
        "sha256", "stored_as", "method", "note")


def build(root: Path, src_of, head: str, arm: str = "pc") -> list[str]:
    run = RUNS[arm]
    store = root / "fetch" / "F0" / run / "copies"
    rows, held = [], []
    for rel in FILES:
        src = src_of(rel)
        if src is None or not src.is_file():
            continue
        for k in (1, 2):
            dst = store / f"{rel}.copy{k}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            b = dst.read_bytes()
            rows.append(dict(utc="2026-11-01T01:10:00Z", fetch="F0", run_id=run, path=rel, drive_id="dry", listing_size=len(b),
                             transfer="dry", decoded_size=len(b), status="COPY", copy=k,
                             sha256=hashlib.sha256(b).hexdigest(), stored_as=str(dst), method="dry copy", note=""))
        held.append(rel)
    with open(root / "fetch" / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    (root / "runlog.txt").write_text(
        f"2026-11-01 01:00 UTC  STANCE7 LAUNCH {run} seed=42 runtime=L4 head={head} "
        f"REPO_REF=claude/screen-trex-r10-phasecritic BEHAVIOR=stance K3_PANEL=True "
        f"K3_PANEL_SIGMA_FILE=/content/mesozoic-labs/docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json "
        f"K3_PANEL_SIGMA_KEY=source_vectors.42 CRITIC_PHASE={PHASE[arm]} RUN_LABEL=\"{LABELS[arm]}\"\n",
        encoding="utf-8")
    return held


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--pins", default=str(T / "pc" / "pins_pc.json"))
    ap.add_argument("--bundle-pc", default=str(T / "dry" / "smoke_bundle" / RUNS["pc"]))
    ap.add_argument("--bundle-off", default=str(T / "dry" / "smoke_bundle_off" / RUNS["off"]))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        print(f"REFUSED: {out} exists")
        return 2
    pos_pc, pos_off = Path(a.bundle_pc), Path(a.bundle_off)
    neg_src = S1 / "foot" / "stance6" / "fetch" / "F0" / "stance6_r10LTSB_s42" / "copies"
    sc_on, sc_off = T / "dry" / "pins_sc" / "on", T / "dry" / "pins_sc" / "off"

    def sc_only(d):
        return lambda rel: (d / "stage_config.json") if rel == "01_stance/stage_config.json" else None

    cases = [  # name, arm (RUN_ID), source, expected faults, required source dir
        ("POS-pc", "pc", lambda rel: pos_pc / rel, [], pos_pc),
        ("POS-off", "off", lambda rel: pos_off / rel, [], pos_off),
        ("X-on-off", "off", lambda rel: pos_pc / rel, [3, 7, 8], pos_pc),
        ("X-off-pc", "pc", lambda rel: pos_off / rel, [3, 7, 8], pos_off),
        ("NEG-pc", "pc", lambda rel: neg_src / f"{rel}.copy1", [1, 3, 7, 8], neg_src),
        ("NEG-off", "off", lambda rel: neg_src / f"{rel}.copy1", [1, 7], neg_src),
        ("SC-on-pc", "pc", sc_only(sc_on), [], sc_on),
        ("SC-on-off", "off", sc_only(sc_on), [3, 8], sc_on),
        ("SC-off-off", "off", sc_only(sc_off), [], sc_off),
        ("SC-off-pc", "pc", sc_only(sc_off), [3, 8], sc_off),
    ]
    rc, summary = 0, []
    for name, arm, src_of, want, need in cases:
        if not need.is_dir():
            summary.append(f"{name}: SKIPPED ({need} absent)")
            print(f"===== {name}: SKIPPED ({need} absent)")
            continue
        root = out / name / "round"
        held = build(root, src_of, a.head, arm)
        cmd = [PY, "-I", "-B", str(T / "pc" / "complete_r10round.py"), "launch-check", "--run", RUNS[arm], "--round",
               str(root), "--runlog", str(root / "runlog.txt"), "--pins", a.pins, "--json", str(out / f"{name}.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
        (out / f"{name}.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
        got = None
        try:
            got = json.loads((out / f"{name}.json").read_text())["faults"]
        except (OSError, ValueError, KeyError):
            pass
        ok = p.returncode == 0 and got == want
        rc = rc if ok else 1
        summary.append(f"{name} ({RUNS[arm]}; files {held}): faults {got}, expected {want}: {'AS EXPECTED' if ok else 'DIFFERS'}"
                       f" (exit {p.returncode})")
        print(f"===== {name} (files held: {held}; exit {p.returncode})")
        print(p.stdout[-6000:])
        if p.stderr.strip():
            print("STDERR:", p.stderr[-2000:])
    print("SUMMARY")
    print("\n".join(summary))
    (out / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
