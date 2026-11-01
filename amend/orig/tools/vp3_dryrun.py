"""PC VP3 dry run (report only): the adapted launch check (pc/complete_r10round.py launch-check) on
  POS: the switch-ON smoke bundle (dry/smoke_bundle/stance7_r10LTSBpc_s42: provenance.json, stage_config.json with the
       launch run block, the smoke ledger), two byte copies each in an F0-shaped store; expected: no fault;
  NEG: STANCE6 s42's F0 files (S1/foot/stance6/fetch/F0/stance6_r10LTSB_s42, read only), copied into an F0-shaped store
       under the PC RUN_ID; expected: facts (1) and (8) FAULT (and (3), (7)).
Each store gets a synthetic RUNLOG (one STANCE7 LAUNCH line, head = the measuring copy's commit) and fetch manifest
COPY rows (10 minutes after the launch).  Writes only under --out.

    PY -I -B vp3_dryrun.py --out DIR --head SHA
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
RUN = "stance7_r10LTSBpc_s42"
FILES = ("provenance.json", "01_stance/stage_config.json", "01_stance/eval_episodes.csv", "zero_action_baseline.json",
         "01_stance/progress.json")
COLS = ("utc", "fetch", "run_id", "path", "drive_id", "listing_size", "transfer", "decoded_size", "status", "copy",
        "sha256", "stored_as", "method", "note")


def build(root: Path, src_of, head: str) -> list[str]:
    store = root / "fetch" / "F0" / RUN / "copies"
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
            rows.append(dict(utc="2026-11-01T01:10:00Z", fetch="F0", run_id=RUN, path=rel, drive_id="dry", listing_size=len(b),
                             transfer="dry", decoded_size=len(b), status="COPY", copy=k,
                             sha256=hashlib.sha256(b).hexdigest(), stored_as=str(dst), method="dry copy", note=""))
        held.append(rel)
    with open(root / "fetch" / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    (root / "runlog.txt").write_text(
        f"2026-11-01 01:00 UTC  STANCE7 LAUNCH {RUN} seed=42 runtime=L4 head={head} "
        f"REPO_REF=claude/screen-trex-r10-phasecritic BEHAVIOR=stance K3_PANEL=True "
        f"K3_PANEL_SIGMA_FILE=/content/mesozoic-labs/docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json "
        f"K3_PANEL_SIGMA_KEY=source_vectors.42 CRITIC_PHASE=True RUN_LABEL=\"r10 LTSB phase-critic s42\"\n",
        encoding="utf-8")
    return held


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--pins", default=str(T / "pc" / "pins_pc.json"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        print(f"REFUSED: {out} exists")
        return 2
    pos_src = T / "dry" / "smoke_bundle" / RUN
    neg_src = S1 / "foot" / "stance6" / "fetch" / "F0" / "stance6_r10LTSB_s42" / "copies"
    rc = 0
    for name, src_of in (("POS", lambda rel: pos_src / rel), ("NEG", lambda rel: neg_src / f"{rel}.copy1")):
        root = out / name / "round"
        held = build(root, src_of, a.head)
        cmd = [PY, "-I", "-B", str(T / "pc" / "complete_r10round.py"), "launch-check", "--run", RUN, "--round",
               str(root), "--runlog", str(root / "runlog.txt"), "--pins", a.pins, "--json", str(out / f"{name}.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
        (out / f"{name}.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
        print(f"===== {name} (files held: {held}; exit {p.returncode})")
        print(p.stdout[-6000:])
        if p.stderr.strip():
            print("STDERR:", p.stderr[-2000:])
        rc = max(rc, p.returncode)
    return rc


if __name__ == "__main__":
    sys.exit(main())
