"""PC dry run (report only): C1-C8 of the adapted completeness tool on a switch-ON run folder written by HEAD's own
writers (dry/smoke_bundle/stance7_r10LTSBpc_s42: 1,024 steps, 2 PPO updates), stored as an F2 fetch store (non-zip
files twice, zips once, COPY rows), with a synthetic RUNLOG (LAUNCH at the bundle's started_at, FOUND-GONE, F2-START)
and the manipulation check filed for the final pair (manip_check.py --run-id ... --label final --out <dry panels>).
Expected: C4 (p) PASS on the real stage_config.json; C8 PASS on a record bound to the real final pair; C3 FAIL (a
1,024-step folder has no R1) and C6 FAIL (2 episodes per evaluation) -- the folder is a smoke, not a run.  Writes only
under --out.

PC AMENDMENT (2026-11-01): --arm off reads a switch-OFF smoke folder (smoke_pc.py train/bundle --switch off) under
stance7_r10LTSBoff_s42: expected C4 (p) PASS (no switch key) and C8 PASS on the stock-critic record manip_check.py
writes for it (the critic reads no phase); C3 and C6 FAIL as above.  --arm pc (default): unchanged.

    PY -I -B f2dry.py --out DIR --copy COPY --head SHA [--arm pc|off] [--src RUN_DIR]
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
RUN = "stance7_r10LTSBpc_s42"
COLS = ("utc", "fetch", "run_id", "path", "drive_id", "listing_size", "transfer", "decoded_size", "status", "copy",
        "sha256", "stored_as", "method", "note")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--arm", choices=("pc", "off"), default="pc")                     # PC AMENDMENT
    ap.add_argument("--src", default=None)
    a = ap.parse_args()
    run = RUN if a.arm == "pc" else "stance7_r10LTSBoff_s42"
    out = Path(a.out).resolve()
    if out.exists():
        print(f"REFUSED: {out} exists")
        return 2
    src = Path(a.src) if a.src else (T / "dry" / ("smoke_bundle" if a.arm == "pc" else "smoke_bundle_off") / run)
    root = out / "round"
    store = root / "fetch" / "F2" / run / "copies"
    rows = []
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(src).as_posix()
        for k in ((1,) if rel.endswith(".zip") else (1, 2)):
            dst = store / f"{rel}.copy{k}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            b = dst.read_bytes()
            rows.append(dict(utc="2026-11-01T23:00:00Z", fetch="F2", run_id=run, path=rel, drive_id="dry",
                             listing_size=len(b), transfer=k, decoded_size=len(b), status="COPY", copy=k,
                             sha256=hashlib.sha256(b).hexdigest(), stored_as=str(dst), method="dry copy", note=""))
    with open(root / "fetch" / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    prov = json.loads((src / "provenance.json").read_text())
    started = prov["sessions"][0]["started_at"][:16].replace("T", " ")
    (root / "runlog.txt").write_text(
        f"{started} UTC  STANCE7 LAUNCH {run} seed=42 runtime=CPU head={a.head} "
        f"CRITIC_PHASE={'True' if a.arm == 'pc' else 'False'}\n"
        f"2026-11-01 16:30 UTC  STANCE7 FOUND-GONE {run} note=\"dry\"\n"
        f"2026-11-01 22:30 UTC  STANCE7 F2-START {run} note=\"dry\"\n", encoding="utf-8")
    panels = out / "panels" / run
    dry_ap = out / "APPROVED_dry"
    dry_ap.write_text(json.dumps({"branch_head": a.head, "note": "DRY-RUN record for the tools only (not an approval)"}))
    zp = store / "01_stance/models/stage1_final.zip.copy1"
    pp = store / "01_stance/models/stage1_final_vecnorm.pkl.copy1"
    sc = store / "01_stance/stage_config.json.copy1"
    mc = subprocess.run([PY, "-I", "-B", str(T / "pc" / "manip_check.py"), "--copy", a.copy, "--approved", str(dry_ap),
                         "--zip", str(zp), "--pkl", str(pp), "--stage-config", str(sc), "--run-id", run, "--label",
                         "final", "--out", str(panels / "final" / "c8")], capture_output=True, text=True, cwd="/tmp")
    (out / "manip_check.txt").write_text(mc.stdout + mc.stderr)
    print(f"manip_check exit {mc.returncode}: {(mc.stdout + mc.stderr)[-600:]}")
    cp = subprocess.run([PY, "-I", "-B", str(T / "pc" / "complete_r10round.py"), "run", "--run", run, "--round",
                         str(root), "--runlog", str(root / "runlog.txt"), "--pins", str(T / "pc" / "pins_pc.json"),
                         "--panels", str(panels), "--json", str(out / "complete.json")],
                        capture_output=True, text=True, cwd="/tmp")
    (out / "complete.txt").write_text(cp.stdout + cp.stderr)
    print(cp.stdout[-9000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
