"""V5 (THIS12 line 858, the criteria's V5): the M1/M2/M3/M4 runners refuse a model whose recorded physics_revision is
not 10 (a synthetic r9 record).

    PY -I -B v5_check.py COPY CKPT_ROOT OUT_DIR [--header-parent | --approved APPROVED]

The copy's commit basis (criteria lean v5: no tool holds the branch HEAD) is passed to every runner: --header-parent
(the header's parent cf9402bc, a validation before the header commit; the judge gets --expect-head cf9402bc) or
--approved APPROVED (the approval record; the default record <ROUND>/APPROVED when neither is given).

CKPT_ROOT holds statue_ckpt.py's pairs statue_r9/ (the r9 plant identity recorded on the zip and the VecNormalize)
and statue_none/ (no plant identity).  For each record and each runner (measure_r10round.py m1, m2, m3; judge_r10ltsb.py
on a synthetic run folder holding that pair as its stage1_final), the runner is called as a run's measurement would
call it, and V5 passes when every call exits 2 (refused), its refusal names the recorded physics_revision, nothing is
rolled (no output folder or judge file is written) and the measuring copy is unchanged.  Each call's argv, exit code,
refusal line and seconds go to OUT_DIR/v5.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
VERSION = "v5_check 1.1 (2026-10-26; v5: the copy's commit basis passed)"
HERE = Path(__file__).resolve().parent
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"


def main() -> int:
    print(VERSION, flush=True)
    copy, ckroot, out = (os.path.realpath(x) for x in sys.argv[1:4])
    basis = sys.argv[4:]
    spec = importlib.util.spec_from_file_location("r10m_common_v5", HERE / "r10m_common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    judge_basis = (["--expect-head", common.HEADER_PARENT] if basis[:1] == ["--header-parent"] else
                   (["--approved", basis[1]] if basis[:1] == ["--approved"] else []))
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    calls = []
    for rec, want in (("r9", "9"), ("none", "None")):
        z = f"{ckroot}/statue_{rec}/stage1_final.zip"
        p = f"{ckroot}/statue_{rec}/stage1_final_vecnorm.pkl"
        # a synthetic run folder for the judge (the notebook's layout, the pair as stage1_final)
        run = outp / f"run_{rec}"
        (run / "01_stance" / "models").mkdir(parents=True)
        shutil.copy2(z, run / "01_stance" / "models" / "stage1_final.zip")
        shutil.copy2(p, run / "01_stance" / "models" / "stage1_final_vecnorm.pkl")
        jdir = outp / f"judged_{rec}"
        jdir.mkdir()
        runner = [PY, "-I", "-B", str(HERE / "measure_r10round.py")]
        argvs = {
            "M1": runner + ["m1", "--copy", copy, "--zip", z, "--pkl", p, "--no-run-records", "--out", str(outp / f"m1_{rec}"),
                            *basis],
            "M2": runner + ["m2", "--copy", copy, "--zip", z, "--pkl", p, "--no-run-records", "--out", str(outp / f"m2_{rec}"),
                            "--statue-json", str(Path(out).parent / "V1" / "m2_statue" / "panel.json"), *basis],
            "M3": runner + ["m3", "--copy", copy, "--zip", z, "--pkl", p, "--no-run-records", "--out", str(outp / f"m3_{rec}"),
                            *basis],
            "M4": [PY, "-I", "-B", str(HERE / "judge_r10ltsb.py"), str(run), "--arm", "LTSB", "--seed", "60", "--steps",
                   "final", "--clone", copy, "--out", str(jdir / "judge_final.json"), *judge_basis],
        }
        for name, argv in argvs.items():
            t0 = time.monotonic()
            proc = subprocess.run(argv, env=env, cwd=out, capture_output=True, text=True)
            err = [ln for ln in proc.stderr.splitlines() if "REFUSED" in ln or "refusing" in ln]
            wrote = (outp / f"{name.lower()}_{rec}").exists() if name != "M4" else any(jdir.iterdir())
            ok = proc.returncode != 0 and bool(err) and f"physics_revision {want}" in err[-1] and not wrote
            if name != "M4":
                ok = ok and proc.returncode == 2
            calls.append({"record": rec, "runner": name, "argv": argv, "returncode": proc.returncode,
                          "refusal": err[-1] if err else None, "stdout_tail": proc.stdout[-500:],
                          "output_written": wrote, "seconds": round(time.monotonic() - t0, 2), "refused_as_V5_needs": ok})
    seal_ok = subprocess.run([PY, "-I", "-B", str(HERE / "measure_r10round.py"), "check-copy", "--copy", copy, *basis],
                             env=env, capture_output=True, text=True).returncode == 0
    res = {"tool": VERSION, "copy": copy, "calls": calls, "copy_unchanged": seal_ok,
           "PASS": seal_ok and all(c["refused_as_V5_needs"] for c in calls)}
    (outp / "v5.json").write_text(json.dumps(res, indent=1) + "\n")
    for c in calls:
        print(f"{c['record']:4s} {c['runner']}: rc={c['returncode']} refused={c['refused_as_V5_needs']} | {c['refusal']}")
    print("copy unchanged:", seal_ok, "PASS:", res["PASS"])
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
