"""Re-measure one panel exactly as measure_tailscreen.measure_one does, writing only under VERIFY/out."""
import importlib.util, json, sys, csv
from pathlib import Path
SCREEN = Path("scratchpad2/tailround/screen")
V = SCREEN / "verify" / "remeasure"
spec = importlib.util.spec_from_file_location("measure_tailscreen", SCREEN / "tools" / "measure_tailscreen.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
import subprocess, os, time
name, label = sys.argv[1], sys.argv[2]
out_dir = V / "out" / name / label
assert not out_dir.exists(), out_dir
out_dir.parent.mkdir(parents=True, exist_ok=True)
z, p = M.checkpoint_pair(name, label)
digest = M.check_copy()
zs, ps = M.sha256_file(z), M.sha256_file(p)
cmd = ["nice", "-n", "19", str(M.C.PY), "-I", "-B", str(M.MEASURE_COPY / M.TRACE_SCRIPT_REL),
       "--species", "tyrannosaurus_rex", "--embodiment", "anatomical", "--model", str(z), "--vecnorm", str(p),
       "--seed", str(M.C.PANEL_SEEDS[0]), "--episodes", str(len(M.C.PANEL_SEEDS)), "--out-dir", str(out_dir)]
env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
logs = V / "logs"; logs.mkdir(exist_ok=True)
t0 = time.monotonic()
proc = subprocess.run(cmd, env=env, cwd=str(logs), capture_output=True, text=True)
digest_after = M.check_copy()
assert M.sha256_file(z) == zs and M.sha256_file(p) == ps
(V / "out" / name / f"{label}_trace_run.json").write_text(json.dumps({"argv": cmd, "rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "seconds": round(time.monotonic()-t0,1), "digest_before": digest, "digest_after": digest_after, "zip_sha256": zs, "pkl_sha256": ps}, indent=1))
assert proc.returncode == 0, proc.stderr[-3000:]
agg, flags, problems = M.reduce_panel(out_dir)
row = {**M.run_identity(name), "label": label, "num_timesteps": M.zip_num_timesteps(z), "zip_sha256": zs, "pkl_sha256": ps,
       **{k: v for k, v in agg.items() if k in M.PANEL_COLUMNS}, "measure_tree_digest": digest}
row = {k: (M._f(v) if isinstance(v, float) or v is None else v) for k, v in row.items()}
with open(V / "out" / name / f"{label}_flags.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=M.FLAG_COLUMNS, lineterminator="\n"); w.writeheader()
    for f in flags:
        w.writerow({k: M._f(v) if isinstance(v, float) or v is None else v for k, v in f.items()})
(V / "out" / name / f"{label}_row.json").write_text(json.dumps({"row": row, "problems": problems, "gate_flagged": agg["gate_flagged"]}, indent=1))
print(json.dumps({"name": name, "label": label, "problems": problems, "cmd": row["cmd"]}))
