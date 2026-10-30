"""LB: one continuation's disk use at E, from measured file sizes (reads only; prints JSON).

    PY -I -B disk_estimate.py <LB tools dir> [<V2 run dir> ...]

Sources [measured, os.stat]:
  - a finished BS2 continuation (S2/r10round/bias_screen2/runs/control_2501, 176 rollouts, S -> 3,538,944): the zip and
    pkl sizes, arm_evidence.jsonl bytes per line, diagnostics.npz bytes per session step, run_meta / check / stage
    files,
    BS2's tmp/<name>, logs/<name>.* and reads/<name> sizes;
  - the parent's F2 records at 12,001,280 (the five records a continuation carries to E, the parent's own sizes at the
    end of the same run: the continuation's records hold the parent's rows <= S plus its own rows in (S, E));
  - optional LB V2 / V4 run folders (no-update runs, for the growth of the records per rollout under this harness).
The estimate = the start copy + 37 periodic pairs + the final pair + the records at E + the evidence (one line per
update
row) + diagnostics (scaled by the session's steps) + the small files + tmp/logs/reads; plus one resume's extra files
(a crashed attempt's records, evidence and the pairs after its resume point stay in its runs/<name>_crashed<k>/; the
pairs up to the resume point are hard links, no extra bytes).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bs_common", sys.argv[1] + "/bs_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

BS2 = Path(C.S2) / "r10round" / "bias_screen2"
RUN = BS2 / "runs" / "control_2501"
st = RUN / "01_stance"
models = st / "models"


def size(p):
    return os.stat(p).st_size


def tree(p):
    return sum(size(Path(d) / f) for d, _, fs in os.walk(p) for f in fs) if Path(p).exists() else 0


out = {"bs2_run": str(RUN)}
periodic_zip = size(models / "stage1_2359296_steps.zip")
start_zip = size(models / "stage1_2097152_steps.zip")
final_zip = size(models / "stage1_final.zip")
periodic_pkl = size(models / "stage1_vecnormalize_2359296_steps.pkl")
start_pkl = size(models / "stage1_vecnormalize_2097152_steps.pkl")
n_periodic = len(C.SAVED_STEPS)
models_bytes = start_zip + start_pkl + n_periodic * (periodic_zip + periodic_pkl) + final_zip + periodic_pkl
out["zip_bytes"] = {"start": start_zip, "periodic": periodic_zip, "final": final_zip}
out["pkl_bytes"] = {"start": start_pkl, "periodic": periodic_pkl}
out["n_periodic_pairs"] = n_periodic
out["models_bytes"] = models_bytes
parent_records = {n: size(C.f2_record(n)) for n in C.PARENT_RECORDS}
out["parent_records_at_E_bytes"] = parent_records
rec_bytes = sum(parent_records.values())
n_lines_bs2 = sum(1 for _ in open(RUN / "arm_evidence.jsonl"))
ev_per_line = size(RUN / "arm_evidence.jsonl") / n_lines_bs2
n_rows = C.expected_update_rows(C.S, C.E)
ev_bytes = int(ev_per_line * n_rows)
out["evidence"] = {"bs2_lines": n_lines_bs2, "bs2_bytes": size(RUN / "arm_evidence.jsonl"),
                   "bytes_per_line": ev_per_line, "lb_lines": n_rows, "lb_bytes": ev_bytes}
bs2_session = 3_538_944 - C.S
diag = size(st / "diagnostics.npz")
diag_bytes = int(diag * (C.E - C.S) / bs2_session)
out["diagnostics"] = {"bs2_bytes": diag, "bs2_session_steps": bs2_session, "lb_bytes_scaled": diag_bytes}
small = sum(size(st / n) for n in ("progress.json", "stage_config.json", "plant_identity.json",
                                    "task_fingerprint.json"))
small += size(RUN / "run_meta.json") + size(RUN / "check.json")
out["small_files_bytes"] = small
logs = sum(size(p) for p in (BS2 / "logs").glob("control_2501.*"))
side = tree(BS2 / "tmp" / "control_2501") + tree(BS2 / "reads" / "control_2501") + logs
out["bs2_tmp_reads_logs_bytes"] = side
# the reads grow with the window (markers.json lists probe rows and evaluations): scale the BS2 read folder by rows
reads_scaled = int(tree(BS2 / "reads" / "control_2501") * n_rows / 175)
out["reads_scaled_bytes"] = reads_scaled
run_bytes = models_bytes + rec_bytes + ev_bytes + diag_bytes + small
out["run_folder_bytes_at_E"] = run_bytes
out["run_folder_MiB_at_E"] = run_bytes / 2**20
resume_extra = rec_bytes + ev_bytes + diag_bytes + 2 * (periodic_zip + periodic_pkl) + small
out["one_resume_extra_bytes"] = resume_extra
total = run_bytes + reads_scaled + side + resume_extra
out["per_run_with_reads_and_one_resume_bytes"] = total
out["per_run_with_reads_and_one_resume_MiB"] = total / 2**20
lb = []
for p in sys.argv[2:]:
    p = Path(p)
    if p.is_dir():
        lb.append({"run": str(p), "bytes": tree(p)})
out["lb_validation_runs"] = lb
print(json.dumps(out, indent=1))
