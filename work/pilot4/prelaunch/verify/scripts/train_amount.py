"""How much was trained in the smoke: zip JSON data (num_timesteps, _n_updates) of every zip in the launch and resume
run copies and of the s42 parent, and the largest TensorBoard step per event file (tensorboard's event reader; no
repository import). Usage: python -I train_amount.py LAUNCH_RUN RESUME_RUN PARENT_ZIP OUT_JSON
"""
import json
import sys
import zipfile
from pathlib import Path

LAUNCH, RESUME, PARENT, OUT = sys.argv[1:5]
res = {"zips": {}, "tensorboard": {}}


def zdata(p):
    d = json.loads(zipfile.ZipFile(p).read("data"))
    lin = d.get("mesozoic_task_lineage")
    if isinstance(lin, dict) and "__type__" not in lin:
        lin_s = lin
    else:
        lin_s = str(lin)[:200]
    return {"num_timesteps": d.get("num_timesteps"), "_n_updates": d.get("_n_updates"),
            "_num_timesteps_at_start": d.get("_num_timesteps_at_start"), "lineage": lin_s}


res["zips"]["parent_s42_8M"] = zdata(PARENT)
for tag, run in (("launch", LAUNCH), ("resume", RESUME)):
    for z in sorted(Path(run, "03_locomotion", "models").glob("*.zip")):
        res["zips"][f"{tag}/{z.name}"] = zdata(z)
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        for ev in sorted(Path(run).rglob("events.out.tfevents.*")):
            steps = []
            for e in EventFileLoader(str(ev)).Load():
                if e.HasField("summary"):
                    steps.append(int(e.step))
            res["tensorboard"][f"{tag}/{ev.relative_to(run)}"] = {"n": len(steps), "min_step": min(steps) if steps else None,
                                                                   "max_step": max(steps) if steps else None}
    except Exception as exc:  # report, do not hide
        res["tensorboard"][f"{tag}/error"] = f"{type(exc).__name__}: {exc}"
json.dump(res, open(OUT, "w"), indent=1, default=str)
for k, v in res["zips"].items():
    print(k, v["num_timesteps"], v["_n_updates"], v["_num_timesteps_at_start"],
          (sorted(v["lineage"].items()) if isinstance(v["lineage"], dict) else v["lineage"]))
for k, v in res["tensorboard"].items():
    print("TB", k, v)
