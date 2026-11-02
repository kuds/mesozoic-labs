"""Report-only: the software and hardware fields each run records (provenance.json and 01_stance/stage_config.json)."""
import json, sys
def fields(prov, stage):
    p = json.load(open(prov)); s = json.load(open(stage))
    dv = p.get("dependency_versions", {})
    g = s.get("gpu", {})
    return (dv.get("stable_baselines3"), dv.get("gymnasium"), dv.get("mujoco"), dv.get("numpy"), p.get("python_version"),
            p.get("platform"), g.get("gpu_model"), g.get("cuda_version"), s.get("library_version"),
            any("torch" in k.lower() for k in list(dv) + list(p) + list(s)))
for label, root in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    print(label, fields(f"{root}/provenance.json.copy1", f"{root}/01_stance/stage_config.json.copy1"))
