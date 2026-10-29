"""Each end trace: made on the run's own E pair (zip and pkl sha256), num_timesteps E, by the pinned M1, on the copy."""
import json, glob, hashlib
ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = []
for p in sorted(glob.glob(ECS + "/traces/*/m1_summary.json")):
    n = p.split("/")[-2]; d = json.load(open(p)); c = d["checkpoint"]; E = 4718592 if n.startswith("s42") else 5767168
    pk = f"{ECS}/runs/{n}/01_stance/models/stage1_vecnormalize_{E}_steps.pkl"
    blob = json.dumps(d["checkpoint"]) + json.dumps(d.get("run_records")) + json.dumps(d.get("trace_run"))
    if pk not in blob or sha(pk) not in blob: bad.append((n, "pkl not named with its sha256"))
    if c.get("num_timesteps") != E: bad.append((n, "num_timesteps", c.get("num_timesteps")))
    if d["run_records"].get("stage_config") != f"{ECS}/runs/{n}/01_stance/stage_config.json": bad.append((n, "stage_config"))
print("checkpoint keys:", sorted(c)); print("traces", len(glob.glob(ECS + "/traces/*/m1_summary.json")), "problems:", bad or "none")
print("argv:", " ".join(d["trace_run"]["argv"])[:900])
