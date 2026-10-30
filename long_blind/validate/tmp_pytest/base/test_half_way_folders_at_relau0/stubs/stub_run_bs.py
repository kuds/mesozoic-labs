
import json, os, sys, time
a = sys.argv[1:]
arm, seed = a[a.index("--arm") + 1], a[a.index("--seed") + 1]
resume = os.path.basename(a[a.index("--resume-from") + 1]) if "--resume-from" in a else "-"
root = os.environ["BS_ROOT"]
name = f"{arm}_{seed}"
S = int(os.environ["BS_STUB_S"])
os.makedirs(f"{root}/attempts", exist_ok=True)
with open(f"{root}/attempts/{name}", "a") as fh:
    fh.write("x\n")
n = sum(1 for _ in open(f"{root}/attempts/{name}"))
plan = json.load(open(f"{root}/plan.json")).get(name, [])
what = plan[n - 1] if n <= len(plan) else "end"
with open(f"{root}/calls.log", "a") as fh:
    fh.write(f"run {name} {what} resume={resume}\n")
print(f"stub attempt {n} of {name}: {what}", flush=True)
if what.startswith("vslow"):
    time.sleep(4.0)
    what = what[5:] or "end"
if what.startswith("slow"):
    time.sleep(1.0)
    what = what[4:] or "end"
if what == "refuse":
    sys.exit(2)
run = f"{root}/runs/{name}"
os.makedirs(f"{run}/01_stance/models")
meta = {"status": "running", "exit": None, "stop_reason": None}
load = S + 2 * 262144 if what.startswith("resumed") else S
if what.startswith("resumed"):
    what = what[7:] or "end"
meta = {"status": "running", "exit": None, "stop_reason": None, "load_step": load}
if what in ("end", "collapse", "checkfail", "hop"):
    meta = {"status": "done", "exit": 0, "stop_reason": {"collapse": "collapse", "hop": "hop_watch"}.get(what, "end")}
    if what == "checkfail":
        meta["stub_check"] = "fail"
elif what == "postfail":
    meta = {"status": "failed", "exit": 5, "stop_reason": "end"}
elif what in ("crash", "earlycrash"):
    meta.update(status="crashed", exit=1, last_step=load + (8192 if what == "earlycrash" else 3 * 131072))
elif what == "kill":
    meta["heartbeat"] = {"rollout_end": S + 2 * 131072}
json.dump(meta, open(f"{run}/run_meta.json", "w"))
with open(f"{run}/01_stance/policy_probe.npz", "w") as fh:
    fh.write("probe")
os.makedirs(f"{root}/tmp/{name}/eval_stage1_x", exist_ok=True)
with open(f"{root}/tmp/{name}/eval_stage1_x/evaluations.npz", "w") as fh:
    fh.write("returns")
if what in ("crash", "earlycrash"):
    sys.exit(1)
if what == "kill":
    os._exit(137)
if what == "postfail":
    sys.exit(5)
sys.exit(0)
