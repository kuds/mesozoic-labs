import json, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/tools")
import ecs_common as C
V = C.VALIDATE + "/fix2_v2"
cur = {f: C.sha256_file(f"{C.TOOLS}/{f}") for f in C.TOOL_FILES}
bad = 0
for name in ("s42_A_1101", "s42_B_1201", "s80_A_1301", "s80_B_1401"):
    m = json.load(open(f"{V}/{name}/run_meta.json"))
    ck = json.load(open(f"{V}/{name}/check.json"))
    S = m["S"]; ds, de = m["digest_start"], m["digest_end"]
    tools = m.get("tool_sha256") or {}
    conds = {
        "status done": m["status"] == "done", "exit 0": m["exit"] == 0, "mode no_update": m["mode"] == "no_update",
        "stop end": m["stop_reason"] == "end", "T = S+262144": m["T"] == S + 262_144 == m["target"],
        "params equal": ds["params"] == de["params"], "adam equal": ds["adam"] == de["adam"],
        "n_updates equal": ds["n_updates"] == de["n_updates"], "torch_threads 1": m.get("torch_threads") == 1,
        "collapse found/patience": (m["collapse"]["found"], m["collapse"]["patience"]) == (1, 10),
        "ent_coef_after_load": m["ent_coef_after_load"] == C.ARM_ENT_COEF_AFTER_LOAD[m["arm"]],
        "tools = current": all(tools.get(f) == h for f, h in cur.items()),
        "code digest": m["code"]["tree_digest"] == C.TREE_DIGEST,
        "check passed": ck.get("passed") is True and ck.get("failures") == [] and ck.get("tool_sha256") == C.sha256_file(C.TOOLS + "/read_ecs.py"),
    }
    ok = all(conds.values()); bad += not ok
    print(name, "OK" if ok else "BAD", {k: v for k, v in conds.items() if not v}, "train_calls", m.get("train_calls"),
          "check keys", sorted(ck)[:8])
print("ALL_OK" if not bad else f"BAD {bad}")
