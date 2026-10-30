"""read.json against the files it names: check.json, ledger, probe, m5.json, trace; and its tool and M5 pins."""
import hashlib, json, glob, os
ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
bad = []
for d in sorted(glob.glob(ECS + "/score/read/*")):
    n = os.path.basename(d); r = json.load(open(d + "/read.json")); run = f"{ECS}/runs/{n}"
    checks = {
        "tool": r["tool_sha256"] == "6f55b3b96bae795d46c3ef693b5b542895186adb4ccb51957076f3421c2fee15",
        "check_sha": r["check_sha256"] == sha(run + "/check.json"),
        "ledger": r["r1"]["ledger_sha256"] == sha(run + "/01_stance/eval_episodes.csv"),
        "m5pins": r["r1"]["m5_sha256"] == "0d89bcb858e19f780c3ce0ac7a5886fc01b33d78164d9ee37d2e50dc1d4b0511" and r["r1"]["m5_common_sha256"] == "ea2c5ea6984b8dc00c1c066a4441a59526c515ceb111da17eaf1688a009226c5" and r["r1"]["m5_pins_ok"] is True,
        "m5json": r["r1"]["m5_json_sha256"] == sha(d + "/m5.json"),
        "probe": r["r2"]["sha256"] == sha(run + "/01_stance/policy_probe.npz"),
        "trace": r["r3"]["status"] == "read" and r["r3"]["sha256"] == sha(f"{ECS}/traces/{n}/m1_summary.json"),
        "c6": r["r1"]["c6_ok"] is True and r["r1"]["c6_counting_failures"] == [] and r["r1"]["c6_tail_missing"] == [],
        "lost": r["lost_written"] is False,
        "kind": r["kind"] == "continuation" and r["name"] == n and r["stop_reason"] == "end",
        "argv": r["r1"]["m5_argv"][3:] == [f"{ECS}/../lean/build/laneB/tools/ledger_r10round.py".replace("/ecs/../", "/"), run + "/01_stance/eval_episodes.csv", "--E", str(r["T"]), "--json", d + "/m5.json"] or r["r1"]["m5_argv"],
    }
    if not all(checks.values()): bad.append((n, [k for k, v in checks.items() if not v]))
    if n == "s42_A_1101": print("m5 argv:", r["r1"]["m5_argv"])
print(len(glob.glob(ECS + "/score/read/*")), "read folders; problems:", bad or "none")
