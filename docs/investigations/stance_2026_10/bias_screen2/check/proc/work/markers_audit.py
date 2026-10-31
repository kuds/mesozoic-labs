"""Markers and cache provenance: reader sha, the loop rows (steps, E row, cache entries and their writers), the ledger
sha now vs at the read, M5's pins; birth times vs ALL_DECIDED."""
import hashlib, json, os, sys
BS2 = sys.argv[1]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
T = os.path.join(BS2, "tools")
reader = sha(os.path.join(T, "read_bs.py"))
deps = hashlib.sha256((sha(os.path.join(T, "bs_common.py")) + sha(os.path.join(T, "bs_hooks.py"))).encode()).hexdigest()
print("reader", reader[:16], "deps", deps[:8])
S, E = 2097152, 3538944
alld_birth = os.stat(os.path.join(BS2, "logs", "ALL_DECIDED"))
QUEUE = ["control_2501", "critic_blind_tail3_2501", "control_2511", "critic_blind_tail3_2511",
         "control_2521", "critic_blind_tail3_2521", "control_2531", "critic_blind_tail3_2531"]
M5_PINS = {"ledger_r10round.py": "0d89bcb858e19f780c3ce0ac7a5886fc01b33d78164d9ee37d2e50dc1d4b0511",
           "r10round_common.py": "ea2c5ea6984b8dc00c1c066a4441a59526c515ceb111da17eaf1688a009226c5"}
M5 = os.environ["S2"] + "/r10round/lean/build/laneB/tools/"
for f, h in M5_PINS.items():
    print("M5 pin", f, sha(M5 + f) == h)
cache_used = set()
problems = []
for name in QUEUE:
    mk = json.load(open(os.path.join(BS2, "reads", name, "markers.json")))
    P = lambda c, m: c or problems.append(f"{name}: {m}")
    P(mk["tool_sha256"] == reader, "markers by another reader")
    P(mk["label"] == name and mk["T"] == E and mk["S"] == S and mk["stop_reason"] == "end", "identity")
    P(mk["ledger"]["sha256"] == sha(mk["ledger"]["file"]), "ledger changed since the read")
    P(mk["probe"]["sha256"] == sha(mk["probe"]["file"]), "probe changed since the read")
    m5argv = mk["ledger"]["m5_argv"]
    P(any(a.endswith("ledger_r10round.py") for a in m5argv), "m5 argv")
    steps = [r["steps"] for r in mk["loop"]]
    P(steps == [S, 2359296, 2621440, 2883584, 3145728, 3407872, E], f"loop steps {steps}")
    rowE = [r for r in mk["loop"] if r["steps"] == E]
    P(len(rowE) == 1, "no loop row at E")
    srcs = []
    for r in mk["loop"]:
        P(r.get("reader_sha256") == reader and r.get("deps_sha256") == deps, f"loop row {r['steps']} by another reader")
        c = r.get("cache")
        if c:
            cache_used.add(c)
            st = os.stat(c)
            srcs.append((r["steps"], os.path.basename(c)[:60], "cache-born-before-ALL_DECIDED" if st.st_mtime < alld_birth.st_mtime else "cache-born-after"))
        z = r.get("zip")
        srcs.append((r["steps"], "zip", z.split("bias_screen2/")[-1] if z else None))
    # E row: num_timesteps recorded in the zip
    e = rowE[0] if rowE else {}
    print(name, "| E row keys:", sorted(k for k in e.keys() if k in ("num_timesteps", "zip_num_timesteps", "zip", "pkl", "zip_sha256", "pkl_sha256")))
    for s_ in srcs:
        print("    ", s_)
    st = os.stat(os.path.join(BS2, "reads", name, "markers.json"))
    P(st.st_mtime > alld_birth.st_mtime, "markers older than ALL_DECIDED")
# every cache entry: writer
bad = []
for f in sorted(os.listdir(os.path.join(BS2, "reads", "_cache"))):
    row = json.load(open(os.path.join(BS2, "reads", "_cache", f)))
    if row.get("reader_sha256") != reader or row.get("deps_sha256") != deps:
        bad.append(f)
print("cache entries", len(os.listdir(os.path.join(BS2, "reads", "_cache"))), "used by the 8 markers", len(cache_used),
      "entries by another reader", bad)
print("PROBLEMS:", problems or "none")
