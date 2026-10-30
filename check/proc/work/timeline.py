"""Per continuation: run start/end, CHECK, decision, re-arm (ctime of the re-armed files), ALL_DECIDED, snapshot,
full read; checks the order.  Uses stat (birth via statx through os.stat on 3.13? fall back to st_birthtime absent)."""
import json, os, subprocess, sys, datetime
BS2 = sys.argv[1]
def birth(p):
    out = subprocess.run(["stat", "-c", "%w", p], capture_output=True, text=True).stdout.strip()
    d, t, z = out.split()
    hms, frac = t.split(".")
    base = datetime.datetime.strptime(d + " " + hms, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    return base + int(frac) / 10 ** len(frac)
def ts(x):
    return datetime.datetime.fromtimestamp(x, datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
dec = {}
for line in open(os.path.join(BS2, "logs", "decided.tsv")):
    t, n, d, why = line.rstrip("\n").split("\t")
    dec[n] = (datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp(), d)
snap = {}
with open(os.path.join(BS2, "logs", "atimes_at_ALL_DECIDED.tsv")) as fh:
    fh.readline()
    for line in fh:
        p, a, m, c, f = line.rstrip("\n").split("\t")
        snap[p] = (int(a), int(m), int(c), int(f))
alld = birth(os.path.join(BS2, "logs", "ALL_DECIDED"))
snapb = birth(os.path.join(BS2, "logs", "atimes_at_ALL_DECIDED.tsv"))
print(f"snapshot born {ts(snapb)}; ALL_DECIDED born {ts(alld)}")
ok = True
for name in ["control_2501", "critic_blind_tail3_2501", "control_2511", "critic_blind_tail3_2511",
             "control_2521", "critic_blind_tail3_2521", "control_2531", "critic_blind_tail3_2531"]:
    rd = os.path.join(BS2, "runs", name)
    meta = os.path.join(rd, "run_meta.json")
    m = json.load(open(meta))
    run_birth = birth(rd)
    # the run's last write among its files (max mtime from the snapshot)
    own = [v for p, v in snap.items() if p.startswith(f"runs/{name}/") and not p.endswith("check.json")]
    last_write = max(v[1] for v in own) / 1e9
    chk = os.path.join(rd, "check.json")
    chk_b, chk_m = birth(chk), os.stat(chk).st_mtime
    chklog_b = birth(os.path.join(BS2, "logs", f"{name}.check.log"))
    rearm_c = [v[2] / 1e9 for p, v in snap.items() if (p.startswith(f"runs/{name}/") or p.startswith(f"tmp/{name}/") or p.startswith(f"logs/{name}.attempt"))]
    rearm_lo, rearm_hi = min(rearm_c), max(rearm_c)
    mk_b = birth(os.path.join(BS2, "reads", name, "markers.json"))
    m5_b = birth(os.path.join(BS2, "reads", name, "m5.json"))
    rdlog_b = birth(os.path.join(BS2, "logs", f"{name}.read.log"))
    d_t, d = dec[name]
    order = [("run dir born", run_birth), ("last run write", last_write), ("check.log born", chklog_b),
             ("check.json born", chk_b), ("check.json mtime", chk_m), ("decided (s)", d_t), ("re-arm ctimes lo", rearm_lo),
             ("re-arm ctimes hi", rearm_hi), ("snapshot born", snapb), ("ALL_DECIDED born", alld),
             ("read.log born", rdlog_b), ("m5.json born", m5_b), ("markers.json born", mk_b)]
    mono = all(order[i][1] <= order[i + 1][1] + 1.0 for i in range(len(order) - 1))  # decided.tsv is whole seconds
    strict = (last_write < chklog_b < chk_b <= chk_m < rearm_lo <= rearm_hi < snapb < alld < rdlog_b <= m5_b < mk_b)
    ok &= strict
    print(f"{name}: start {m['utc_start'][11:]} " + " | ".join(f"{k} {ts(v)}" for k, v in order[1:]) + f" | {d} | order {'OK' if strict else 'VIOLATED'}")
print("ALL ORDERS OK" if ok else "SOME ORDER VIOLATED")
