"""Runner order and concurrency from run_waves_ecs.log, against the fixed queue (BL 8) and run_meta's own times."""
import re, json, datetime as dt
ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()
Q = [f"{s}_{a}_{x}" for j in range(6) for s, a, base in (("s42", "A", 1101), ("s42", "B", 1201), ("s80", "A", 1301), ("s80", "B", 1401)) for x in [base + 10 * j]]
log = open(ECS + "/logs/run_waves_ecs.log").read().splitlines()
starts, exits, kinds = [], {}, {}
running, maxrun = set(), 0
for l in log:
    m = re.match(r"\[run_waves_ecs (\S+)\] (.*)$", l); t, msg = ts(m.group(1)), m.group(2)
    kinds.setdefault(msg.split(" ")[0].rstrip(":"), 0)
    if msg.startswith("start "):
        n = msg.split()[1]; starts.append((n, t)); running.add(n); maxrun = max(maxrun, len(running))
        assert msg.endswith(f"({n}.attempt1.log)"), msg
    m2 = re.match(r"(\S+) exited (\d+)$", msg)
    if m2: exits[m2.group(1)] = (t, int(m2.group(2))); running.discard(m2.group(1))
print("start order == queue:", [n for n, _ in starts] == Q, "; starts", len(starts), "; max concurrent", maxrun)
print("exits all 0:", all(v[1] == 0 for v in exits.values()), len(exits))
other = [l for l in log if not re.search(r"\] (launch:|queue:|start |\S+ exited |\S+: COUNTED|all 24 decided|\S+: end trace|\S+: read_ecs|score_ecs.py exit)", l)]
print("log lines of any other kind (relaunch, crash, refusal, halt, pause, orphan wait, VOID):", other or "none")
print("launch lines:", sum("launch:" in l for l in log), "; queue lines:", sum("] queue:" in l for l in log))
# run_meta times inside the runner's start/exit window
odd = []
for n, t0 in starts:
    m = json.load(open(f"{ECS}/runs/{n}/run_meta.json"))
    a, b = ts(m["utc_start"]), ts(m["utc_end"])
    if not (t0 <= a and b <= exits[n][0]): odd.append((n, m["utc_start"], m["utc_end"]))
print("run_meta utc_start/utc_end outside the runner's start/exit:", odd or "none")
# arm balance at each point in time: decided count per arm never differs by more than 3
dec = [l.split("\t") for l in open(ECS + "/logs/decided.tsv").read().splitlines()]
ca = cb = 0; mx = 0
for t, n, c, _ in dec:
    if "_A_" in n: ca += 1
    else: cb += 1
    mx = max(mx, abs(ca - cb))
print("max |A decided - B decided| over time:", mx, "; final", ca, cb)
