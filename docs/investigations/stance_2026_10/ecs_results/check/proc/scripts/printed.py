"""(d) Every per-continuation report-only figure in score.txt against its source: m5.json (M5's own lists), the run's
ledger and probe (my own reduction), read.json, and the trace's m1_summary.json.  Also POSITIVE/NEGATIVE from m5.json
and the end trace's checkpoint (the E pair)."""
import csv, hashlib, json, math, re
import numpy as np
ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
W = {"s42": (3145728, 4718592), "s80": (4194304, 5767168)}
lines = {}
for l in open(ECS + "/score/score.txt").read().splitlines():
    m = re.match(r"^  (s(?:42|80)_[AB]_\d+)\s+(COUNTED|STOPPED|LOST)\s+(\S+)\s+T ([\d,]+)\s+(.*)$", l)
    if m: lines[m.group(1)] = m.groups()
assert len(lines) == 24, len(lines)
probs = []; rows = []
for name, (nm, cls, outc, Tstr, rest) in lines.items():
    s = name[:3]; S, E = W[s]; T = int(Tstr.replace(",", ""))
    P = []
    def want(c, msg):
        if not c: P.append(msg)
    want(T == E, f"T {T}")
    rd = json.load(open(f"{ECS}/score/read/{name}/read.json"))
    m5p = f"{ECS}/score/read/{name}/m5.json"; m5 = json.load(open(m5p))
    want(rd["r1"]["m5_json_sha256"] == sha(m5p), "read.json m5_json sha256 != m5.json")
    led = f"{ECS}/runs/{name}/01_stance/eval_episodes.csv"
    want(m5["sha256"] == sha(led) == rd["r1"]["ledger_sha256"], "ledger sha256")
    want(m5["c6"]["passed"] is True and m5["c6"]["failures"] == [] and m5["c6"]["E"] == T, f"C6 {m5['c6']}")
    # my reduction of M5's per-evaluation counts: ANY-flagged = any >= 5 of the evaluation's 10 rows (CRIT6 M5 (a))
    per = {int(k): v for k, v in m5["per_evaluation"].items()}
    win = sorted(k for k in per if S < k <= T)
    want(win == [S + 131072 * j for j in range(1, 13)], f"window evaluations {win}")
    want(all(per[k]["n"] == 10 for k in win), "n != 10")
    my_any = [k for k in win if per[k]["any"] >= 5]
    my_tail = [k for k in win if per[k]["tail"] >= 5]
    want(my_any == [k for k in m5["any_flagged"] if S < k <= T], f"any list: mine {my_any} vs M5 {m5['any_flagged']}")
    want(my_tail == [k for k in m5["tail_flagged"] if S < k <= T], "tail list vs M5")
    # alarms: M5's own alarm steps whose two evaluations lie in (S, T]
    my_aal = [k for k in m5["any_alarms"] if S < k - 131072 and k <= T]
    my_tal = [k for k in m5["tail_alarms"] if S < k - 131072 and k <= T]
    # my own alarm: two consecutive ANY-flagged evaluations (for cross-check only)
    my_aal2 = [k for k in my_any if (k - 131072) in my_any]
    my_tal2 = [k for k in my_tail if (k - 131072) in my_tail]
    pos = bool(my_any)
    want(outc == ("POSITIVE" if pos else "NEGATIVE"), f"outcome printed {outc}, mine {pos}")
    want(cls == "COUNTED", cls)
    first = f"{my_any[0]:,}" if my_any else f"none (censored at T = {T:,})"
    tail_missing = [f for f in m5["c6"]["failures"] if "TAIL input missing" in f]
    # returns: my own mean of the ledger's return column at T
    with open(led, newline="") as fh:
        rr = [float(r["return"]) for r in csv.DictReader(fh) if r["timesteps"] and float(r["timesteps"]) == T]
    want(len(rr) == 10, f"{len(rr)} ledger rows at T")
    ret = sum(rr) / len(rr)
    # std: probe std_per_actuator, first and last row in (S, T), mean over the 15 actuators
    with np.load(f"{ECS}/runs/{name}/01_stance/policy_probe.npz", allow_pickle=False) as z:
        u = np.asarray(z["update_timesteps"]).astype(np.int64); sd = np.asarray(z["std_per_actuator"], dtype=np.float64)
    idx = [i for i in range(len(u)) if S < u[i] < T]
    want(sd.shape[1] == 15 and len(idx) == 191, f"std shape {sd.shape}, rows {len(idx)}")
    std0, std1 = float(np.mean(sd[idx[0]])), float(np.mean(sd[idx[-1]]))
    # trace
    tp = f"{ECS}/traces/{name}/m1_summary.json"; tr = json.load(open(tp))
    ezip = f"{ECS}/runs/{name}/01_stance/models/stage1_{E}_steps.zip"
    want(tr["checkpoint"]["path"] == ezip and tr["checkpoint"]["sha256"] == sha(ezip), f"trace checkpoint {tr['checkpoint'].get('path')}")
    want(tr["checkpoint"].get("num_timesteps") in (E, None), f"trace num_timesteps {tr['checkpoint'].get('num_timesteps')}")
    want(tr["seeds"] == list(range(3042, 3062)) and tr["n_episodes"] == 20, "trace seeds")
    want(tr["tool_sha256"] == "b293a511d8beb5b6d6f24cec8744e601382748ce9b2f89e6bb653b6c97f5840c", "trace tool")
    want(tr["copy_tree_digest_after"] == "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1", "trace copy")
    want(tr["physics_revision"] == 10, "physics")
    exp = (f"first ANY flag {first}; flagged {len(my_any)}; ANY ALARM {my_aal}; TAIL-flagged {my_tail}; TAIL ALARM {my_tal};"
           f" TAIL-input missing (report only) {len(tail_missing)}; mean std first/last row {std0:.4f}/{std1:.4f}; "
           f"mean return at T {ret:.1f}; R3 CMD tail_3 {tr['cmd_a']['14']}, tail flags {len(tr['tail_flags'])}, falls "
           f"{len(tr['fall_flags'])} ({tp} sha256 {sha(tp)})")
    want(rest == exp, f"line differs:\n   printed  {rest}\n   expected {exp}")
    want(my_aal == my_aal2 and my_tal == my_tal2, f"alarm cross-check: M5 {my_aal}/{my_tal} vs consecutive {my_aal2}/{my_tal2}")
    # straddling alarms and flags at or before S (must be none: the parent's evaluation at S is flag-free)
    want(all(k > S for k in m5["any_flagged"] if k > S - 1) and not [k for k in m5["any_flagged"] if k <= S], f"flags <= S {[k for k in m5['any_flagged'] if k <= S]}")
    want(rd["r1"]["any_alarms_straddling_S"] == [] and rd["r1"]["tail_alarms_straddling_S"] == [], "straddling")
    rows.append((name, outc, len(my_any), my_any[:1], len(tr['tail_flags']), round(tr['cmd_a']['14'], 4)))
    if P: probs.append((name, P))
for r in rows: print(*r)
print("PROBLEMS:", probs if probs else "none")
pa = sum(1 for r in rows if r[0].startswith("s42_A") and r[1] == "POSITIVE"); pb = sum(1 for r in rows if r[0].startswith("s42_B") and r[1] == "POSITIVE")
qa = sum(1 for r in rows if r[0].startswith("s80_A") and r[1] == "POSITIVE"); qb = sum(1 for r in rows if r[0].startswith("s80_B") and r[1] == "POSITIVE")
print(f"s42 a={pa} b={pb}; s80 a={qa} b={qb}")
