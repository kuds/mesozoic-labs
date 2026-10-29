"""(c) Every sha256 in criteria.txt's PINS section, parsed from the text itself, recomputed from its file."""
import hashlib, re, json
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ECS = S2 + "/r10round/ecs"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
txt = open(ECS + "/criteria.txt").read().splitlines()
i0 = next(i for i, l in enumerate(txt) if l.startswith("PINS"))
i1 = next(i for i, l in enumerate(txt) if l.startswith("  V1-V5"))
n = 0; bad = []
for l in txt[i0:i1]:
    m = re.match(r"^\s{4}(\S+(?: \S+)?)\s+([0-9a-f]{64})$", l)
    if not m: continue
    key, want = m.groups()
    if key.startswith(("s42 ", "s80 ")):
        st, f = key.split(" ")
        rid = {"s42": "stance6_r10LTSB_s42", "s80": "stance6_r10LTSB_s80"}[st]
        path = f"{S1}/foot/stance6/fetch/F2/{rid}/copies/01_stance/" + (("models/" + f) if f.endswith((".zip", ".pkl")) else f) + ".copy1"
    elif key.startswith("lane"):
        path = f"{S2}/r10round/lean/build/{key}"
    else:
        path = f"{ECS}/{key}"
    got = sha(path); n += 1
    print(("OK  " if got == want else "BAD ") + key, want[:16])
    if got != want: bad.append((key, got, want))
# the v3 text the pins filled, and the code digest line
v3 = sha(ECS + "/v3/criteria_ecs_v3.txt")
print("v3/criteria_ecs_v3.txt", v3, v3 == "90a9f9aa6c3d87947817aceb93e23b1bb107214c22e0cdd294e7790056ad6d58")
crit = sha(ECS + "/criteria.txt"); appr = json.load(open(ECS + "/APPROVED"))
print("criteria.txt", crit, "== APPROVED criteria_sha256:", crit == appr["criteria_sha256"])
print("final/criteria_ecs_final.txt equal bytes:", open(ECS + "/criteria.txt", "rb").read() == open(ECS + "/final/criteria_ecs_final.txt", "rb").read())
for k, v in appr["tool_sha256"].items():
    got = sha(f"{ECS}/tools/{k}"); print("APPROVED tool", k, got == v)
print("APPROVED sha256", sha(ECS + "/APPROVED"))
print("CRIT6", sha(S1 + "/foot/stance6/criteria.txt"))
print(f"{n} pins parsed; bad: {bad or 'none'}")
