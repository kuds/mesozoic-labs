"""List leaf differences in stage_config.json, progress.json and the zip's data policy/buffer classes. usage: A B"""
import json, sys, zipfile, re
A, B = sys.argv[1] + "/run", sys.argv[2] + "/run"
out = []
def walk(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k): walk(x.get(k), y.get(k), f"{path}.{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q: walk(p, q, f"{path}[{i}]")
    else:
        out.append((path, str(x)[:120], str(y)[:120]))
for f in ["stage_config.json", "progress.json"]:
    walk(json.load(open(f"{A}/{f}")), json.load(open(f"{B}/{f}")), f)
za = json.loads(zipfile.ZipFile(f"{A}/models/stage1_final.zip").read("data")); zb = json.loads(zipfile.ZipFile(f"{B}/models/stage1_final.zip").read("data"))
for k in ["policy_class", "rollout_buffer_class"]:
    walk(za[k], zb[k], k)
addr = re.compile(r" at 0x[0-9a-f]+>")
for p, x, y in out:
    kind = "ADDRESS-ONLY" if addr.sub(">", x) == addr.sub(">", y) else "VALUE"
    print(kind, p, "|", x, "|", y)
