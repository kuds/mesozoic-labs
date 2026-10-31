import sys, json
p = sys.argv[1]
lines = open(p).read().splitlines()
print("n lines", len(lines))
for i in (0, 1, len(lines) - 1):
    d = json.loads(lines[i])
    print(f"-- line {i}")
    for k, v in d.items():
        s = json.dumps(v)
        print(f"  {k}: {s[:150]}")
