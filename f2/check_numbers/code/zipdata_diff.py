"""Compare the top-level JSON fields of the SB3 'data' member of two model zips (JSON text only; embedded
serialized blobs are compared as opaque strings and never decoded)."""
import json, sys, zipfile
def data(p):
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("data").decode())
a, b = data(sys.argv[1]), data(sys.argv[2])
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k), b.get(k)
    if va != vb:
        sa, sb = json.dumps(va)[:160], json.dumps(vb)[:160]
        print("DIFF", k, "|", sa, "|", sb)
print("same keys:", sum(1 for k in set(a) & set(b) if a[k] == b[k]), "of", len(set(a) | set(b)))
for k in ("num_timesteps", "_n_updates", "seed"):
    print(k, a.get(k), b.get(k))
