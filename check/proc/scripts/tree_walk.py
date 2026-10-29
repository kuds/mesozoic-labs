"""My own file walk: sorted 'relpath<TAB>sha256' lines (symlink -> 'symlink:<target>'), seal excluded; sha256 of the
joined lines + trailing newline, and file count.  Written from the seal's documented format, independent code."""
import hashlib, os, sys, json
def walk(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel == ".r10m_seal.json":
                continue
            if os.path.islink(p):
                out[rel] = "symlink:" + os.readlink(p)
            else:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for c in iter(lambda: f.read(1 << 20), b""):
                        h.update(c)
                out[rel] = h.hexdigest()
    return out
res = {}
for root in sys.argv[1:]:
    m = walk(root)
    lines = sorted(f"{k}\t{v}" for k, v in m.items())
    dig = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    res[root] = (dig, len(lines), m)
    print(root, dig, len(lines))
    pyc = [k for k in m if "__pycache__" in k or k.endswith(".pyc")]
    print("  pycache/pyc files:", len(pyc))
if len(sys.argv) == 3:
    a, b = res[sys.argv[1]][2], res[sys.argv[2]][2]
    print("only in first:", sorted(set(a) - set(b))[:20])
    print("only in second:", sorted(set(b) - set(a))[:20])
    print("differ:", sorted(k for k in set(a) & set(b) if a[k] != b[k])[:20])
