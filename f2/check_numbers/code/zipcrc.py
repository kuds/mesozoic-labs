"""Compare zip member CRC32/size of two zip files (central directory only; nothing is extracted or loaded)."""
import sys, zipfile
a, b = sys.argv[1], sys.argv[2]
def members(p):
    with zipfile.ZipFile(p) as z:
        return {i.filename: (i.CRC, i.file_size) for i in z.infolist()}
ma, mb = members(a), members(b)
for k in sorted(set(ma) | set(mb)):
    va, vb = ma.get(k), mb.get(k)
    print(("SAME " if va == vb else "DIFF ") + k, va, vb)
