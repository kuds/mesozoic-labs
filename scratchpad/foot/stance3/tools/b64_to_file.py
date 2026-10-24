"""Decode one hand-transcribed base64 text file (an inline Drive download's `content`) to DST and verify it.

usage: python3 -I b64_to_file.py SRC_B64 DST EXPECT_SIZE
Whitespace in SRC_B64 is ignored; strict base64.  Checks decoded size == Drive fileSize; zip/npz: zipfile.testzip;
.json: json parse.  Never imports, unpickles or executes the payload.  Prints one JSON line (path, bytes, sha256).
"""
import base64, hashlib, json, os, sys, zipfile

src, dst, want_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    text = "".join(f.read().split())
raw = base64.b64decode(text, validate=True)
res = {"path": dst, "bytes": len(raw), "expect": want_size, "sha256": hashlib.sha256(raw).hexdigest()}
if len(raw) != want_size:
    res["error"] = f"decoded {len(raw)} bytes, Drive fileSize {want_size}"
    print(json.dumps(res)); sys.exit(1)
os.makedirs(os.path.dirname(dst), exist_ok=True)
tmp = dst + ".part"
with open(tmp, "wb") as f:
    f.write(raw)
os.replace(tmp, dst)
if dst.endswith((".zip", ".npz")):
    with zipfile.ZipFile(dst) as z:
        bad = z.testzip()
    res["testzip"] = "OK" if bad is None else f"BAD:{bad}"
if dst.endswith(".json"):
    json.loads(raw.decode("utf-8"))
    res["json"] = "parsed"
print(json.dumps(res))
