"""Decode one Drive download JSON ({content, id, mimeType, title}) to DST, verify, report.

usage: python3 -I decode_verify.py SRC_JSON DST EXPECT_ID EXPECT_TITLE EXPECT_SIZE
Checks id, title, decoded size == Drive fileSize; runs zipfile.testzip (zip/npz); prints JSON line.
Never imports or executes the payload.
"""
import base64, hashlib, json, os, sys, zipfile

src, dst, want_id, want_title, want_size = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
with open(src) as f:
    d = json.load(f)
assert set(d) >= {"content", "id", "title"}, sorted(d)
assert d["id"] == want_id, (d["id"], want_id)
assert d["title"] == want_title, (d["title"], want_title)
raw = base64.b64decode(d["content"], validate=True)
assert len(raw) == want_size, f"decoded {len(raw)} bytes, Drive fileSize {want_size}"
tmp = dst + ".part"
with open(tmp, "wb") as f:
    f.write(raw)
os.replace(tmp, dst)
with open(dst, "rb") as f:
    disk = f.read()
assert disk == raw
res = {"path": dst, "bytes": len(disk), "sha256": hashlib.sha256(disk).hexdigest(), "mime": d.get("mimeType")}
if dst.endswith((".zip", ".npz")):
    with zipfile.ZipFile(dst) as z:
        bad = z.testzip()
        res["testzip"] = "OK" if bad is None else f"BAD:{bad}"
        res["members"] = [(i.filename, i.file_size) for i in z.infolist()]
    assert bad is None, bad
print(json.dumps(res))
