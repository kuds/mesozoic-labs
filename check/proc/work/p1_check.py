import json, hashlib, sys
RL, APP = sys.argv[1], sys.argv[2]
lines = open(RL, encoding="utf-8").read().split("\n")
raw = open(APP, "rb").read()
fileobj = json.loads(raw)
hits = [i for i, l in enumerate(lines, 1) if "184f9ab91f284149f15ac192b4a5f318fa01485fbaf0068df64bb03ba5029ad2" in l]
print("RUNLOG lines naming APPROVED's sha256:", hits)
launch = [i for i, l in enumerate(lines, 1) if "BS2 LAUNCHED" in l]
print("RUNLOG lines with 'BS2 LAUNCHED':", launch)
l788 = lines[hits[0] - 1]
start = l788.index('{"criteria_sha256"')
dec = json.JSONDecoder()
obj, end = dec.raw_decode(l788[start:])
txt = l788[start:start + end]
print("RUNLOG JSON parses; equal to APPROVED parsed:", obj == fileobj, "; key order equal:", list(obj) == list(fileobj))
print("RUNLOG text byte-equal to the file:", txt.encode() == raw, "; sha256 of the RUNLOG text:", hashlib.sha256(txt.encode()).hexdigest()[:16])
for ind in (None, 0, 1, 2):
    for tail in (b"", b"\n"):
        cand = json.dumps(obj, indent=ind).encode() + tail
        if hashlib.sha256(cand).hexdigest() == hashlib.sha256(raw).hexdigest():
            print(f"json.dumps(RUNLOG object, indent={ind}) + {tail!r} reproduces APPROVED byte for byte (sha256 184f9ab9...)")
print("criteria sha in the same entry:", "291ff4573fa1b6030c739ddc1480d2260162c1b2e274c9af9933d6400183066f" in l788)
print("entry header:", l788[:40])
print("file has trailing newline:", raw.endswith(b"\n"), "size", len(raw))
