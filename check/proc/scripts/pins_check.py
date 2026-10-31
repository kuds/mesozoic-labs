import hashlib, json, os, stat, sys
PS = "/home/user/work/phase_screen"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
ap = json.load(open(f"{PS}/APPROVED"))
ok = True
rows = []
def chk(name, path, want):
    global ok
    got = sha(path)
    m = oct(stat.S_IMODE(os.stat(path).st_mode))
    r = "OK" if got == want else "MISMATCH"
    if r != "OK": ok = False
    rows.append((name, path, want, got, m, r))
chk("criteria", f"{PS}/criteria.txt", ap["criteria_sha256"])
chk("screen", f"{PS}/SCREEN.json", ap["screen_sha256"])
for t, w in ap["tool_sha256"].items():
    chk(t, f"{PS}/tools/{t}", w)
for r in rows: print("\t".join(r))
print("APPROVED_sha256\t" + sha(f"{PS}/APPROVED"))
for p in ("criteria.txt", "SCREEN.json", "APPROVED"):
    print("mode", p, oct(stat.S_IMODE(os.stat(f"{PS}/{p}").st_mode)))
print("ALL_PINS_OK" if ok else "PIN_FAILURE")
print("code_tree_digest_in_APPROVED", ap["code_tree_digest"])
