import hashlib, re, sys
RL = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance4/live/run_log.txt"
PS = "/home/user/work/phase_screen"
raw = open(RL, "rb").read()
lines = raw.split(b"\n")
ap = open(f"{PS}/APPROVED", "rb").read()
aps = hashlib.sha256(ap).hexdigest()
print("APPROVED bytes len", len(ap), "ends_with_newline", ap.endswith(b"\n"))
apl = ap.rstrip(b"\n")
pin_idx = [i for i, l in enumerate(lines) if aps.encode() in l]
exact_idx = [i for i, l in enumerate(lines) if l == apl]
launch_idx = [i for i, l in enumerate(lines) if b"LAUNCHING run_waves_bs.sh" in l or b"PS LAUNCHED at 2026-10-31T15:56:58Z" in l]
print("lines with APPROVED sha256 (1-based):", [i + 1 for i in pin_idx])
print("lines exactly equal to APPROVED bytes (1-based):", [i + 1 for i in exact_idx])
print("launch lines (1-based):", [i + 1 for i in launch_idx])
for i in pin_idx:
    print("pin line says 'exact bytes follow on the next line':", b"exact bytes follow on the next line" in lines[i], "; next line equals APPROVED:", lines[i + 1] == apl)
print("pin before launch:", max(pin_idx + exact_idx) < min(launch_idx))
# timestamps monotone around
ts = []
for i, l in enumerate(lines):
    m = re.match(rb"^(2026-\d\d-\d\d \d\d:\d\d:\d\d) UTC", l)
    if m: ts.append((i + 1, m.group(1).decode()))
bad = [(a, b) for a, b in zip(ts, ts[1:]) if b[1] < a[1]]
print("timestamp-ordered entries:", len(ts), "out-of-order pairs:", bad[-5:], "count", len(bad))
# VG tool sha lines
vg = {}
for i, l in enumerate(lines):
    m = re.match(rb"^  ([0-9a-f]{64})  (SPEC\.txt|vg_sigma\.py|tests/\S+)$", l)
    if m: vg.setdefault(m.group(2).decode(), []).append((i + 1, m.group(1).decode()))
alld = [i + 1 for i, l in enumerate(lines) if b"ALL_DECIDED at" in l]
print("ALL_DECIDED entry lines:", alld)
print("VG lines:", vg)
import os
for name, occ in vg.items():
    p = f"/home/user/work/vg_tool/{name}"
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "MISSING"
    last = occ[-1]
    print(name, "logged_line", last[0], "logged", last[1], "now", got, "EQUAL" if got == last[1] else "DIFF", "before_ALL_DECIDED_entry", last[0] < min(alld))
