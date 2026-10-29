"""Every file under each run folder (no content read): which have atime > mtime (read after the last write), and
when; grouped by whether the read came before the 'all 24 decided' line (12:05:20Z)."""
import os, glob, subprocess, datetime as dt, collections
ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
T_DEC = dt.datetime(2026, 10, 29, 12, 5, 20, tzinfo=dt.timezone.utc).timestamp()
def iso(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
before = collections.Counter(); after = collections.Counter(); never = collections.Counter(); ex = {}
for run in sorted(glob.glob(ECS + "/runs/*")):
    for dp, dn, fn in os.walk(run):
        for f in fn:
            p = os.path.join(dp, f); s = os.stat(p)
            rel = os.path.relpath(p, run)
            key = rel if "models/" not in rel else "models/" + ("E-pair" if any(x in f for x in ("4718592", "5767168")) else ("start-pair" if any(x in f for x in ("3145728_", "4194304_")) and "s42" in run and "3145728" in f or ("s80" in run and "4194304" in f) else ("final" if "final" in f else "other-ckpt")))
            if s.st_atime > s.st_mtime + 0.0005:
                if s.st_atime < T_DEC: before[key] += 1; ex.setdefault(("b", key), (os.path.basename(run), iso(s.st_mtime), iso(s.st_atime)))
                else: after[key] += 1; ex.setdefault(("a", key), (os.path.basename(run), iso(s.st_mtime), iso(s.st_atime)))
            else:
                never[key] += 1
print("read after last write, BEFORE the decided line:")
for k, v in sorted(before.items()): print(f"  {k:40s} {v:3d}  e.g. {ex[('b', k)]}")
print("read after last write, AFTER the decided line:")
for k, v in sorted(after.items()): print(f"  {k:40s} {v:3d}  e.g. {ex[('a', k)]}")
print("no read after last write (atime <= mtime):")
for k, v in sorted(never.items()): print(f"  {k:40s} {v:3d}")
