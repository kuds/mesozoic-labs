"""VP1 M1 comparison (report only): per-step CSV sha256 (csv_sha256.txt) and the M1 reductions of
(a) the HEAD copy's M1 on STANCE6 s80's final F2 pair vs the round's panel (S1/foot/stance6/panels/.../final/m1), and
(b) the HEAD copy's M1 on the zero-phase-column widened pair vs (a).  Reads only; prints."""
import json
import re
import sys
from pathlib import Path

S1 = Path(re.search(r'^S1 = Path\("(.*)"\)$', Path("/home/user/work/pc_study/tools/pc/pc_common.py").read_text(), re.M).group(1))
ROUND = S1 / "foot/stance6/panels/stance6_r10LTSB_s80/final/m1"
F = Path(sys.argv[1])
KEYS = ("cmd_a", "cmd_all_actuators", "tail_flags", "fall_flags", "tail_flags_full_horizon_by_actuator_sign",
        "any_actuator_over_bar_episodes", "episodes", "physics_revision", "hip_roll", "cmd_a_episodes",
        "episodes_without_window", "drive_backed")


def csvs(d):
    rows = {}
    for ln in (d / "csv_sha256.txt").read_text().split("\n"):
        if ln.strip():
            h, name = ln.split()[0], ln.split()[-1]
            rows[Path(name).name] = h
    return rows


def cmp(a, b, label):
    ok = True
    ca, cb = csvs(a), csvs(b)
    same = ca == cb
    print(f"{label}: per-step CSV sha256 ({len(ca)} vs {len(cb)} files) equal: {same}")
    ok &= same and len(ca) == 20
    sa = json.loads((a / "m1_summary.json").read_text())
    sb = json.loads((b / "m1_summary.json").read_text())
    for k in KEYS:
        if k in sa or k in sb:
            e = sa.get(k) == sb.get(k)
            ok &= e
            print(f"  {k}: equal {e}")
    ta = json.loads((a / "episode_trace.json").read_text())
    tb = json.loads((b / "episode_trace.json").read_text())
    ea, eb = ta.get("episodes"), tb.get("episodes")
    e = ea == eb
    print(f"  episode_trace.json 'episodes' equal: {e}")
    ok &= e
    fa, fb = (a / "flags.csv").read_bytes(), (b / "flags.csv").read_bytes()
    print(f"  flags.csv bytes equal: {fa == fb}")
    ok &= fa == fb
    print(f"  copy commits {sa.get('copy_seal', {}).get('commit', '')[:12]} {sb.get('copy_seal', {}).get('commit', '')[:12]}")
    print(f"{label}: {'EQUAL' if ok else 'DIFFERENT'}")
    return ok


r1 = cmp(ROUND, F / "vp1/m1_stock", "VP1 (a) M1 round panel (8737cd3d copy) vs HEAD copy, stock s80 final")
r2 = cmp(F / "vp1/m1_stock", F / "vp1/m1_widened", "VP1 (c) M1 HEAD copy stock vs widened (zero phase columns)")
print(f"VP1 M1 parts: {'PASS' if r1 and r2 else 'FAIL'}")
