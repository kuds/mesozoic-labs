"""The lead's check of the candidate experiments' key figures, from the raw episode files (28 Nov 2026).

Reads the session scratch folder's raw records (candidates/E1, E2, E3, E5, E5_pitch0) and this folder's committed
result files, recomputes the figures the report (design/CANDIDATES_2026_11_28.md) and finding F31 quote, and prints
them. It uses none of the experiments' own summarisers. E3's winding temperatures go through the build's
rl/analysis/thermal_ambient.py (method A, 8 min time constant, 35 C room); E3's loop RMS is recomputed here from the
stored per-segment mean squared drive torques. E2(a) is recomputed from e2a_grid.csv, its row-level table (E2(a) is
DESK on the logged torques and speeds). Nothing is simulated. Output: lead_check_2026-11-28.txt beside this file.

Run (needs mujoco, which gait_report imports; from a scratch working directory):
    python -B lead_check.py > lead_check_2026-11-28.txt
"""

import csv
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
BUILD = RES.parents[2]
SCR = Path(os.environ.get("TRIKE_SCRATCH",
                          "/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad"))
CAND = SCR / "candidates"


def head(t):
    print(f"\n== {t}")


# ---------------------------------------------------------------- E1: episodes, poses and falls from the raw records
head("E1 (raw episode records)")
eps = [json.load(open(p)) for p in sorted(glob.glob(str(CAND / "E1" / "episodes" / "*.json")))]
print("episodes", len(eps), "poses", sum(e["n_poses"] for e in eps), "falls", sum(e["fell"] for e in eps),
      "unstable", sum(e["unstable"] for e in eps))
falls = defaultdict(int)
for e in eps:
    falls[(e["controller"], e["condition"])] += e["fell"]
print("falls by controller and condition", {f"{k[0]} {k[1]}": v for k, v in sorted(falls.items()) if v})

# ---------------------------------------------------------------- E2(a): judged rows from the row-level table
head("E2(a) (e2a_grid.csv)")
rows = [r for r in csv.DictReader(open(RES / "E2" / "e2a_grid.csv")) if r["judged"] == "True"]
ok = lambda r: r["C1_pass"] == "True" and r["C2_pass"] == "True" and r["speed_pass"] == "True"  # noqa: E731
g = defaultdict(list)
for r in rows:
    g[(r["placement"], r["r"], r["transmission"])].append(r)
print("judged rows (placement x r x transmission)", len(g), "PASS", sum(all(ok(r) for r in v) for v in g.values()))
for (pl, rr, tr), v in sorted(g.items()):
    if pl != "pitch" and rr not in ("1.0", "1.25", "1.3"):
        continue
    fails = sorted({(r["policy"], r["block"], r["speed_worst_peak_drive"], round(float(r["speed_peak_max_rpm"]), 2))
                    for r in v if r["speed_pass"] != "True"})
    print(f"{pl:5s} r={rr:4s} {tr:5s} C1 {max(float(r['C1_c']) for r in v):6.1f} C  C2 "
          f"{max(float(r['C2']) for r in v):.3f}  p99 {max(float(r['speed_p99_max_rpm']) for r in v):5.1f}  peak "
          f"{max(float(r['speed_peak_max_rpm']) for r in v):6.1f} rpm  speed fails {fails}")

# ---------------------------------------------------------------- E2(b): reduced drives' peaks from the raw records
head("E2(b) (raw episode records, judged runs at 0.25 ms; B1 at 1 ms)")
for run in ("B1_dt0.001", "rod_dt0.00025", "belt_dt0.00025", "cable_dt0.00025"):
    ee = [json.load(open(p)) for p in glob.glob(str(CAND / "E2" / "episodes" / run / "*" / "*.json"))]
    pk, over = [], 0
    for e in ee:
        cap = e.get("cap_actuators") or [a for a in e["actuators"] if "roll" not in a]
        p = max(e["rpm_peak"][e["actuators"].index(a)] for a in cap)
        pk.append(p)
        over += p > 91.3
    print(f"{run:16s} episodes {len(ee)} falls {sum(e['fell'] for e in ee)} unstable {sum(e['unstable'] for e in ee)} "
          f"peak of the reduced (B1: pitch and knee) drives {max(pk):.2f} rpm; episodes over 91.3 rpm {over}")

# ---------------------------------------------------------------- E3: C1 through the build's thermal model, C2 here
head("E3 (fore128 policies; C1 via rl/analysis/thermal_ambient.py, method A, 8 min, 35 C; C2 = loop RMS / rating)")
sys.path[:0] = [str(BUILD / "rl" / "analysis"), str(BUILD / "rl"), str(BUILD / "rl" / "analytics")]
import gait_report as gr  # noqa: E402
import thermal_ambient as TA  # noqa: E402

RATING = {"X4-36-L": 7.5, "X6-60": 12.0}
dt = gr.drive_table()
for run in ("fore128-s0", "fore128-s1", "fore128-s2", "fore128-s3"):
    for seed in (700000, 710000):
        p = (TA.D22 / "demo" / run / "demo_course.json") if seed == 700000 else \
            (CAND / "E3" / "demo" / run / f"demo_course_seed{seed}.json")
        st = json.loads(p.read_text())
        therm, per_joint, groups = TA.thermal_inputs(st)
        with TA._cached_fits():
            temps = TA.temperatures(therm, per_joint, 35.0, "A", tcs=(480.0,))
        k = TA.tc_key(480.0)
        jn = max(temps, key=lambda j: temps[j][k]["peak_c"])
        segs, loop = st["segments"], [tuple(x) for x in st["loop"]]
        worst = max((math.sqrt(sum(segs[n]["tau_drive_sq_mean"][j] * s for n, s in loop) / sum(s for _, s in loop))
                     / RATING[dt[groups[name]]["drive"]], name)
                    for j, name in enumerate(segs["stand"]["joints"]))
        print(f"{run} {seed}: C1 {temps[jn][k]['peak_c']:.2f} C ({jn}); C2 {worst[0]:.4f} ({worst[1]})")

# ---------------------------------------------------------------- E5: both branches and the frozen trot, raw records
head("E5 (raw episode records)")
JC = ("J1", "J2", "J3", "J4")


def e5_sets(folder, sid):
    ee = [json.load(open(p)) for p in glob.glob(str(folder / f"{sid}__*.json"))]
    out = {}
    for lo in sorted({(e["seed"] // 100) * 100 + 1 for e in ee}):
        s = [e for e in ee if lo <= e["seed"] <= lo + 9]
        j3 = [e for e in s if e["condition"] == "J3"]
        slow, knee = defaultdict(int), defaultdict(float)
        for e in s:
            slow[e["condition"]] += e["slow_gait"]
            knee[e["condition"]] = max(knee[e["condition"]], e["worst_hind_knee"])
        x4 = max(p for e in s for a, p in zip(e["actuators"], e["rpm_peak"]) if "fore_flexion" not in a)
        out[f"{lo}-{lo + 9}"] = dict(n=len(s), falls_J=sum(e["fell"] for e in s if e["condition"] in JC),
                                     J3=sum(e["record"]["forward_speed_mps"] for e in j3) / len(j3),
                                     slow_max=max(slow.values()), knee_max=max(knee.values()),
                                     knee_per_cond=(min(knee.values()), max(knee.values())), x4_peak=x4)
    return out


for size in ("0.60", "0.50"):
    for branch, base, sid in (("pitched", CAND / "E5", f"H{size}-s1-"), ("pitch 0", CAND / "E5_pitch0", f"H{size}-p0-s1-")):
        ch = json.load(open(RES / "E5" / (f"e5_choice_{size}.json" if branch == "pitched"
                                          else f"pitch0/e5p0_choice_{size}.json")))
        cid = ch["chosen_id"]
        for stage in ("heldout", "heldout_frozen"):
            folder = base / "episodes" / size / stage
            pat = cid if stage == "heldout" else "*"
            for setname, v in e5_sets(folder, pat).items():
                print(f"{size} {branch:7s} {stage:14s} {cid if stage == 'heldout' else 'frozen':18s} {setname}: n {v['n']} "
                      f"falls J1-J4 {v['falls_J']} J3 {v['J3']:.5f} m/s slow-gait max/cond {v['slow_max']} worst knee "
                      f"{v['knee_max']:.3f} (per condition {v['knee_per_cond'][0]:.3f}-{v['knee_per_cond'][1]:.3f}) "
                      f"X4 peak {v['x4_peak']:.2f} rpm")

head("E5 search budget (tuning CSVs and JSONs)")
ids = {}
for size in ("0.60", "0.50"):
    for branch, f in (("pitched", RES / "E5" / f"e5_tuning_{size}.csv"), ("pitch 0", RES / "E5" / "pitch0" / f"e5p0_tuning_{size}.csv")):
        rr = list(csv.DictReader(open(f)))
        key = next(k for k in rr[0] if k in ("id", "setting_id", "setting"))
        vals = {tuple((k, r[k]) for k in ("period_s", "duty", "body_pitch_deg", "lift_m", "max_reach_m") if k in r) for r in rr}
        ids[(size, branch)] = vals
        j = json.load(open(str(f).replace(".csv", ".json")))
        print(f"{size} {branch}: distinct settings by value {len(vals)}; budget record {j.get('budget')}")
    print(f"{size} union of distinct settings {len(ids[(size, 'pitched')] | ids[(size, 'pitch 0')])}, shared "
          f"{len(ids[(size, 'pitched')] & ids[(size, 'pitch 0')])}")

# ---------------------------------------------------------------- runners: copies against recorded hashes
head("runners/ copies against the sha256 the result files record")
rec = {}
for f in glob.glob(str(RES / "**" / "*.json"), recursive=True):
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and k.endswith(".py") and len(v) == 64:
                    rec.setdefault(os.path.basename(k), set()).add(v)
                if k == "provenance" and isinstance(v, dict) and "runner" in v:
                    for kk, vv in v.get("sha256", {}).items():
                        if kk.endswith(os.path.basename(v["runner"])):
                            rec.setdefault(os.path.basename(v["runner"]), set()).add(vv)
                if k in ("script", "script_sha256"):
                    pass
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    d = json.load(open(f))
    walk(d)
    if isinstance(d, dict) and "provenance" in d and isinstance(d["provenance"], dict) and "script_sha256" in d["provenance"]:
        rec.setdefault(os.path.basename(d["provenance"]["script"]), set()).add(d["provenance"]["script_sha256"])
for p in sorted(glob.glob(str(HERE.parent / "runners" / "*" / "*.py"))):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    r = rec.get(os.path.basename(p))
    print(f"{os.path.relpath(p, HERE.parent)}: {h}  {'recorded, MATCH' if r and h in r else ('recorded, DIFFERS' if r else 'no hash recorded')}")
