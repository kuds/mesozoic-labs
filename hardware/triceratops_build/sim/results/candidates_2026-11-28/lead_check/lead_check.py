"""The lead's check of key figures of the candidate experiments E1-E5 (28 Nov 2026).

It recomputes the figures listed in the report's section 7 (design/CANDIDATES_2026_11_28.md, "The lead's check of
key figures"), and only those, without the experiments' own summarisers. Sources:
- the session scratch folder's raw episode records (candidates/E1, E2, E3, E5, E5_pitch0);
- the build's stored D22 demo courses (rl/analysis/results/d22/demo/, E3's seed0 700000 for the fore128 and fore160
  policies);
- the committed result files in sim/results/candidates_2026-11-28/: e2a_grid.csv (E2(a)'s row-level table, DESK on
  the logged torques and speeds), the E5 choice files and tuning CSVs and JSONs, and the provenance blocks;
- the runner copies in ../runners/ and their scratch originals.
E3's winding temperatures go through the build's rl/analysis/thermal_ambient.py (method A, 8 min time constant,
35 C room); E3's loop RMS is recomputed here from the stored per-segment mean squared drive torques. E2(a)'s per-row
pass flags are checked against the thresholds before they are tallied. It also runs the candidate checks' unit tests.
Nothing is simulated.

Run from a scratch working directory (needs mujoco, which gait_report imports, and pytest):
    python -B <build>/sim/results/candidates_2026-11-28/lead_check/lead_check.py \
        > <build>/sim/results/candidates_2026-11-28/lead_check/lead_check_2026-11-28.txt
"""

import csv
import filecmp
import glob
import hashlib
import json
import math
import os
import subprocess
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

# ---------------------------------------------------------------- E2(a): every judged row, from the row-level table
head("E2(a) (e2a_grid.csv: each row holds its fastest reduced drive only)")
rows = [r for r in csv.DictReader(open(RES / "E2" / "e2a_grid.csv")) if r["judged"] == "True"]
bad_flags = sum((r["C1_pass"] == "True") != (float(r["C1_c"]) <= 90.0)
                or (r["C2_pass"] == "True") != (float(r["C2"]) <= 0.80)
                or (r["speed_pass"] == "True") != (float(r["speed_p99_max_rpm"]) <= 83.0
                                                   and float(r["speed_peak_max_rpm"]) <= 91.3) for r in rows)
print("per-row pass flags that disagree with the thresholds (C1 <= 90 C, C2 <= 0.80, p99 <= 83 and peak <= 91.3 rpm):",
      bad_flags)


def ok(r):
    return r["C1_pass"] == "True" and r["C2_pass"] == "True" and r["speed_pass"] == "True"


g = defaultdict(list)
for r in rows:
    g[(r["placement"], r["r"], r["transmission"])].append(r)
print("judged rows (placement x r x transmission)", len(g), "PASS", sum(all(ok(r) for r in v) for v in g.values()))
for (pl, rr, tr), v in sorted(g.items()):
    fails = sorted({(r["policy"], r["block"], r["speed_worst_peak_drive"], round(float(r["speed_peak_max_rpm"]), 2))
                    for r in v if r["speed_pass"] != "True"})
    print(f"{pl:5s} r={rr:4s} {tr:5s} C1 {max(float(r['C1_c']) for r in v):6.1f} C  C2 "
          f"{max(float(r['C2']) for r in v):.3f}  p99 {max(float(r['speed_p99_max_rpm']) for r in v):5.1f}  peak "
          f"{max(float(r['speed_peak_max_rpm']) for r in v):6.1f} rpm  speed fails {fails}")
print("first r at which each policy fails the speed check, by placement and block (any transmission):")
for pl in ("pitch", "knees", "both"):
    for pol in sorted({r["policy"] for r in rows}):
        for blk in ("A", "B"):
            rs = sorted({float(r["r"]) for r in rows if r["placement"] == pl and r["policy"] == pol
                         and r["block"] == blk and r["speed_pass"] != "True"})
            print(f"  {pl:5s} {pol} {blk}: {rs[0] if rs else 'none'}")

# ---------------------------------------------------------------- E2(b): all seven runs from the raw records
head("E2(b) (raw episode records; the reduced drives are each record's cap_actuators, the eight non-roll drives)")
E2B_CONDS = ("walk_0.25", "push_left_0.5", "push_right_0.5")
for run in ("B1_dt0.001", "B025_dt0.00025", "rod_dt0.00025", "belt_dt0.00025", "cable_dt0.00025", "R_dt0.00025",
            "belt_1span_dt0.00025"):
    allee = [json.load(open(p)) for p in glob.glob(str(CAND / "E2" / "episodes" / run / "*" / "*.json"))]
    for label, ee in (("all", allee), ("E2(b) conditions", [e for e in allee if e["condition"] in E2B_CONDS])):
        if label == "E2(b) conditions" and len(ee) == len(allee):
            continue
        pk, over = [], 0
        for e in ee:
            cap = e.get("cap_actuators") or [a for a in e["actuators"] if "roll" not in a]
            p = max(e["rpm_peak"][e["actuators"].index(a)] for a in cap)
            pk.append(p)
            over += p > 91.3
        print(f"{run:21s} {label:16s} episodes {len(ee)} falls {sum(e['fell'] for e in ee)} unstable "
              f"{sum(e['unstable'] for e in ee)}; reduced drives' peak {max(pk):.2f} rpm; "
              f"episodes over 91.3 rpm {over}")

# ---------------------------------------------------------------- E3: all eleven policies
head("E3 (C1 via rl/analysis/thermal_ambient.py, method A, 8 min, 35 C; C2 = worst drive's loop RMS / rating)")
sys.path[:0] = [str(BUILD / "rl" / "analysis"), str(BUILD / "rl"), str(BUILD / "rl" / "analytics")]
import gait_report as gr  # noqa: E402
import thermal_ambient as TA  # noqa: E402

RATING = {"X4-36-L": 7.5, "X6-60": 12.0}
dt = gr.drive_table()
POLICIES = [f"fore128-s{i}" for i in range(4)] + [f"fore160-s{i}" for i in range(4)] + ["run3-s0", "run3-s1", "run2b"]
meet = []
for run in POLICIES:
    res = []
    for seed in (700000, 710000):
        p = CAND / "E3" / "demo" / run / f"demo_course_seed{seed}.json"
        if not p.exists():
            p = TA.D22 / "demo" / run / "demo_course.json"
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
        res.append((temps[jn][k]["peak_c"], worst[0]))
        print(f"{run} {seed}: C1 {temps[jn][k]['peak_c']:.2f} C ({jn}); C2 {worst[0]:.4f} ({worst[1]}); "
              f"source {os.path.relpath(p, BUILD) if str(p).startswith(str(BUILD)) else 'scratch ' + p.name}")
    if all(c1 <= 90.0 and c2 <= 0.80 for c1, c2 in res):
        meet.append(run)
print(f"MEET on both courses: {len(meet)} of {len(POLICIES)}: {meet}")

# ---------------------------------------------------------------- E5: both branches and the frozen trot, raw records
head("E5 (raw episode records)")
JC = ("J1", "J2", "J3", "J4")


def e5_sets(folder, pattern):
    ee = [json.load(open(p)) for p in glob.glob(str(folder / f"{pattern}__*.json"))]
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
    for branch, base, choice in (("pitched", CAND / "E5", RES / "E5" / f"e5_choice_{size}.json"),
                                 ("pitch 0", CAND / "E5_pitch0", RES / "E5" / "pitch0" / f"e5p0_choice_{size}.json")):
        cid = json.load(open(choice))["chosen_id"]
        for stage, label, pattern in (("heldout", f"{branch} {cid}", cid), ("heldout_frozen", "frozen trot", "*")):
            for setname, v in e5_sets(base / "episodes" / size / stage, pattern).items():
                print(f"{size} {label:26s} {setname}: n {v['n']} falls J1-J4 {v['falls_J']} J3 {v['J3']:.5f} m/s "
                      f"slow-gait max/cond {v['slow_max']} worst knee {v['knee_max']:.3f} (per condition "
                      f"{v['knee_per_cond'][0]:.3f}-{v['knee_per_cond'][1]:.3f}) X4 peak {v['x4_peak']:.2f} rpm")

head("E5 search budget (tuning CSVs and JSONs; the prereg counts the budget over all stages, at most 60 per size)")
SETTING_KEYS = ("period_s", "duty", "body_pitch_deg", "lift_m", "max_reach_m")
for size in ("0.60", "0.50"):
    vals, runs = {}, {}
    for branch, f in (("pitched", RES / "E5" / f"e5_tuning_{size}.csv"),
                      ("pitch 0", RES / "E5" / "pitch0" / f"e5p0_tuning_{size}.csv")):
        rr = list(csv.DictReader(open(f)))
        vals[branch] = {tuple(r[k] for k in SETTING_KEYS) for r in rr}
        runs[branch] = len(rr)
        j = json.load(open(str(f).replace(".csv", ".json")))
        print(f"{size} {branch}: setting runs over stages {len(rr)}; distinct settings {len(vals[branch])}; "
              f"budget record {j.get('budget')}")
    shared = vals["pitched"] & vals["pitch 0"]
    print(f"{size} both branches: distinct {len(vals['pitched'] | vals['pitch 0'])} ({len(shared)} shared); setting "
          f"runs over stages {runs['pitched'] + runs['pitch 0']}, {runs['pitched'] + runs['pitch 0'] - len(shared)} "
          f"with each shared setting counted once")

# ---------------------------------------------------------------- runners: copies against hashes and originals
head("runners/ copies: against the sha256 the result files record, and byte for byte against the scratch originals")
rec = defaultdict(set)


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and k.endswith(".py") and len(v) == 64:
                rec[os.path.basename(k)].add(v)
            if k == "provenance" and isinstance(v, dict):
                if "runner" in v:
                    name = os.path.basename(v["runner"])
                    for kk, vv in v.get("sha256", {}).items():
                        if kk.endswith(name):
                            rec[name].add(vv)
                if "script_sha256" in v:
                    rec[os.path.basename(v["script"])].add(v["script_sha256"])
            walk(v)
    elif isinstance(o, list):
        for x in o:
            walk(x)


for f in glob.glob(str(RES / "**" / "*.json"), recursive=True):
    walk(json.load(open(f)))
ORIG = {"E1": CAND / "E1", "E2": CAND / "E2" / "code", "E3": CAND / "E3" / "code", "E4": CAND / "E4",
        "E5": CAND / "E5" / "code", "E5_pitch0": CAND / "E5_pitch0" / "code"}
for p in sorted(glob.glob(str(RES / "runners" / "*" / "*.py"))):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    r = rec.get(os.path.basename(p))
    state = "no hash recorded" if not r else ("recorded, MATCH" if r == {h} else f"recorded, DIFFERS {sorted(r)}")
    orig = ORIG[Path(p).parent.name] / Path(p).name
    same = orig.exists() and filecmp.cmp(p, orig, shallow=False)
    print(f"{os.path.relpath(p, RES)}: {h}  {state}; scratch original {'identical' if same else 'DIFFERS OR MISSING'}")

# ---------------------------------------------------------------- the candidate checks' unit tests
head("sim/tests/test_candidate_checks.py")
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
t = subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q",
                    str(BUILD / "sim" / "tests" / "test_candidate_checks.py")], capture_output=True, text=True, env=env)
print([ln for ln in t.stdout.splitlines() if ln.strip()][-1], f"(pytest exit {t.returncode})")
