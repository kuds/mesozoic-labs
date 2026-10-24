"""validate_k3r.py (REPORT ONLY; numpy only, reads files, writes nothing unless OUT_JSON is given): compare
k3readout5r.py's --arm LTSB outputs with stance5's skeptic-checked K3 values.  Per episode where the reference kept
lengths and terminations:
  verify_1/{p,q}*.jsonl (vroll.py: full, hip_pitch, nohp, hpD, hpC), k3diag/rollouts/npz (k3roll.py: hip_pitch,
  pulse_anti, pulse_anti20), verify_2/probe/*.npz (v2_probe.py: full, hpD; and the probe arrays, from which the response
  and the stagger sd are recomputed with v2_probe_ana.py's reduction at full precision).
Counts also against ablation_matrix.csv, pulse.csv and v2_probe.json (3 decimals for the response, 4 for own sigma,
0.1 mm for the stagger sd).  The output checked for each slot is the newest one written by the CURRENT tool (selected by
its identity.tool_sha256, never by file name); every other output of the slot (k3r_<steps>.json, k3r_<steps>.<n>.json,
possibly from an earlier tool) is compared with it readout by readout (determinism; fields an older tool did not write
are ignored).  Own hip-pitch sigma of s52 / s60 at 6.3M is read from the tool's --check-only logs when present.  The
LTSBr end-to-end tests (selftest/t8_ltsbr_e2e and fix_r2/e2e: byte copies of replication judge files under
K3R_TEST_ROOT) and the review-fix round's tests (fix_r2/tests.sh: rc.txt, the refused JSONs and the reward-tamper
smoke output under fix_r2/, and fix_r2/args_tests.txt) are checked too.

    python3 validate_k3r.py [OUT_JSON]      (prints a markdown table; OUT_JSON gets the rows; never overwritten)
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

SP = "scratchpad"
K3 = SP + "/foot/stance5/k3diag"
HERE = SP + "/foot/stance5/replication/k3readouts"
OUT = HERE + "/out"
FIX = HERE + "/fix_r2"
SLOTS = [("stance5_LTSB_s42", "6291456"), ("stance5_LTSB_s42", "7340032"), ("stance5_LTSB_s42", "final"),
         ("stance5_LTSB_s52", "final"), ("stance5_LTSB_s60", "final")]
W0 = 200


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


TOOL_SHA = sha(HERE + "/k3readout5r.py")
RUNNER_SHA = sha(HERE + "/k3readout_one5r.sh")


def outputs(run, st, out=OUT):
    """Every output of the slot: k3r_<st>.json first, then k3r_<st>.<n>.json in order of n."""
    first = os.path.join(out, run, f"k3r_{st}.json")
    reps = []
    for p in glob.glob(os.path.join(out, run, f"k3r_{st}.*.json")):
        m = re.fullmatch(rf"k3r_{re.escape(st)}\.(\d+)\.json", os.path.basename(p))
        if m:
            reps.append((int(m.group(1)), p))
    return ([first] if os.path.exists(first) else []) + [p for _, p in sorted(reps)]


def pick(paths):
    """(primary, others): the newest output written by the current tool, else the newest output."""
    cur = [p for p in paths if json.load(open(p))["identity"]["tool_sha256"] == TOOL_SHA]
    primary = cur[-1] if cur else paths[-1]
    return primary, [p for p in paths if p != primary]


def readouts(d):
    """The readouts of an output (no timings, stamps or tool identity): what a repeat must reproduce."""
    d = json.loads(json.dumps(d))
    keep = {k: d.get(k) for k in ("falls_of", "conditions", "kick", "response", "own_sigma", "stagger")}
    for blk in ("conditions", "kick"):
        for r in keep[blk].values():
            r.pop("seconds", None)
    i = d["identity"]
    keep["identity"] = {k: i.get(k) for k in ("judge_file_sha256", "model_sha256", "vecnorm_sha256")}
    keep["identity"]["sigma_vector"] = i["sigma"]["vector"]
    keep["identity"]["sigma_file_sha256"] = i["sigma"]["file_sha256"]
    return keep


def covered(old, new):
    """True when every field of old is in new with the same value (new may hold fields an older tool did not write)."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and covered(v, new[k]) for k, v in old.items())
    return old == new


v1 = {}
for f in sorted(glob.glob(K3 + "/verify_1/[pq]*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        v1[(d["run"], d["steps"], d["cond"])] = (d, os.path.basename(f))
matrix = {(r["run"], r["steps"]): r for r in csv.DictReader(open(K3 + "/rollouts/tables/ablation_matrix.csv"))}
pulse = {(r["run"], r["steps"], r["cond"]): r for r in csv.DictReader(open(K3 + "/rollouts/tables/pulse.csv"))}
gain = {(r["run"], r["steps"]): r for r in csv.DictReader(open(K3 + "/rollouts/tables/policy_gain.csv"))}
v2 = {(r["run"], r["steps"], r["cond"]): r for r in json.load(open(K3 + "/verify_2/v2_probe.json"))}


def npz(path):
    return np.load(path) if os.path.exists(path) else None


def v2_reduce(d):
    """v2_probe_ana.py: window 200 .. L-1 (L >= 1000) or 200 .. L-151; skip hi <= 260; pooled."""
    L = d["length"]
    jq, st, dq = [], [], []
    for e in range(len(L)):
        hi = int(L[e]) if L[e] >= 1000 else int(L[e]) - 150
        if hi <= W0 + 60:
            continue
        sl = slice(W0, hi)
        jq.append(d["Jq"][e, sl].astype(np.float64))
        st.append(d["stagger"][e, sl].astype(np.float64))
        dq.append(d["dq"][e, sl].astype(np.float64))
    return (float(np.median(np.concatenate(jq))), 1000 * float(np.std(np.concatenate(st))),
            1000 * float(np.std(np.concatenate(dq))))


rows = []


def add(run, st, readout, expected, got, match, source, note=""):
    rows.append({"run": run, "steps": st, "readout": readout, "expected": expected, "got": got,
                 "match": None if match is None else bool(match), "source": source, "note": note})


def ep_compare(run, st, cond, mine, ref_lengths, ref_terms, source):
    ok = list(mine["lengths"]) == [int(x) for x in ref_lengths] and list(mine["terms"]) == [str(x) for x in ref_terms]
    add(run, st, f"{cond} per-episode lengths + terms", "identical", "identical" if ok else "DIFFER", ok, source)


timings = {}
ratio_hp = {}
for run, st in SLOTS:
    paths = outputs(run, st)
    if not paths:
        add(run, st, "output", "present", "missing", False, os.path.join(OUT, run, f"k3r_{st}.json"))
        continue
    path, others = pick(paths)
    d = json.load(open(path))
    for p in paths:
        dd = json.load(open(p))
        timings[os.path.relpath(p, OUT)] = {"total": dd["timing_seconds"]["total_seconds"],
                                            "per_condition": {k: v for k, v in dd["timing_seconds"].items() if k != "total_seconds"},
                                            "tool_sha256": dd["identity"]["tool_sha256"], "started_utc": dd["identity"]["started_utc"]}
    add(run, st, f"checked output {os.path.basename(path)} written by the current tool", TOOL_SHA[:8],
        d["identity"]["tool_sha256"][:8], d["identity"]["tool_sha256"] == TOOL_SHA, "identity.tool_sha256")
    for p in others:
        o = json.load(open(p))
        same = covered(readouts(o), readouts(d))
        add(run, st, f"{os.path.basename(p)} (tool {o['identity']['tool_sha256'][:8]}) readouts == {os.path.basename(path)}",
            "identical", "identical" if same else "DIFFER", same, "determinism")
    C, KK = d["conditions"], d["kick"]
    full = C["full"]
    jfd = json.load(open(d["identity"]["judge_file"]))
    jf = jfd["noise_matched_control_joint"]
    add(run, st, "full falls", str(jf["falls"]), str(full["falls"]), full["falls"] == jf["falls"], "judge file")
    add(run, st, "full matches_judge (seed, length, termination per episode)", "true", str(full["matches_judge"]).lower(),
        full["matches_judge"] is True, "judge file episodes_detail")
    add(run, st, "full reward max abs diff vs judge", "0", f"{full['reward_max_abs_diff_vs_judge']:.3g}",
        full["reward_max_abs_diff_vs_judge"] < 1e-6, "judge file episodes_detail")
    add(run, st, "full reward_matches_judge (< 1e-6, part of valid)", "true", str(d.get("reward_matches_judge")).lower(),
        d.get("reward_matches_judge") is True, "tool")
    if (run, st, "full") in v1:
        r, fn = v1[(run, st, "full")]
        ep_compare(run, st, "full", full, r["lengths"], r["terms"], f"verify_1/{fn}")
    add(run, st, "valid", "true", str(d["valid"]).lower(), d["valid"] is True, "tool")
    add(run, st, "addendum_slot (stance5 reference slot)", "reference", str(d.get("addendum_slot")),
        d.get("addendum_slot") == "reference", "addendum WHERE AND WHEN")
    add(run, st, "judge file judge_sha256 == pinned judge_stance5.py", "sha256:d9286724",
        str(jfd.get("judge_sha256"))[:15], jfd.get("judge_sha256", "").startswith("sha256:d9286724550b"), "judge file")
    # stagger: falls inside the addendum window, from the judge file's own per-episode lengths
    jl = [int(e["length"]) for e in jf["episodes_detail"]]
    want = sum(W0 < L < 1000 for L in jl)
    sfull = d["stagger"]["full"]
    add(run, st, "stagger falls_in_window (full; 200 < L < 1000)", str(want), str(sfull.get("falls_in_window")),
        sfull.get("falls_in_window") == want, "judge file episodes_detail lengths")
    line = (f"stagger_sd={sfull['sd_mm']}mm stagger_sd_v2_window={sfull['sd_mm_v2_window']}mm "
            f"stagger_falls_in_window={sfull.get('falls_in_window')}")
    add(run, st, "summary line carries both stagger figures and falls_in_window", line, line if line in d["summary"]
        else "absent", line in d["summary"], "tool summary")
    # noise conditions
    for cond in ("hip_pitch", "nohp", "hpD", "hpC"):
        got = C[cond]["falls"]
        refs = []
        if (run, st, cond) in v1:
            r, fn = v1[(run, st, cond)]
            refs.append((r["falls"], f"verify_1/{fn}"))
            ep_compare(run, st, cond, C[cond], r["lengths"], r["terms"], f"verify_1/{fn}")
        if cond == "hip_pitch" and (run, st) in matrix and matrix[(run, st)]["hip_pitch"] != "":
            refs.append((int(matrix[(run, st)]["hip_pitch"]), "ablation_matrix.csv"))
            z = npz(f"{K3}/rollouts/npz/{run}__{st}__hip_pitch.npz")
            if z is not None:
                ep_compare(run, st, cond, C[cond], z["length"], z["term"], "rollouts/npz (k3roll hip_pitch)")
        if cond == "hpD" and (run, st, "hpD") in v2:
            refs.append((v2[(run, st, "hpD")]["falls"], "v2_probe.json"))
            z = npz(f"{K3}/verify_2/probe/{run}__{st}__hpD.npz")
            if z is not None:
                ok = list(C[cond]["lengths"]) == [int(x) for x in z["length"]]
                add(run, st, f"{cond} per-episode lengths", "identical", "identical" if ok else "DIFFER", ok,
                    "verify_2/probe npz (v2_probe hpD)")
        if refs:
            add(run, st, f"{cond} falls", " / ".join(f"{v} ({s})" for v, s in refs), str(got),
                all(v == got for v, _ in refs), "; ".join(s for _, s in refs))
        else:
            add(run, st, f"{cond} falls", "no reference (not rolled by k3diag)", str(got), None, "-", "new value")
    # kicks
    for cond, label in (("pulse_anti", "kick +/-0.3"), ("pulse_anti20", "kick +/-0.2")):
        got = KK[cond]["falls"]
        p = pulse.get((run, st, cond))
        if p is not None:
            add(run, st, f"{label} falls", f"{p['falls']} (pulse.csv)", str(got), int(p["falls"]) == got, "pulse.csv")
            z = npz(f"{K3}/rollouts/npz/{run}__{st}__{cond}.npz")
            if z is not None:
                ep_compare(run, st, cond, KK[cond], z["length"], z["term"], f"rollouts/npz (k3roll {cond})")
                pre = int((np.asarray(z["length"]) <= 400).sum())
                add(run, st, f"{label} falls_before_kick (L <= 400)", str(pre), str(KK[cond].get("falls_before_kick")),
                    KK[cond].get("falls_before_kick") == pre, f"rollouts/npz (k3roll {cond}) lengths")
        else:
            add(run, st, f"{label} falls", "no reference", str(got), None, "-", "new value")
    # response, stagger, own sigma
    resp = d["response"]
    ref = v2.get((run, st, "full"))
    z = npz(f"{K3}/verify_2/probe/{run}__{st}__full.npz")
    if ref is not None:
        add(run, st, "direct response Jq median (3 dp, per rad)", f"{ref['Jq_median']:.3f}",
            f"{resp['Jq_median_3dp']:.3f}", round(resp["Jq_median"], 3) == ref["Jq_median"], "v2_probe.json")
        add(run, st, "stagger sd, v2 window (mm, 0.1)", f"{ref['stagger_sd_mm']:.1f}",
            f"{d['stagger']['full']['sd_mm_v2_window']:.2f}",
            round(d["stagger"]["full"]["sd_mm_v2_window"], 1) == ref["stagger_sd_mm"], "v2_probe.json")
        add(run, st, "own hip-pitch sigma (4 dp)", f"{ref['own_sigma_hp']:.4f}",
            f"{d['own_sigma']['hip_pitch_mean']:.4f}", round(d["own_sigma"]["hip_pitch_mean"], 4) == ref["own_sigma_hp"],
            "v2_probe.json")
    if z is not None:
        jq, sd, dq = v2_reduce(z)
        add(run, st, "direct response Jq median, full precision", repr(jq), repr(resp["Jq_median"]),
            jq == resp["Jq_median"], "verify_2/probe npz, v2_probe_ana reduction")
        add(run, st, "stagger sd v2 window, full precision (mm)", f"{sd:.6f}",
            f"{d['stagger']['full']['sd_mm_v2_window']:.2f}", round(sd, 2) == d["stagger"]["full"]["sd_mm_v2_window"],
            "verify_2/probe npz")
        add(run, st, "hip-pitch r-l angle sd, v2 window (mrad)", f"{dq:.6f}",
            f"{d['stagger']['full']['dq_sd_mrad_v2_window']:.2f}", round(dq, 2) == d["stagger"]["full"]["dq_sd_mrad_v2_window"],
            "verify_2/probe npz")
        ok = [int(x) for x in z["length"]] == list(full["lengths"])
        add(run, st, "full per-episode lengths vs v2 probe", "identical", "identical" if ok else "DIFFER", ok,
            "verify_2/probe npz")
    if st == "6291456":
        v = d["own_sigma"]["hip_pitch_mean"]
        add(run, st, "own hip-pitch sigma in the band's 6.3M range", "0.020-0.022", f"{v:.4f}",
            0.0195 <= v <= 0.0225, "report.md sec. 2 line 266")
    if st == "final" and run.endswith(("s52", "s42")):
        want = {"s52": "0.0139", "s42": "0.0148"}[run[-3:]]
        v = d["own_sigma"]["hip_pitch_mean"]
        add(run, st, "own hip-pitch sigma", want, f"{v:.4f}", f"{v:.4f}" == want, "report.md sec. 4 line 313")
    own = d["own_sigma"]
    jo = own["judge_over_own_hip_pitch"]
    add(run, st, "judge sigma / own sigma, hip pitch r, l (report only)", "-", f"{jo[0]:.2f}, {jo[1]:.2f}", None, "tool")
    sv = d["identity"]["sigma"]["vector"]
    ratio_hp[(run, st)] = 0.5 * (sv[3] + sv[7]) / own["hip_pitch_mean"]
    g = gain.get((run, st))
    add(run, st, "stagger sd, addendum window 200..L-1 (mm; report only)",
        f"k3gain window {1000 * float(g['stagger_sd_m']):.1f}" if g else "-",
        f"{d['stagger']['full']['sd_mm']:.2f}", None, "policy_gain.csv (other window)",
        "different windows; the v2-window row is the check")
    add(run, st, "stagger: per-episode max of abs(stagger) over 200..L-1, largest episode (mm; report only)", "-",
        f"{d['stagger']['full']['absmax_mm_max']}", None, "tool")

# report.md sec. 2 line 267: "K3's hip-pitch noise is therefore 6.9-10.2x the band policies' own" (mean of the judge's r/l
# hip-pitch sigma over the own hip-pitch sigma); the range over the validated band slots
if len(ratio_hp) == len(SLOTS):
    lo, hi = min(ratio_hp.values()), max(ratio_hp.values())
    add("band", "5 slots", "judge / own hip-pitch sigma, range over the validated slots (1 dp)", "6.9-10.2",
        f"{lo:.1f}-{hi:.1f} ({min(ratio_hp, key=ratio_hp.get)[0][-3:]} {min(ratio_hp, key=ratio_hp.get)[1]} to "
        f"{max(ratio_hp, key=ratio_hp.get)[0][-3:]} {max(ratio_hp, key=ratio_hp.get)[1]})",
        (round(lo, 1), round(hi, 1)) == (6.9, 10.2), "report.md sec. 2 line 267")

# own hip-pitch sigma of the other band seeds at 6.3M (report.md sec. 2: "0.020-0.022 ... in all three seeds"), from the
# tool's --check-only logs (the checkpoint loaded by the judge's load_checkpoint; nothing rolled)
for run in ("stance5_LTSB_s52", "stance5_LTSB_s60"):
    logs = sorted(glob.glob(os.path.join(HERE, "logs", f"{run}_k3r_6291456_*_check.log")))
    val = None
    for lp in logs:
        for line in open(lp):
            if line.startswith('{"check_only": "pass"'):
                val = json.loads(line)["own_hip_pitch_mean"]
    if val is None:
        add(run, "6291456", "own hip-pitch sigma in the band's 6.3M range (check-only)", "0.020-0.022", "not run", None, "-")
    else:
        add(run, "6291456", "own hip-pitch sigma in the band's 6.3M range (check-only)", "0.020-0.022", f"{val:.4f}",
            0.0195 <= val <= 0.0225, "report.md sec. 2 line 266; " + os.path.basename(logs[-1]))

# the LTSBr path end to end (selftest/t8_ltsbr_e2e, tool b69ff098, and fix_r2/e2e, the current tool: byte copies of
# replication judge files; K3R_TEST_ROOT marks the output invalid by design, so what is checked is matches_judge,
# reward_matches_judge and the sidecar sigma; a current-tool repeat of a t8 slot must reproduce its readouts)
e2e_seen = {}
for src in ("selftest/t8_ltsbr_e2e", "fix_r2/e2e"):
    for p in sorted(glob.glob(os.path.join(HERE, src, "out", "*", "k3r_*.json"))):
        d = json.load(open(p))
        i = d["identity"]
        run, st = i["run"], i["steps"]
        orig = os.path.join(SP, "foot/stance5/replication/judged", run, f"judge_{st}.json")
        same_copy = hashlib.sha256(open(orig, "rb").read()).hexdigest() == i["judge_file_sha256"]
        tag = f"LTSBr e2e ({src}, tool {i['tool_sha256'][:8]})"
        add(run, st, f"{tag}: judge file copy byte-identical to R/judged", "true", str(same_copy).lower(), same_copy, src)
        add(run, st, f"{tag}: sigma from the sidecar (sha256 89778b3f...) == judge file sigma_vector", "true",
            str(i["judge_file_sigma_vector_equal"]).lower(),
            i["judge_file_sigma_vector_equal"] and i["sigma"]["file_sha256"].startswith("89778b3f"), src)
        full = d["conditions"]["full"]
        add(run, st, f"{tag}: full matches_judge", "true", str(full["matches_judge"]).lower(),
            full["matches_judge"] is True, src)
        add(run, st, f"{tag}: full reward max abs diff vs judge", "0", f"{full['reward_max_abs_diff_vs_judge']:.3g}",
            full["reward_max_abs_diff_vs_judge"] < 1e-6, src)
        add(run, st, f"{tag}: full falls / judge falls", str(full["judge_falls"]), str(full["falls"]),
            full["falls"] == full["judge_falls"], src)
        add(run, st, f"{tag}: valid (false by design under K3R_TEST_ROOT)", "false", str(d["valid"]).lower(),
            d["valid"] is False, src)
        if src == "fix_r2/e2e":
            add(run, st, f"{tag}: written by the current tool", TOOL_SHA[:8], i["tool_sha256"][:8],
                i["tool_sha256"] == TOOL_SHA, src)
            add(run, st, f"{tag}: reward_matches_judge", "true", str(d.get("reward_matches_judge")).lower(),
                d.get("reward_matches_judge") is True, src)
            add(run, st, f"{tag}: addendum_slot (not an addendum slot)", "no", str(d.get("addendum_slot")),
                d.get("addendum_slot") == "no", src)
            prev = e2e_seen.get((run, st))
            if prev is not None:
                same = covered(readouts(json.load(open(prev))), readouts(d))
                add(run, st, f"{tag}: readouts == {os.path.relpath(prev, HERE)}", "identical",
                    "identical" if same else "DIFFER", same, "determinism")
        e2e_seen.setdefault((run, st), p)
        timings[os.path.relpath(p, HERE)] = {"total": d["timing_seconds"]["total_seconds"],
                                             "per_condition": {k: v for k, v in d["timing_seconds"].items()
                                                               if k != "total_seconds"},
                                             "tool_sha256": i["tool_sha256"], "started_utc": i["started_utc"]}

# the review-fix round's tests (fix_r2/tests.sh; fix_r2/args_tests.txt from the runner's argument checks)
if os.path.exists(FIX + "/rc.txt"):
    rc = dict(line.split() for line in open(FIX + "/rc.txt") if line.strip())
    rc = {k: int(v) for k, v in rc.items()}

    def txt(name, ext):
        p = os.path.join(FIX, name + ext)
        return open(p).read() if os.path.exists(p) else ""

    for name in sorted(k for k in rc if k.startswith("conf_")):
        msg = "is not a test directory inside" in txt(name, ".err")
        add("fix", "-", f"K3R_TEST_ROOT confinement {name}: exit 2, refused message", "2, message",
            f"{rc[name]}, {'message' if msg else 'no message'}", rc[name] == 2 and msg, "fix_r2/tests.sh")
    written = txt("conf_written", ".txt").split()
    add("fix", "-", "K3R_TEST_ROOT confinement: nothing written under out/ (or /tmp/out, R/out)", "nothing",
        ", ".join(written) or "nothing", not written, "fix_r2/conf_written.txt")
    o = txt("note", ".out")
    ok = rc.get("note") == 0 and "addendum_slot=no CHECK-ONLY pass" in o and "NOTE: stance5_LTSB_s70 4194304" in o
    add("fix", "-", "slot outside the addendum (s70 4194304 copy): NOTE and addendum_slot=no", "rc 0, NOTE, no",
        f"rc {rc.get('note')}", ok, "fix_r2/tests.sh note")
    ref = os.path.join(FIX, "c_jsha/out/stance5_LTSB_s70/refused_4194304.json")
    reasons = json.load(open(ref))["reasons"] if os.path.exists(ref) else []
    ok = rc.get("jsha") == 3 and any("judge_sha256 'sha256:06d39f7d" in r for r in reasons)
    add("fix", "-", "judge file recording another judge_sha256: refused, refused JSON", "rc 3, reason",
        f"rc {rc.get('jsha')}, {len(reasons)} reason(s)", ok, "fix_r2/c_jsha")
    ref = os.path.join(FIX, "r_missing/out/stance5_LTSB_s70/refused_6291456.json")
    reasons = json.load(open(ref))["reasons"] if os.path.exists(ref) else []
    ok = (rc.get("r_missing") == 3 and any(r.endswith("judge_6291456.json missing") for r in reasons)
          and "k3readout refused (rc=3)" in txt("r_missing", ".err"))
    add("fix", "-", "runner, judge file missing: the tool refuses and writes the refused JSON; stderr says so",
        "rc 3, refused JSON, stderr", f"rc {rc.get('r_missing')}, {'JSON' if reasons else 'no JSON'}", ok,
        "fix_r2/runner_testroot.sh")
    ok = rc.get("r_crash") == 1 and "k3readout FAILED rc=1, see " in txt("r_crash", ".err")
    add("fix", "-", "runner, tool crash: stderr 'k3readout FAILED rc=1, see <log>'", "rc 1, message",
        f"rc {rc.get('r_crash')}", ok, "fix_r2/runner_crash.sh")
    rp = os.path.join(FIX, "c_reward/out/stance5_LTSB_s42/k3r_final.json")
    if os.path.exists(rp):
        d = json.load(open(rp))
        f = d["conditions"]["full"]
        ok = (rc.get("reward") == 4 and d["matches_judge"] is True and d["reward_matches_judge"] is False
              and d["valid"] is False and abs(f["reward_max_abs_diff_vs_judge"] - 1e-3) < 1e-6)
        got = (f"rc {rc.get('reward')}, matches {d['matches_judge']}, reward_matches {d['reward_matches_judge']}, "
               f"valid {d['valid']}, diff {f['reward_max_abs_diff_vs_judge']:.3g}")
    else:
        ok, got = False, "no output"
    add("fix", "-", "reward tamper (+1e-3, smoke 1): matches true, reward_matches false, valid false, exit 4",
        "rc 4, True, False, False, 0.001", got, ok, "fix_r2/c_reward")
if os.path.exists(FIX + "/rc_selftest.txt"):  # the builder's refusal self-tests t1-t7 re-run with the current tool
    want = {"t1_missing": "judge_6291456.json missing", "t2_model": "model sha256 ", "t3_sigma": "sigma differs from",
            "t4_sidecar": "judge file sigma file ", "t5_head": "judge file was judged at 0000000", "t6_foreign": "mode 'quick'",
            "t7_sigma_r": "sigma differs from"}
    rcs = {k: int(v) for k, v in (line.split() for line in open(FIX + "/rc_selftest.txt") if line.strip())}
    for name, frag in want.items():
        lp = os.path.join(FIX, f"st_{name}.log")
        k3r = [line for line in open(lp) if line.startswith("K3R ")] if os.path.exists(lp) else []
        ok = rcs.get(name) == 3 and len(k3r) == 1 and "REFUSED" in k3r[0] and frag in k3r[0] and "(recorded " in open(lp).read()
        add("fix", "-", f"self-test {name} re-run with the current tool: exit 3, refused JSON, reason '{frag.strip()}'",
            "3, reason", f"{rcs.get(name)}, {'reason' if ok else 'see ' + os.path.basename(lp)}", ok, "fix_r2/selftests_again.sh")
if os.path.exists(FIX + "/args_tests.txt"):
    for line in open(FIX + "/args_tests.txt"):
        m = re.fullmatch(r"args=\[(.*)\] rc=(\d+)", line.strip())
        if m:
            add("fix", "-", f"runner arguments [{m.group(1).strip()}] (timeout 5 s): exit 2, no hang", "2", m.group(2),
                m.group(2) == "2", "fix_r2/args_tests.txt")

print("| run | steps | readout | expected | got | match | source |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['run'].replace('stance5_LTSB_', '')} | {r['steps']} | {r['readout']} | {r['expected']} | {r['got']} | "
          f"{'info' if r['match'] is None else 'yes' if r['match'] else 'NO'} | {r['source']} |")
print()
checked = [r for r in rows if r["match"] is not None]
print("all checked rows match:", all(r["match"] for r in checked), "checked rows:", len(checked),
      "mismatches:", sum(not r["match"] for r in checked), "info rows:", len(rows) - len(checked))
print("timings (s):", json.dumps(timings))
if len(sys.argv) > 1:
    with open(sys.argv[1], "x") as fh:
        json.dump({"rows": rows, "timings": timings}, fh, indent=1)
