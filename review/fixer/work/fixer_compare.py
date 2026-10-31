"""Fixer: compare the fixer's re-run of V1 / V2 / V3 (<PS>/validate/fixer) with the builder's (<PS>/validate), files
only, and check the two run_bs.py changes (P-B1: the TensorBoard folder; P-i3: the end-of-run actor phase check).

    PY -I -B fixer_compare.py   -> exit 0 when every expectation holds
"""
import base64
import csv
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

PS = Path("/home/user/work/phase_screen")
OLD, NEW = PS / "validate", PS / "validate" / "fixer"
ok = True


def expect(cond, text):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {text}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meta(run):
    return json.loads((run / "run_meta.json").read_text())


def evidence(run):
    rows = [json.loads(x) for x in (run / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    for r in rows:
        r.pop("clock", None)
    return rows


def ledger(run):
    with open(run / "01_stance" / "eval_episodes.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    keep = [k for k in rows[0] if not any(t in k for t in ("seconds", "wall", "utc", "time_"))]
    return keep, [[r[k] for k in keep] for r in rows]


def probe(run):
    with np.load(run / "01_stance" / "policy_probe.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def zip_data(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("data")), {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()
                                            if n != "data"}


VOLATILE = {"start_time"}                       # wall clock at learn()'s start


def data_diff(do, dn):
    """Keys of SB3's saved data that differ, after: the class entries compared by their pickled class only (their
    JSON repr carries object addresses), ep_info_buffer by its episodes' (r, l) (its t is wall time), start_time
    dropped (wall clock)."""
    out = []
    for k in sorted(set(do) | set(dn)):
        a, b = do.get(k), dn.get(k)
        if k in VOLATILE:
            continue
        if isinstance(a, dict) and isinstance(b, dict) and ":serialized:" in a and k.endswith("_class"):
            a, b = a[":serialized:"], b[":serialized:"]
        elif k == "ep_info_buffer" and isinstance(a, dict) and isinstance(b, dict):
            a, b = ([(e["r"], e["l"]) for e in pickle.loads(base64.b64decode(x[":serialized:"]))]  # noqa: S301
                    for x in (a, b))
        if a != b:
            out.append(k)
    return out


print("V1 (save-loaded-only, seed 9401): the fixer's run against the builder's")
for arm in ("control", "critic_phase"):
    a, b = meta(OLD / "v1" / f"{arm}_9401"), meta(NEW / "v1" / f"{arm}_9401")
    expect(b["status"] == "done" and b["exit"] == 0, f"{arm}: status {b['status']} exit {b['exit']}")
    for k in ("params_equal", "adam_equal", "vecnormalize_equal", "num_timesteps_S", "saved_pkl_stock_classes",
              "bank_tail3_equal", "phase_bitwise", "load_equal_start", "saved_equal_hooked"):
        expect(b["v1"].get(k) is True and a["v1"].get(k) == b["v1"].get(k), f"{arm}: v1.{k} {b['v1'].get(k)}")
    for k in ("loaded", "hooked", "saved_zip", "start_zip"):
        expect(a["v1"][k] == b["v1"][k], f"{arm}: v1.{k} digests equal the builder's")
    zo, zn = (r / "v1" / f"{arm}_9401" / "saved" / "stage1_2097152_steps.zip" for r in (OLD, NEW))
    (do, mo), (dn, mn) = zip_data(zo), zip_data(zn)
    expect(mo == mn and data_diff(do, dn) == [] and dn.get("tensorboard_log") == do.get("tensorboard_log"),
           f"{arm}: the saved zip equals the builder's (members other than data; data after the volatile keys; "
           f"tensorboard_log unchanged in V1: {dn.get('tensorboard_log')!r})")
    po, pn = (r / "v1" / f"{arm}_9401" / "saved" / "stage1_vecnormalize_2097152_steps.pkl" for r in (OLD, NEW))
    expect(sha(po) == sha(pn), f"{arm}: the saved pkl equals the builder's byte for byte")
    expect(a["digest_load"] == b["digest_load"] and a["digest_start"] == b["digest_start"],
           f"{arm}: digest_load / digest_start equal the builder's")
    expect("tensorboard_log" not in b, f"{arm}: V1 records no TensorBoard redirect (it never calls learn())")

print("V2 (no-update to 2,359,296, seeds 9401 and 9411): the fixer's runs against the builder's")
for seed in (9401, 9411):
    for arm in ("control", "critic_phase"):
        name = f"{arm}_{seed}"
        ro, rn = OLD / "v2" / name, NEW / "v2" / name
        a, b = meta(ro), meta(rn)
        expect(b["status"] == "done" and b["exit"] == 0 and b["stop_reason"] == "end",
               f"{name}: status {b['status']} exit {b['exit']} stop {b['stop_reason']}")
        expect(b["no_update_check"]["pass"] is True and b["digest_end"]["n_updates"] == 2471,
               f"{name}: no optimizer step (no_update_check pass, n_updates {b['digest_end']['n_updates']})")
        for k in ("digest_load", "digest_start", "digest_end"):
            expect(a[k] == b[k], f"{name}: {k} equals the builder's")
        expect(b["arm_state_end"] == "in force", f"{name}: arm_state_end {b['arm_state_end']!r}")
        ev_a, ev_b = evidence(ro), evidence(rn)
        expect(len(ev_b) == 31 and ev_a == ev_b, f"{name}: arm_evidence.jsonl, {len(ev_b)} rollouts, equal to the "
                                                 f"builder's (clock fields dropped)")
        sa, sb = dict(a["arm_evidence"]), dict(b["arm_evidence"])
        expect(sa == sb, f"{name}: run_meta arm_evidence summary equal")
        la, lb = ledger(ro), ledger(rn)
        expect(la == lb, f"{name}: eval_episodes.csv equal ({len(lb[1])} rows x {len(lb[0])} columns, time columns "
                         f"dropped)")
        pa, pb = probe(ro), probe(rn)
        same = sorted(k for k in pa if k in pb and np.array_equal(pa[k], pb[k], equal_nan=pa[k].dtype.kind == "f"))
        diff = sorted(k for k in set(pa) | set(pb) if k not in same and k != "bank_build_seconds")   # wall time
        expect(not diff, f"{name}: policy_probe.npz {len(same)} arrays equal; differing {diff}")
        ck = json.loads((rn / "check.json").read_text())
        expect(ck.get("passed") is True and ck.get("failures") == [],
               f"{name}: CHECK passed {ck.get('passed')} failures {ck.get('failures')}")
        # P-B1: the TensorBoard folder
        tb = b.get("tensorboard_log") or {}
        want = str(PS / "tmp" / f"no_update_{name}" / "tb")
        expect(tb.get("set") == want and str(tb.get("loaded", "")).startswith("/tmp/tb_buffer/"),
               f"{name}: run_meta tensorboard_log loaded {tb.get('loaded')!r} -> set {tb.get('set')!r}")
        events = sorted((Path(want) / "PPO_0").glob("events.out.tfevents.*"))
        mine = [e for e in events if e.stat().st_mtime >= (NEW / "v2" / name / "run_meta.json").stat().st_mtime - 3600]
        expect(len(events) == 1 and len(mine) == 1 and events[0].stat().st_size > 0,
               f"{name}: one events file under <PS>/tmp/no_update_{name}/tb/PPO_0 "
               f"({events[0].name if events else None}, {events[0].stat().st_size if events else 0} B)")
        # the saved periodic pair at 2,359,296: tensors equal the builder's; only data.tensorboard_log differs
        zo = ro / "01_stance" / "models" / "stage1_2359296_steps.zip"
        zn = rn / "01_stance" / "models" / "stage1_2359296_steps.zip"
        do, mo = zip_data(zo)
        dn, mn = zip_data(zn)
        expect(mo == mn, f"{name}: the saved 2,359,296 zip's members other than data equal the builder's "
                         f"({len(mn)} members)")
        keys = data_diff(do, dn)
        expect(keys == ["tensorboard_log"] and dn["tensorboard_log"] == want,
               f"{name}: the zip's data differs from the builder's only in tensorboard_log ({keys})")
        po = ro / "01_stance" / "models" / "stage1_vecnormalize_2359296_steps.pkl"
        pn = rn / "01_stance" / "models" / "stage1_vecnormalize_2359296_steps.pkl"
        expect(sha(po) == sha(pn), f"{name}: the saved 2,359,296 VecNormalize pkl equals the builder's byte for byte")
        # P-i3: the end-of-run actor phase check
        if arm == "critic_phase":
            pse = b.get("phase_state_end") or {}
            expect(pse.get("policy_phase_cols_unchanged") is True and pse.get("policy_phase_adam_max_abs") == 0.0,
                   f"{name}: phase_state_end actor phase columns unchanged {pse.get('policy_phase_cols_unchanged')}, "
                   f"Adam max {pse.get('policy_phase_adam_max_abs')}; value_phase_cols_max_abs "
                   f"{pse.get('value_phase_cols_max_abs')} (0: no update)")
        else:
            expect(b.get("phase_state_end") is None, f"{name}: no phase_state_end for control")
print("V3: the builder's v3_check.py on the fixer's outputs (see logs/val/v3_check.out) and the 2,359,296 markers")
for seed in (9401, 9411):
    for arm in ("control", "critic_phase"):
        name = f"{arm}_{seed}"
        mo = json.loads((OLD / "v2" / name / "read" / "markers.json").read_text())
        mn = json.loads((NEW / "v2" / name / "read" / "markers.json").read_text())
        lo = {r["steps"]: r for r in mo["loop"]}
        ln = {r["steps"]: r for r in mn["loop"]}
        for st in sorted(ln):
            ro_, rn_ = dict(lo[st]), dict(ln[st])
            for r in (ro_, rn_):
                for k in ("zip", "pkl", "zip_sha256", "pkl_sha256", "cache", "cache_key", "seconds", "utc",
                          "cpu_seconds"):
                    r.pop(k, None)
            same = json.dumps(ro_, sort_keys=True) == json.dumps(rn_, sort_keys=True)
            if not same:
                dk = sorted(k for k in set(ro_) | set(rn_) if ro_.get(k) != rn_.get(k))
            expect(same, f"{name}: loop row at {st:,} equal to the builder's" + ("" if same else f" (differ: {dk})"))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
