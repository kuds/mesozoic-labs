import json, hashlib, csv, sys, os, collections
S = sys.argv[1]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()
coef = json.load(open(f"{S}/calib/coef.json"))
PIN = {"s42": ("d3c90db8c107ce1fef6b63efa3e8cdd2271e6e437a3abdb5b30f57b3f46b8df2", "3903cbcfe43559c8f5dccc0b2d6ed01f3c3989f039f966b12ed6aaa7ecbca118"),
       "s80": ("649db94c026db3925a43643dfcd58299877fbe7783bca25e7b7b6ca504a5a3cc", "98fe304328256720f59658da9a0c2b7ccae89dd0cb5ad5798a1768383adc06d9")}
ARM = {"control": (0.0, 0.90), "b90": (coef["COEF_90"], 0.90), "b60": (coef["COEF_60"], 0.60)}
STACK = {"python": "3.13.17", "torch": "2.13.0+cpu", "numpy": "2.5.4", "gymnasium": "1.4.0", "stable_baselines3": "2.9.0", "mujoco": "3.10.0"}
LABELS = ["4325376", "4456448", "4587520", "4718592", "4849664", "4980736", "5111808", "5242880", "final"]
APPROVED_SHA = sha(f"{S}/APPROVED")
print("APPROVED sha", APPROVED_SHA)
names = [f"{st}_{a}_r{sd}" for st in ("s42", "s80") for a in ("control", "b90", "b60") for sd in (101, 202, 303)]
runs_present = sorted(os.listdir(f"{S}/runs"))
print("runs dirs:", len(runs_present), "extra:", set(runs_present) - set(names), "missing:", set(names) - set(runs_present))
rows = list(csv.DictReader(open(f"{S}/panels.csv")))
print("panels rows", len(rows))
keys = collections.Counter((r["name"], r["label"]) for r in rows)
print("dup (name,label):", [k for k, c in keys.items() if c > 1])
probs = []
starts = {}
for n in names:
    st, a, r = n.split("_"); sd = int(r[1:])
    m = json.load(open(f"{S}/runs/{n}/run_meta.json"))
    c, b = ARM[a]
    chk = {
        "status": m.get("status") == "done",
        "name": m.get("name") == n,
        "state": m.get("state") == st,
        "arm": m.get("arm") == a,
        "seed": m.get("rep_seed") == sd,
        "coef": m.get("coef") == c,
        "b": m.get("b") == b,
        "bound_record": (m.get("bound_record") or {}).get("coef") == c and (m.get("bound_record") or {}).get("bound") == b and (m.get("bound_record") or {}).get("index") == 14 and (m.get("bound_record") or {}).get("side") == "upper",
        "argv": m["argv"][1:] == ["--state", st, "--arm", a, "--rep-seed", str(sd), "--steps", "1048576"],
        "start_zip": m["start"]["zip_sha256"] == PIN[st][0] and sha(m["start"]["zip"]) == PIN[st][0],
        "start_pkl": m["start"]["pkl_sha256"] == PIN[st][1] and sha(m["start"]["pkl"]) == PIN[st][1],
        "start_ts": m["start"]["num_timesteps"] == 4194304,
        "head": m["clone"]["head"] == "4c0c8da76d14f4b482f1fc63e0cf57af79b0184f" and m["clone"]["status"] == "",
        "changed": m["clone"]["changed_files"] == ["environments/shared/mirror_kl_ppo.py", "environments/shared/tests/test_mirror_kl_ppo.py"],
        "stack": m["stack"] == STACK,
        "criteria": m["criteria_sha256"] == "9930ee722bf07f21256c0998656568d544ef7413c18d4dfa823338370bcb1b6a",
        "coefjson": m["coef_json_sha256"] == "f6d90f99fc09c09577af9dc2bf4e00fde6537cc8c4fc631af055eaf26880c18d",
        "screen_common": m["screen_common_sha256"] == "c9925334108f6d5f1ab4300bb9c3e00bdb775c5fc237fc1d08f07b97d550607e",
        "approved": m.get("approved_sha256") == APPROVED_SHA,
        "final_ts": m.get("final_checkpoint_num_timesteps") == 5242880 and m["final_digest"]["num_timesteps"] == 5242880,
        "final_zip": sha(f"{S}/runs/{n}/stage1_final.zip") == m["final_sha256"]["stage1_final.zip"],
        "final_pkl": sha(f"{S}/runs/{n}/stage1_final_vecnorm.pkl") == m["final_sha256"]["stage1_final_vecnorm.pkl"],
        "recipe": m["recipe_checks"]["ppo_equal"] and m["recipe_checks"]["curriculum_equal"] and m["recipe_checks"]["run_seed"] == int(st[1:]),
        "train_coef": m["final_update_train"]["train/tail3_bound_coef"] == c and m["final_update_train"]["train/tail3_bound_b"] == b,
    }
    starts[n] = (m["utc_start"], m["utc_end"], m["start"]["digest"]["all"])
    bad = [k for k, v in chk.items() if not v]
    # panels
    nr = [r for r in rows if r["name"] == n]
    labs = sorted(r["label"] for r in nr)
    pb = []
    for r in nr:
        zp = f"{S}/runs/{n}/stage1_final.zip" if r["label"] == "final" else f"{S}/runs/{n}/stage1_{r['label']}_steps.zip"
        pp = f"{S}/runs/{n}/stage1_final_vecnorm.pkl" if r["label"] == "final" else f"{S}/runs/{n}/stage1_vecnormalize_{r['label']}_steps.pkl"
        if sha(zp) != r["zip_sha256"]: pb.append(("zip", r["label"]))
        if sha(pp) != r["pkl_sha256"]: pb.append(("pkl", r["label"]))
        if r["label"] == "final" and r["zip_sha256"] != m["final_sha256"]["stage1_final.zip"]: pb.append(("finalmeta", r["label"]))
        if (r["state"], r["arm"], r["rep_seed"]) != (st, a, str(sd)) or float(r["coef"]) != c or float(r["b"]) != b: pb.append(("rowmeta", r["label"]))
        if r["n_episodes"] != "20" or r["seeds"] != "3042-3061": pb.append(("eps", r["label"], r["n_episodes"], r["seeds"]))
        if r["measure_tree_digest"] != "cf12a36b865a9172361c42372b31425d5c3545591447c67bb27e10ce8d5e3bc2": pb.append(("tree", r["label"]))
        if json.loads(r["stack"]) != STACK: pb.append(("stack", r["label"]))
        if r["label"] != "final" and r["num_timesteps"] != r["label"]: pb.append(("ts", r["label"], r["num_timesteps"]))
        if r["label"] == "final" and r["num_timesteps"] != "5242880": pb.append(("ts", r["label"], r["num_timesteps"]))
        tj = f"{S}/panels/{n}/{r['label']}/episode_trace.json"
        if os.path.exists(tj):
            if sha(tj) != r["trace_json_sha256"]: pb.append(("tracejson", r["label"]))
        else: pb.append(("notrace", r["label"]))
    ok_l = labs == sorted(LABELS)
    pdirs = sorted(os.listdir(f"{S}/panels/{n}"))
    print(f"{n:18s} meta_bad={bad} labels_ok={ok_l} n={len(nr)} panel_dirs={len(pdirs)} panel_probs={pb} start={m['utc_start']} end={m['utc_end']} final_zip={m['final_sha256']['stage1_final.zip'][:12]}")
# start digest identical across replicates of a state
for st in ("s42", "s80"):
    print(st, "start digests:", set(v[2] for k, v in starts.items() if k.startswith(st)))
# duplicate checkpoint hashes across runs (e.g., copied finals)
allfin = collections.Counter()
for n in names:
    m = json.load(open(f"{S}/runs/{n}/run_meta.json"))
    allfin[m["final_sha256"]["stage1_final.zip"]] += 1
print("duplicate final zips across runs:", [k for k, v in allfin.items() if v > 1])
zips = collections.Counter(r["zip_sha256"] for r in rows)
print("duplicate zip sha across panel rows:", [(k[:12], v) for k, v in zips.items() if v > 1])
