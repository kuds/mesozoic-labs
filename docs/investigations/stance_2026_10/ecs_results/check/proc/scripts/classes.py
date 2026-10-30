"""(a) CLASSES: my own checks of run_meta.json, check.json and R4 for the 24 continuations.  Reads only run_meta,
check.json, the probe columns update_timesteps / ent_coef / learning_rate (allow_pickle=False), the parent F2 probes,
and the start pair copies' sha256.  Pins are typed in from criteria.txt PINS / APPROVED (not imported from the tools)."""
import hashlib, json, os, sys
import numpy as np
ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
F2 = S1 + "/foot/stance6/fetch/F2"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
CRIT = "04813918e494ae2d4845819c7ef123bc4bf74634093faa439e572c831277214e"
TREE = "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1"
COMMIT = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"
APPROVED = json.load(open(ECS + "/APPROVED"))
APPROVED_SHA = sha(ECS + "/APPROVED")
READ_PIN = "6f55b3b96bae795d46c3ef693b5b542895186adb4ccb51957076f3421c2fee15"
ST = {"s42": dict(S=3145728, E=4718592, run="stance6_r10LTSB_s42",
                  zip="3c054bd738304965bbbcfde588982635ed047436311a267b91f8855e02edad18",
                  pkl="997f31421d805f6e6cd366b9a9bd05451ed7ab679fd615e7567d259a310f22e0",
                  probe="4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11",
                  rec={"eval_episodes.csv": "d1bcb78964691cf194e6276f06740d1f24aa69ad36831a046ad6307c1c4db11e",
                       "evaluations.npz": "3e11136120622b90cc63203c46332004a9989188a0b8fe91620dbabb9f2563e9",
                       "gate_progress.npz": "4097099b6880fa234a38e1b43819a21a8531f39ca0aa1daefaaeeb35a06c5a0e",
                       "train_episodes.csv": "14b2f9e49b67e9c90cc79362f740e1768de5a2d159035176d5299b9c476d851d",
                       "policy_probe.npz": "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11"}),
      "s80": dict(S=4194304, E=5767168, run="stance6_r10LTSB_s80",
                  zip="51197dbd7f90e481a99214ba7bd63ebbe92804367c2e5edab988c0b4c4840096",
                  pkl="f405244b94662bdffae9bbe1b047fda9b34bb62012bec9491ed09131ef6155cb",
                  probe="0e61919dc9a0a2a508095cec5e14a673c2d909b61420d9b6c95c74d7562b3957",
                  rec={"eval_episodes.csv": "b677ce762e739513c18eaa35ffb7b149c0532753746259d5037356580e934d13",
                       "evaluations.npz": "89e18b1006deb304b368284d6106299fa219cce9e7f1d73c3ef37f5120c39be8",
                       "gate_progress.npz": "02dc43115adff9096f175cc6387f01e81cc816aca54c7070b1e8a9b6400cb65c",
                       "train_episodes.csv": "fd65c2058c4b6970c5fca7a32a6461131f41a768ec792b5d625ea8a5049af8c6",
                       "policy_probe.npz": "0e61919dc9a0a2a508095cec5e14a673c2d909b61420d9b6c95c74d7562b3957"}),}
SEEDS = {("s42", "A"): range(1101, 1152, 10), ("s42", "B"): range(1201, 1252, 10),
         ("s80", "A"): range(1301, 1352, 10), ("s80", "B"): range(1401, 1452, 10)}
NAMES = [f"{s}_{a}_{x}" for (s, a), xs in SEEDS.items() for x in xs]
assert len(NAMES) == 24
def probe(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("update_timesteps", "ent_coef", "learning_rate")}
parent = {}
for s, d in ST.items():
    p = f"{F2}/{d['run']}/copies/01_stance/policy_probe.npz.copy1"
    h = sha(p)
    print(f"parent probe {s}: {p} sha256 {h} pin-equal {h == d['probe']}")
    parent[s] = probe(p)
problems = []
summary = []
for name in NAMES:
    s, a, x = name.split("_"); x = int(x); d = ST[s]; S, E = d["S"], d["E"]
    run = f"{ECS}/runs/{name}"
    m = json.load(open(run + "/run_meta.json"))
    c = json.load(open(run + "/check.json"))
    P = []
    def want(cond, msg):
        if not cond: P.append(msg)
    want(m.get("schema") == "ecs-run/1", "schema")
    want(m.get("name") == name, f"name {m.get('name')}")
    want(m.get("mode") == "continuation", "mode")
    want(m.get("status") == "done", f"status {m.get('status')}")
    want(m.get("exit") == 0 and type(m.get("exit")) is int, f"exit {m.get('exit')}")
    want((m.get("start"), m.get("arm"), m.get("seed")) == (s, a, x), "start/arm/seed")
    want(m.get("argv")[1:] == ["--start", s, "--arm", a, "--seed", str(x)], f"argv {m.get('argv')}")
    want(m.get("S") == S and m.get("E") == E and m.get("target") == E, "S/E/target")
    want(m.get("T") == E and m.get("num_timesteps_end") == E, f"T {m.get('T')}")
    want(m.get("stop_reason") == "end", f"stop {m.get('stop_reason')}")
    sae = m.get("stop_at_end") or {}
    want(sae.get("flag") is True and sae.get("step") == E and sae.get("target") == E, f"stop_at_end {sae}")
    code = m.get("code") or {}
    want(code.get("commit") == COMMIT and code.get("tree_digest") == TREE and code.get("files") == 1406, f"code {code}")
    want(m.get("code_seal_after") == "unchanged", f"code_seal_after {m.get('code_seal_after')}")
    want(m.get("modules_outside_code") == [], "modules_outside_code")
    sp = m.get("start_pair") or {}
    want(sp.get("zip_sha256") == d["zip"] and sp.get("pkl_sha256") == d["pkl"], "start pair recorded")
    zp = f"{run}/01_stance/models/stage1_{S}_steps.zip"; pp = f"{run}/01_stance/models/stage1_vecnormalize_{S}_steps.pkl"
    want(sha(zp) == d["zip"] and sha(pp) == d["pkl"], "start pair copy on disk sha256")
    want(m.get("parent_records") == d["rec"], f"parent_records {m.get('parent_records')}")
    want(m.get("criteria_sha256") == CRIT, "criteria_sha256")
    want(m.get("approved_sha256") == APPROVED_SHA, f"approved_sha256 {m.get('approved_sha256')} vs {APPROVED_SHA}")
    want(m.get("tool_sha256") == APPROVED["tool_sha256"], "tool_sha256 vs APPROVED")
    want(m.get("torch_threads") == 1 and type(m.get("torch_threads")) is int, f"torch_threads {m.get('torch_threads')}")
    env = m.get("environment") or {}
    want(all(env.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "env threads")
    want(env.get("GITHUB_SHA") == COMMIT, "GITHUB_SHA")
    col = m.get("collapse") or {}
    want(col.get("found") == 1 and col.get("patience") == 10 and col.get("tripped") is False, f"collapse {col}")
    want(m.get("arm_ppo") == ("unchanged" if a == "A" else {"ent_coef": 0.0, "ent_coef_end": 0.0}), f"arm_ppo {m.get('arm_ppo')}")
    want(m.get("ent_coef_after_load") == (0.005 if a == "A" else 0.0), f"ent_after_load {m.get('ent_coef_after_load')}")
    pa = m.get("ppo_applied") or {}
    want((pa.get("ent_coef"), pa.get("ent_coef_end")) == ((0.005, 0.0) if a == "A" else (0.0, 0.0)), f"ppo_applied {pa}")
    want((m.get("recipe_v5") or {}).get("problems") == [], "recipe_v5")
    d0, d1 = m.get("digest_start") or {}, m.get("digest_end") or {}
    want(d0.get("params") and d1.get("params") and d0["params"] != d1["params"], "params digest unchanged")
    want(isinstance(d0.get("n_updates"), int) and isinstance(d1.get("n_updates"), int) and d1["n_updates"] > d0["n_updates"], "n_updates")
    want(d0.get("num_timesteps") == S and d1.get("num_timesteps") == E, "digest num_timesteps")
    hb = m.get("heartbeat") or {}
    want(hb.get("rollout_end") == E - 8192, f"heartbeat {hb}")
    want("error" not in m and "traceback" not in m, "error/traceback present")
    # check.json
    want(c.get("passed") is True and c.get("failures") == [], f"check passed {c.get('passed')} {c.get('failures')}")
    want(c.get("mode") == "continuation" and c.get("name") == name, "check mode/name")
    want(c.get("tool_sha256") == READ_PIN, "check tool sha")
    want((c.get("start"), c.get("arm"), c.get("seed"), c.get("S"), c.get("E"), c.get("T"), c.get("stop_reason")) == (s, a, x, S, E, E, "end"), "check identity")
    src = c.get("sources") or {}
    want(src.get("run_meta.json", [None, None])[1] == sha(run + "/run_meta.json"), "check.json not current: run_meta")
    want(src.get("policy_probe.npz", [None, None])[1] == sha(run + "/01_stance/policy_probe.npz"), "check.json not current: probe")
    want(src.get("parent policy_probe.npz", [None, None])[1] == d["probe"], "check parent probe sha")
    # my own R4
    r = probe(run + "/01_stance/policy_probe.npz"); pr = parent[s]
    u = r["update_timesteps"].astype(np.int64)
    idx = [i for i in range(len(u)) if S < u[i] < E]
    steps = [int(u[i]) for i in idx]
    exp = [S + 8192 * k for k in range(1, (E - S) // 8192)]
    want(len(exp) == 191 and steps == exp, f"R4 steps: {len(steps)} rows")
    want(not any(int(v) >= E for v in u), "R4 rows at/after T")
    pu = pr["update_timesteps"].astype(np.int64)
    pidx = []
    for st_ in exp:
        w = np.nonzero(pu == st_)[0]
        want(len(w) == 1, f"parent rows at {st_}: {len(w)}")
        pidx.append(int(w[0]) if len(w) else 0)
    ent = r["ent_coef"].astype(np.float32)[idx]; lr = r["learning_rate"].astype(np.float32)[idx]
    pent = pr["ent_coef"].astype(np.float32)[pidx]; plr = pr["learning_rate"].astype(np.float32)[pidx]
    if a == "A":
        nb_ent = int(np.sum(~(ent == pent)))
    else:
        nb_ent = int(np.sum(~(ent == np.float32(0.0))))
    nb_lr = int(np.sum(~(lr == plr)))
    want(nb_ent == 0, f"R4 ent mismatches {nb_ent}")
    want(nb_lr == 0, f"R4 lr mismatches {nb_lr}")
    # raw dtypes and exact (not just float32) equality, for information
    exact_ent = bool(np.array_equal(r["ent_coef"][idx], (pr["ent_coef"][pidx] if a == "A" else np.zeros(191))))
    exact_lr = bool(np.array_equal(r["learning_rate"][idx], pr["learning_rate"][pidx]))
    n_seeded = int(np.sum(u <= S))
    summary.append((name, len(steps), steps[0], steps[-1], float(ent[0]), float(ent[-1]), float(lr[0]), float(lr[-1]),
                    nb_ent, nb_lr, exact_ent, exact_lr, n_seeded, str(r["ent_coef"].dtype), d1["n_updates"] - d0["n_updates"],
                    m.get("utc_start"), m.get("utc_end"), m.get("wall_seconds")))
    if P:
        problems.append((name, P))
for row in summary:
    print(" ".join(str(v) for v in row))
print("PROBLEMS:", problems if problems else "none")
