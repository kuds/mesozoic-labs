"""Per-continuation run_meta / check.json audit against the pins as they are now (recomputed), SCREEN.json and the
approved limits.  Reads only; prints a table and every deviation."""
import hashlib, json, os, re, sys, datetime

BS2 = sys.argv[1]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

S, E, LENGTH = 2_097_152, 3_538_944, 1_441_792
assert S + LENGTH == E and LENGTH <= 1_500_000
APPROVED = json.load(open(os.path.join(BS2, "APPROVED")))
now = {
    "approved": sha(os.path.join(BS2, "APPROVED")),
    "criteria": sha(os.path.join(BS2, "criteria.txt")),
    "screen": sha(os.path.join(BS2, "SCREEN.json")),
    "tools": {t: sha(os.path.join(BS2, "tools", t)) for t in
              ("run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py", "score_bs.py")},
}
problems = []
def want(cond, msg):
    if not cond:
        problems.append(msg)

want(now["approved"] == "184f9ab91f284149f15ac192b4a5f318fa01485fbaf0068df64bb03ba5029ad2", "APPROVED sha changed")
want(APPROVED["criteria_sha256"] == now["criteria"], "APPROVED criteria pin != criteria.txt now")
want(APPROVED["screen_sha256"] == now["screen"], "APPROVED screen pin != SCREEN.json now")
want(APPROVED["code_tree_digest"] == "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1", "code digest pin")
for t, h in now["tools"].items():
    want(APPROVED["tool_sha256"].get(t) == h, f"APPROVED tool pin {t} != file now")
want(set(APPROVED["tool_sha256"]) == set(now["tools"]), "APPROVED tool set")
# criteria PINS lines
crit = open(os.path.join(BS2, "criteria.txt")).read()
for rel in ("SCREEN.json", "tools/score_bs.py", "tools/bs_common.py", "tools/bs_hooks.py", "tools/run_bs.py",
            "tools/run_waves_bs.sh", "tools/read_bs.py"):
    h = sha(os.path.join(BS2, rel))
    want(f"{rel}" in crit and h in crit, f"criteria PINS lacks {rel} {h}")
    line = [l for l in crit.splitlines() if re.fullmatch(r"\s{4}" + re.escape(rel) + r"\s+[0-9a-f]{64}", l)]
    want(len(line) == 1 and line[0].split()[-1] == h, f"criteria PINS line for {rel}: {line}")
h_t = sha(os.path.join(BS2, "tools/tests/test_score_bs.py"))
line = [l for l in crit.splitlines() if l.strip().startswith("tools/tests/test_score_bs.py ")]
want(len(line) == 1 and line[0].split()[-1] == h_t, f"criteria PINS test_score_bs.py line {line} vs {h_t}")

screen = json.load(open(os.path.join(BS2, "SCREEN.json")))
queue = [(a, int(s)) for a, s in screen["queue"]]
want(screen["intervention"] == "critic_blind_tail3", "intervention")
want(len(queue) == 8 and sorted({a for a, _ in queue}) == ["control", "critic_blind_tail3"], "queue arms/size")
want(sorted(s for a, s in queue if a == "control") == [2501, 2511, 2521, 2531] ==
     sorted(s for a, s in queue if a != "control"), "paired seeds")
names = [f"{a}_{s}" for a, s in queue]
runs_now = sorted(os.listdir(os.path.join(BS2, "runs")))
want(runs_now == sorted(names), f"runs/ holds {runs_now}")
want(not os.path.exists(os.path.join(BS2, "lost.json")), "lost.json exists")
for f in ("HALTED", "PAUSED_disk"):
    want(not os.path.exists(os.path.join(BS2, "logs", f)), f"logs/{f} exists")
want(os.path.getsize(os.path.join(BS2, "logs", "crash_ledger.tsv")) == 0, "crash ledger not empty")

rows = []
appr_utc = datetime.datetime(2026, 10, 30, 12, 53, 50, tzinfo=datetime.timezone.utc)
for (arm, seed), name in zip(queue, names):
    rd = os.path.join(BS2, "runs", name)
    m = json.load(open(os.path.join(rd, "run_meta.json")))
    c = json.load(open(os.path.join(rd, "check.json")))
    P = lambda cond, msg: want(cond, f"{name}: {msg}")
    P(m["name"] == name and m["arm"] == arm and m["seed"] == seed, "name/arm/seed")
    P(m["mode"] == "continuation", "mode")
    P(m["status"] == "done" and m["exit"] == 0, f"status {m['status']} exit {m['exit']}")
    P(m["stop_reason"] == "end" and m["T"] == E and m["E"] == E and m["target"] == E, "stop/T/E")
    P(m["S"] == S, "S")
    P(m["num_timesteps_end"] == E, "num_timesteps_end")
    P(m["stop_at_end"]["flag"] is True and m["stop_at_end"]["step"] == E, "stop_at_end")
    P(m["collapse"]["tripped"] is False, "collapse tripped")
    P(m["criteria_sha256"] == now["criteria"], "criteria sha")
    P(m["approved_sha256"] == now["approved"], "approved sha")
    P(m["screen_sha256"] == now["screen"], "screen sha")
    P(m["code"]["tree_digest"] == APPROVED["code_tree_digest"] and m["code"]["files"] == 1406 and
      m["code"]["commit"] == "8737cd3d86431de1bdd0fcdb49a0d956a45e084d", "code")
    P(m["code"]["dir"].endswith("/r10round/bias_screen/code"), "code dir")
    for t, h in m["tool_sha256"].items():
        P(now["tools"].get(t) == h, f"tool {t} in run_meta {h[:12]} != now")
    P(set(m["tool_sha256"]) == {"run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py"},
      f"tool set {sorted(m['tool_sha256'])}")
    P(m["code_seal_after"] == "unchanged", "code seal after")
    P(m["modules_outside_code"] == [], "modules outside code")
    P(m["digest_end"]["n_updates"] > m["digest_start"]["n_updates"], "n_updates grew")
    P(m["digest_end"]["params"] != m["digest_start"]["params"], "params changed")
    P(m["digest_start"]["num_timesteps"] == S and m["digest_end"]["num_timesteps"] == E, "digest steps")
    P(m["start_pair"]["zip_sha256"] == "83b6c88edb933005adcb2089bdac67654f4e77809ca5506f4e47364995ef1b61" and
      m["start_pair"]["pkl_sha256"] == "991d434f385187e87301755ddf9a12fe2b3adb65cf186e7e32b175b3574bed90", "start pair")
    P(m["arm_state_end"] == "in force", "arm state end")
    ap = m["arm_applied"]
    want_cls = "CriticBlindTail3MlpExtractor" if arm == "critic_blind_tail3" else "MlpExtractor"
    P(ap["arm"] == arm and ap["mlp_extractor_class"] == want_cls and ap["mlp_extractor_class_before"] == "MlpExtractor",
      f"classes {ap.get('mlp_extractor_class_before')} -> {ap.get('mlp_extractor_class')}")
    P(ap["obs_rms_class"] == "SymmetricRunningMeanStd", "obs_rms class")
    P(ap["gae_lambda"] == 0.95 and ap["rollout_buffer_gae_lambda"] == 0.95, "gae")
    P(m["arm_ppo"] == "unchanged", "arm_ppo")
    ae = m["arm_evidence"]
    P(ae["n"] == 175 and ae["steps_first_last"] == [S + 8192, S + 175 * 8192], f"evidence n/steps {ae['n']} {ae['steps_first_last']}")
    P(ae["state_problems"] == [], "state problems")
    t0 = datetime.datetime.strptime(m["utc_start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    P(t0 > appr_utc, "started before APPROVED")
    # copies at the pinned sha
    for k, v in m["copies"].items():
        P(v["from"].startswith(os.environ["S1"] + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/"), f"copy source {k}")
    # check.json
    P(c["passed"] is True and c["failures"] == [], "check passed")
    P(c["tool_sha256"] == now["tools"]["read_bs.py"], "check.json made by another read_bs.py")
    P(c["name"] == name and c["arm"] == arm and c["seed"] == seed and c["S"] == S and c["E"] == E and c["T"] == E
      and c["stop_reason"] == "end", "check identity")
    for k, (path, h) in c["sources"].items():
        if k.startswith("parent"):
            continue
        P(sha(path) == h, f"{k} changed since CHECK")
    P(c["items"]["probe_detail"]["rows_in_window"] == 175 and c["items"]["probe_detail"]["rows_expected"] == 175,
      "probe rows")
    P(all(v == [] for k, v in c["items"].items() if k != "probe_detail"), "check items")
    # check.log
    cl = open(os.path.join(BS2, "logs", f"{name}.check.log")).read()
    P('"check": "pass"' in cl and '"failures": []' in cl, "check.log")
    # checkpoints present (14) and final pair
    mods = sorted(os.listdir(os.path.join(rd, "01_stance", "models")))
    need = {"stage1_final.zip", "stage1_final_vecnorm.pkl"} | {f"stage1_{s}_steps.zip" for s in range(S, E, 262144)} | \
           {f"stage1_vecnormalize_{s}_steps.pkl" for s in range(S, E, 262144)}
    P(need <= set(mods), f"models missing {need - set(mods)}")
    extra = set(mods) - need
    rows.append((name, m["utc_start"], m["utc_end"], m["status"], m["exit"], m["stop_reason"], m["T"],
                 m["digest_start"]["n_updates"], m["digest_end"]["n_updates"], m["learn_seconds"],
                 m["heartbeat"]["rollout_end"], sorted(extra)))
print("now:", json.dumps(now, indent=1))
for r in rows:
    print("\t".join(str(x) for x in r))
print("PROBLEMS:", problems if problems else "none")
