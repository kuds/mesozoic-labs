"""Build the two synthetic BS2-shaped dry-run roots for score_bs.py (no simulation, no training).

    PY -I -B make_dryrun.py <BS2>

  <BS2>/criteria_work/dryrun_root/parent_as_all/   the criteria's dry run (VALIDATIONS): the parent's own reference
      markers (<BS2>/validate/v3final/parent_s42/markers.json, read_bs.py --reference) as every one of the 8 queue
      entries, relabelled as continuations; the blind entries' loop rows get the arm's critic form ("blind+raw") and
      effective slope 0.0 (the scorer refuses any other form for that arm).  Expected: line 3, p 70/70, control rise
      +0.57878; SEC-CMD 8 equal values, p 70/70; SEC-G and SEC-MARGIN MISSING in all 8 (no loop row at E).
  <BS2>/criteria_work/dryrun_root/synthetic_E/     the same, plus a SYNTHETIC loop row at E per entry (a copy of the
      parent's 3,145,728 row with steps = num_timesteps = E and OWN.P.G / b_star / c0 set below), to exercise SEC-G and
      SEC-MARGIN end to end.  Expected: line 3, p 70/70 (the primaries are the parent's); SEC-G p 1/70 SUPPORTED and
      surviving Bonferroni; SEC-MARGIN p 70/70 NOT SUPPORTED; SEC-CMD p 70/70; G(E) > G(S) blind 4 of 4, control 2 of 4.
Each root: SCREEN.json and criteria.txt copied from <BS2> (the installed ones), runs/<name>/{run_meta.json,
check.json, 01_stance/eval_episodes.csv (the parent's truncated ledger, copied)}, reads/<name>/markers.json.  No
APPROVED is written anywhere (the reading is PROVISIONAL by design).  Writes only under criteria_work/dryrun_root.
"""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

BS2 = Path(sys.argv[1]).resolve()
OUT = BS2 / "review" / "rereview" / "work" / "dryrun_root"   # re-reviewer copy: only OUT differs
TOOLS = BS2 / "tools"
PARENT = BS2 / "validate" / "v3final" / "parent_s42" / "markers.json"
S, E = 2097152, 3538944
SEEDS = (2501, 2511, 2521, 2531)
ARMS = ("control", "critic_blind_tail3")
QUEUE = [(a, s) for s in SEEDS for a in ARMS]
# the synthetic end-pair values (synthetic_E only): G at E and b* at E per seed, per arm
G_E = {"control": (-0.9, -0.7, -0.5, -0.3), "critic_blind_tail3": (-0.2, -0.1, 0.0, 0.1)}
B_E = {"control": (1.9, 1.7, 1.5, 1.3), "critic_blind_tail3": (1.0, 0.9, 0.8, 0.7)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    return path


def require_under(p, root):
    p, root = Path(p).resolve(), Path(root).resolve()
    assert p == root or root in p.parents, p
    return p


def meta(arm, seed, crit_sha, screen_sha):
    blind = arm == "critic_blind_tail3"
    ev = {"n": 175, "buffer_gae_lambda": [0.95], "max_adv_err_vs_gae095": 3.8e-6, "min_dmu_actor": 0.166,
          "max_dv_predict_values": 0.0 if blind else 0.29, "max_dv_forward": 0.0 if blind else 0.29,
          "max_dv_evaluate_actions": 0.0 if blind else 0.29, "max_dv_mirrored": 0.0 if blind else 0.30,
          "min_dv_predict_values": 0.0 if blind else 0.0706, "min_buffer_values_vs_raw_value_mlp": 0.0246 if blind
          else 0.0, "state_problems": []}
    return {"schema": "bs-run/1", "name": f"{arm}_{seed}", "arm": arm, "seed": seed, "T": E, "stop_reason": "end",
            "status": "done", "_dry_run": "SYNTHETIC run_meta for the scorer's dry run; no run exists",
            "arm_evidence": ev, "arm_applied": {"arm": arm, "mlp_extractor_class": "CriticBlindTail3MlpExtractor"
                                                if blind else "MlpExtractor"}, "arm_state_end": "in force",
            "criteria_sha256": crit_sha, "screen_sha256": screen_sha, "approved_sha256": None,
            "tool_sha256": {t: sha(TOOLS / t) for t in ("run_bs.py", "read_bs.py", "run_waves_bs.sh",
                                                        "bs_common.py", "bs_hooks.py")}}


def as_continuation(pm, arm, seed, i, synthetic_e):
    m = copy.deepcopy(pm)
    name = f"{arm}_{seed}"
    m.update(kind="continuation", label=name, arm=arm, seed=seed, stop_reason="end",
             _dry_run="the parent's reference markers relabelled as this continuation (scorer dry run)")
    m.pop("reproduction", None)
    if synthetic_e:
        row = copy.deepcopy(next(r for r in m["loop"] if r["steps"] == 3145728))
        G, b = G_E[arm][i], B_E[arm][i]
        row.update(steps=E, num_timesteps=E, _dry_run="SYNTHETIC loop row at E (a copy of the 3,145,728 row with "
                                                      "OWN.P.G, c0, b_star set)")
        row["OWN"]["P"].update(G=G, b_star=b, c0=1.0 - G - b)
        m["loop"].append(row)
    if arm == "critic_blind_tail3":
        for row in m["loop"]:
            row["critic_form"] = "blind+raw"
            for d in row["critic_slope_obs6"].values():
                d["effective"] = {**d["effective"], "dV_dq3_per_deg": 0.0}
    return m


def build(root, synthetic_e):
    root = require_under(OUT / root, OUT)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    shutil.copyfile(BS2 / "SCREEN.json", root / "SCREEN.json")
    shutil.copyfile(BS2 / "criteria.txt", root / "criteria.txt")
    crit_sha, screen_sha = sha(root / "criteria.txt"), sha(root / "SCREEN.json")
    pm = json.loads(PARENT.read_text(encoding="utf-8"))
    src_ledger = Path(pm["ledger"]["file"])
    assert sha(src_ledger) == pm["ledger"]["sha256"]
    for arm, seed in QUEUE:
        name = f"{arm}_{seed}"
        rd = root / "runs" / name
        mp = write(rd / "run_meta.json", meta(arm, seed, crit_sha, screen_sha))
        led = rd / "01_stance" / "eval_episodes.csv"
        led.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_ledger, led)
        write(rd / "check.json", {
            "schema": "bs-check/1", "tool_sha256": sha(TOOLS / "read_bs.py"), "name": name, "mode": "continuation",
            "passed": True, "failures": [], "arm": arm, "seed": seed, "S": S, "E": E, "T": E, "end": E,
            "stop_reason": "end", "_dry_run": "SYNTHETIC check.json",
            "sources": {"run_meta.json": [str(mp), sha(mp)],
                        "policy_probe.npz": [pm["probe"]["file"], pm["probe"]["sha256"]]}})
        m = as_continuation(pm, arm, seed, SEEDS.index(seed), synthetic_e)
        m["ledger"] = {**m["ledger"], "file": str(led), "sha256": sha(led)}
        write(root / "reads" / name / "markers.json", m)
    print(f"built {root} ({len(QUEUE)} entries; synthetic loop row at E: {synthetic_e})")


build("parent_as_all", False)
build("synthetic_E", True)
