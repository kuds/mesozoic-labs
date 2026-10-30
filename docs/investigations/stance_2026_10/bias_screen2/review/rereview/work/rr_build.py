"""Re-reviewer's two synthetic roots for score_bs.py (no simulation, no training).  Writes only under OUT.
  rr_A: every entry COUNTED, pinned by a temp APPROVED inside the root (so the reading can be FINAL); varied primaries
        with one exact tie across arms; varied loop rows at E and CMD_L at E.
  rr_B: LOST (control_2511), STOPPED at 3,407,872 (critic_blind_tail3_2521), UNREADABLE (control_2531: one NaN row),
        COUNTED without its final pair (critic_blind_tail3_2501: D7), control_2501 without the evaluation at E (SEC-CMD
        MISSING); also pinned, so the only PROVISIONAL item should be the D7 one.
Base markers: <BS2>/validate/v3final/parent_s42/markers.json (as the adapter's make_dryrun.py)."""
import copy, hashlib, json, math, shutil, sys
from fractions import Fraction
from pathlib import Path

BS2 = Path(sys.argv[1]).resolve()
OUT = BS2 / "review" / "rereview" / "work" / "rr_roots"
TOOLS = BS2 / "tools"
PARENT = BS2 / "validate" / "v3final" / "parent_s42" / "markers.json"
S, E, ROLL = 2097152, 3538944, 8192
SEEDS = (2501, 2511, 2521, 2531)
ARMS = ("control", "critic_blind_tail3")
QUEUE = [(a, s) for s in SEEDS for a in ARMS]
B_S = Fraction(135930855, 1073741824)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8"); return path


def under(p):
    p = Path(p).resolve(); assert OUT.resolve() in p.parents, p; return p


def meta(arm, seed, crit, screen, appr, T, stop):
    blind = arm == "critic_blind_tail3"
    ev = {"n": (T - S) // ROLL - 1 if stop == "collapse" else 175, "buffer_gae_lambda": [0.95],
          "max_adv_err_vs_gae095": 3.8e-6, "min_dmu_actor": 0.166,
          "max_dv_predict_values": 0.0 if blind else 0.29, "max_dv_forward": 0.0 if blind else 0.29,
          "max_dv_evaluate_actions": 0.0 if blind else 0.29, "max_dv_mirrored": 0.0 if blind else 0.30,
          "min_dv_predict_values": 0.0 if blind else 0.0706,
          "min_buffer_values_vs_raw_value_mlp": 0.0246 if blind else 0.0, "state_problems": []}
    return {"schema": "bs-run/1", "name": f"{arm}_{seed}", "arm": arm, "seed": seed, "T": T, "stop_reason": stop,
            "status": "done", "_rr": "SYNTHETIC (re-review)", "arm_evidence": ev,
            "arm_applied": {"arm": arm, "mlp_extractor_class": "CriticBlindTail3MlpExtractor" if blind
                            else "MlpExtractor"}, "arm_state_end": "in force",
            "criteria_sha256": crit, "screen_sha256": screen, "approved_sha256": appr,
            "tool_sha256": {t: sha(TOOLS / t) for t in ("run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py",
                                                        "bs_hooks.py")}}


def set_primary(m, target, jitter, nan_row=None):
    """The 8 primary rows set so their exact mean is `target` (a float): rows target + jitter_i with sum(jitter)=0."""
    st = m["probe"]["steps"]
    col = m["probe"]["rollout_mu14_mean"]
    steps = [E - ROLL * k for k in range(8, 0, -1)]
    for j, s in enumerate(steps):
        col[st.index(s)] = target + jitter[j]
    if nan_row is not None:
        col[st.index(steps[nan_row])] = float("nan")


def build(rootname, spec):
    root = under(OUT / rootname)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    shutil.copyfile(BS2 / "SCREEN.json", root / "SCREEN.json")
    shutil.copyfile(BS2 / "criteria.txt", root / "criteria.txt")
    crit, screen = sha(root / "criteria.txt"), sha(root / "SCREEN.json")
    tools = ("run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py", "score_bs.py")
    write(root / "APPROVED", {"criteria_sha256": crit, "screen_sha256": screen,
                              "code_tree_digest": "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1",
                              "tool_sha256": {t: sha(TOOLS / t) for t in tools}, "approved_by": "rereview temp root",
                              "utc": "synthetic"})
    appr = sha(root / "APPROVED")
    pm = json.loads(PARENT.read_text(encoding="utf-8"))
    src_ledger = Path(pm["ledger"]["file"])
    lost = {}
    for i, (arm, seed) in enumerate(QUEUE):
        name = f"{arm}_{seed}"
        sp = spec[name]
        if sp.get("lost"):
            lost[name] = {"cause": "crashes", "detail": "synthetic", "by": "rr_build", "utc": "x"}
            continue
        T = sp.get("T", E)
        stop = "end" if T == E else "collapse"
        rd = root / "runs" / name
        mp = write(rd / "run_meta.json", meta(arm, seed, crit, screen, appr, T, stop))
        led = rd / "01_stance" / "eval_episodes.csv"
        led.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_ledger, led)
        write(rd / "check.json", {
            "schema": "bs-check/1", "tool_sha256": sha(TOOLS / "read_bs.py"), "name": name, "mode": "continuation",
            "passed": True, "failures": [], "arm": arm, "seed": seed, "S": S, "E": E, "T": T, "end": T,
            "stop_reason": stop, "_rr": "SYNTHETIC check.json",
            "sources": {"run_meta.json": [str(mp), sha(mp)],
                        "policy_probe.npz": [pm["probe"]["file"], pm["probe"]["sha256"]]}})
        m = copy.deepcopy(pm)
        m.update(kind="continuation", label=name, arm=arm, seed=seed, stop_reason=stop, T=T, _rr="SYNTHETIC")
        m.pop("reproduction", None)
        if T == E:
            set_primary(m, float(B_S) + sp["rise"], sp.get("jitter", [0.0] * 8), sp.get("nan_row"))
        else:   # STOPPED: probe rows S + 8,192 k for k < (T - S) / 8,192
            n = (T - S) // ROLL - 1
            for key, v in list(m["probe"].items()):
                if isinstance(v, list) and len(v) == 175:
                    m["probe"][key] = v[:n]
            m["probe"]["n_rows"] = n
            m["probe"]["T"] = T
        if sp.get("loopE", True) and T == E:
            row = copy.deepcopy(next(r for r in m["loop"] if r["steps"] == 3145728))
            row.update(steps=E, num_timesteps=E, _rr="SYNTHETIC loop row at E")
            row["OWN"]["P"].update(G=sp["G"], b_star=sp["b"], c0=1.0 - sp["G"] - sp["b"])
            m["loop"].append(row)
        if T == E:
            pe = m["ledger"]["direct"]["per_evaluation"]
            if sp.get("cmd") is None:
                pe.pop(str(E), None)
            else:
                pe[str(E)] = {**pe[str(E)], "cmd_l": sp["cmd"]}
        else:
            pe = m["ledger"]["direct"]["per_evaluation"]
            for k in [k for k in pe if int(k) > T]:
                pe.pop(k)
        if arm == "critic_blind_tail3":
            for row in m["loop"]:
                row["critic_form"] = "blind+raw"
                for d in row["critic_slope_obs6"].values():
                    d["effective"] = {**d["effective"], "dV_dq3_per_deg": 0.0}
        m["ledger"] = {**m["ledger"], "file": str(led), "sha256": sha(led)}
        write(root / "reads" / name / "markers.json", m)
    if lost:
        import importlib.util
        spc = importlib.util.spec_from_file_location("bsc", str(TOOLS / "bs_common.py"))
        bsc = importlib.util.module_from_spec(spc); spc.loader.exec_module(bsc)
        write(root / "lost.json", {"schema": bsc.LOST_SCHEMA, "lost": lost})
    print(f"built {root}")


J = [0.011, -0.007, 0.003, -0.005, 0.009, -0.004, 0.001, -0.008]   # sum 0
SPEC_A = {
    "control_2501": dict(rise=0.31, jitter=J, G=-0.40, b=1.10, cmd=0.71),
    "critic_blind_tail3_2501": dict(rise=0.05, jitter=J[::-1], G=-0.10, b=1.30, cmd=0.42),
    "control_2511": dict(rise=0.62, jitter=J, G=-0.80, b=0.95, cmd=0.88),
    "critic_blind_tail3_2511": dict(rise=0.31, jitter=J, G=-0.35, b=1.20, cmd=0.71),   # tie with control_2501 (primary, cmd)
    "control_2521": dict(rise=0.18, jitter=[0.0] * 8, G=-0.20, b=1.40, cmd=0.55),
    "critic_blind_tail3_2521": dict(rise=0.12, jitter=[0.0] * 8, G=0.05, b=1.50, cmd=0.50),
    "control_2531": dict(rise=0.47, jitter=J, G=-0.60, b=1.00, cmd=0.80),
    "critic_blind_tail3_2531": dict(rise=0.33, jitter=J, G=-0.30, b=1.05, cmd=0.66),
}
SPEC_B = {
    "control_2501": dict(rise=0.40, jitter=J, G=-0.40, b=1.10, cmd=None),
    "critic_blind_tail3_2501": dict(rise=0.02, jitter=J, G=None, b=None, loopE=False, cmd=0.30),
    "control_2511": dict(lost=True),
    "critic_blind_tail3_2511": dict(rise=0.10, jitter=J, G=0.10, b=1.6, cmd=0.35),
    "control_2521": dict(rise=0.25, jitter=J, G=-0.50, b=1.0, cmd=0.70),
    "critic_blind_tail3_2521": dict(T=3407872),
    "control_2531": dict(rise=0.55, jitter=J, G=-0.70, b=0.9, cmd=0.85, nan_row=3),
    "critic_blind_tail3_2531": dict(rise=0.08, jitter=J, G=0.20, b=1.7, cmd=0.33),
}
build("rr_A", SPEC_A)
build("rr_B", SPEC_B)
SPEC_C = {   # REDUCED; SEC-G and SEC-CMD at 1/70 (Bonferroni)
    "control_2501": dict(rise=0.50, jitter=J, G=-0.60, b=1.10, cmd=0.80),
    "critic_blind_tail3_2501": dict(rise=0.05, jitter=J, G=-0.10, b=1.00, cmd=0.40),
    "control_2511": dict(rise=0.60, jitter=J, G=-0.70, b=1.20, cmd=0.85),
    "critic_blind_tail3_2511": dict(rise=0.10, jitter=J, G=0.00, b=1.30, cmd=0.45),
    "control_2521": dict(rise=0.45, jitter=J, G=-0.50, b=0.90, cmd=0.75),
    "critic_blind_tail3_2521": dict(rise=0.02, jitter=J, G=0.10, b=1.15, cmd=0.35),
    "control_2531": dict(rise=0.70, jitter=J, G=-0.80, b=1.05, cmd=0.90),
    "critic_blind_tail3_2531": dict(rise=0.15, jitter=J, G=-0.20, b=0.95, cmd=0.50),
}
SPEC_D = {   # NOT INFORMATIVE (median rise 0.135) with filled p 1/70 (the line-1 addition)
    "control_2501": dict(rise=0.10, jitter=J, G=-0.6, b=1.1, cmd=0.3),
    "critic_blind_tail3_2501": dict(rise=0.00, jitter=J, G=-0.5, b=1.2, cmd=0.2),
    "control_2511": dict(rise=0.15, jitter=J, G=-0.6, b=1.1, cmd=0.3),
    "critic_blind_tail3_2511": dict(rise=0.01, jitter=J, G=-0.5, b=1.2, cmd=0.2),
    "control_2521": dict(rise=0.12, jitter=J, G=-0.6, b=1.1, cmd=0.3),
    "critic_blind_tail3_2521": dict(rise=0.02, jitter=J, G=-0.5, b=1.2, cmd=0.2),
    "control_2531": dict(rise=0.30, jitter=J, G=-0.6, b=1.1, cmd=0.3),
    "critic_blind_tail3_2531": dict(rise=0.03, jitter=J, G=-0.5, b=1.2, cmd=0.2),
}
build("rr_C", SPEC_C)
build("rr_D", SPEC_D)
