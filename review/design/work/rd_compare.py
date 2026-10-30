"""Design reviewer: random synthetic BS2 roots scored by score_bs.score (the pinned candidate, or a mutant copy) and by
rd_scorer.score (independent, from the criteria text); every reading field compared.

    PY -I -B rd_compare.py <n_roots> <seed> [<score_bs.py path>]   -> prints a summary; exit 0 if all agree

Roots are written under <this folder>/roots/ (removed and rewritten per root).  Nothing outside this folder is written.
The code-copy measurement (score_bs.code_seal_items) is replaced by a stub returning [] (it only feeds PROVISIONAL).
"""
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TOOLS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/"
             "bias_screen2/tools")
sys.path.insert(0, str(HERE))
import rd_scorer as RD  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load("bs_common", TOOLS / "bs_common.py")
R = load("read_bs", TOOLS / "read_bs.py")
SB_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else TOOLS / "score_bs.py"
SB = load("score_bs_under_test", SB_PATH)
SB.code_seal_items = lambda d: []
assert (C.S, C.E) == (RD.S, RD.E)

S, E, ROLL = RD.S, RD.E, RD.ROLL
B_S = float(RD.B_S)
FAKE_SHA = "cd" * 32


def w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_root(root, rng):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    w(root / "SCREEN.json", {"schema": C.SCREEN_SCHEMA, "intervention": RD.BLIND,
                             "queue": [list(q) for q in RD.QUEUE]})
    scheme = rng.choice(["tie", "cont", "gate", "sep", "sep", "cont2"])
    allow_undecided = rng.random() < 0.08
    p_odd = rng.choice([0.0, 0.0, 0.1, 0.25])
    lost = {}
    vals = {}
    # the arm values (the primary's intended level)
    for arm in (RD.CTRL, RD.BLIND):
        for i, seed in enumerate(RD.SEEDS):
            if scheme == "tie":
                v = B_S + rng.randint(-3, 40) / 64
            elif scheme == "cont":
                v = B_S + rng.gauss(0.3 if arm == RD.CTRL else 0.2, 0.15)
            elif scheme == "cont2":
                v = B_S + rng.gauss(0.25, 0.12)
            elif scheme == "gate":
                v = B_S + 0.2 + rng.choice([-1, 1]) * rng.choice([1e-9, 1e-6, 1e-3, 0.02])
            else:  # sep: control high, blind low, with a random swap or near-ties
                v = B_S + (0.5 + 0.01 * i if arm == RD.CTRL else 0.1 + 0.01 * i)
                if rng.random() < 0.3:
                    v += rng.choice([0.35, -0.35, 0.0])
            vals[(arm, seed)] = v
    for arm, seed in RD.QUEUE:
        nm = RD.name(arm, seed)
        u = rng.random()
        if allow_undecided and u < 0.15:
            kind = "UNDECIDED"
        elif u < p_odd * 0.3:
            kind = "LOST"
        elif u < p_odd * 0.6:
            kind = "STOPPED"
        elif u < p_odd:
            kind = "UNREADABLE"
        else:
            kind = "COUNTED"
        if kind == "LOST":
            lost[nm] = {"cause": "crashes", "detail": "3 own crashes"}
            continue
        if kind == "UNDECIDED":
            continue
        T = E if kind != "STOPPED" else rng.choice([S + 131_072 * k for k in range(4, 11)])
        stop = "collapse" if kind == "STOPPED" else "end"
        rd = root / "runs" / nm
        meta = w(rd / "run_meta.json", {"name": nm, "arm": arm, "seed": seed})
        w(rd / "check.json", {"schema": R.CHECK_SCHEMA, "tool_sha256": "00" * 32, "name": nm, "mode": "continuation",
                              "passed": True, "failures": [], "arm": arm, "seed": seed, "S": S, "E": E, "T": T,
                              "end": E, "stop_reason": stop,
                              "sources": {"run_meta.json": [str(meta), sha(meta)],
                                          "policy_probe.npz": ["p", FAKE_SHA]}})
        steps = [S + ROLL * k for k in range(1, (T - S) // ROLL)]
        v = vals[(arm, seed)]
        mu = []
        for s in steps:
            if s in RD.PRIMARY_STEPS:
                mu.append(v + rng.choice([0.0, rng.uniform(-0.05, 0.05), rng.randint(-4, 4) / 256]))
            else:
                mu.append(rng.uniform(-1.0, 2.0))          # rows outside the primary: unrelated values
        if kind == "UNREADABLE":
            mu[steps.index(rng.choice(RD.PRIMARY_STEPS))] = float("nan")
        n = len(steps)
        probe = {"file": "p", "sha256": FAKE_SHA, "label": nm, "S": S, "T": T, "n_rows": n, "steps": steps,
                 "rollout_mu14_mean": mu, "tail3_statue_settled_mean": [0.0] * n, "trailing8_mean": [0.0] * n,
                 "trailing8_first_above": {}, "trailing8_max": 0.0, "std14": [0.2] * n, "obs6_mean_deg": [0.0] * n}
        blind = arm == RD.BLIND
        loop = []
        pair_steps = [S, *[x for x in C.SAVED_STEPS if x < T], T]
        drop_E = rng.random() < 0.06
        for s in pair_steps:
            if s == T and drop_E:
                continue
            if s == T:
                g = rng.choice([rng.randint(-8, 8) / 8, rng.gauss(0, 0.6), float("nan") if rng.random() < 0.05
                                else rng.gauss(0, 0.6)])
                b = rng.choice([rng.randint(0, 16) / 8, rng.gauss(1.0, 0.5), float("nan") if rng.random() < 0.05
                                else rng.gauss(1.0, 0.5)])
            else:
                g, b = rng.gauss(-0.5, 0.3), rng.gauss(1.3, 0.3)
            crit = {k: {"raw": {"dV_dq3_per_deg": 0.01}, "effective": {"dV_dq3_per_deg": 0.0 if blind else 0.01}}
                    for k in ("REF", "OWN", "ONPOLICY")}
            loop.append({"steps": s, "num_timesteps": s, "critic_form": "blind+raw" if blind else "plain",
                         "OWN": {"P": {"G": g if finite(g) else None, "c0": 0.1, "b_star": b if finite(b) else None},
                                 "L": {"G": 9.9}, "b_star_L": -9.9},
                         "REF": {"P": {"G": 9.9}}, "critic_slope_obs6": crit})
        per = {}
        for t in range(S + 131_072, T + 1, 131_072):
            if t == E and rng.random() < 0.06:
                continue
            cm = rng.choice([rng.randint(0, 16) / 16, rng.uniform(0, 1)])
            if t == E and rng.random() < 0.04:
                cm = None
            per[str(t)] = {"n": 10, "cmd_l": cm, "cmd_l_max": 1.0 - (cm or 0)}
        direct = {"per_evaluation": per, "firsts": {}, "cmd_l_max": 0.9, "cmd_l_last": -0.9}
        w(root / "reads" / nm / "markers.json", {
            "schema": R.MARKERS_SCHEMA, "tool_sha256": "00" * 32, "label": nm, "kind": "continuation", "arm": arm,
            "seed": seed, "S": S, "T": T, "stop_reason": stop, "probe": probe,
            "ledger": {"file": str(root / "no_ledger.csv"), "sha256": "00" * 32, "direct": direct, "m5": {}},
            "loop": loop})
    if lost:
        w(root / "lost.json", {"schema": C.LOST_SCHEMA, "lost": lost})
    return scheme


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def cmp(root):
    """[(field, theirs, mine)] disagreements."""
    try:
        res, lines, code = SB.score(root)
    except Exception as exc:  # noqa: BLE001 - a crash is a disagreement
        return [("exception", repr(exc)[:200], None)], None
    mine = RD.score(root)
    bad = []
    if mine["status"] == "incomplete":
        if res.get("status") != "incomplete":
            bad.append(("status", res.get("status"), "incomplete"))
        if any(k in res for k in ("line", "p_filled")):
            bad.append(("outcome printed while incomplete", True, False))
        return bad, mine
    if res.get("status") == "incomplete":
        return [("status", "incomplete", "complete")], mine

    def eq(field, theirs, ours):
        if theirs != ours:
            bad.append((field, theirs, ours))
    eq("line", res["line"], mine["line"])
    eq("additions", res["additions"], mine["additions"])
    eq("p_filled", (res["p_filled"]["count"], res["p_filled"]["total"]), mine["p_filled"])
    pc = res["p_as_counted"]
    eq("p_as_counted", None if pc is None else (pc["count"], pc["total"]), mine["p_as_counted"])
    eq("informative", res["informative"], mine["informative"])
    eq("control_rise", res["control_rise"], None if mine["control_rise"] is None else float(mine["control_rise"]))
    eq("control_median", res["control_median"], None if mine["control_median"] is None
       else float(mine["control_median"]))
    eq("n_missing", res["n_missing"], mine["missing"][RD.CTRL] + mine["missing"][RD.BLIND])
    for c in res["classes"]:
        eq("primary " + c["name"], c["primary_exact"], None if mine["primaries"][c["name"]] is None
           else str(mine["primaries"][c["name"]]))
    for s in res["secondaries"]:
        m = mine["secondaries"][s["key"]]
        eq(s["key"] + " p_filled", (s["p_filled"]["count"], s["p_filled"]["total"]), m["p_filled"])
        spc = s["p_as_counted"]
        eq(s["key"] + " p_as_counted", None if spc is None else (spc["count"], spc["total"]), m["p_as_counted"])
        eq(s["key"] + " verdict", s["verdict"], m["verdict"])
        eq(s["key"] + " notes", s["notes"], m["notes"])
        eq(s["key"] + " n_read", s["n_read"], m["n_read"])
    # the line text verbatim for line 1 (the texts of lines 2 / 3 are compared with criteria.txt elsewhere)
    if mine["line"] == 1:
        eq("line 1 text", res["line_text"], RD.LINE1)
    return bad, mine


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    tally = {"roots": 0, "disagree": 0, "lines": {1: 0, 2: 0, 3: 0}, "incomplete": 0, "add1": 0, "add3": 0,
             "sec_supported": 0, "sec_bonf": 0, "missing_any": 0, "schemes": {}}
    first_bad = []
    root = HERE / "roots" / f"r{seed}"
    for k in range(n):
        scheme = make_root(root, rng)
        bad, mine = cmp(root)
        tally["roots"] += 1
        tally["schemes"][scheme] = tally["schemes"].get(scheme, 0) + 1
        if mine and mine["status"] == "complete":
            tally["lines"][mine["line"]] += 1
            tally["add1"] += RD.ADD1 in mine["additions"]
            tally["add3"] += RD.ADD3 in mine["additions"]
            tally["sec_supported"] += sum(v["supported"] for v in mine["secondaries"].values())
            tally["sec_bonf"] += sum(RD.BONF in v["notes"] for v in mine["secondaries"].values())
            tally["missing_any"] += (mine["missing"][RD.CTRL] + mine["missing"][RD.BLIND]) > 0
        elif mine:
            tally["incomplete"] += 1
        if bad:
            tally["disagree"] += 1
            if len(first_bad) < 3:
                first_bad.append((k, scheme, bad[:4]))
    shutil.rmtree(root)
    print("scorer under test:", SB_PATH, hashlib.sha256(SB_PATH.read_bytes()).hexdigest())
    print(json.dumps(tally, default=str))
    for fb in first_bad:
        print("DISAGREE", fb)
    return 0 if tally["disagree"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
