"""The foot study (decision D5): each foot variant (build_feet60.py) standing, crawling and trotting, under the model's
contact and the stiffer impratio-10 setting. Simulation only. Rules: PREREG_FEET60.md (development runs use other
seeds and are not judged).

- Stand: check_stand.run (the 60 cm standing check's code path, model swapped) and check_stand60.judge.
- Crawl: crawl.run with crawl_judged60.PARAMS60 (the 60 cm crawl, no tuning) and crawl.judge (C1-C6). C5 measures the
  ankle pin's travel, which a rolling foot moves without sliding, so the study adds C5c: each foot's sliding at its
  contacts (the sole's velocity at the mean contact point, horizontal, integrated) between 0.10 s after touchdown and
  lift-off, at most 10 mm, the same window and limit as C5.
- Trot: the frozen model-based trot (walk/walk60.py) on the variant, the walk's four judged conditions (friction 1.0
  and 0.4, each under both contact settings), 10 ms delay, noise and randomisation on, rules W1-W4 of
  walk/PREREG_WALK60.md, with slip measured at the contacts (gait_metrics.record_episode slip_at_contact=True).

Usage: python feet60_study.py --models feet60 --out results/feet60_<date> [--crawl-seeds 1,2,3,4]
       [--stand-seeds 1,2] [--walk-seeds 1,2] [--variants pitch,roll_k2.032,...] [--workers 4]
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WALK = HERE.parent / "walk"
for p in (HERE, WALK):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
WALK_CONDITIONS = {"J1": (1.0, None), "J2": (0.4, None), "J3": (1.0, 10.0), "J4": (0.4, 10.0)}
SLIDE_LIMIT = 0.010


def model_file(models: Path, variant: str, impratio: bool, tmp: Path) -> Path:
    path = models / f"trike60_foot_{variant}.xml"
    if not impratio:
        return path
    text = path.read_text()
    assert text.count("<option ") == 1 and "impratio" not in text
    out = tmp / f"trike60_foot_{variant}_impratio10.xml"
    out.write_text(text.replace("<option ", '<option impratio="10" ', 1))
    return out


# ------------------------------------------------------------------------------------------------ stand
class _View:
    def __init__(self, path):
        self.path = path

    def __truediv__(self, name):
        return self.path


def stand_job(args):
    models, variant, imp, seed = args
    import check_stand as cs
    import check_stand60 as c60
    with tempfile.TemporaryDirectory() as td:
        orig = cs.GEN
        try:
            cs.GEN = _View(model_file(Path(models), variant, imp, Path(td)))
            r = cs.run("60cm", "closed", seed)
        finally:
            cs.GEN = orig
    ledger = json.loads((HERE / "generated" / "mass_ledger_trike60_C_closed.json").read_text())
    rules, derived = c60.judge(r, ledger)
    rules = {k: bool(v) for k, v in rules.items() if not k.endswith("_flag_ok")}
    return ("stand", variant, imp, seed, {"rules": rules, "derived": _jsonable(derived)})


# ------------------------------------------------------------------------------------------------ crawl
class Slider:
    """After every physics step: each sole's horizontal sliding speed at its contacts, integrated (m)."""

    def __init__(self, model):
        import mujoco
        self.mj = mujoco
        self.soles = [model.geom(f"{c}_sole").id for c in CORNERS]
        self.cum = np.zeros(4)
        self.t, self.hist = [], []
        self.vel6 = np.zeros(6)

    def __call__(self, m, d):
        dt = m.opt.timestep
        if d.ncon:
            gg = d.contact.geom[:d.ncon]
            for i, g in enumerate(self.soles):
                hit = (gg[:, 0] == g) | (gg[:, 1] == g)
                if hit.any():
                    self.mj.mj_objectVelocity(m, d, self.mj.mjtObj.mjOBJ_GEOM, g, self.vel6, 0)
                    r = d.contact.pos[:d.ncon][hit].mean(axis=0) - d.geom_xpos[g]
                    v = self.vel6[3:] + np.cross(self.vel6[:3], r)
                    self.cum[i] += math.hypot(v[0], v[1]) * dt
        self.t.append(d.time)
        self.hist.append(self.cum.copy())


def crawl_job(args):
    models, variant, imp, seed = args
    import mujoco
    import crawl
    import crawl_judged60 as cj
    with tempfile.TemporaryDirectory() as td:
        path = model_file(Path(models), variant, imp, Path(td))
        slider = Slider(mujoco.MjModel.from_xml_path(str(path)))
        r = crawl.run(path, seed, cj.PARAMS60, step_hook=slider)
    rules, info = crawl.judge(r)
    t, hist = np.array(slider.t), np.array(slider.hist)
    windows = {i: [(0.0, None)] for i in range(4)}
    for ph in r["phases"]:
        L = ph["leg_index"]
        windows[L][-1] = (windows[L][-1][0], ph["swing_start"])
        windows[L].append((ph["swing_end"], None))
    slides = []
    for i in range(4):
        windows[i][-1] = (windows[i][-1][0], t[-1])
        for td_, lo in windows[i]:
            a, b = np.searchsorted(t, td_ + 0.10), np.searchsorted(t, lo, side="right") - 1
            if b > a:
                slides.append(float(hist[b, i] - hist[a, i]))
    rules = {k: bool(v) for k, v in rules.items()}
    rules["C5c"] = max(slides) <= SLIDE_LIMIT
    out = {"rules": rules, "slide_max_m": max(slides), "slide_median_m": float(np.median(slides)),
           "info": _jsonable(info)}
    return ("crawl", variant, imp, seed, out)


# ------------------------------------------------------------------------------------------------ trot
def walk_job(args):
    models, variant, cond, seed = args
    import gait_metrics
    import walk60
    fr, imp = WALK_CONDITIONS[cond]
    env = walk60.TrikeEnv(walk60.plant_config(model_path=str(Path(models) / f"trike60_foot_{variant}.xml"),
                                              friction_range=(fr, fr)))
    if imp is not None:
        env.model.opt.impratio = imp
    ctrl = walk60.TrotController(env, walk60.WalkParams())
    log = gait_metrics.record_episode(env, ctrl.act, seed=seed, command=0.25, delay_steps=10, slip_at_contact=True)
    ms = gait_metrics.measures(log, 0.25)
    out = {"fell": bool(log["fell"]), "termination": log["termination"], "speed_mps": ms["forward_speed_mps"],
           "heat_worst": max(d["rms_over_continuous"] for d in ms["drives"].values()),
           "slip_ratio": ms["slip_ratio"], "touchdowns_per_s": ms["touchdowns_per_s"], "power_w": ms["power_w"],
           "at_limit_nominal_worst": max(d["at_limit_nominal"] for d in ms["drives"].values())}
    return ("walk", variant, cond, seed, out)


def walk_rules(eps):
    sp = [e["speed_mps"] for e in eps]
    feet = eps[0]["slip_ratio"].keys()
    slip = {c: float(np.mean([e["slip_ratio"][c] for e in eps if e["slip_ratio"][c] is not None] or [np.nan]))
            for c in feet}
    return {"W1": not any(e["fell"] for e in eps),
            "W2": bool(np.mean(sp) >= 0.225 and min(sp) >= 0.20),
            "W3": bool(max(e["heat_worst"] for e in eps) < 1.0),
            "W4": bool(np.nanmax(list(slip.values())) <= 0.25)}


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(HERE / "feet60"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="pitch,roll_k2.032,capsule_r16,ball_r30")
    ap.add_argument("--stand-seeds", default="1,2")
    ap.add_argument("--crawl-seeds", default="1,2,3,4")
    ap.add_argument("--walk-seeds", default="1,2")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    seeds = lambda s: [int(x) for x in s.split(",") if x]  # noqa: E731
    V = a.variants.split(",")
    jobs = ([(stand_job, (a.models, v, imp, s)) for v in V for imp in (False, True) for s in seeds(a.stand_seeds)]
            + [(walk_job, (a.models, v, c, s)) for v in V for c in WALK_CONDITIONS for s in seeds(a.walk_seeds)]
            + [(crawl_job, (a.models, v, imp, s)) for v in V for imp in (False, True) for s in seeds(a.crawl_seeds)])
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = [pool.apply_async(f, (x,)) for f, x in jobs]
        rows = [r.get() for r in res]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for v in V:
        s = summary[v] = {}
        for imp in (False, True):
            key = "impratio10" if imp else "model"
            st = [r[4] for r in rows if r[0] == "stand" and r[1] == v and r[2] == imp]
            cr = [r[4] for r in rows if r[0] == "crawl" and r[1] == v and r[2] == imp]
            s[f"stand_{key}"] = {k: all(x["rules"][k] for x in st) for k in st[0]["rules"]} if st else None
            s[f"stand_{key}_roll_max_nm"] = max(x["derived"]["roll_max_nm"] for x in st) if st else None
            s[f"crawl_{key}"] = {k: sum(x["rules"][k] for x in cr) for k in cr[0]["rules"]} if cr else None
            s[f"crawl_{key}_n"] = len(cr)
            s[f"crawl_{key}_slide_max_mm"] = 1000 * max(x["slide_max_m"] for x in cr) if cr else None
        for c in WALK_CONDITIONS:
            eps = [r[4] for r in rows if r[0] == "walk" and r[1] == v and r[2] == c]
            if eps:
                s[f"walk_{c}"] = {"rules": walk_rules(eps), "speeds": [round(e["speed_mps"], 3) for e in eps],
                                  "heat_worst": max(e["heat_worst"] for e in eps),
                                  "slip_worst": max(x for e in eps for x in e["slip_ratio"].values() if x is not None)
                                  if not any(e["fell"] for e in eps) else None}
    res_out = {"simulation_only": True, "tool": "sim/feet60_study.py", "args": vars(a), "wall_s": time.time() - t0,
               "summary": summary, "rows": rows}
    (out / "results.json").write_text(json.dumps(_jsonable(res_out), indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
