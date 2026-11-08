"""Gait tuning for the animal-like walk (Froude 0.05): lower the binding drive ratio without changing a drive.

The Froude 0.05 walk of the behavior-envelope screen (``envelope.FAST``: stride 0.80 H, duty 0.56, lift 0.11 H,
crouch 0.03 H) sets the binding limit on every body chosen for verification (DECISION_LOG P16): the knee's speed,
0.93-0.95 of its planning limit (the 1.15 speed margin of ``sizing.SPEED_MARGIN`` included).  The owner asked for
gait tuning before any motor change.  This module searches the walk's free parameters at the same Froude number
for the lowest worst drive ratio, with each body's drives fixed (its sized record; nothing is re-sized), subject to:

* the swing toe clears a hard floor by at least 0.04 H over the middle of the swing (BEHAVIOR_SPEC.md section 1;
  ``carpet.toe_catch``, hard floor),
* a duty factor of at least 0.55 (walking, BEHAVIOR_SPEC.md section 1),
* IK holds and the reference centre of pressure stays on the pad in single support (``envelope.fast_walk``'s rule:
  more than 10 mm outside makes the drive loads meaningless),
* for the finalists, the round-two walking clearance gates (``clearance.screen``, gates "r2", the tuned walk
  standing in for both reference walks).

Search: a grid over swing lift, duty, stride and crouch, then a coordinate pattern search from the best feasible
grid point.  The walk generator, inverse dynamics and drive ratios are the envelope screen's
(``turning.generate`` at zero yaw rate, ``turning.inverse_dynamics``, ``sizing._motor_space``,
``envelope._ratios``).  Reported alongside: lateral sway (peak to peak, / H) and the peak horizontal acceleration
of the centre-of-mass reference, against BEHAVIOR_SPEC's 0.12 H and 1.5 m/s^2.

This is a reference-walk (Stage 0) result: it shows the walk a controller could aim for, not that a controller
achieves it (Stage 1 does that).

Run from hardware/trex_form_study:
    python -m designlab.gaittune --body 60-N-T-10 --out round2/gait_tune
    python -m designlab.gaittune --report --out round2/gait_tune
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import carpet, clearance, envelope, gait, round2, sizing, turning

BODIES = {"60-N-T-10": "narrow_carpet_60", "60-N-H-10": "narrow_carpet_60", "75-M-H-10": "feet_75"}
GRID = dict(lift_frac=(0.08, 0.09, 0.10, 0.11), duty=(0.55, 0.57, 0.60), stride_rel=(0.70, 0.75, 0.80, 0.85),
            crouch=(0.02, 0.03, 0.05))
BOUNDS = dict(lift_frac=(0.08, 0.12), duty=(0.55, 0.65), stride_rel=(0.60, 0.95), crouch=(0.01, 0.07))
STEPS = dict(lift_frac=0.005, duty=0.01, stride_rel=0.025, crouch=0.01)
MIN_DUTY = 0.55
COP_RULE_MM = -10.0
SWAY_LIMIT_H = 0.12
ACCEL_LIMIT = envelope.STOP_DECEL  # 1.5 m/s^2, BEHAVIOR_SPEC section 1
N_FINALISTS = 6
PATTERN_EVALS = 36


class Body:
    """One chosen body, built once per process from its sized record (the same build as ``refine``)."""

    def __init__(self, cid: str, scratch: Path, r2: Path = Path("round2")):
        from .pipeline import build_candidate

        gait.set_accuracy(round2.GAIT_ACCURACY)
        self.cid, self.case = cid, BODIES[cid]
        self.rec = json.loads((r2 / "refine" / self.case / "candidates" / f"{cid}.json").read_text())
        scratch.mkdir(parents=True, exist_ok=True)
        res = build_candidate(round2.candidate(cid, tuple(tuple(o) for o in self.rec.get("overrides", ()))), scratch,
                              selection=dict(self.rec["selection"]), B=self.rec["B"], write=False)
        self.M, self.q0, self.design = res["model"], res["q_home"], res["design"]
        self.H, self.layout, self.bus = self.design.sk.H, self.design.cand.leg_layout, self.rec["selection"]["bus"]
        self.xml = str(scratch / f"{cid}.xml")
        self.hard_target = carpet.screening_values()["swing_targets"]["hard_frac_H"] * self.H

    def evaluate(self, spec: gait.GaitSpec, full: bool = False) -> dict:
        t0 = time.time()
        tr = turning.generate(self.M, self.q0, self.H, spec, yaw_rate=0.0)
        idr = turning.inverse_dynamics(self.M, tr)
        single = [k for k, ph in enumerate(tr["phases"]) if len(ph) == 1]
        cm = np.asarray(idr["cop_margin"], float)
        cop_min = float(np.nanmin(cm[single])) if single and np.isfinite(cm[single]).any() else float("nan")
        ms = sizing._motor_space(self.M.names, idr["tau"], idr["omega"], self.layout)
        r = envelope._ratios(self.design, self.bus, ms)
        worst = max(r.items(), key=lambda kv: kv[1]["worst"])
        toe = carpet.toe_catch(self.M, tr, self.H)
        toe_mm = float(toe["hard_floor"]["mid_swing_min_mm"])
        com = np.asarray(tr["com_ref"], float)
        dt = float(tr["dt"])
        acc = np.gradient(np.gradient(com[:, :2], dt, axis=0), dt, axis=0)
        acc_h = np.linalg.norm(acc, axis=1)
        info = tr["info"]
        ik = bool(tr.get("ik_ok", True))
        out = dict(spec={k: float(getattr(spec, k)) for k in ("froude", "lift_frac", "duty", "stride_rel", "crouch",
                                                               "sway_gain")},
                   speed_ms=float(info["speed"]), cadence_steps_s=float(info["cadence_hz"]), stride_m=float(info["stride"]),
                   ik_ok=ik, min_cop_margin_single_mm=1000 * cop_min,
                   cop_on_pad=bool(np.isfinite(cop_min) and 1000 * cop_min > COP_RULE_MM),
                   toe_mid_swing_mm=toe_mm, toe_target_mm=1000 * self.hard_target,
                   toe_ok=bool(toe_mm >= 1000 * self.hard_target - 1e-2), duty_ok=bool(spec.duty >= MIN_DUTY - 1e-9),
                   sway_pp_H=float(2 * info["sway_amp"] / self.H),
                   com_accel_peak_ms2=float(np.max(acc_h)),
                   ratios={g: {k: float(v[k]) for k in ("cont_ratio", "peak_ratio", "speed_ratio", "worst")}
                           for g, v in r.items()},
                   worst_group=worst[0], worst_ratio=float(worst[1]["worst"]))
        out["feasible"] = bool(out["ik_ok"] and out["cop_on_pad"] and out["toe_ok"] and out["duty_ok"])
        if full:
            cl = clearance.screen(self.xml, self.q0, {"nominal": tr, "brisk": tr}, H=self.H, gates="r2")["summary"]
            mg = cl.get("joint_range_margins") or {}
            tight = min(((j, v) for j, r_ in mg.items() for v in (r_.get("margin_min"), r_.get("margin_max"))
                         if v is not None), key=lambda t: t[1], default=(None, None))
            out["clearance_pass"] = cl["pass"]
            out["walking_gates"] = all(cl["pass"].get(k) is True for k in round2.WALK_GATES)
            out["tightest_joint"] = list(tight)
            out["feasible_full"] = bool(out["feasible"] and out["walking_gates"])
        out["seconds"] = time.time() - t0
        return out


def _objective(e: dict) -> float:
    if e["feasible"]:
        return e["worst_ratio"]
    pen = 0.0
    if not e["ik_ok"]:
        pen += 1.0
    if not e["cop_on_pad"]:
        pen += 0.5 + max(0.0, (COP_RULE_MM - e["min_cop_margin_single_mm"]) / 100.0)
    if not e["toe_ok"]:
        pen += 0.5 + max(0.0, (e["toe_target_mm"] - e["toe_mid_swing_mm"]) / 10.0)
    return 1.0 + e["worst_ratio"] + pen


def _spec(base: gait.GaitSpec, x: dict) -> gait.GaitSpec:
    return replace(base, **{k: float(v) for k, v in x.items()})


def tune(cid: str, out: Path, scratch: Path, log=print) -> dict:
    """Grid, finalists with the clearance gates, then a pattern search from the best feasible point."""
    body = Body(cid, scratch / cid)
    base = envelope.FAST
    rec = dict(id=cid, case=body.case, H_m=body.H, bus=body.bus, drives=body.rec["selection"].get("joints"),
               baseline=body.evaluate(base, full=True), grid=[], finalists=[], pattern=[])
    log(cid, "baseline", rec["baseline"]["worst_group"], round(rec["baseline"]["worst_ratio"], 4))
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        x = dict(zip(keys, vals, strict=True))
        e = body.evaluate(_spec(base, x))
        rec["grid"].append(e)
        log(cid, "grid", x, e["worst_group"], round(e["worst_ratio"], 4), "feasible" if e["feasible"] else "infeasible")
    feas = sorted((e for e in rec["grid"] if e["feasible"]), key=lambda e: e["worst_ratio"])
    for e in feas[:N_FINALISTS]:
        x = {k: e["spec"][k] for k in keys}
        f = body.evaluate(_spec(base, x), full=True)
        rec["finalists"].append(f)
        log(cid, "finalist", x, round(f["worst_ratio"], 4), "gates" if f["walking_gates"] else "GATES FAIL", f["tightest_joint"])
    ok = [f for f in rec["finalists"] if f["feasible_full"]]
    if not ok:
        rec["chosen"] = None
        (out / f"{cid}.json").write_text(json.dumps(rec, indent=1))
        return rec
    best = min(ok, key=lambda f: f["worst_ratio"])
    x = {k: best["spec"][k] for k in keys}
    fx = _objective(best)
    steps = dict(STEPS)
    evals = 0
    for _ in range(3):  # step halvings
        improved = True
        while improved and evals < PATTERN_EVALS:
            improved = False
            for k in keys:
                for sgn in (-1.0, 1.0):
                    y = dict(x)
                    y[k] = float(np.clip(round(x[k] + sgn * steps[k], 4), *BOUNDS[k]))
                    if y[k] == x[k]:
                        continue
                    e = body.evaluate(_spec(base, y))
                    evals += 1
                    rec["pattern"].append(e)
                    fy = _objective(e)
                    log(cid, "pattern", y, round(e["worst_ratio"], 4), "feasible" if e["feasible"] else "infeasible")
                    if fy < fx - 1e-4:
                        x, fx, improved = y, fy, True
        steps = {k: v / 2 for k, v in steps.items()}
    chosen = body.evaluate(_spec(base, x), full=True)
    if not chosen["feasible_full"]:  # the pattern search ignores the clearance gates; fall back to the best finalist
        chosen = best
    rec["chosen"] = chosen
    (out / f"{cid}.json").write_text(json.dumps(rec, indent=1))
    log(cid, "chosen", chosen["spec"], chosen["worst_group"], round(chosen["worst_ratio"], 4))
    return rec


GROUP_LABEL = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
               "ankle_a": "ankle A", "ankle_b": "ankle B", "neck": "neck", "tail": "tail"}


def report(out: Path) -> str:
    rows = []
    for cid in BODIES:
        p = out / f"{cid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        b, c = d["baseline"], d.get("chosen")
        if not c:
            rows.append(f"| `{cid}` | no feasible tuned walk | | | | | | |")
            continue
        s = c["spec"]
        rows.append(
            f"| `{cid}` | lift {s['lift_frac']:.3f} H, duty {s['duty']:.2f}, stride {s['stride_rel']:.3f} H, crouch "
            f"{s['crouch']:.3f} H | {GROUP_LABEL[b['worst_group']]} {b['worst_ratio']:.3f} → "
            f"{GROUP_LABEL[c['worst_group']]} {c['worst_ratio']:.3f} | knee speed {b['ratios']['knee']['speed_ratio']:.3f} → "
            f"{c['ratios']['knee']['speed_ratio']:.3f} | {b['toe_mid_swing_mm']:.1f} → {c['toe_mid_swing_mm']:.1f} mm "
            f"(≥ {c['toe_target_mm']:.1f}) | {b['min_cop_margin_single_mm']:.1f} → {c['min_cop_margin_single_mm']:.1f} mm | "
            f"{b['cadence_steps_s']:.2f} → {c['cadence_steps_s']:.2f} | {b['sway_pp_H']:.3f} → {c['sway_pp_H']:.3f} H; "
            f"{b['com_accel_peak_ms2']:.2f} → {c['com_accel_peak_ms2']:.2f} m/s² |")
    md = ["# Gait tuning for the animal-like walk (Froude 0.05)", "",
          "Generated by `python -m designlab.gaittune` (method: module docstring). Each body keeps its drives; only the "
          "Froude 0.05 reference walk changes. Baseline: `envelope.FAST` (lift 0.11 H, duty 0.56, stride 0.80 H, "
          "crouch 0.03 H).", "",
          "| Body | Tuned walk | Worst drive ratio | Knee speed ratio | Toe clearance, hard floor | CoP margin (single support) | Cadence, steps/s | Sway (p-p) and peak CoM acceleration |",
          "|---|---|---|---|---|---|---|---|", *rows, "",
          f"Ratios include the screens' margins (speed ×{sizing.SPEED_MARGIN}, peak ×{sizing.PEAK_MARGIN}). Spec limits: "
          f"sway {SWAY_LIMIT_H} H peak to peak, acceleration {ACCEL_LIMIT} m/s². A reference walk, not a controller "
          "result."]
    text = "\n".join(md) + "\n"
    (out / "GAIT_TUNE.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--body", choices=sorted(BODIES))
    ap.add_argument("--out", default="round2/gait_tune")
    ap.add_argument("--scratch", default="round2/gait_tune/models")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.body:
        tune(a.body, out, Path(a.scratch), log=lambda *s: print(*s, flush=True))
    if a.report or not a.body:
        print(report(out))


if __name__ == "__main__":
    main()
