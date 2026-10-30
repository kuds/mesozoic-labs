"""Static (network-only) decomposition of the loop gain into a normaliser part and a weight part, on stored states.

PYTHONPATH=<policies/code_copy>:<this folder> python -B analyze_static.py > static.txt   (writes static.json)

No simulation and no training: it reads ckpts/episodes.jsonl and ckpts/states/<label>_3042.npy (run_ckpts.py, seed
3042, the observations at steps 500..1000) and stats.json (read_stats.py), loads each checkpoint's network (the
harness's loader) and evaluates the verifier's tail_3 angle sweep under different statistics on FIXED states, so only
(weights, statistics) vary:
  G(W, N | X)  = loop gain of weights W normalised with statistics N, on states X (median over the 6 states)
For every checkpoint c of every continuation (and the start S):
  own          G(W_c, N_c | X_c)          (= the episode's own sweep; checked)
  sN_full      G(W_c, N_S | X_c)          all 63 dims from the start
  sN_tail      G(W_c, N_c[tail<-S] | X_c) the 8 tail dims from the start
  sN_t3q       G(W_c, N_c[6<-S] | X_c)    tail_3 angle only from the start
  Jz           d cmd / d z_t3 = slope / scale (z = normalised tail_3 angle), own statistics
  startW_cN    G(W_S, N_c | X_c), G(W_S, N_c[tail<-c] ...) i.e. the start's network with c's statistics
  startW_sN    G(W_S, N_S | X_c)          the start's network with its own statistics on c's states
and the 2x2 decomposition of G(W_c,N_c) - G(W_S,N_S) on the common states X_c and on the start's states X_S.
Regression across checkpoints: gain on the tail_3 angle scale (s42 and s80 separately), and on log scale.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from probe_common import NAMES, SE, START_RUN, T3Q, TAIL_DIMS, Agent, label, make_factory, steps_of, sweep_states

HERE = Path(__file__).resolve().parent


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(coef[0]), float(coef[1]), float(r2)


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    st = json.load(open(HERE / "stats.json"))["runs"]
    eps = 1e-8
    rows = {}
    for line in open(HERE / "ckpts" / "episodes.jsonl"):
        r = json.loads(line)
        if r["seed"] == 3042:
            rows[r["label"]] = r
    factory = make_factory()
    res: dict = {}
    for lin in ("s42", "s80"):
        s0, e0 = SE[lin]
        srun = START_RUN[lin]
        mS, vS = np.array(st[srun][str(s0)]["mean"]), np.array(st[srun][str(s0)]["var"])
        XS = np.load(HERE / "ckpts" / "states" / f"start_{lin}_3042.npy")
        start = Agent(srun, s0, factory)
        res[lin] = []
        ckpts = [(srun, s0)] + [(r, t) for r in NAMES if r.startswith(lin) for t in steps_of(lin)[1:]]
        for run, t in ckpts:
            lab = label(run, t)
            X = np.load(HERE / "ckpts" / "states" / f"{lab}_3042.npy")
            mc, vc = np.array(st[run][str(t)]["mean"]), np.array(st[run][str(t)]["var"])
            ag = start if (run, t) == (srun, s0) else Agent(run, t, factory)
            out = {"label": lab, "run": run, "step": t, "lin": lin,
                   "scale_t3q_per_deg": math.radians(1) / math.sqrt(vc[T3Q] + eps),
                   "mean_t3q_deg": math.degrees(mc[T3Q]),
                   "obs_t3q_deg_states": float(np.degrees(np.median(X[:, T3Q])))}
            ag.set_stats(None)
            g_own = sweep_states(ag, X)
            out["own"] = g_own["loop_gain"]
            out["own_at0"] = g_own["tail3"]["at0"]
            out["own_head_slope"] = g_own["head_pitch"]["slope_per_deg"]
            out["own_episode"] = rows[lab]["sweep"]["loop_gain"]
            out["Jz"] = g_own["tail3"]["slope_per_deg"] / out["scale_t3q_per_deg"]
            ag.set_stats(mS, vS)
            out["sN_full"] = sweep_states(ag, X)["loop_gain"]
            out["sN_full_on_XS"] = sweep_states(ag, XS)["loop_gain"]
            ag.set_stats(mS, vS, dims=TAIL_DIMS)
            out["sN_tail"] = sweep_states(ag, X)["loop_gain"]
            ag.set_stats(mS, vS, dims=(T3Q,))
            out["sN_t3q"] = sweep_states(ag, X)["loop_gain"]
            ag.set_stats(None)
            out["own_on_XS"] = sweep_states(ag, XS)["loop_gain"]
            # the start's network on c's states, with c's statistics (full, tail, t3q) and with its own
            start.set_stats(mc, vc)
            out["startW_cN"] = sweep_states(start, X)["loop_gain"]
            out["startW_cN_on_XS"] = sweep_states(start, XS)["loop_gain"]
            start.set_stats(mc, vc, dims=TAIL_DIMS)
            out["startW_cN_tail"] = sweep_states(start, X)["loop_gain"]
            start.set_stats(mc, vc, dims=(T3Q,))
            out["startW_cN_t3q"] = sweep_states(start, X)["loop_gain"]
            start.set_stats(None)
            out["startW_sN"] = sweep_states(start, X)["loop_gain"]
            out["startW_sN_on_XS"] = sweep_states(start, XS)["loop_gain"]
            res[lin].append(out)
            print(f"# {lab} own {out['own']:+.3f} sN_full {out['sN_full']:+.3f} startW_cN {out['startW_cN']:+.3f}",
                  file=sys.stderr, flush=True)
    json.dump(res, open(HERE / "static.json", "w"), indent=1)

    # ---------------- report
    print("STATIC DECOMPOSITION OF THE tail_3 LOOP GAIN (analyze_static.py; network evaluation on stored seed-3042 states)")
    print("loop gain = d(raw tail_3 command)/d(observed tail_3 angle, deg) x 5.183 deg/unit (plant).  Medians over 6 states.")
    for lin in ("s42", "s80"):
        R = res[lin]
        S = R[0]
        print(f"\n===== {lin}   start: own {S['own']:+.3f}  (episode sweep {S['own_episode']:+.3f}); scale {S['scale_t3q_per_deg']:.3f}/deg; "
              f"mean {S['mean_t3q_deg']:+.2f} deg; Jz {S['Jz']:+.4f} per z")
        print("  per step (median over the 12 continuations):")
        hdr = ("step", "scale/deg", "mu deg", "own", "Jz", "W_c N_S", "W_c tail_S", "W_c t3q_S", "W_S N_c", "W_S tail_c",
               "W_S t3q_c", "W_S N_S|X_c")
        print("   " + " ".join(f"{h:>11s}" for h in hdr))
        for t in steps_of(lin):
            rr = [r for r in R if r["step"] == t]
            f = lambda k: np.median([r[k] for r in rr])  # noqa: E731
            print("   " + " ".join(f"{v:>11.4g}" for v in (t, f("scale_t3q_per_deg"), f("mean_t3q_deg"), f("own"), f("Jz"),
                                                          f("sN_full"), f("sN_tail"), f("sN_t3q"), f("startW_cN"),
                                                          f("startW_cN_tail"), f("startW_cN_t3q"), f("startW_sN"))))
        ends = [r for r in R if r["step"] == SE[lin][1]]
        print("  per end checkpoint:")
        print("   " + " ".join(f"{h:>11s}" for h in ("run", "own", "W_c N_S", "W_c tail_S", "W_c t3q_S", "W_S N_c",
                                                         "W_S tail_c", "W_S t3q_c", "Jz", "Jz/Jz_S")))
        for r in ends:
            print("   " + f"{r['run']:>11s} " + " ".join(f"{r[k]:>+11.3f}" for k in ("own", "sN_full", "sN_tail", "sN_t3q",
                                                                                     "startW_cN", "startW_cN_tail",
                                                                                     "startW_cN_t3q", "Jz"))
                  + f" {r['Jz'] / S['Jz']:>11.2f}")
        for k, txt in (("own", "end own"), ("sN_full", "end weights, start stats (all)"),
                       ("sN_tail", "end weights, start stats (tail 8)"), ("sN_t3q", "end weights, start stats (tail_3 angle)"),
                       ("startW_cN", "start weights, end stats (all)"), ("startW_cN_tail", "start weights, end stats (tail 8)"),
                       ("startW_cN_t3q", "start weights, end stats (tail_3 angle)")):
            v = np.array([r[k] for r in ends])
            print(f"    {txt:42s} median {np.median(v):+.3f} [{v.min():+.3f}, {v.max():+.3f}]  vs start {S['own']:+.3f}: "
                  f"{int((v > S['own']).sum())} above / {int((v < S['own']).sum())} below")
        # 2x2 decomposition on common states
        for xs, ksuf in (("X_end (end's states)", ""), ("X_S (start's states)", "_on_XS")):
            Gcc = np.array([r["own" + ksuf] for r in ends])
            Gcs = np.array([r["sN_full" + ksuf] for r in ends])
            Gsc = np.array([r["startW_cN" + ksuf] for r in ends])
            Gss = np.array([r["startW_sN" + ksuf] for r in ends])
            tot = Gcc - Gss
            print(f"  2x2 on {xs}: G(Wc,Nc) {np.median(Gcc):+.3f}  G(Wc,NS) {np.median(Gcs):+.3f}  G(WS,Nc) {np.median(Gsc):+.3f}  "
                  f"G(WS,NS) {np.median(Gss):+.3f}")
            print(f"     total change {np.median(tot):+.3f}; normaliser part at end weights G(Wc,Nc)-G(Wc,NS) {np.median(Gcc - Gcs):+.3f} "
                  f"({int((Gcc - Gcs > 0).sum())}/12 >0); at start weights G(WS,Nc)-G(WS,NS) {np.median(Gsc - Gss):+.3f} "
                  f"({int((Gsc - Gss > 0).sum())}/12 >0)")
            print(f"     weight part at start stats G(Wc,NS)-G(WS,NS) {np.median(Gcs - Gss):+.3f} ({int((Gcs - Gss > 0).sum())}/12 >0); "
                  f"at end stats G(Wc,Nc)-G(WS,Nc) {np.median(Gcc - Gsc):+.3f} ({int((Gcc - Gsc > 0).sum())}/12 >0)")
            sh = np.median(0.5 * ((Gcc - Gcs) + (Gsc - Gss)) / np.where(tot == 0, np.nan, tot))
            print(f"     symmetric (Shapley) normaliser share of the total change, median over runs: {sh:+.2f}")
        # regression of gain on scale across all checkpoints of the lineage
        allr = R
        x = np.array([r["scale_t3q_per_deg"] for r in allr])
        y = np.array([r["own"] for r in allr])
        b, a0, r2 = ols(x, y)
        print(f"  regression over {len(allr)} checkpoints (start + 6 steps x 12 runs): gain = {b:+.3f} x scale {a0:+.3f}, R^2 {r2:.2f}; "
              f"Spearman {spearman(x, y):+.2f}")
        if np.all(y > 0):
            b2, a2, r22 = ols(np.log(x), np.log(y))
            print(f"    log gain on log scale: slope {b2:+.2f} (a pure scale effect would give +1), R^2 {r22:.2f}")
        jz = np.array([r["Jz"] for r in allr])
        print(f"    Jz (sensitivity per normalised unit) over the same checkpoints: start {S['Jz']:+.4f}; "
              f"ends median {np.median([r['Jz'] for r in ends]):+.4f}; Spearman(Jz, gain) {spearman(jz, y):+.2f}; "
              f"Spearman(scale, Jz) {spearman(x, jz):+.2f}")
        # within-step: does the gain's spread over runs track the scale's spread?
        for t in steps_of(lin)[1:]:
            rr = [r for r in R if r["step"] == t]
            xx = [r["scale_t3q_per_deg"] for r in rr]
            yy = [r["own"] for r in rr]
            print(f"    step {t}: across 12 runs Spearman(scale, gain) {spearman(xx, yy):+.2f}  "
                  f"scale range {min(xx):.3f}-{max(xx):.3f}  gain range {min(yy):+.3f}..{max(yy):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
