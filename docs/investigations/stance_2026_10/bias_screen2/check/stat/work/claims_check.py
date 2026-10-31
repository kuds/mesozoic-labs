"""BS2 statistic lens: the coordinator's nine report-only claims, against the raw records and the markers.
Own code; imports nothing from the screen's tools.  Reads only (numpy allow_pickle False); stdout is captured by the
caller into check/stat/.  No Monte Carlo: the one probability below is a deterministic quadrature."""
import csv
import hashlib
import json
import math
import statistics
from fractions import Fraction as Fr

import numpy as np
from scipy import integrate, stats
from pathlib import Path

_BS2 = Path(__file__).resolve().parents[3]                 # work -> stat -> check -> bias_screen2
S2 = str(_BS2.parents[1])                                   # bias_screen2 -> r10round -> the session scratchpad
S1 = str(_BS2.parents[3] / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad")
BS2 = str(_BS2)
F2_PROBE = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1"
S, E, ROLL = 2_097_152, 3_538_944, 8192
PRIMARY_STEPS = [3_473_408 + ROLL * j for j in range(8)]
EVAL_STEPS = [S + 131_072 * k for k in range(1, 12)]
SEEDS = (2501, 2511, 2521, 2531)
ARMS = ("control", "critic_blind_tail3")
B_S = Fr(135930855, 1073741824)
TAIL = ("tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def probe(path):
    with open(path, "rb") as fh:
        z = np.load(fh, allow_pickle=False)
        return np.array(z["update_timesteps"]), np.array(z["rollout_mu_mean"])


def mean8(u, m, steps):
    vals = []
    for s in steps:
        i = np.nonzero(u == s)[0]
        assert len(i) == 1, s
        vals.append(Fr(float(m[i[0], 14])))
    return sum(vals, Fr(0)) / 8


def traj(u, m):
    out = {}
    for t in EVAL_STEPS:
        steps = PRIMARY_STEPS if t == E else [t - ROLL * j for j in range(7, -1, -1)]
        out[t] = mean8(u, m, steps)
    return out


def med(xs):
    xs = sorted(xs)
    return (xs[1] + xs[2]) / 2 if len(xs) == 4 else statistics.median(xs)


def main():
    names = [f"{a}_{s}" for s in SEEDS for a in ARMS]
    ctrl = [f"control_{s}" for s in SEEDS]
    blnd = [f"critic_blind_tail3_{s}" for s in SEEDS]
    mk = {n: json.load(open(f"{BS2}/reads/{n}/markers.json")) for n in names}

    # (1) and (2): primaries, rises, trajectories --------------------------------------------------------------
    pu, pm = probe(F2_PROBE)
    ptraj = traj(pu, pm)
    p_endE = mean8(pu, pm, [E - ROLL * j for j in range(7, -1, -1)])
    tr = {}
    for n in names:
        u, m = probe(f"{BS2}/runs/{n}/01_stance/policy_probe.npz")
        tr[n] = traj(u, m)
    print("(1) primary rises from B_S")
    for n in names:
        print(f"   {n:24s} primary {float(tr[n][E]):+.5f} rise {float(tr[n][E] - B_S):+.5f}")
    rb = [tr[n][E] - B_S for n in blnd]
    rc = [tr[n][E] - B_S for n in ctrl]
    print(f"   blind rises {float(min(rb)):+.5f} .. {float(max(rb)):+.5f}; max |rise| {float(max(abs(x) for x in rb)):.5f}")
    print(f"   control rises {float(min(rc)):+.5f} .. {float(max(rc)):+.5f}")
    print(f"   min control primary {float(min(tr[n][E] for n in ctrl)):+.5f} > max blind {float(max(tr[n][E] for n in blnd)):+.5f}:"
          f" {min(tr[n][E] for n in ctrl) > max(tr[n][E] for n in blnd)}")
    dev = [(abs(float(tr[n][t] - B_S)), n, t) for n in blnd for t in EVAL_STEPS]
    print(f"   blind arm, the 11 evaluation-step 8-row means: max |value - B_S| {max(dev)[0]:.5f} ({max(dev)[1]} at {max(dev)[2]})")
    print("(2) trajectory: parent vs control median vs blind median (8-row means; at E the primary's rows)")
    print(f"   parent's mean over the 8 rows ENDING at E (3,481,600..3,538,944), for reference: {float(p_endE):+.5f}")
    for t in EVAL_STEPS:
        mc = med([tr[n][t] for n in ctrl])
        mb = med([tr[n][t] for n in blnd])
        cs = sorted(float(tr[n][t]) for n in ctrl)
        print(f"   {t:>9,d} parent {float(ptraj[t]):+.4f} ctrl-med {float(mc):+.4f} (diff {float(mc - ptraj[t]):+.4f}; "
              f"ctrl {cs[0]:+.4f}..{cs[-1]:+.4f}) blind-med {float(mb):+.4f}")
    mcE = med([tr[n][E] for n in ctrl])
    print(f"   at E: control median rise {float(mcE - B_S):+.5f} = {float((mcE - B_S) / (ptraj[E] - B_S)):.3f} of the parent's "
          f"{float(ptraj[E] - B_S):+.5f}")
    for n in ctrl:
        print(f"   {n}: above the parent at E? {tr[n][E] > ptraj[E]} ({float(tr[n][E]):+.4f} vs {float(ptraj[E]):+.4f})")
    sc = json.load(open(BS2 + "/score/score.json"))
    ok = all(abs(sc["parent_trajectory"][str(t)] - float(ptraj[t])) == 0 for t in EVAL_STEPS)
    ok2 = all(sc["trajectory_arm_medians"][str(t)]["control"] == float(med([tr[n][t] for n in ctrl])) and
              sc["trajectory_arm_medians"][str(t)]["critic_blind_tail3"] == float(med([tr[n][t] for n in blnd]))
              for t in EVAL_STEPS)
    ok3 = all(dict((int(a), b) for a, b in sc["report_only"][n]["trajectory"])[t] == float(tr[n][t])
              for n in names for t in EVAL_STEPS)
    print(f"   score.json parent_trajectory == mine: {ok}; trajectory_arm_medians == mine: {ok2}; per-run trajectories == mine: {ok3}")

    # (3), (4), (7): the raw ledgers ---------------------------------------------------------------------------
    print("(3)(4)(7) from eval_episodes.csv (window (S, E])")
    win_ret = {}
    for n in names:
        p = f"{BS2}/runs/{n}/01_stance/eval_episodes.csv"
        with open(p, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if S < float(r["timesteps"]) <= E]
        assert len(rows) == 110, (n, len(rows))
        by = {}
        for r in rows:
            by.setdefault(int(float(r["timesteps"])), []).append(r)
        assert sorted(by) == EVAL_STEPS and all(len(v) == 10 for v in by.values())
        over = [(int(float(r["timesteps"])), r["saturated_actuator"], float(r["saturated_actuator_fraction"]))
                for r in rows if r["saturated_actuator_fraction"] not in ("", "nan") and
                float(r["saturated_actuator_fraction"]) > 0.10]
        per_over = {}
        for t, a, f in over:
            per_over.setdefault(t, []).append(a)
        tail_flag = [t for t, a in per_over.items() if sum(1 for x in a if x in TAIL) >= 5]
        cmdE = sum(float(r["tail_3_pitch_act_command_mean"]) for r in by[E]) / 10
        rets = [float(r["return"]) for r in rows]
        win_ret[n] = sum(rets) / len(rets)
        falls = [(int(float(r["timesteps"])), r["length"], r["cause"]) for r in rows
                 if str(r["full_horizon"]).strip().lower() not in ("true", "1", "1.0")]
        lens = sorted({r["length"] for r in rows})
        first_tail = min((t for t, a in per_over.items() if any(x in TAIL for x in a)), default=None)
        print(f"   {n:24s} over-bar per eval {dict(sorted((t, len(a)) for t, a in per_over.items()))} "
              f"actuators {sorted({x for a in per_over.values() for x in a})}; first tail over-bar {first_tail}; "
              f">=5 tail over-bar (M5 TAIL-flag rule) at {sorted(tail_flag)}; CMD_L(E) {cmdE:+.5f}; window mean return "
              f"{win_ret[n]:.2f}; falls {falls}; lengths {lens[:3]}{'...' if len(lens) > 3 else ''}")
        m5 = mk[n]["ledger"]["m5"]
        print(f"   {'':24s} markers M5 any_flagged {m5['any_flagged']} tail_flagged {m5['tail_flagged']} any_alarms "
              f"{m5['any_alarms']} tail_alarms {m5['tail_alarms']}; markers firsts {mk[n]['ledger']['direct']['firsts']}")
        rep = sc["report_only"][n]["ledger"]
        print(f"   {'':24s} score.json mean_return_window {rep['mean_return_window']!r} vs mine {win_ret[n]!r}")
    rc_ = [win_ret[n] for n in ctrl]
    rb_ = [win_ret[n] for n in blnd]
    print(f"   window mean return: control median {med(rc_):.2f} (values {[round(x, 1) for x in rc_]}), blind median "
          f"{med(rb_):.2f} (values {[round(x, 1) for x in rb_]}); per pair blind - control "
          f"{[round(win_ret[b] - win_ret[c], 1) for c, b in zip(ctrl, blnd)]}")
    cmdc = [mk[n]["ledger"]["direct"]["per_evaluation"]["3538944"]["cmd_l"] for n in ctrl]
    cmdb = [mk[n]["ledger"]["direct"]["per_evaluation"]["3538944"]["cmd_l"] for n in blnd]
    print(f"(4) CMD_L at E: control {min(cmdc):.5f}..{max(cmdc):.5f}; blind {min(cmdb):.5f}..{max(cmdb):.5f}")

    # (5), (6), (8): the loop rows -------------------------------------------------------------------------------
    print("(5)(6)(8) loop rows (markers; each pair's files re-hashed)")
    allc0 = {a: [] for a in ARMS}
    allb = {a: [] for a in ARMS}
    slopes = []
    for n in names:
        arm = "control" if n.startswith("control") else "critic_blind_tail3"
        rows = sorted(mk[n]["loop"], key=lambda r: r["steps"])
        steps = [r["steps"] for r in rows]
        assert steps == [S, 2_359_296, 2_621_440, 2_883_584, 3_145_728, 3_407_872, E], steps
        bad = []
        for r in rows:
            if sha256(r["zip"]) != r["zip_sha256"] or sha256(r["pkl"]) != r["pkl_sha256"]:
                bad.append(r["steps"])
            P = r["OWN"]["P"]
            if P["b_star"] != 1 - P["G"] - P["c0"]:
                bad.append(("bstar", r["steps"], P["b_star"] - (1 - P["G"] - P["c0"])))
        G = [r["OWN"]["P"]["G"] for r in rows]
        c0 = [r["OWN"]["P"]["c0"] for r in rows]
        bs = [r["OWN"]["P"]["b_star"] for r in rows]
        sl = [r["critic_slope_obs6"]["ONPOLICY"]["effective"]["dV_dq3_per_deg"] for r in rows]
        slr = [r["critic_slope_obs6"]["ONPOLICY"]["raw"]["dV_dq3_per_deg"] for r in rows]
        allc0[arm] += [(x, n, s) for x, s in zip(c0[1:], steps[1:])]
        allb[arm] += [(x, n, s) for x, s in zip(bs[1:], steps[1:])]
        if arm == "control":
            slopes += [(x, n, s) for x, s in zip(sl, steps)]
        print(f"   {n:24s} sha/bstar problems {bad}; forms {sorted({r.get('critic_form') for r in rows})}")
        print(f"   {'':24s} G  {[round(x, 3) for x in G]}  G(E) > G(S): {G[-1] > G[0]}")
        print(f"   {'':24s} c0 {[round(x, 3) for x in c0]}")
        print(f"   {'':24s} b* {[round(x, 3) for x in bs]}")
        print(f"   {'':24s} on-policy slope eff {[round(x, 5) for x in sl]} raw {[round(x, 5) for x in slr]}")
    for a in ARMS:
        atE_c0 = [x for x, n, s in allc0[a] if s == E]
        atE_b = [x for x, n, s in allb[a] if s == E]
        print(f"   {a}: c0 at E {min(atE_c0):.3f}..{max(atE_c0):.3f}; c0 over the 6 window pairs min {min(allc0[a])} max "
              f"{max(allc0[a])}")
        print(f"   {a}: b* at E {min(atE_b):+.5f}..{max(atE_b):+.5f}; b* over the 6 window pairs min {min(allb[a])} max "
              f"{max(allb[a])}")
    print(f"   control on-policy slope over all 7 pairs x 4: min {min(slopes)} max {max(slopes)}; all > 0: "
          f"{all(x > 0 for x, _, _ in slopes)}")
    for s in SEEDS:
        gc = [r for r in mk[f"control_{s}"]["loop"] if r["steps"] == E][0]["OWN"]["P"]["G"]
        gb = [r for r in mk[f"critic_blind_tail3_{s}"]["loop"] if r["steps"] == E][0]["OWN"]["P"]["G"]
        print(f"   (6) seed {s}: G(E) control {gc:+.5f} blind {gb:+.5f} blind lower: {gb < gc}")

    # (9) SDs and the spread caveat --------------------------------------------------------------------------
    pc = [float(tr[n][E]) for n in ctrl]
    pb = [float(tr[n][E]) for n in blnd]
    sdc, sdb = statistics.stdev(pc), statistics.stdev(pb)
    print(f"(9) SD control {sdc:.5f} blind {sdb:.5f} ratio {sdc / sdb:.3f}; gap min control - max blind "
          f"{min(pc) - max(pb):.5f} = {(min(pc) - max(pb)) / sdb:.1f} blind SDs = {(min(pc) - max(pb)) / sdc:.2f} control SDs")

    def p_sep(ratio):
        """P(min of 4 N(0, ratio^2) > max of 4 N(0, 1)): equal means, control SD = ratio x blind SD."""
        f = lambda z: 4 * stats.norm.pdf(z) * stats.norm.cdf(z) ** 3 * stats.norm.sf(z / ratio) ** 4
        v, err = integrate.quad(f, -12, 12, epsabs=1e-13, epsrel=1e-11, limit=400)
        return v, err
    for ratio in (1.0, 2.0, 3.0, sdc / sdb, 5.0, 7.0, 20.0, 1e6):
        v, err = p_sep(ratio)
        print(f"   equal means, SD ratio {ratio:10.3f}: P(complete separation, i.e. p = 1/70) = {v:.5f} (quad err {err:.1e})")
    print(f"   references: 1/70 = {1 / 70:.5f}, 1/16 = {1 / 16:.5f}")
    # post hoc, report only: Welch t on the 8 primaries (not pre-registered)
    w = stats.ttest_ind(pc, pb, equal_var=False, alternative="greater")
    se = math.sqrt(sdc ** 2 / 4 + sdb ** 2 / 4)
    dfw = (sdc ** 2 / 4 + sdb ** 2 / 4) ** 2 / ((sdc ** 2 / 4) ** 2 / 3 + (sdb ** 2 / 4) ** 2 / 3)
    print(f"   post hoc only: Welch t {w.statistic:.3f} df {dfw:.2f} one-sided p {w.pvalue:.4f} (diff {statistics.mean(pc) - statistics.mean(pb):.5f}, SE {se:.5f})")


if __name__ == "__main__":
    main()
