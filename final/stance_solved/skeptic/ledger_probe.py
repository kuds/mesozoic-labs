"""Report-only readouts from existing F2 copies (no simulation, no checkpoint load).

Per run:
  A. action noise: policy_probe.npz std_per_actuator (allow_pickle=False) at the first, ~4.19M and last update;
     ent_coef at the last update.
  B. eval ledger: head tip (last 8 evaluations: mean, share of episodes under 0.90 m, lowest episode min);
     head-tip trajectory; short episodes with cause and screen_v2_reasons; unclean episodes in the last 8;
     foot slip and CoP in the last 8; returns.
  C. in-run K3 panel: falls per panel and test (own noise / joint noise).
"""
import csv
import math
import sys

import numpy as np

ROOT = "/home/user/work/pc_study/round/fetch/F2"
RUNS = ["stance7_r10LTSBpc_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBpc_s80",
        "stance7_r10LTSBoff_s60", "stance7_r10LTSBoff_s42", "stance7_r10LTSBoff_s80"]


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def probe(run):
    p = f"{ROOT}/{run}/copies/01_stance/policy_probe.npz.copy1"
    z = np.load(p, allow_pickle=False)
    t = z["update_timesteps"]
    s = z["std_per_actuator"]
    ent = z["ent_coef"]
    out = []
    for target in (int(t[0]), 4194304, 8388608, int(t[-1])):
        i = int(np.argmin(np.abs(t - target)))
        row = s[i]
        out.append(f"t={int(t[i])}: mean std {float(row.mean()):.4f} (log {float(np.log(row).mean()):+.3f}) "
                   f"min {float(row.min()):.4f} max {float(row.max()):.4f}")
    print("  A probe std_per_actuator: " + " | ".join(out))
    print(f"  A ent_coef first {float(ent[0]):.5g} last {float(ent[-1]):.5g}; last-update std per actuator: "
          + " ".join(f"{v:.3f}" for v in s[-1]))


def ledger(run):
    p = f"{ROOT}/{run}/copies/01_stance/eval_episodes.csv.copy1"
    rows = list(csv.DictReader(open(p, newline="")))
    by_t = {}
    for r in rows:
        by_t.setdefault(int(float(r["timesteps"])), []).append(r)
    ts = sorted(by_t)
    last8 = [r for t in ts[-8:] for r in by_t[t]]
    ht = [f(r["head_tip_height_mean_m"]) for r in last8]
    hmin = [f(r["head_tip_height_min_m"]) for r in last8]
    under = sum(1 for v in ht if v is not None and v < 0.90)
    print(f"  B head tip last8: mean {mean(ht):.3f} m, episodes with mean < 0.90 m: {under}/{len(ht)}, "
          f"lowest episode mean {min(v for v in ht if v is not None):.3f}, lowest episode min "
          f"{min(v for v in hmin if v is not None):.3f}")
    traj = []
    for target in (131072, 1048576, 2097152, 4194304, 6291456, 8388608, 10485760, ts[-1]):
        t = min(ts, key=lambda x: abs(x - target))
        traj.append(f"{t}:{mean(f(r['head_tip_height_mean_m']) for r in by_t[t]):.3f}")
    print("  B head tip trajectory (eval mean): " + " ".join(traj))
    short = [r for r in rows if int(float(r["length"])) < 1000]
    for r in short:
        print(f"  B short episode t={r['timesteps']} len={r['length']} cause={r['cause']!r} "
              f"reasons={r['screen_v2_reasons']!r} head_min={r['head_tip_height_min_m']}")
    unclean8 = [r for r in last8 if str(r["screen_v2_clean"]).strip().lower() not in ("1", "true", "1.0")]
    print(f"  B last8 unclean episodes {len(unclean8)}/{len(last8)}: "
          + "; ".join(sorted(set(r['screen_v2_reasons'] for r in unclean8)))[:600])
    unclean_all = [r for r in rows if str(r["screen_v2_clean"]).strip().lower() not in ("1", "true", "1.0")]
    print(f"  B all unclean episodes {len(unclean_all)}/{len(rows)}")
    for c in ("r_slip_m", "l_slip_m", "max_sole_cop_fore_aft", "return"):
        print(f"  B last8 {c}: mean {mean(f(r[c]) for r in last8):.4f} max "
              f"{max(v for v in (f(r[c]) for r in last8) if v is not None):.4f}")


def k3(run):
    p = f"{ROOT}/{run}/copies/01_stance/k3_panel.csv.copy1"
    rows = list(csv.DictReader(open(p, newline="")))
    agg = {}
    for r in rows:
        k = (int(r["timesteps"]), r["test"])
        a = agg.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += int(float(r["fall"]))
    parts = []
    for (t, test) in sorted(agg):
        n, falls = agg[(t, test)]
        parts.append(f"{t}/{test}:{falls}/{n}")
    print("  C K3 panel falls: " + " ".join(parts))
    tot = {}
    for (t, test), (n, falls) in agg.items():
        a = tot.setdefault(test, [0, 0])
        a[0] += n
        a[1] += falls
    print("  C K3 totals: " + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(tot.items())))


def main():
    for run in RUNS:
        print(f"=== {run}")
        for fn in (probe, ledger, k3):
            try:
                fn(run)
            except (OSError, KeyError, ValueError) as exc:
                print(f"  {fn.__name__}: not found or unreadable: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
