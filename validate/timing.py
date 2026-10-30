"""BS2 timing (files only): the no-update V2 runs of BS2 (4 at once, the screen's slot count, on an otherwise idle
machine) give the per-rollout cost of the rollout and the evaluation at the new start; the update's cost comes from
BS's 8 REAL continuations (S2/r10round/bias_screen/runs, finished; their per-rollout evidence clocks include the update
between two rollout ends) minus BS's own no-update V2 runs (BS/validate/v2final), both [measured there].  Estimate for
one BS2 continuation (176 rollouts, 175 updates, 11 evaluations) and for 8 at 4 at a time.

    PY -I -B timing.py BS2_V2_DIR
Derived from BS's validate/timing.py (which had to infer the update from the ECS; BS's own real runs now measure it).
"""
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

BS1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/"
           "bias_screen")
GRID = 131_072
N_ROLL, N_EVAL = 176, 11


def t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def per_run(run, S):
    meta = json.loads((run / "run_meta.json").read_text())
    ev = [json.loads(x) for x in (run / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    cpu = [e["clock"]["cpu_seconds"] for e in ev]
    wall = [t(e["clock"]["utc"]) for e in ev]
    steps = [e["num_timesteps"] for e in ev]
    d_cpu = [b - a for a, b in zip(cpu, cpu[1:], strict=False)]
    d_wall = [b - a for a, b in zip(wall, wall[1:], strict=False)]
    nxt = steps[1:]
    plain = [d for d, s in zip(d_cpu, nxt, strict=True) if (s - S) % GRID != 0]
    with_eval = [d for d, s in zip(d_cpu, nxt, strict=True) if (s - S) % GRID == 0]
    plain_w = [d for d, s in zip(d_wall, nxt, strict=True) if (s - S) % GRID != 0]
    return dict(run=run.name, n=len(ev), cpu_total=meta.get("cpu_seconds"), wall_total=meta.get("wall_seconds"),
                learn_seconds=meta.get("learn_seconds"), first_rollout_end_cpu=cpu[0],
                cpu_per_rollout_median=st.median(plain),
                cpu_rollout_with_eval=st.median(with_eval) if with_eval else None,
                wall_per_rollout_median=st.median(plain_w), n_updates=[meta["digest_start"]["n_updates"],
                                                                       meta["digest_end"]["n_updates"]])


def summarise(rows):
    roll = st.median(r["cpu_per_rollout_median"] for r in rows)
    ev = [r["cpu_rollout_with_eval"] for r in rows if r["cpu_rollout_with_eval"]]
    setup = st.median(r["first_rollout_end_cpu"] - r["cpu_per_rollout_median"] for r in rows)
    return roll, (st.median(ev) - roll) if ev else 0.0, setup


bs2 = [per_run(r, 2_097_152) for r in sorted(Path(sys.argv[1]).iterdir()) if (r / "arm_evidence.jsonl").is_file()]
bs1_nu = [per_run(r, 1_048_576) for r in sorted((BS1 / "validate" / "v2final").iterdir())]
bs1_real = [per_run(r, 1_048_576) for r in sorted((BS1 / "runs").iterdir()) if (r / "arm_evidence.jsonl").is_file()]
for name, rows in (("BS2 no-update V2 (4 at once)", bs2), ("BS no-update V2 (2 at once)", bs1_nu),
                   ("BS real continuations (4 at once)", bs1_real)):
    print(name)
    for r in rows:
        print("  " + json.dumps(r))
r2, e2, s2 = summarise(bs2)
r1, e1, s1 = summarise(bs1_nu)
rr, er, sr = summarise(bs1_real)
upd = rr - r1
print(f"BS2 no-update: {r2:.2f} CPU-s per rollout, +{e2:.2f} CPU-s per evaluation, setup {s2:.1f} CPU-s [measured]")
print(f"BS no-update:  {r1:.2f} CPU-s per rollout, +{e1:.2f} per evaluation, setup {s1:.1f} [measured there]")
print(f"BS real:       {rr:.2f} CPU-s per rollout incl. the update, +{er:.2f} per evaluation [measured there]; "
      f"the update's share {upd:.2f} CPU-s per rollout [measured there: real minus no-update]")
real_tot = [r["cpu_total"] for r in bs1_real]
real_wall = [r["wall_total"] for r in bs1_real]
print(f"BS real totals: CPU {min(real_tot):.0f}-{max(real_tot):.0f} s, wall {min(real_wall):.0f}-{max(real_wall):.0f} "
      "s "
      f"(median {st.median(real_wall) / 60:.1f} min) per continuation, 4 at a time")
one = s2 + N_ROLL * r2 + (N_ROLL - 1) * upd + N_EVAL * e2
ratio = (st.median(real_tot)) * (r2 + upd) / (r1 + upd)
print(f"ONE BS2 CONTINUATION (estimate): {one:.0f} CPU-s = {one / 60:.0f} min on one CPU "
      f"(setup + {N_ROLL} x {r2:.2f} + {N_ROLL - 1} x {upd:.2f} update + {N_EVAL} x {e2:.2f}); cross-check, BS's real "
      f"median {st.median(real_tot):.0f} CPU-s scaled by the per-rollout ratio: {ratio:.0f} CPU-s [inferred]")
wall_ratio = st.median(real_wall) / st.median(real_tot)
for k in (0, 1, 2, 3):
    f = max(1.0, (4 + k) / 4)
    print(f"8 continuations, 4 at a time, 2 waves, {k} other busy process(es) on the 4 CPUs: about "
          f"{2 * one * wall_ratio * f / 3600:.2f} h wall (x{f:.2f}; BS's wall/CPU {wall_ratio:.3f}) [inferred]")
bs_window = (t("2026-10-30T08:49:37Z") - t("2026-10-30T06:00:19Z")) / 3600
bs_reads = (t("2026-10-30T09:02:40Z") - t("2026-10-30T08:49:39Z")) / 60
print(f"BS's own 8 continuations took {bs_window:.2f} h wall (launch 06:00:19Z, last exit 08:49:37Z; BS/logs/"
      f"run_waves_bs.log lines 1 and 25), then the 8 full reads {bs_reads:.1f} min (lines 27-35; run_waves_bs.exit "
      f"09:02:40Z) [measured there]")
