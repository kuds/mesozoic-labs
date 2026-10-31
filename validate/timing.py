"""PS timing (files only): PS's no-update V2 runs (4 at once: control and critic_phase at two seeds, on an otherwise
idle machine) give the per-rollout cost of the rollout and the evaluation for each arm (critic_phase with its extra
evidence);
the update's cost is BS2's [measured there]: its 8 REAL continuations from the same S (BS2/runs, finished, 4 at once)
minus BS2's no-update V2 runs (BS2/validate/v2final, 4 at once).  Estimate for one PS continuation (176 rollouts, 175
updates, 11 evaluations) per arm and for 8 at 4 at a time; BS2's own measured wall time beside it.

    PY -I -B timing.py PS_V2_DIR
Derived from BS2's validate/timing.py (which used BS's real continuations for the update).
"""
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

BS2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/"
           "bias_screen2")
S = 2_097_152
GRID = 131_072
N_ROLL, N_EVAL = 176, 11


def t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def per_run(run):
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
                first_rollout_end_cpu=cpu[0], cpu_per_rollout_median=st.median(plain),
                cpu_rollout_with_eval=st.median(with_eval) if with_eval else None,
                wall_per_rollout_median=st.median(plain_w),
                n_updates=[meta["digest_start"]["n_updates"], meta["digest_end"]["n_updates"]])


def summarise(rows):
    roll = st.median(r["cpu_per_rollout_median"] for r in rows)
    ev = [r["cpu_rollout_with_eval"] for r in rows if r["cpu_rollout_with_eval"]]
    setup = st.median(r["first_rollout_end_cpu"] - r["cpu_per_rollout_median"] for r in rows)
    return roll, (st.median(ev) - roll) if ev else 0.0, setup


ps = [per_run(r) for r in sorted(Path(sys.argv[1]).iterdir()) if (r / "arm_evidence.jsonl").is_file()]
bs2_nu = [per_run(r) for r in sorted((BS2 / "validate" / "v2final").iterdir()) if (r / "arm_evidence.jsonl").is_file()]
bs2_real = [per_run(r) for r in sorted((BS2 / "runs").iterdir()) if (r / "arm_evidence.jsonl").is_file()]
for name, rows in (("PS no-update V2 (4 at once)", ps), ("BS2 no-update V2 (4 at once)", bs2_nu),
                   ("BS2 real continuations (4 at once)", bs2_real)):
    print(name)
    for r in rows:
        print("  " + json.dumps(r))
rn, en, sn = summarise(bs2_nu)
rr, er, sr = summarise(bs2_real)
upd = rr - rn
print(f"BS2 no-update: {rn:.2f} CPU-s per rollout, +{en:.2f} per evaluation, setup {sn:.1f} [measured there]")
print(f"BS2 real:      {rr:.2f} CPU-s per rollout incl. the update, +{er:.2f} per evaluation [measured there]; the "
      f"update's share {upd:.2f} CPU-s per rollout [measured there: real minus no-update]")
real_tot = [r["cpu_total"] for r in bs2_real]
real_wall = [r["wall_total"] for r in bs2_real]
wall_ratio = st.median(real_wall) / st.median(real_tot)
print(f"BS2 real totals: CPU {min(real_tot):.0f}-{max(real_tot):.0f} s, wall {min(real_wall):.0f}-{max(real_wall):.0f} "
      f"s (median {st.median(real_wall) / 60:.1f} min) per continuation, 4 at a time; wall/CPU {wall_ratio:.3f}")
ones = {}
for arm in ("control", "critic_phase"):
    rows = [r for r in ps if r["run"].startswith(arm + "_")]
    r2, e2, s2 = summarise(rows)
    one = s2 + N_ROLL * r2 + (N_ROLL - 1) * upd + N_EVAL * e2
    ones[arm] = one
    print(f"PS {arm} no-update: {r2:.2f} CPU-s per rollout, +{e2:.2f} per evaluation, setup {s2:.1f} [measured]; "
          f"ONE CONTINUATION (estimate) {one:.0f} CPU-s = {one / 60:.0f} min (setup + {N_ROLL} x {r2:.2f} + "
          f"{N_ROLL - 1} x {upd:.2f} update + {N_EVAL} x {e2:.2f}) [inferred]")
wave = max(ones.values())
ps_ratio = st.median(r["wall_total"] / r["cpu_total"] for r in ps)
print(f"PS V2 wall/CPU {ps_ratio:.3f} (the machine was not idle during V2); at that ratio 8 continuations take about "
      f"{2 * wave * ps_ratio / 3600:.2f} h wall [inferred]")
for k in (0, 1, 2, 3):
    f = max(1.0, (4 + k) / 4)
    print(f"8 continuations (4 + 4), 4 at a time, 2 waves, {k} other busy process(es) on the 4 CPUs: about "
          f"{2 * wave * wall_ratio * f / 3600:.2f} h wall (x{f:.2f}) [inferred]")
lines = (BS2 / "logs" / "run_waves_bs.log").read_text().splitlines()
start = t(lines[0][14:34])
last_exit = max(t(x[14:34]) for x in lines if " exited " in x)
reads_end = max(t(x[14:34]) for x in lines if "read_bs (the markers) exit" in x)
print(f"BS2's own 8 continuations took {(last_exit - start) / 3600:.2f} h wall (launch {lines[0][14:34]}, last exit "
      f"at {datetime.fromtimestamp(last_exit, timezone.utc):%H:%M:%S}Z), then the 8 full reads "
      f"{(reads_end - last_exit) / 60:.1f} "
      f"min (BS2/logs/run_waves_bs.log) [measured there]")
