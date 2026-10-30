"""LB timing (files only): the cost of one continuation S -> E and of 4 at once, and the reader's cost per saved pair.

    PY -I -B timing.py <LB dir>

Inputs [measured]:
  - LB's no-update runs (validate/v2/*, validate/v4/*): CPU-s per rollout (the evidence clocks: process_time at each
    rollout end), per evaluation, setup; run at 2 processes of this build beside other workflows' processes (load 6-9 on
    4 CPUs: CPU-s, not wall, is the comparable measure);
  - BS2's no-update V2 runs (bias_screen2/validate/v2final, 4 at once) and BS2's 8 REAL continuations
    (bias_screen2/runs, 4 at once, 175 updates each): the update's CPU-s per rollout = real minus no-update; BS2's
    epochs per train (n_updates grown / 175) give the cost per PPO epoch; BS2's wall / CPU ratio at 4 at once;
  - the parent's epochs per train over LB's window (validate/logs/parent_schedule.out: 6.73) and BS2's (7.35-7.99):
    the update's cost over the window is taken between the two;
  - the reader's CPU-s per saved pair: its loop cache entries (cpu_seconds of loop_markers_one) under LB/reads/_cache
    and BS2/reads/_cache (55 entries, 4 reads at once).
Outputs an estimate for one LB continuation (1,209 rollouts, 1,208 updates, 75 evaluations) and 4 at once, and the
read schedule's cost (every saved pair: 39 per run, S shared; every 1,048,576: 11).
"""
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

LB = Path(sys.argv[1])
S2 = LB.parents[1]
BS2 = S2 / "r10round" / "bias_screen2"
GRID, S = 131_072, 2_097_152
N_ROLL, N_UPD, N_EVAL = 1_209, 1_208, 75


def t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def per_run(run):
    meta = json.loads((run / "run_meta.json").read_text())
    ev = [json.loads(x) for x in (run / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    load = meta.get("load_step", S)
    ev = [e for e in ev if e["num_timesteps"] > load]            # this attempt's own rollouts only
    cpu = [e["clock"]["cpu_seconds"] for e in ev]
    wall = [t(e["clock"]["utc"]) for e in ev]
    steps = [e["num_timesteps"] for e in ev]
    d_cpu = [b - a for a, b in zip(cpu, cpu[1:], strict=False)]
    d_wall = [b - a for a, b in zip(wall, wall[1:], strict=False)]
    nxt = steps[1:]
    plain = [d for d, s in zip(d_cpu, nxt, strict=True) if s % GRID != 0]
    with_eval = [d for d, s in zip(d_cpu, nxt, strict=True) if s % GRID == 0]
    plain_w = [d for d, s in zip(d_wall, nxt, strict=True) if s % GRID != 0]
    d0, d1 = meta.get("digest_start") or {}, meta.get("digest_end") or {}
    return dict(run=run.name, n=len(ev), cpu_total=meta.get("cpu_seconds"), wall_total=meta.get("wall_seconds"),
                first_rollout_end_cpu=cpu[0] if cpu else None,
                cpu_per_rollout_median=st.median(plain) if plain else None,
                cpu_rollout_with_eval=st.median(with_eval) if with_eval else None,
                wall_per_rollout_median=st.median(plain_w) if plain_w else None,
                n_updates=[d0.get("n_updates"), d1.get("n_updates")], T=meta.get("T"), load=load)


def summarise(rows):
    rows = [r for r in rows if r["cpu_per_rollout_median"]]
    roll = st.median(r["cpu_per_rollout_median"] for r in rows)
    ev = [r["cpu_rollout_with_eval"] for r in rows if r["cpu_rollout_with_eval"]]
    setup = st.median(r["first_rollout_end_cpu"] - r["cpu_per_rollout_median"] for r in rows)
    return roll, (st.median(ev) - roll) if ev else 0.0, setup


def runs(folder):
    return [per_run(r) for r in sorted(Path(folder).iterdir()) if (r / "arm_evidence.jsonl").is_file()]


lb = runs(LB / "validate" / "v2") + runs(LB / "validate" / "v4")
bs2_nu = runs(BS2 / "validate" / "v2final")
bs2_real = runs(BS2 / "runs")
for name, rows in (("LB no-update (V2, V4; 2 of this build's processes beside other workflows')", lb),
                   ("BS2 no-update V2 (4 at once)", bs2_nu), ("BS2 real continuations (4 at once)", bs2_real)):
    print(name)
    for r in rows:
        print("  " + json.dumps(r))
rl, el, sl = summarise(lb)
r0, e0, s0 = summarise(bs2_nu)
rr, er, sr = summarise(bs2_real)
upd = rr - r0
# one train per evidence line: the last rollout (at the target) ends in StopAtEnd before on_rollout_end, no train
ep = [(r["n_updates"][1] - r["n_updates"][0]) / r["n"] for r in bs2_real]
per_epoch = upd / st.median(ep)
print(f"LB no-update:   {rl:.2f} CPU-s per rollout, +{el:.2f} per evaluation, setup {sl:.1f} [measured]")
print(f"BS2 no-update:  {r0:.2f} CPU-s per rollout, +{e0:.2f} per evaluation, setup {s0:.1f} [measured there]")
print(f"BS2 real:       {rr:.2f} CPU-s per rollout incl. the update; the update {upd:.2f} CPU-s per rollout [real "
      f"minus no-update]; epochs per train {min(ep):.2f}-{max(ep):.2f} (median {st.median(ep):.2f}); "
      f"{per_epoch:.3f} CPU-s per epoch [measured there]")
real_tot = [r["cpu_total"] for r in bs2_real]
real_wall = [r["wall_total"] for r in bs2_real]
wall_ratio = st.median(real_wall) / st.median(real_tot)
print(f"BS2 real totals: CPU {min(real_tot):.0f}-{max(real_tot):.0f} s, wall {min(real_wall):.0f}-"
      f"{max(real_wall):.0f} s (median {st.median(real_wall) / 60:.1f} min for 176 rollouts), wall / CPU "
      f"{wall_ratio:.3f} at 4 at once")
roll = r0              # BS2's no-update rollout at 4 at once (LB's own is measured under other workflows' load)
for label, epochs in (("parent's 6.73 epochs per train", 6.73), (f"BS2's {st.median(ep):.2f}", st.median(ep)),
                      ("10 (no early stop)", 10.0)):
    one = s0 + N_ROLL * roll + N_UPD * per_epoch * epochs + N_EVAL * e0
    print(f"ONE LB CONTINUATION ({label}): {one:.0f} CPU-s = {one / 3600:.2f} h on one CPU; 4 at once on a free "
          f"machine about {one * wall_ratio / 3600:.2f} h wall [inferred: BS2's wall / CPU ratio]")
    for k in (1, 2):
        print(f"    with {k} other busy process(es) on the 4 CPUs: about {one * wall_ratio * (4 + k) / 4 / 3600:.2f} h")
cache = []
for d in (LB / "reads" / "_cache", BS2 / "reads" / "_cache"):
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        row = json.loads(f.read_text())
        cache.append((d.parent.parent.name, row.get("num_timesteps"), row.get("cpu_seconds"), row.get("critic_form")))
lbc = [c[2] for c in cache if c[0] == "long_blind" and c[2]]
bsc = [c[2] for c in cache if c[0] == "bias_screen2" and c[2]]
print(f"READER: CPU-s per pair (loop_markers_one): LB {len(lbc)} entries {min(lbc) if lbc else None}-"
      f"{max(lbc) if lbc else None} (median {st.median(lbc) if lbc else None}); BS2 {len(bsc)} entries "
      f"{min(bsc):.1f}-{max(bsc):.1f} (median {st.median(bsc):.1f}, 4 reads at once) [measured]")
pair = st.median(bsc + lbc)
for label, n in (("every saved pair (LOOP_READ_EVERY = 262,144): 39 pairs, 38 new per run", 38),
                 ("every 1,048,576 (the parent's F2 grid): 11 pairs, 10 new per run", 10)):
    print(f"READ SCHEDULE {label}: {n * pair / 60:.0f} CPU-min per run; 4 runs 4 at once about {n * pair / 60:.0f} "
          f"min wall (+ M5 and the probe markers, about 1 min) [inferred]")
