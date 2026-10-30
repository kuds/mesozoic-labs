"""LB design numbers (exact, no simulation): why no test can decide with 3 + 1 (or 2 + 2) continuations, what the
counts can bound, and the cost and disk arithmetic.  Read only; prints; writes nothing.  Every input is a constant with
its source named beside it.
Usage: PY -I -B design_numbers.py
"""
from fractions import Fraction
from math import comb


def fisher_one_sided_min(n_blind, n_ctrl):
    """The smallest one-sided Fisher exact p for 'blind rails less often' at n_blind + n_ctrl: 0 of n_blind rail and
    every control rails (hypergeometric, the margins fixed)."""
    n, k = n_blind + n_ctrl, n_ctrl   # k rails in all, all in the control arm
    return Fraction(1, comb(n, k))


print("A. THE SMALLEST ATTAINABLE ONE-SIDED P (exact), 'blind rails less often than control'")
for nb, nc in ((3, 1), (2, 2), (4, 0)):
    if nc == 0:
        print(f"   {nb} + {nc}: no within-study comparison")
        continue
    p = fisher_one_sided_min(nb, nc)
    print(f"   {nb} + {nc}: Fisher exact, 0 of {nb} against {nc} of {nc}: p = {p} = {float(p):.4f}")
print("   (a rank or permutation test on any continuous outcome: 3 + 1 -> 1/4, 2 + 2 -> 1/6; neither can reach 0.05)")
for nb, nc in ((3, 1), (2, 2)):
    print(f"   permutation minimum {nb} + {nc}: 1/C({nb + nc},{nc}) = 1/{comb(nb + nc, nc)} = {1 / comb(nb + nc, nc):.4f}")

print("\nB. WHAT THE BLIND ARM'S COUNT CAN BOUND (one-sided Clopper-Pearson, exact for 0 events)")
for n in (2, 3, 4):
    up = 1 - 0.05 ** (1 / n)
    print(f"   0 of {n} railed: the per-run rail chance under blinding is below {up:.3f} at 95% (upper bound); "
          f"0 of {n} has chance {0.5 ** n:.3f} if that chance is 0.5")
print("   chance to see at least one blind rail (an 'escape') if each blind run escapes with chance q:")
for q in (0.1, 0.2, 0.3, 0.5, 0.7):
    print(f"     q {q:.1f}: 2 runs {1 - (1 - q) ** 2:.3f}, 3 runs {1 - (1 - q) ** 3:.3f}, 4 runs {1 - (1 - q) ** 4:.3f}")

print("\nC. THE CONTROL'S CHANCE TO RAIL, FROM THE RECORD (context, not a test)")
for k, nm in ((12, "ECS: 12 of 12 s42 continuations from 3,145,728 any-flagged by E (ECS score.txt; RUNLOG l740)"),
              (4, "BS2: 4 of 4 controls climbed (rise +0.29 to +0.67 by 3,538,944; 2 of 4 over the bar)")):
    print(f"   {nm}: one-sided 95% lower bound {0.05 ** (1 / k):.3f}; two-sided 95% lower bound {0.025 ** (1 / k):.3f}")
for p in (0.95, 0.9, 0.8, 0.7):
    print(f"   if one control rails with chance {p:.2f}: P(control does not rail) {1 - p:.2f} with 1 control; "
          f"P(neither of 2 rails) {(1 - p) ** 2:.4f}; P(at least one of 2 fails to rail) {1 - p ** 2:.3f}")
print("   if blinding had no lasting effect and each blind run railed with the record's chance p, P(0 of 3 blind "
      "rail):")
for p in (0.95, 0.9, 0.8, 0.735, 0.6, 0.5):
    print(f"     p {p:.3f}: 0 of 3 {(1 - p) ** 3:.4f}; 0 of 2 {(1 - p) ** 2:.4f}")

print("\nD. COMPUTE (inferred from BS2's measured wall times; LB's harness builder's TIMING supersedes)")
S, E = 2097152, 12001280
roll_bs2, roll_lb = (3538944 - 2097152) // 8192, (E - S) // 8192
for mins in (82, 85, 89):
    per = mins * 60 / roll_bs2
    print(f"   BS2 {mins} min for {roll_bs2} rollouts with 4 running -> {per:.1f} s per rollout -> LB {roll_lb} rollouts "
          f"{per * roll_lb / 3600:.2f} h")
print(f"   V4 (harness builder, load 6-7): 32 s per rollout -> {32 * roll_lb / 3600:.2f} h")
print(f"   evaluations: BS2 11 in {roll_bs2} rollouts; LB 75 in {roll_lb} (the same ratio, {11 / roll_bs2:.4f} vs "
      f"{75 / roll_lb:.4f}), so the per-rollout figure includes them")

print("\nE. DISK (bytes; the harness builder's disk_estimate.out and the measurement builder's NOTES)")
GiB = 1024 ** 3
run_est = 193386567       # per run with reads and one resume [measured estimate, LB validate/logs/disk_estimate.out]
pair = 4033138 + 4917      # one periodic pair: zip + pkl [measured, the same file]
floor_meas = 3 * GiB       # measure_r10round.py's floor (r10m_common.DISK_FLOOR_BYTES; THIS12 line 958)
need_meas = floor_meas + 20 * 1024 ** 2 + 64 * 1024 ** 2   # floor + one M1 output + margin (measure_lb.py)
free_now = int(2.9e9)      # df -h /tmp 17:36 UTC, 2.9G avail [measured, CMD_LOG entry 0001]
launch_floor = int(1.5 * GiB) + 4 * 200 * 1024 ** 2   # bs_common.LAUNCH_DISK_BYTES
print(f"   per run {run_est:,} B ({run_est / 1024 ** 2:.1f} MiB); 4 runs {4 * run_est:,} B ({4 * run_est / GiB:.2f} GiB)")
print(f"   the runner's launch floor {launch_floor:,} B ({launch_floor / GiB:.2f} GiB); pause floor {int(1.5 * GiB):,} B")
print(f"   the measurement needs {need_meas:,} B free at each launch ({need_meas / 1e9:.2f} GB); measure_r10round.py "
      f"refuses below {floor_meas:,} B")
after = free_now - 4 * run_est
print(f"   free now about {free_now / 1e9:.2f} GB -> after 4 runs about {after / 1e9:.2f} GB [inferred]; short of the "
      f"measurement's need by {(need_meas - after) / 1e9:.2f} GB")
prunable = 37 - 9           # periodic pairs not on the 1,048,576 loop schedule (which holds J and R1)
print(f"   periodic pairs per run 37; on the 1,048,576 schedule 9 (incl. J 4,194,304 ... 10,485,760 and R1); the other "
      f"{prunable} x {pair:,} B = {prunable * pair / 1e6:.0f} MB per run, {4 * prunable * pair / 1e6:.0f} MB for 4 "
      f"(fewer if M5 names RESOLUTION checkpoints off that grid)")
print(f"   so pruning inside LB/runs after the reads gives about {(after + 4 * prunable * pair) / 1e9:.2f} GB: still short "
      f"by about {(need_meas - after - 4 * prunable * pair) / 1e9:.2f} GB -> space must be freed elsewhere first "
      f"(the coordinator, with the maintainer's word for other studies' files), or the reading waits (INCOMPLETE)")
# Revised (18:15 UTC): the loop schedule stays every 262,144 (all periodic pairs read, then cached); after the reads only
# the measurement's labels are needed: J up to E (5 periodic: 4,194,304, 5,242,880, 6,291,456, 8,388,608, 10,485,760,
# R1 among them) and the final pair, plus any RESOLUTION checkpoint.
pr2 = 37 - 5
validate_mb = 176e6        # du -sm LB/validate at 17:58 UTC [measured, CMD_LOG entry 0061]
print(f"   (revised) with every pair read first: the periodic pairs the measurement does not name, {pr2} x {pair:,} B = "
      f"{pr2 * pair / 1e6:.0f} MB per run, {4 * pr2 * pair / 1e6:.0f} MB for 4 (fewer if RESOLUTION checkpoints are "
      f"named); with LB/validate's {validate_mb / 1e6:.0f} MB: {(4 * pr2 * pair + validate_mb) / 1e6:.0f} MB inside LB; "
      f"after the runs about {after / 1e9:.2f} GB + that = {(after + 4 * pr2 * pair + validate_mb) / 1e9:.2f} GB, short "
      f"of {need_meas / 1e9:.2f} GB by {(need_meas - after - 4 * pr2 * pair - validate_mb) / 1e9:.2f} GB")
