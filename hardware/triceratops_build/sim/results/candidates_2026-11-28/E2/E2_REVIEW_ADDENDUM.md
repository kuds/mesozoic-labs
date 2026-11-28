# E2 transmission and speed cap: fixes after the independent review (28 Nov 2026)

**DESK readings of E2's stored outputs; no new episode; nothing measured.** This addendum answers review findings
E2-R1 to E2-R4. Every reading in it was chosen after E2's results were seen, so all of it is **changed after a
result**. No verdict, rule, seed or threshold of `design/PREREG_CANDIDATES_2026_11_28.md` (section 4 E2; section 3
items 3-6, 13, 14) is changed.

`E2_summary.md` and the other E2 files are left as they were, since no existing file is edited. Their sha256 values
are in `e2_review_addendum.json` under `files_left_unchanged_sha256` (`E2_summary.md` `75f4ffe2458ea55d`). **Where
this addendum and `E2_summary.md` differ, this addendum holds.** The passages it replaces are named below.

- Runner: `/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E2/code/e2_review.py`,
  sha256 `9c46836b49ee6513`, recorded before it ran.
- Log: `/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E2/logs/e2_review.log`.
- Commit `fb1e80c8`. `sim/candidate_checks.py` is unchanged (`008aa308debdf1b6`).
- New files here: `e2_review_addendum.json` (every reading below, with provenance) and `e2_review_speed_failures.csv`
  (all 44 failing policy, block, r, placement and drive entries of the E2(a) speed check).

## Outcome (unchanged)

- **E2(a):** 0 of 45 judged rows PASS.
- **E2(b):** rod, belt and cable are each COMPATIBLE WITH TODAY'S POLICIES, 4 of 4 policies. This holds under the
  pre-registered torque-envelope cap, which **did not hold the drives to 83 rpm** (E2-R2 below).
- **The R1 numbers were checked.** The runner recomputed E2(a)'s speed check from the B1 1 ms episode files with its
  own percentile code. It printed: `max |dp99| 0 rpm, max |dpeak| 0 rpm, pass flags 120 of 120 equal`.

## Findings

| Finding | Disposition | What was done |
|---|---|---|
| E2-R1: "every speed failure is s3" is false at the knees and both | **Accepted** | Confirmed: s0 and s1 also fail at r = 1.25-1.3 on the left fore elbow (an X6-60). The claim is narrowed to hip pitch below. The verdict is unchanged. |
| E2-R2: the E2(b) headline leaves out that the cap did not hold 83 rpm | **Accepted** | Confirmed from `e2b_summary.json` and the per-episode files. The headline is qualified below. E2(a) stays the reading on speed. |
| E2-R3: the X6 peak allowance of 1.1 x 108 rpm was chosen after the output | **Accepted** | Labelled "changed after a result", with the pre-registration's wording beside it. The plain 108 rpm comparison is added, also "changed after a result". The script's sha256 is given. |
| E2-R4: "Committed results" | **Accepted** | `git status --short` shows the folder as untracked (`?? hardware/triceratops_build/sim/results/candidates_2026-11-28/`). These are result files, untracked, for the owner to commit. |

## E2-R1: where the E2(a) speed check fails (DESK on SIMULATION logs)

The data are the B1 1 ms logs: walking at 0.25 m/s and standing, weighted 0.75 / 0.25, with the four episodes of
each seed block pooled. The limits are p99 <= 83 rpm and peak <= 91.3 rpm on each reduced drive. Each entry below
is the first r at which a drive fails, from the runner's output.

| Placement | Policy | Drive | Block A: first failing r, peak rpm | Block B: first failing r, peak rpm |
|---|---|---|---|---|
| pitch | s3 | right fore hip pitch (X4) | 1.2, 99.17 | 1.1, 94.18 |
| knees and both | s3 | left fore elbow (X6-60) | 1.0, 96.60 | 1.0, 94.91 |
| knees and both | s1 | left fore elbow (X6-60) | 1.25, 92.18 | 1.3, 93.21 |
| knees and both | s0 | left fore elbow (X6-60) | 1.3, 93.85 | 1.3, 91.37 |
| both | s3 | right fore hip pitch (X4) | 1.2, 99.17 | 1.1, 94.18 |

- **s0 and s1 fail on the peak only.** Their p99 there is 58.6-64.7 rpm. s0 on block B at r = 1.3 is over the
  limit by 0.07 rpm.
- **s2 has no speed failure in any row.**
- **No failure at hip pitch comes from a policy other than s3.** The runner printed
  `non-s3 failures at hip pitch: []`.

**Replaces, in `E2_summary.md` ("Where it fails"), the bullet "Every speed failure comes from policy s3 alone", and
the same line in the runner's report:**

> At hip pitch every speed failure is s3's right fore hip pitch. At the knees and both, s3 fails from r = 1.0, s1
> from r = 1.25 (block A) and s0 from r = 1.3. All three fail on the left fore elbow (X6-60), at the 83 rpm limit
> the pre-registration applies to it (item 6).

**Replaces the runner's report line "Placement both behaves the same way, with worse speeds":**

> Placement "both" has the same heat figures as hip pitch, with worse speeds (86.0 / 115.9 rpm at r = 1.2, s3). On
> the speed check, s3 fails from r = 1.0, s1 from r = 1.25 and s0 from r = 1.3, on their left fore elbows.

**Per policy, both placement (supporting, not the verdict; `e2a_summary.json` `per_policy_all_pass`, every
transmission alike):**

| r | Policies that pass all three checks |
|---|---|
| 1.2 | s0, s1, s2 |
| 1.25 | s0, s2 |
| 1.3 | s2 only |

At the knees alone, s2 and s3 fail C1 and C2 in every row.

## E2-R2: drive speeds in the E2(b) runs (SIMULATION)

These readings are for the eight reduced drives, drive side. Speeds are pooled per condition over the 8 seeds,
from 2 s to 20 s, sampled at every physics step. The conditions are walking at 0.25 m/s and the two 0.5 m/s pushes.
E2(b) has no standing condition, so this compares numbers with E2(a)'s limits; it is not E2(a)'s check. The at-cap
share is one drive's share of samples at or over 83 rpm in one episode, counted over the whole episode (E2
summary, change 3).

| Run | Worst p99, rpm | Worst peak, rpm | Walk 0.25 only: p99 / peak, rpm | Episodes with a reduced drive's peak > 91.3 rpm | Largest at-cap share |
|---|---|---|---|---|---|
| B1 (r = 1, no cap) | 75.3 | 108.67 | 75.0 / 96.6 | 22 of 96 | - |
| B025 (r = 1, no cap) | 68.7 | 105.3 | 68.1 / 93.74 | 18 of 96 | - |
| R (supporting) | 82.8 | 94.44 | 82.8 / 94.44 | 6 of 96 | 0.765 % |
| **rod** | **83.3** | **114.8** | 82.6 / 106.06 | 24 of 96 | 1.24 % |
| **belt** | **83.9** | **104.0** | 83.8 / 104.0 | 25 of 96 | 1.787 % |
| **cable** | 82.3 | **110.21** | 82.2 / 102.77 | 24 of 96 | 1.124 % |
| belt_1span (supporting) | 83.5 | 149.66 | 83.3 / 120.26 | 26 of 96 | 1.487 % |

- **Every worst figure is fore128-s3's,** on its left fore elbow or right fore hip pitch.
- **Against E2(a)'s numbers,** rod and belt are over on both p99 and peak, and cable on peak. On walking alone,
  all three are over on peak, and belt also on p99.
- **The cap worked as pre-registered.** `SpeedCap` (frozen code) cuts the drive's torque in the direction it turns
  from 0.95 to 1.0 of 83 rpm, and it can always brake. It is a torque envelope, not a hard speed limit. Impacts and
  the series spring can still drive the rotor past 83 rpm. That is how it was pre-registered and run, so this is not
  a deviation.
- **Context:** today's plant at r = 1 (B1) already peaks at 108.67 rpm, on s3's X6 elbow (push right).

**Replaces the E2(b) headline (the runner's short answer, and the E2(b) table's verdict lines read alone):**

> E2(b): rod, belt and cable are each COMPATIBLE WITH TODAY'S POLICIES (4 of 4) under the pre-registered
> torque-envelope cap, which did not hold the drives to 83 rpm. The judged variants reach p99 up to 83.9 rpm and
> peaks of 104-115 rpm, and a reduced drive passed 91.3 rpm in 24-25 of each variant's 96 episodes. This is not
> evidence that today's policies run within an 83 rpm drive limit. E2(a) stays the reading on speed.

**Replaces, in `E2_summary.md` ("What it means"), "the 83 rpm cap at r = 1.25 is tolerated", and the report's "the
capped, reduced plant is tolerated":**

> In simulation, today's four policies tolerate a reduced, compliant plant whose drives are torque-limited above
> 0.95 x 83 rpm. The drives still overspeed (above), so this does not show that the policies would run within an
> 83 rpm limit.

## E2-R3: the X6 reading, supporting (changed after a result)

**Pre-registration wording (section 3 item 6):** "the 108 rpm figure is shown beside it".

- **As run:** the X6 passes at p99 <= 108 rpm and peak <= 1.1 x 108 = 118.8 rpm; the X4s stay at 83 / 91.3 rpm.
  This is **changed after a result**. `e2a_x6_beside.py` chose the 1.1 factor for the X6 and the pass/fail framing
  after `e2a_summary.json` was written (script mtime 07:08, summary 07:06). The script's sha256 is
  `9f649f170beb09f6`, also recorded in `e2_checks.json` (`scripts`). Its output file,
  `e2a_speed_x6_allowance_supporting.csv`, carries no provenance of its own. This file and `e2_review_addendum.json`
  give it.
- **Plain 108 rpm (added here, also changed after a result):** the X6 passes at p99 and peak <= 108 rpm.

| Reading | Rows where the speed check alone passes on all 8 policy-blocks | Rows that pass C1, C2 and this speed reading |
|---|---|---|
| As run (108 / 118.8) | pitch r 1.0, knees r 1.0-1.2, both r 1.0 | none |
| Plain 108 | pitch r 1.0, knees r 1.0-1.1, both r 1.0 | none |

Neither reading is judged, and no verdict changes.

## E2-R4: wording

The E2 files are **result files in `sim/results/candidates_2026-11-28/E2/` (untracked, for the owner to
commit)**. No commit holds them, and HEAD `fb1e80c8` has no E2 file. The pre-registration's "Committed results"
names the folder's intended role, not its state in git.

## What it means, corrected (desk and simulation; no hardware; nothing adopted)

- **For A (60 cm, today's drives).**
  - (a) still matches the pre-registered outcome "C1 and C2 pass only where the speed check fails". With today's
    four policies, the 83 rpm limit blocks the reduction route at 60 cm.
  - **At hip pitch, the block is s3 alone.** A hip-pitch reduction would want either a policy other than s3 or a
    policy trained with the cap and the reduction. At r = 1.2-1.3, s0, s1 and s2 pass all three checks on both
    blocks (per policy, supporting).
  - **For a knee or both-joint reduction at r >= 1.25,** "a policy other than s3" is not enough on its own. s1
    (r = 1.25, block A) and s0 (r = 1.3) also fail, on the left fore elbow (X6-60), at the 83 rpm limit applied to
    it.
  - A knee-only reduction leaves every policy's C1 and C2 as they are. In every judged knee row of
    `e2a_grid.csv` (3 transmissions x 5 r x 8 policy-blocks), each policy's deciding drive is a hip pitch.
  - (b) shows that today's policies tolerate the reduced, compliant, torque-capped plant in straight walking and
    0.5 m/s side pushes. That plant still overspeeds (judged variants up to 83.9 rpm p99 and 104-115 rpm peak), so
    (b) gives no evidence that the policies would stay within an 83 rpm drive limit.
- **On cables.**
  - Unchanged: in simulation, rod, belt and cable cannot be told apart at these stiffnesses, which are upper bounds.
  - Wear, pretension loss and construction stretch were not modelled.
  - The lead's "within the 83 rpm drive limit" does not hold for s3 at any placement for r = 1.2-1.3. At hip pitch
    and r = 1.1 it fails on block B only. With the knees reduced (knees or both), it also fails for s1 from r = 1.25
    and for s0 at r = 1.3.
- **For B (50 cm, L1).** Not tested in E2. The inference that a reduction there would fail the speed check by more
  is unchanged, and it is an inference, not a result.
