# The lead's rulings of 15 Nov 2026 on the builds' independent reviews

Each block below is the addendum the lead appended to a package's brief before that package's fixer started, so
the fixer worked from it. The reviewers' findings they answer are in `checkpoint/reviews/`. Where a ruling changes
a pre-registered rule, the change is written into `STAGE2_75_PLAN.md` with its label (its last section lists
them); the Track B points (package CE) go into `STAGE2_TRACKB_PLAN.md`, which is still a draft. A working paper,
not a rule.


## Round A, package M (moves)

These settle the points the reviewers handed up. Where a ruling and a reviewer's proposed fix differ, the ruling wins.
The lead writes the matching words into `STAGE2_75_PLAN.md`; you do not edit that file.

**M-R1. The two causal ZMP limits each read their own frames** (mechanics finding 1, bit-identity finding 1). Take
the reviewer's recommended fix, as follows.
- `zmp_consistency` (limit 2.5 mm, unchanged) is the largest |e| over the output frames with phase < command + 1.
- `post_window_zmp` (limit 4.0 mm, unchanged) reads the frames from one stride after the command on, as now.
- Keep the all-frames value in every entry as a recorded field, `zmp_consistency_all_m` (and its history if the
  history is kept per pass), gated by nothing.
- `RULES['S2A-gates']` says exactly: "consistency error over the first stride after the command at most 2.5 mm; ZMP
  error from one stride after it at most 4.0 mm (windows fixed on 15 Nov after the scratch builds of V0, before any
  set was frozen and before any closed-loop episode on a Stage 2 set: STAGE2_75_PLAN.md A7)".
- This is a rule amended after a prescribed-motion result and is labelled so in the plan. Neither constant moves.
- Update `test_causal_fade_window` to the windowed reading, and restructure `test_stop_and_turn_from_v0` so that one
  failed gate cannot hide the other readings (bit-identity finding 1, part 2). Expected under this reading: V0 turn
  2.04 mm / 2.76 mm, V0 stop 1.62 / 0.53. If your build does not reproduce these to the digits the reviewer logged,
  stop and report; do not tune anything.

**M-R2. The acceleration reading** (mechanics finding 2). Take the reviewer's fix, both parts together: second
differences on the frames' own time grid, then interpolation for the filter; the filter is given the walk (or the
stand) before the first frame and after the last as its history; both peaks are read over the move's own frames
only, and `accel_filter` says so. The gate's constant does not move. Expected for V0's stop: low-passed 0.623, raw
1.90 m/s^2.

**M-R3. The 1.0 rad/s ladder turn** (mechanics finding 3). Option (a): no code change. Rule A11's third outcome
already covers it: the turn is built, written with `pass_: false` and its numbers, and not frozen; the set builder
records "no reference within the heading rule" for that rate, and it goes to the owner in the report. The
slowest-clock gate stays on every rate.

**M-R4. Everything else.** Fix bit-identity findings 2 to 7 and mechanics findings 4 and 5 as the reviewers propose.
For bit-identity finding 4, restore `gait.ACCURACY` on exit wherever this module sets it, and say in the docstring
which calls set it. For finding 6, the default suite writes nothing under the study tree (use `tmp_path`).

**M-R5. After your last edit to `moves.py`**, make `stage2/sets/legacy_check.json` again for both picks (about 55 min
of CPU, one heavy process at a time through `tools/lowprio.sh`), and report the tool's own last line. The record must
show 36 of 36 equal.


## Round A, package B (the bar's runner)

These settle the points the reviewers handed up. Where a ruling and a reviewer's proposed fix differ, the ruling wins.
The lead writes any matching words into `STAGE2_75_PLAN.md`; you do not edit that file.

**B-R1. The attribution units stand where the pre-registered order puts them** (judge finding 1, blocker). The rule
stands and the code changes: take the reviewer's fix in full (`run_75cm.sh` runs T3.1 and T4.1 with `--attribute
none`, then one attribution step per set in the set order before T3.2 / T4.2; in `bar.evaluate` a unit's episodes
finish before any attribution unit starts; the order test asserts it).

**B-R2. The turn item's window** (judge finding 2). Option (a): the joint range and the leg gap are read over
[t_c, t_c + at_s + 3 s], as C5 says. The drives' peak speed stays over the whole episode, because the recorder holds
the episode's peak only and `control.run` is not edited; `RULES['turn']` says so in these words: "the drives' peak
speed is read over the whole episode (the recorder holds the episode's peak only), which is at least as strict as
the window". The record carries `drive_speed_span: "episode"`. The lead adds the same sentence to C5 as a labelled
note. Add the test with a lead-in excursion.

**B-R3. Toe clearance, C3: the lead amends the reading, and you build it** (judge findings 5 and 15; review of R,
finding 1; R's fixer). On Stage 1's stock walks with no servo delay the reading as first written (middle half of the
swing's own duration) is 0.010 to 0.017 H under Stage 1's and fails the R0 row in six of six episodes, because the 2 %
load rule puts the touchdown 7 to 17 ms after the reference's and the 75 % edge on the falling flank. The lead amends
C3 before any set is frozen, with a label. Build exactly this in `designlab/stage2/readings.py` (it is in your
worktree through the merge of R; merge the working branch first so that you start from R as merged):

- Contact, air runs, the swing / air-gap rule, *interrupted* and the window rule ("each swing wholly inside the
  window") do not change.
- A swing is read as the lowest point of the foot over **[t_lo + 0.25 T_p, t_lo + 0.75 T_p]**: `t_lo` is the swing's
  own lift-off time; `T_p` is the planned duration, on the simulated clock (converted exactly as C3 already converts
  the shortest planned swing), of the reference swing of that foot that overlaps the air run the longest, or the
  track's shortest planned swing when it overlaps none. The value is interpolated at both edges and taken at every
  update between, as now.
- The interval is read to its end even when the swing's own touchdown comes before it: the foot is then on the floor
  and the reading says so (an early touchdown inside the middle half fails the term). A swing whose interval runs
  past the window's end is not read and not counted.
- The reading as first written stays in every record beside it as `toe_clearance_own_duration_m`: reported, never
  judged. Stage 1's reading stays as it is.
- Per episode the record also holds, for the swing that gave the smallest judged reading: its own duration, `T_p`,
  the time of the minimum as a fraction of `T_p` from `t_lo`, and whether an edge gave it.
- The count rule (ruling of the lead on the R fixer's point): a reference swing counts toward a foot's number only if
  no window edge cuts the own swing that overlaps it. "Its number of reference swings less one" then applies to that
  count.
- The term keeps its name, `toe_clearance_own_swing`, and its limit, 0.04 H. Its rule text ends: "(middle half of the
  planned swing counted from the own lift-off; amended on 15 Nov after the reread of Stage 1's stock walks, before
  any Stage 2 set was frozen: STAGE2_75_PLAN.md C3)".
- Tests: one per clause above (a pure delay leaves the reading where it was; a late soft touchdown no longer lowers
  it; a touchdown at 48 % of the planned swing fails; a swing cut by a window edge is neither read nor counted), and
  keep every existing test of `readings.py` green or change it with the reason in a comment.
- Evidence the lead will check against his own implementation: the R reviewer's table again under the amended
  reading (75-M-H-10, stock `walk/nominal`, `walk/brisk`, `walk/fr05`, seeds 3042 to 3044, hard floor, no delay,
  rows R2/42 and R0, last 10 s; S1 view, which rule R4 allows). R's fixer measured about 0.0477 H (R0, nominal),
  0.0517 H (R0, brisk), 0.0435 and 0.0483 H (R2/42); report yours to four places with the reading as first written
  beside it. Do not tune anything to these.
- Also fix finding 5: tell the lead the exact record paths of the toe readings so that C3's Check line names them.

**B-R4. Scratch-set episodes and rule R4** (judge finding 16). Rule R4 protects the three registered sets. Episodes on
a scratch or fixture set built from Stage 1's gaits are allowed for testing the runner under R0 rows only, never
under an R2 row. Every such closed-loop episode of a Stage 2 move run so far (the builder's smoke on D0, the judge
reviewer's 15, the provenance reviewer's, yours) is listed in summary form in
`stage2/runs/2026-11-15-75cm/dev_episodes.json` under a key `scratch_sets`: set, rows, conditions, seeds or counts,
who ran them and why. Take the counts from the two reviews' `commands_run` and the builder's logs; where a count
cannot be recovered say so in the entry.

**B-R5. Provenance findings 1 to 5.** Take each fix as the reviewer wrote it. For finding 2 agree the label's field
name with `setio` as merged (`designlab/stage2/setio.py`); if `setio` has no such field, read
`set.json`'s `freezes[i].change_after_result` (a string, absent when none) and say so in your report so that work
package S writes it.

**B-R6. Provenance finding 6, traces under a short disk.** The second option: the trace is always written and hashed;
when it is not kept the record holds `trace = {sha256, rows, kept: false, reason}` and the session holds
`no_traces_for`. Rule R3's hash is never dropped.

**B-R7. Everything else.** Fix judge findings 3, 4, 6 to 14 and provenance findings 7 to 14 as the reviewers propose.
Where a fix would change the wording of a pre-registered rule rather than make the code match it, do not make it:
list it for the lead with the rule's id.

**B-R8. For `COST.md`** (judge finding 8): measure each controller's rate after a warm-up episode that is not
counted, on episodes that run their full length, and record the load (other processes) at the time.


## Round A, package G (regression record)

**G-R1. The stride period of a K-stride window is exact for every K** (finding 2). Take the reviewer's exact form: in
`add_periodic` compute `stride = (ref.n // K) * ref.step` once and use it for `frame_info`'s period and the
segment's `stride_s`; raise `ValueError` when `K < 1` or `ref.n % K`. With no key (K = 1) this is `n * step`, the
expression of `Reference.period`, so the default path must stay bit-identical: show it with the old-against-new
comparison of `add_periodic` on all 20 references of both bodies and with the regression. Add K = 3 and a refused
`n % K != 0` case to `test_window_strides_key`. (The six V6 windows of the look ladder all have 160 frames per
stride, so none is refused.) This is the last edit to `control.py` in this round.

**G-R2. After that edit** make the regression record again, fast and full tiers, at the edit's commit, one heavy
process at a time through `tools/lowprio.sh`, and report the tool's own last line. The lead makes it once more at
code state A.

**G-R3. Findings 1 and 3** as the reviewer proposes. For 3 the record and every text you own say how many distinct
committed episodes the full tier proves (248) beside the count of comparisons (256); tell the lead where a
pre-registered text quotes 256.


## Round C, package CE (the training env)

These settle the points the reviewer handed up. Where a ruling and the reviewer's proposed fix differ, the ruling
wins. The lead writes the matching words into `STAGE2_TRACKB_PLAN.md` (still a draft, in force only once settled and
merged; no Track B run has been made); you do not edit that file.

**CE-R1. The reward's four readings stand as coded** (finding 1, blocker). The lead amends W1, W2, W3 (a) and W5 to
the code's readings, tagged [lead, before any Track B run]: the duty term reads a foot only where the reference's
own share over the same stride lies inside the band; progress is min(1, max(0, (v_ref_bar - v_bar) / (1.05 v_cmd)) /
0.10) with v_ref_bar as `r5core.REWARD_RULE` defines it, restarted at the segment's first frame and read after one
stride; clearance (a) times the swing by the planned length of the reference's swing in progress, on the reference
clock; positions are those of the last physics step's start. The reason is one the rules already state: a playback of
the reference scores every penalty zero. Your part: add the last sentence to `r5core.REWARD_RULE`, and make W2's
check an equality of the env description's `reward_rule` text (a test that fails when the formula text and the rule
text differ). No formula changes.

**CE-R2. The commanded heading in a turn in place** (finding 2). Take the proposal: inside a finite segment that is
no finite turn and that enters or leaves a gait with a yaw rate, the command stays E4's and the commanded heading
follows the reference's own yaw; the next segment's `heading0` is its value at the segment's end. Add the in-place
plan on S1 arrays to the playback test. For the finite turns nothing changes in the code: the bar judges the heading
against the command's own integral, so a playback carries the heading penalty of its ramps. Add one test that
measures that penalty on a finite turn of the S1 arrays and pins it, and give the number in your report; the lead
writes the sentence into W1.

**CE-R3. A later freeze of the same set** (finding 3). Take the fix as written (rule K4).

**CE-R4. Scripted head targets keep the bar's joint margin** (finding 4). Rule: no scripted target of the head lies
within 0.15 rad of a joint limit (the bar's 0.10 rad margin plus 0.05 rad for tracking). Compute the numbers from
each -X body's own ranges and report them; for the jaw's [0, 0.6] rad that gives `JAW_OPEN_RAD = (0.3, 0.45)`, and
the gaze target is cut the same way from the neck yaw's range. Add the test on both -X bodies (a playback with head
events at their extremes scores range 0 and meets the bar's 0.10 rad). This narrows how far the jaw opens in a
script; it is a look choice the lead reports to the owner.

**CE-R5. The rest.** Findings 5, 6 and 7 as the reviewer proposes; for 6 take a rung's target speed from its name, as
the bar does, so that training and the bar apply the same duty band at 0.30 m/s. Finding 8: no code change; the lead
adds the numbers to T1 and J4. Finding 9: add the refusal as written (`allow_75cm_round=False`, rule X1).
