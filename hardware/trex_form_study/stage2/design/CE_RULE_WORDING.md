# Work package CE: wording for the Track B rules draft, after the review of the env

For the lead, to paste into `STAGE2_TRACKB_PLAN.md` (the draft in the worktree `prereg-b`; this file edits nothing
there). It follows the lead's rulings of 16 Nov 2026 on the nine findings of
`stage2/design/checkpoint/reviews/roundC__review_CE.json`. Each entry gives the old sentence as the draft holds it, the
exact new sentence, and one of two labels:

- **Label A**: "a wording change made before any training run; the code is unchanged by it".
- **Label B**: "a wording change made before any training run; the code was changed with it" (the finding and the
  commit on `wt/wpc-ce` are named).

No training run, no policy episode and no panel was run for this file. The numbers under "Why" are from kinematic
playbacks (a script's own poses written into the plant's data, no physics step) on Stage 1's stock arrays of
`75-M-H-10` and, where it says so, on the frozen set V0 of `75-M-H-10`; and from the env on Stage 1's arrays. The
tests named are in `tests/test_stage2_r5lib.py`, `tests/test_stage2_r5core.py` and `tests/test_stage2_r5env.py`.

Entries 1 to 4 are the review's blocker (the reward's readings stand as coded; the rule text follows them). Entry 5 is
the turn in place, 6 the head script's ranges, 7 the numbers the rules did not state, 8 the later freeze, 9 rule X1.
Entry 1b needs the lead's word: it says where the draft's W1 ("every positive term 1 and every penalty 0") does not
hold, which the review listed for the walking turns and did not list for the tracking term. Entries 5c and 8 add to a
rule's check what the code now records.

---

## 1. W1 (form)

### 1a. Which instant the reward reads

**Old** (W1): "each term in [0, 1], read on the true state at the end of every env step (c the reference clock then,
c0 at its start)."

**New**: "each term in [0, 1], read on the true state at the end of every env step (c the reference clock then, c0 at
its start). Joint angles are those at the env step's end. Body positions and orientations (the CoM, the trunk's
attitude and yaw, the feet, the head, the feet's lowest points and slip speeds) and the foot loads are the plant's own
after the env step's last physics step, which the simulator computes at that physics step's start, 2 ms before the env
step's end: the plant's data is read and never recomputed, because a second forward pass would change the solver's
warm start and gate Z1 would no longer hold."

**Label A.**

Why: the review's blocker, part 4. In the env, after 15 env steps of noise actions (sigma 0.1), the foot term read
0.9046, which is the term of the pose 2 ms before the step's end to 1e-9; the term of the pose at the step's end is
0.9072. The joint term is the one of the angles at the step's end. Test:
`test_the_reward_reads_positions_as_of_the_last_physics_steps_start`.

### 1b. What a playback scores (needs the lead's word)

**Old** (W1): "A zero action on a kinematic playback scores every positive term 1 and every penalty 0."

**New**: "A kinematic playback of the reference scores the imitation terms (joint, foot, contact, head) 1, the
tracking term at least 0.95, and every penalty 0, with one exception: inside a finite turn the `heading` penalty
charges the playback, because the commanded heading is the integral of a rectangular yaw-rate pulse (E4) while the
reference's own yaw ramps. On set V0 of `75-M-H-10` (kinematic playback) the reference's yaw is up to 9.0 degrees
behind the commanded heading in the quick turn, 5.4 degrees at 0.5 rad/s and 3.8 degrees at 0.35 rad/s, and the
penalty reaches 0.35, 0.14 and 0.04; in the left turns it is above zero on 98, 148 and 207 of the task's 1,499 env
steps (task means 0.021, 0.013, 0.006), in the right turns on one step more each."

**Label A** (the code is as the lead ruled it: walking turns keep rule E4).

Why: two places where the draft's sentence is not what the code does.

- *The heading penalty in a walking turn.* The review listed it under its finding on the turn in place and computed
  8.4 degrees for the quick turn. Measured on V0 by a kinematic playback of every task script (no physics step; a
  script of this session, read-only on the main checkout's set, kept nowhere and no record): `turn_q90_left` and
  `_right` heading up to 0.351, `turn_0.5_*` up to 0.140, `turn_0.35_*` up to 0.045, every other penalty of every one
  of V0's 15 available tasks 0 (progress below 1e-4).
- *The tracking term.* The review did not list it. Stage 1's term compares the CoM's speed over the 20 ms step with
  the reference's speed at the step's start; on a playback it is not exactly 1. Smallest values seen: 0.9938 on
  Stage 1's walk task, 0.989 to 0.994 on V0's walks, 0.962 in V0's quick turn, 0.951 in Stage 1's turn in place. The
  tests ask for at least 0.99 on the walks and the stop and 0.95 on the turn in place.

If the lead would rather have the heading penalty score a playback 0 in walking turns too, that is a code change
(the commanded heading following the reference there as well) and a change to E4; it is not made here.

---

## 2. W2 (the table and its check)

### 2a. The duty row

**Old** (W2's table): "| duty, slip, impact, tilt | 0.02 each | Stage 1's (W3 for the duty band) | Stage 1 |"

**New**, two rows:

"| duty | 0.02 | W3 | Stage 1, read against the reference's own contact share (W3) |"

"| slip, impact, tilt | 0.02 each | Stage 1's | Stage 1 |"

**Label A.**

### 2b. The check

**Old** (W2): "Check: `settings.env_configs[].reward` and the env description's `reward_scales` in every `run.json` equal
this table."

**New**: "Check: `settings.env_configs[].reward` and the env description's `reward_scales` in every `run.json` equal
this table, and the env description's `reward_rule` is the text of `r5core.REWARD_RULE` at the run's commit, which
states every term's formula as coded (W3's and W5's included)."

**Label A.**

Why: the review's blocker: W2's check compared weights and scales only, so a run with another formula would have
passed it.

---

## 3. W3 (clearance and duty)

### 3a. Clearance (a)

**Old** (W3): "`clearance` is the larger of (a) for a foot in the air between 25 % and 75 % of the planned swing time
since its **own** lift-off and lower than 0.04 H: the shortfall over 0.04 H;"

**New**: "`clearance` is the larger of (a) for a foot not in contact at the env step's end whose `u = (c - c_lift) / T`
lies between 0.25 and 0.75 and whose lowest point is lower than 0.04 H: the shortfall over 0.04 H. `c_lift` is the
reference clock at the foot's **own** lift-off (the physics step in which its contact last ended; at an episode's
start, for the foot the reference has in swing, the clock less the reference's swing progress x T, and for the other
foot the clock). `T` is the planned length, in reference time, of the reference's swing in progress at c; in double
support, of its next swing; past its last swing, of that one. Each swing has its own length, also inside a change of
gait;"

**Label A.**

Why: with one swing time per gait, a playback of Stage 1's brisk task was charged inside both changes of gait (13 env
steps, up to 0.78): a foot was "between 25 % and 75 % of its swing" by the constant while the reference had it near
the floor. As coded the playback scores 0 at every env step. Test:
`test_clearance_a_times_a_swing_by_the_references_swing_in_progress`.

### 3b. Duty: the formula and the reference's own share

**Old** (W3): "`duty`'s band is 0.55-0.72 while the commanded target speed is below 0.30 m/s (owner decision 2), else
0.55-0.65."

**New**: "`duty` is `min(1, sum over the feet read of max(0, lo - d_f, d_f - hi) / 0.05)`, d_f the foot's contact
share over the last stride (the physics steps of the last `round(stride period / 1.05 / 20 ms)` env steps); zero
while the reference stands and until that many env steps of the episode have been played. A foot is read only where
the reference's own support share for it over the same physics steps lies inside the band, so that a start, a stop or
a change, where the reference itself is outside the band, is not charged. The band [lo, hi] is 0.55-0.72 while the
commanded gait's target speed is below 0.30 m/s (owner decision 2), else 0.55-0.65."

**Label A.**

Why: Stage 1's formula without the condition charged the reference's own playback in every start and stop: 84 of 918
env steps of Stage 1's walk task and 147 of 1,479 of its brisk task, up to 1.0 (task means 0.088 and 0.079). As coded
the playback scores 0 at every env step, and a body that keeps both feet down while the reference steps is charged in
full. Tests: `test_duty_reads_a_foot_only_where_the_references_own_share_is_inside_the_band`,
`test_stage_1s_penalties_and_the_action_rate`.

### 3c. Duty: what "target speed" is

**New**, added after 3b's last sentence: "The target speed is the speed a rung's name holds, as the bar takes it
(`v0.3` is 0.30 m/s and is not below it, whatever float its entry computed); for a gait whose name holds no speed (the
nominal walk) it is the entry's own."

**Label B** (the review's minor finding on the 0.30 m/s comparison; commit `0dc39b8`).

Why: one rung is exactly 0.30 m/s, and V0's entries show that an entry's float can be one ulp off its name (`v0.5`
holds 0.49999999999999994). The code compared the entry's float; it now reads `Script.target_speed`. Tests:
`test_the_duty_bands_target_speed_is_the_speed_a_rungs_name_holds`,
`test_a_rung_at_0_30_m_s_keeps_the_usual_duty_band`.

---

## 4. W5 (progress, heading, range)

### 4a. Progress

**Old** (W5): "`progress`: only inside a periodic walking segment, from one stride after its first frame, with a
positive command: `min(1, max(0, 1 - v_bar / (1.05 v_cmd)) / 0.10)`, v_bar the bias-corrected exponential mean (time
constant: the stride period over 1.05) of the CoM's speed along the commanded heading (the bar's walk item fails 4.8 %
short of the commanded distance; Stage 1's tracking term charges 0.2 % of the reward for 5 %)."

**New**: "`progress`: only inside a periodic walking segment, from one stride after its first frame, with a positive
command, and once the two means have run one stride of reference time: `min(1, max(0, (v_ref_bar - v_bar) / (1.05
v_cmd)) / 0.10)`. v_bar is the bias-corrected exponential mean (time constant: the stride period over 1.05; one sample
per env step; restarted at the segment's first frame, and started where an episode starts inside the segment) of the
CoM's speed over the env step along the commanded heading. v_ref_bar is the same mean over the same env steps of the
reference's own CoM speed along its stride direction at the commanded clock: 1.05 x the reference CoM's progress over
the reference time the env step covered, over that reference time. The reference's mean over a stride is 1.05 v_cmd
and its speed ripple inside a stride is in both means, so a playback is charged nothing; a clock held slow leaves the
reference's sample as it is and is charged in full (the bar's walk item fails 4.8 % short of the commanded distance;
Stage 1's tracking term charges 0.2 % of the reward for 5 %)."

**Label A.**

Why: against the constant `1.05 v_cmd` the playback's own speed ripple was charged up to 0.286 on Stage 1's nominal
walk (mean 0.059) and up to 0.168 on its brisk walk. As coded a playback scores below 1e-4, and a clock held 5 % slow
scores 0.42 to 0.58. Tests: `test_progress_is_the_shortfall_against_the_references_own_mean_speed`,
`test_progress_charges_a_slow_clock_in_a_steady_walk`.

### 4b. Heading

**Old** (W5): "`heading`: `clip((|yaw - yaw_home - commanded heading(c)| - 3 deg) / 17 deg, 0, 1)` on the true yaw, zero
in a hold."

**New**: "`heading`: `clip((|yaw - yaw_home - commanded heading(c)| - 3 deg) / 17 deg, 0, 1)` on the true yaw, the
difference wrapped to +-180 degrees, zero in a hold. The commanded heading is E4's (in a turn-in-place task the
reference's own yaw). `yaw_home` is the true yaw at the episode's start less what the reference itself had turned by
the start's reference time, so that a mid-task start reads as a playback from the track's start would."

**Label A** for the wrap and `yaw_home`; the turn in place is entry 5.

### 4c. The closing sentence

**Old** (W5): "The reference itself therefore scores zero."

**New**: "The reference itself therefore scores zero on `range`, also with a head command: a scripted head target
keeps 0.15 rad from a joint limit (E5). On `heading` a playback scores zero everywhere but inside a finite turn (W1)."

**Label B** for the head command (entry 6); the heading clause follows entry 1b.

---

## 5. E4 and T1 (the turn in place)

### 5a. E4

**Old** (E4): "The commanded heading is the integral of w over reference time."

**New**: "The commanded heading is the integral of w over reference time, except in the turn-in-place tasks
(`inplace_left`, `inplace_right`), where it is the reference's own yaw less its yaw at the track's start, at every
reference time; the command is the same. A start into the turn commands 0.5 rad/s from its first frame while the
reference ramps up from rest, so the integral ran ahead of the reference through the whole turn and the heading
penalty charged the reference itself. A walking turn keeps the integral, and so does a plan that is given as a plan
(a bar condition's): only a task's definition names the reference's yaw."

**Label B** (the review's major finding on the turn in place, the lead's ruling 2; commit `0dc39b8`).

Why: on Stage 1's arrays the integral is 71.6 degrees ahead of the reference at the first and the last frame of the
periodic part on `75-M-H-10` (64.3 degrees on `60-N-T-10`) and level again only at the plan's end. A kinematic
playback of that plan under the integral: heading penalty mean 0.73, at 1.0 on 69 % of the env steps, the heading
input at its clip on 51 %. Under the task's definition: heading 0 at every env step of the whole task, the input at
zero, every imitation term 1, tracking at least 0.95, every penalty 0. Tests:
`test_in_a_turn_in_place_the_commanded_heading_is_the_references_own_yaw`,
`test_a_playback_of_a_turn_in_place_scores_heading_zero`.

### 5b. T1's row

**Old** (T1's table): "| `inplace_left`, `inplace_right` | `start_inplace_0.5_<dir>`, 3 strides of
`inplace_0.5_<dir>`, `stop_inplace_0.5_<dir>` | 0.5 each |"

**New**: "| `inplace_left`, `inplace_right` | `start_inplace_0.5_<dir>`, 3 strides of `inplace_0.5_<dir>`,
`stop_inplace_0.5_<dir>`; the commanded heading is the reference's own yaw (E4) | 0.5 each |"

**Label B** (as 5a).

### 5c. T1's check

**Old** (T1): "Check: `settings.tasks` (the resolved table: names, plans with their counts of windows, weights) equals
this table less the absent tasks, with that reading where `set.json` `rules` names cut 3."

**New**: "Check: `settings.tasks` (the resolved table: names, plans with their counts of windows, weights, and each
task's `heading`, which is `reference` for the two turn-in-place tasks and `command` for every other) equals this
table less the absent tasks, with that reading where `set.json` `rules` names cut 3."

**Label B** (the resolved table gained the key `heading`; commit `0dc39b8`).

---

## 6. E5 and J4 (the head script's ranges)

### 6a. E5

**Old** (E5): "a train of gaze shifts for the whole episode (yaw target uniform on +-0.5 rad, minimum-jerk over 0.8-1.2
s, held 2-6 s) and one jaw event at a time uniform in the episode's middle half (open to 0.3-0.6 rad over 0.4-0.6 s,
held 1.0-1.5 s, closed over the same time); both always present."

**New**: "a train of gaze shifts for the whole episode (yaw target uniform on +-0.45 rad, minimum-jerk over 0.8-1.2
s, held 2-6 s) and one jaw event at a time uniform in the episode's middle half (open to 0.3-0.45 rad over 0.4-0.6 s,
held 1.0-1.5 s, closed over the same time); both always present. A scripted head target keeps 0.15 rad from a joint
limit: on both `-X` bodies the neck yaw's range is +-0.6 rad and the jaw's 0 to 0.6 rad (closed is the end the jaw
rests on by design and is no limit), so a head that obeys its command keeps the 0.10 rad of joint range that W5's
`range` term and the bar's `joint_range` ask, with 0.05 rad left for the servo's tracking."

**Label B** (the review's major finding on the jaw, the lead's ruling 4; commit `ad64656`). The tag of E5 that says
"head script H1's numbers accepted as drafted" needs "[lead, 16 Nov 2026, before any Track B run: the gaze and jaw
ranges]" beside it.

Why: as drafted a jaw sitting exactly on a commanded opening above 0.5 rad was inside the 0.10 rad margin of its
upper limit: a playback obeying a 0.6 rad opening scores `range` 1.0 with a margin of 0.000 rad. The ruling names the
jaw; the same sentence ("scripted head targets keep 0.15 rad from a joint limit") moves the gaze target from +-0.5 to
+-0.45 rad, because +-0.5 rad leaves 0.100 rad on a range of +-0.6 rad. Tests on both bodies with a head:
`test_no_scripted_head_target_lies_within_0_15_rad_of_a_joint_limit` (H1 over 300 episode seeds and 300 of
training's draws: no target within 0.15 rad; a core is refused if the script's numbers come closer),
`test_a_playback_obeying_the_head_command_scores_range_zero` (range 0, margins 0.150 rad on the neck yaw and the
jaw).

### 6b. J4

**Old** (J4): "With probability 0.5 a train of gaze shifts for the whole episode: yaw target uniform on +-0.5 rad,
minimum-jerk over 0.8-1.2 s, held 2-6 s. With probability 0.25 one jaw event: open to 0.3-0.6 rad over 0.4-0.6 s, held
1.0-1.5 s, closed over the same time."

**New**: "With probability 0.5 a train of gaze shifts for the whole episode, the first starting at the episode's first
instant: yaw target uniform on +-0.45 rad, minimum-jerk over 0.8-1.2 s, held 2-6 s. With probability 0.25 one jaw
event, starting at a time uniform in the episode's middle half as H1's does (E5): open to 0.3-0.45 rad over 0.4-0.6 s,
held 1.0-1.5 s, closed over the same time. The ranges keep 0.15 rad from a joint limit (E5)."

**Label B** for the two ranges (as 6a); **Label A** for the two times (entry 7c).

---

## 7. Numbers the rules did not state (the review's minor finding; they stand)

### 7a. T1: the stop task's hold

**Old** (T1's table): "| `stop` | L, 2.5 strides of `v0.4`, `stop2_v0.4`, a hold | 2 |"

**New**: "| `stop` | L, 2.5 strides of `v0.4`, `stop2_v0.4`, a 4 s hold | 2 |"

**Old** (T1's table): "| `start_stop` | the `stop` task, a 2 s hold, `start`, 2 windows of `nominal` | 1 |"

**New**: "| `start_stop` | the `stop` task's plan without its 4 s hold, a 2 s hold, `start`, 2 windows of `nominal` |
1 |"

**Label A.** (`r5lib.STOP_HOLD_S` is 4.0 s of reference time: it covers the bar's stop item, whose hold after the rest
is 2.0 s plus its end margin at the slowest clock. `START_STOP_HOLD_S` is 2.0.)

### 7b. T1: the stand's plan

**Old** (T1's table): "| `stand` | a 12 s hold of the home stance (10 s played) | 1 |"

**New**: "| `stand` | a 12 s hold of the home stance, then `start` (10 s played: the start is never reached; the hold
is of the start's first frame, as in Stage 1's stand task, so a set without a usable `start` has no `stand` task) | 1
|"

**Label A.**

### 7c. J4: when a head event starts

In entry 6b's new J4: "the first starting at the episode's first instant" and "starting at a time uniform in the
episode's middle half as H1's does (E5)".

**Label A.** (`r5core.JAW_AT` is (0.25, 0.75) of the episode's length; a gaze train's first shift has `t0 = 0`.)

---

## 8. K4 (a later freeze)

K4's sentence stands as written. The env did not follow it and now does (**the code was changed**, the lead's ruling
3; commit `d1e667c`). Two sentences to add, if the lead wants K4 to say how the env decides:

**New**, after "and both freeze indices are recorded": "The env makes this comparison itself when it is built from the
stored settings (`r5env.FREEZE_RULE`): under the latest freeze every file its tasks use, and the nominal walk's, must
have the sha256 and the entry hash that the named freeze's manifest holds, and the same tasks must be in the table (a
task is in the table under a freeze exactly when that freeze's manifest holds every file of its plan); settings that
name their tasks are read for those tasks alone. A refusal lists every difference. The env description holds both
freezes (`set_freeze_started`, `set_freeze_now`: index and manifest sha256)."

**New**, in K4's Check, after "`freeze_index`": "; the env description's `set_freeze_started` and `set_freeze_now`".

**Label B.**

What "the resolved task table is unchanged" means in the code is a reading the lead may want to see: the env has only
the stored settings (a manifest hash), so it rebuilds the named freeze's table from that freeze's manifest in
`set.json`. With env configs that name no tasks (the default of `r5env.env_configs`), a later freeze that makes a
further task available (a change or a turn frozen later) is refused, as T2 says it must change no table of the run;
the refusal names the task and the file. Tests:
`test_a_later_freeze_with_the_same_files_and_task_table_is_accepted`,
`test_a_later_freeze_that_changes_what_the_run_uses_is_refused_with_what_differs`.

---

## 9. X1 (the 75 cm round's sets)

**Old** (X1): "Until task 3's first unit of the 75 cm round has run, no R2-mode, policy or training episode runs on V0,
F25F9 or F25F9cl;"

**New**, added after X1's first sentence: "[lead, 16 Nov 2026, before any Track B run] The 75 cm round's panels
finished on 15 Nov 2026 (task 3's and task 4's units have all run); this limit is lifted for the frozen 75 cm sets."

**Label A** (no code refused a set; `g8a.py`'s docstring now says where the rule stands, commit `ebae071`).

The first episodes run under it: `g8a.bar_form` on V0 of `75-M-H-10`, `walk/v0.4`, seed 3042, at 0 ms and 6 ms, the
whole condition (10,539 physics steps each): 2 of 2 identical. It is a smoke and no record; it wrote no file.
