# Work package CE: wording for the Track B rules draft, after the review of the env

For the lead, to paste into `STAGE2_TRACKB_PLAN.md` (the draft in the worktree `prereg-b`; this file edits nothing
there). It follows the lead's rulings of 16 Nov 2026 on the nine findings of
`stage2/design/checkpoint/reviews/roundC__review_CE.json`, and, since the env's second fix round of 16 Nov 2026, the
eight minor findings of `stage2/design/checkpoint/reviews/roundC__review_fix_CE.md` with the lead's rulings on what was
handed up (`stage2/design/checkpoint/STATUS.md`, the update of 16 Nov 2026, 03:02 UTC; `TRACKB_RULINGS3_16NOV.md`,
ruling 13), and, since the third fix round of the same day, the three points of wording in the independent review of
the second fix round (its findings 2, 3 and 4). Each entry gives the old sentence as the draft holds it, the exact new
sentence, and one of two labels:

- **Label A**: "a wording change made before any training run; the code is unchanged by it".
- **Label B**: "a wording change made before any training run; the code was changed with it" (the finding and the
  commit on `wt/wpc-ce` are named).

The old sentences were compared in the third fix round with the draft as commit `c4a0ff8` of `wt/prereg-b` holds it
(that branch's last change to the draft then; its tip was `0d10a94`): all 19 are there word for word. Entry 9 quotes
none, because the draft already holds what it asked for.

No training run, no policy episode and no panel was run for this file. The numbers under "Why" are from kinematic
playbacks (a script's own poses written into the plant's data, no physics step) on Stage 1's stock arrays of
`75-M-H-10` and, where it says so, on the frozen set V0 of `75-M-H-10`; and from the env on Stage 1's arrays (resets,
and a few env steps of the zero action or of noise actions, none of them a record). The playbacks and the env
readings were run again on 16 Nov 2026 in the second fix round, with the code as it then stood; the set V0 was read
from the main checkout and nothing was written there. The tests named are in `tests/test_stage2_r5lib.py`,
`tests/test_stage2_r5core.py` and `tests/test_stage2_r5env.py`.

Entries 1 to 4 are the review's blocker (the reward's readings stand as coded; the rule text follows them). Entry 5 is
the turn in place, 6 the head script's ranges and the head command in R6, 7 the numbers the rules did not state, 8 the
later freeze, 9 rule X1, 10 the mid-task start (T3). Entry 1b carries the lead's ruling on W1: it says where the
draft's W1 ("every positive term 1 and every penalty 0") does not hold. Entries 5c and 8 add to a rule's check what
the code now records.

What the second fix round changed in this file: entry 1a (two sentences on the instants), entry 1b (rewritten to the
lead's ruling, with the review's two corrections), entry 4a (one clause), entry 6c (new), entry 8 (rewritten: the
code changed again) and entry 10 (new). The other entries are as the review of the fix round read them.

What the third fix round changed in this file (no code changed with it): entry 3a (the swing length of a track that
plans no complete swing, and `T` not above zero), entry 3b (the bounds of duty's stride), entry 8 (K4's sentence on
the resume says "as built", with the reason) and entry 9 (withdrawn: the lead wrote X1's lifting into the draft).

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
warm start and gate Z1 would no longer hold. What is read at every physics step (the foot loads, and with them a
foot's contact, for contact, duty, flight, impact, touchdown, a foot's own lift-off and clearance (b); the foot bodies'
heights for touchdown's speed; the actuator forces for servo) is the plant's own after that physics step in the same
way, computed at its start, and the reference's flags beside it are read at the reference clock after that physics
step. In an episode's first env step the displacement behind tracking's speed and yaw rate and behind progress's
sample runs from the reading at the reset, which is of the episode's start, to the step's reading 18 ms later, and is
divided by 20 ms like every other; this matters only for a start in motion."

**Label A.**

Why: the review's blocker, part 4. In the env, after 15 env steps of noise actions (sigma 0.1), the foot term read
0.9046, which is the term of the pose 2 ms before the step's end to 1e-9; the term of the pose at the step's end is
0.9072. The joint term is the one of the angles at the step's end. Test:
`test_the_reward_reads_positions_as_of_the_last_physics_steps_start`.

The last two sentences were added in the second fix round (the review of the fix round, findings 3a and 3e); the code
is unchanged by them and `r5core.REWARD_RULE` holds both (commit `b4c223e`).

- *Every physics step.* In the env over 15 env steps of noise actions from a mid-task start, each physics step's foot
  loads equal a forward pass on the state before that physics step to 3e-9 body weights and differ from a pass on the
  state after it by up to 1.2 body weights; the foot bodies' heights equal the earlier pass's to 1e-12 m and differ
  from the later one's by up to 0.9 mm. Test:
  `test_what_is_read_at_every_physics_step_is_the_plants_own_computed_at_that_steps_start`.
- *The first env step.* At a start in motion (mid-stride of Stage 1's brisk walk, zero action, sensor noise off) the
  first step's tracking term is the one of the CoM's displacement over the first nine physics steps, over 20 ms, to
  1e-9: a speed of 0.303 m/s, where the displacement over all ten physics steps gives 0.338 m/s and the reference's own
  speed is 0.326 m/s (terms 0.9913 and 0.9970). A stance start begins at rest but for the start noise (at most 0.02
  m/s on the root's velocity), so the 2 ms that are left out hold almost no displacement. Test:
  `test_the_first_env_steps_displacement_runs_from_the_reset_and_spans_18_ms`.

### 1b. What a playback scores (the lead's ruling of 16 Nov 2026)

**Old** (W1): "A zero action on a kinematic playback scores every positive term 1 and every penalty 0."

**New**: "A kinematic playback of the reference scores the imitation terms (joint, foot, contact, head) 1, the
tracking term close to 1 and every penalty below 1e-4, with one exception: inside a finite walking turn a playback's
`heading` penalty is not zero. Rule E4 stays in walking turns: the commanded heading there is the integral of a
rectangular yaw-rate pulse while the reference's own yaw ramps, and the bar judges the heading against the command, so
the penalty points the same way as the bar. Measured on kinematic playbacks of two sets, Stage 1's stock arrays of
`75-M-H-10` and the frozen set V0 of `75-M-H-10`, and on no other: the tracking term is at least 0.989 on the walks
of both sets, 0.962 in V0's quick turn and 0.951 in Stage 1's turn in place (V0 has no turn in place and no 1.0 rad/s
turn). On V0 the reference's yaw is up to 9.0 degrees behind the commanded heading in the quick turn, 5.4 degrees at
0.5 rad/s and 3.8 degrees at 0.35 rad/s, and the heading penalty reaches 0.351, 0.139 and 0.0447; in the left turns it
is above zero on 98, 148 and 207 of the 1,499 env steps played (task means 0.021, 0.013, 0.006), in the right turns
on one step more each."

**Label A** (the code is as the lead ruled it: walking turns keep rule E4).

The lead's ruling (`stage2/design/checkpoint/STATUS.md`, 16 Nov 2026, 03:02 UTC, ruling 1; `TRACKB_RULINGS3_16NOV.md`,
ruling 13): rule E4 stays in walking turns, because the bar judges the heading against the command; W1's sentence
says that a playback's heading penalty is not zero there, with the review's two corrections: the tracking figure is
scoped to the two sets it was measured on, and "every penalty 0" becomes "below 1e-4".

Why: three places where the draft's sentence is not what the code does. The numbers are from kinematic playbacks of
every task script of the two sets from its first frame, one env step short of the episode's length (no physics step,
no record; V0 read from the main checkout), run on 16 Nov 2026 with the code of commit `4523ed6`.

- *The heading penalty in a walking turn.* On V0: `turn_q90_left` and `_right` up to 0.351 (the reference's yaw up to
  8.97 degrees behind the command), `turn_0.5_*` up to 0.139 (5.37 degrees), `turn_0.35_*` up to 0.0447 (3.76 degrees);
  zero in every other task of V0 and in every task of Stage 1's arrays.
- *The tracking term.* Stage 1's term compares the CoM's speed over the 20 ms step with the reference's speed at the
  step's start; on a playback it is not exactly 1. Smallest values: on Stage 1's arrays 0.9938 (walk and stop tasks)
  and 0.9936 (brisk task); on V0 0.9891 (`walk_v0.5`) to 0.9938 (`walk_v0.35`) on the six walks and 0.9929 in the stop
  tasks, 0.978 in the 0.35 rad/s turns, 0.966 in the 0.5 rad/s turns, 0.962 in the quick turn; 0.951 in Stage 1's turn
  in place (left 0.9507, right 0.9515; the arrays under design D5's names in a test's set). The tests ask for at
  least 0.99 on the walks and the stop and 0.95 on the turn in place.
- *The `progress` penalty.* It is not exactly zero on a playback: up to 2.1e-05 in V0's walks and walking turns and
  up to 8.3e-06 on Stage 1's walk and brisk tasks. Every other penalty of every one of V0's 15 available tasks and of
  Stage 1's four is 0 (heading inside the walking turns apart). The tests ask for below 1e-4.

---

## 2. W2 (the table and its check)

### 2a. The duty row

**Old** (W2's table): "| duty, slip, impact, tilt | 0.02 each | Stage 1's (W3 for the duty band) | Stage 1 |"

**New**, two rows:

"| duty | 0.02 | W3 | Stage 1, read against the reference's own contact share (W3) |"

"| slip, impact, tilt | 0.02 each | Stage 1's | Stage 1 |"

**Label A.**

### 2b. The check

**Old** (W2): "Check: `settings.env_configs[].reward` and the env description's `reward_scales` in every `run.json`
equal this table."

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
gait; a track that plans no complete swing has its gait's, (1 - duty) x the stride period. Where `T` is not above
zero, (a) is not read;"

**Label A.**

Why: with one swing time per gait, a playback of Stage 1's brisk task was charged inside both changes of gait (13 env
steps, up to 0.78): a foot was "between 25 % and 75 % of its swing" by the constant while the reference had it near
the floor. As coded the playback scores 0 at every env step. Test:
`test_clearance_a_times_a_swing_by_the_references_swing_in_progress`.

The words from "a track that plans no complete swing" to "(a) is not read" were added in the third fix round (the
review of the second fix round, finding 3); the code is unchanged by them. `r5core.REWARD_RULE` holds both clauses,
the second since commit `b4c223e`. The second is a guard: `u` divides by `T`; (b) is read all the same. Test:
`test_clearance_a_is_not_read_without_a_swing_length`.

### 3b. Duty: the formula and the reference's own share

**Old** (W3): "`duty`'s band is 0.55-0.72 while the commanded target speed is below 0.30 m/s (owner decision 2), else
0.55-0.65."

**New**: "`duty` is `min(1, sum over the feet read of max(0, lo - d_f, d_f - hi) / 0.05)`, d_f the foot's contact
share over the last stride (the physics steps of the last n env steps, `n = round(stride period / 1.05 / 20 ms)`, kept
between 1 and 256 env steps: the history the env keeps); zero while the reference stands and until n env steps of the
episode have been played. A foot is read only where the reference's own support share for it over the same physics
steps lies inside the band, so that a start, a stop or a change, where the reference itself is outside the band, is
not charged. The band [lo, hi] is 0.55-0.72 while the commanded gait's target speed is below 0.30 m/s (owner decision
2), else 0.55-0.65."

**Label A.**

Why: Stage 1's formula without the condition charged the reference's own playback in every start and stop: 84 of 918
env steps of Stage 1's walk task and 147 of 1,479 of its brisk task, up to 1.0 (task means 0.088 and 0.079). As coded
the playback scores 0 at every env step, and a body that keeps both feet down while the reference steps is charged in
full. Tests: `test_duty_reads_a_foot_only_where_the_references_own_share_is_inside_the_band`,
`test_stage_1s_penalties_and_the_action_rate`.

The bounds of n ("kept between 1 and 256 env steps: the history the env keeps") were added in the third fix round
(the review of the second fix round, finding 3); the code is unchanged by them and `r5core.REWARD_RULE` holds them
since commit `b4c223e` (`r5core.HISTORY_STEPS` is 256, 5.12 s of env steps). Test:
`test_dutys_stride_is_kept_between_1_and_256_env_steps`.

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
and its speed ripple inside a stride is in both means, so a playback is charged below 1e-4 (W1); a clock held slow
leaves the reference's sample as it is and is charged in full (the bar's walk item fails 4.8 % short of the commanded
distance; Stage 1's tracking term charges 0.2 % of the reward for 5 %)."

**Label A.** (The second fix round changed "is charged nothing" to "is charged below 1e-4 (W1)": entry 1b.)

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

## 6. E5 and J4 (the head script's ranges; the head command in R6)

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

### 6c. E5: what the head command is added to in R6

**Old** (E5): "the commanded yaw and jaw angles are added to the reference's neck-yaw and jaw targets before the
policy's offsets."

**New**: "in R5 the commanded yaw and jaw angles are added to the reference's neck-yaw and jaw targets before the
policy's offsets; in R6, which has no reference controller, they are added to the settled home pose's neck-yaw and jaw
angles before the policy's term (E2: the joint targets are `clip(home + head command + a x span, range)`). The settled
jaw rests a little open (0.0015 rad on `75-M-H-10-X`, 0.0007 rad on `60-N-T-10-X`), so an R6 jaw target reaches 0.4515
and 0.4507 rad and keeps 0.1485 and 0.1493 rad from the jaw's limit, where the script's own value keeps 0.15 rad."

**Label A** (added in the second fix round: the review of the fix round, finding 3f; `r5core.HEAD_RULE` says it since
commit `b4c223e`; `Core.targets` is unchanged).

Why: the draft's sentence is R5's. In R6 the policy sets joint targets about the settled home pose, and the head
command is added there. The neck yaw settles within 1e-7 rad of zero on both bodies. The `range` term asks 0.10 rad
of the jaw, so it charges neither. E2's formula in the draft, `clip(home + a x span, range)`, is the one without a
head command. Test: `test_in_r6_the_head_overlay_is_added_to_the_settled_home_pose` (both `-X` bodies).

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

K4's sentence stands as written. The env did not follow it and now does (**the code was changed**, twice: the lead's
ruling 3 on the first review, commit `d1e667c`; then findings 1, 2 and 7 of the review of the fix round and the
lead's ruling 3 of 16 Nov 2026, 03:02 UTC, commits `011f6e2` and `4523ed6`). Sentences to add, if the lead wants K4
to say how the env decides:

**New**, after "and both freeze indices are recorded": "The env makes this comparison itself when it is built from the
stored settings (`r5env.FREEZE_RULE`), whether the freeze they name is the set's latest or an earlier one: as the set
is read today, every file its tasks use, and the nominal walk's, must have the sha256 and the entry hash that the named
freeze's manifest holds, and the same tasks must be in the table (a task is in the table under a freeze exactly when
that freeze's manifest holds every file of its plan); settings that name their tasks are read for those tasks alone.
A job's env configs name the tasks the set holds when the run starts (`r5env.env_configs`), so a later freeze that
makes a further task usable is accepted and the run plays on with its own tasks. A refusal lists every difference.
The env's library also refuses a set whose `set.json` `crouch_frac`, which no freeze covers and with which the
track's LIPM frequency is computed, differs from the `crouch_frac` of a walk entry it reads, as the bar does
(`r5lib.CROUCH_RULE`). The env description holds both freezes (`set_freeze_started`, `set_freeze_now`: index and
manifest sha256), the task names (`task_names`) and the files those tasks use with their hashes (`task_files`). On a
resume the trainer compares the stored env configs and, of the description of the envs as built, `config`,
`task_names`, `tasks`, `task_files` and `set_freeze_started` (`r5env.RESUME_KEYS`), not the description whole: an
accepted later freeze changes its `set_freeze`, `set_freeze_now` and `inputs.freeze`, and, where it made further tasks
usable, its `inputs.files` and `absent` (`r5env.LATER_FREEZE_KEYS`). "As built" is the env made from its stored config
before any task weight is set on it, in the first session and in the resumed one alike: `tasks` holds the task
weights in force, and those are run state (K5), which M4's fix changes."

**New**, in K4's Check, after "`freeze_index`": "; the env description's `set_freeze_started`, `set_freeze_now`,
`task_names` and `task_files`".

**Label B.**

What the code does, for the lead to see:

- *The named freeze.* The settings hold a manifest hash alone, so the env rebuilds the named freeze's table from that
  freeze's manifest in `set.json`, and checks that this manifest is the one the hash names. `set_freeze_started` is
  the named freeze by the index of the first freeze of the set that holds that manifest; it does not move when the set
  is frozen again. (`sets.freeze` appends no freeze whose manifest an earlier one holds, so on a set that command wrote
  the index is the one the run started on. A config that stored the index as well would not need this reading; no such
  field was added.)
- *The latest freeze.* The comparison is made when the named freeze is the set's latest too: a set's records can be
  edited after its latest freeze without a new one. A set frozen once whose change into the 0.40 m/s walk was then
  recorded as not available was accepted with one task fewer before; it is refused now, with the task and the reason.
- *Task names.* With env configs that name no tasks, a later freeze that makes a further task usable is refused, as
  T2 says it must change no task table of the run; the refusal names the task and the file. `r5env.env_configs` now
  stores the names of the tasks the set holds when it is called, so the stored configs are read for those tasks alone
  and the same later freeze is accepted: the run's table is unchanged and the new task is not played. A config made by
  hand with `tasks=()` still behaves as before.
- *A named task that left the table* (it was in the table under the named freeze and is absent now) is refused with
  the study's `InputError`, naming the task and the set's reason; a name that is no row of T1's table is a
  `ValueError`, as before.
- *The crouch.* `set.json`'s `crouch_frac` 0.03 edited to 0.08 with the entries left at 0.03 was accepted before, also
  across a later freeze, and moved the track's LIPM frequency from 4.828 to 4.979 rad/s (the review of the fix round,
  finding 2, on a test's set; it says that only the critic's DCM-error inputs move with it). It is refused now with
  both values.
- *"As built"* (added to the sentence in the third fix round: the review of the second fix round, finding 4; the code
  is unchanged and `R5Env.describe`'s docstring has said it since commit `4523ed6`). `set_task_weights` changes the
  weights inside the description's `tasks` and nothing else of `r5env.RESUME_KEYS`, and no other key of the
  description; a trainer that compared `tasks` of a live env after M4's fix with what the first session stored would
  refuse its own run. Test: `test_of_what_a_resume_compares_a_change_of_the_task_weights_moves_the_weights_alone`.

Tests: `test_a_later_freeze_with_the_same_files_and_task_table_is_accepted`,
`test_a_later_freeze_that_changes_what_the_run_uses_is_refused_with_what_differs`,
`test_the_task_table_is_compared_also_when_the_named_freeze_is_the_latest`,
`test_a_task_that_leaves_the_table_under_a_later_freeze_is_refused`,
`test_env_configs_name_their_tasks_so_a_later_freeze_that_adds_a_task_is_accepted`,
`test_set_freeze_started_is_the_index_the_run_started_on`,
`test_a_set_whose_crouch_differs_from_a_walk_entrys_is_refused`. The real set V0 of `75-M-H-10` builds with its
freeze and its 15 tasks named (a build, no episode).

---

## 9. X1 (the 75 cm round's sets): in the draft already, nothing to paste

**No wording change is asked.** The draft holds the lifting in the lead's own words since commit `405c218` of
`wt/prereg-b` (16 Nov 2026; a commit of that branch, not of this one): X1's first sentence there reads "Until task 3's
first unit of the 75 cm round has run, no R2-mode, policy or training episode runs on V0, F25F9 or F25F9cl (lifted for
the frozen 75 cm sets on 16 Nov 2026, after the panels)", with its tag and the lead's amendment 13 of
`STAGE2_75_PLAN.md` beside it.

Until the third fix round this entry quoted that sentence as it stood before the lifting and proposed a sentence of
its own for the same lifting, to be added after it ("The 75 cm round's panels finished on 15 Nov 2026 (task 3's and
task 4's units have all run); this limit is lifted for the frozen 75 cm sets."). That proposal is withdrawn (the
review of the second fix round, finding 2: the quoted sentence is no longer in the draft as quoted). On the code's
side nothing depends on it: no code refused a set, and `g8a.py`'s docstring says where the rule stands (commit
`ebae071`).

The first episodes run under the lifted rule: `g8a.bar_form` on V0 of `75-M-H-10`, `walk/v0.4`, seed 3042, at 0 ms
and 6 ms, the whole condition (10,539 physics steps each): 2 of 2 identical. It is a smoke and no record; it wrote no
file.

---

## 10. T3 (the mid-task start)

**Old** (T3): "the plant is set to that frame (rates x 1.05) with Stage 1's start noise (0.01 rad on the joints, 0.02
m/s on the root's velocity), the estimator is reset from that state, and the delay queue holds the k commands the zero
action would have given before it."

**New**: "the plant is set to that frame (rates x 1.05) with Stage 1's start noise (0.01 rad on the joints, 0.02 m/s on
the root's velocity), the estimator is reset from that state, and the delay queue holds the k commands the zero action
would have given before it. The servos are then set to the oldest of those commands (without a delay, k = 0: to the
command the zero action would have given one physics step before the start), the forward pass is computed once on
that, and the estimator is reset again from the same state with a sensor reading of that pass (its foot loads,
accelerometer and contact flags): the plant's first pass ran with the servos at their home targets, and its foot loads
are not those of a body in mid-stride."

**Label B** (the first review's minor finding on the estimator's first reading at a mid-task start; commit `225f4b3`
of the first fix round. The first version of this file had no entry for it; `r5env.MID_START_RULE` has said it since
that commit).

Why: the code was changed in the first fix round and T3 was not. Before the change, without a delay the servos stayed
at their home targets through the reset, and with a delay the estimator kept the reading of the first pass. At the
five starts of the test (Stage 1's brisk task, sensor noise off) the pass with the servos at home puts 2.5 to 4.4 body
weights on the loaded foot; the pass that is read now puts at most 0.76 body weights on a foot without a delay and at
most 2.06 with k = 5 (one start of the five; the others at most 1.41). The estimator is still reset from the known
state (its angles and its base position are the frame's), and the first observation's foot loads and the estimator's
and the core's first contact flags are that pass's. In R6 the commands are the reference's joint angles (E2: "its
action history is set to the actions that give the reference's targets"), and the same pass is read. Tests:
`test_mid_start_first_reading_is_of_the_zero_actions_command` (k = 0 and 5),
`test_mid_start_queue_holds_the_references_commands`, `test_r6_mid_start_history`.

The same commit made the code keep T3's "never inside a hold" for one more draw; T3's words on it stand as written
(**no wording change**). The uniform draw is laid along the spans a start may land in, and a draw at the very end of
the last span, which can end where a hold begins, now gives the last instant before it (`r5env.mid_start_time`).
Before, such a draw raised in the reset or, carried one rounding step past the end, started at the last span's first
instant. Tests: `test_mid_start_time_stays_inside_its_span`,
`test_the_envs_own_draw_at_the_end_of_the_last_span_starts_just_before_the_hold` (the env's own draw; added in the
second fix round, commit `4523ed6`).
