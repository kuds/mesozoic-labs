# Training artifact names and stance replays

Human-facing training output uses the model display names from
`configs/species_manifest.toml`, through `model_display_name()`: the species
display name followed by the embodiment label, such as "Tyrannosaurus Rex
(Anatomical)" or "Compsognathus Longipes (Robot)" (decision D-D33). Stage
headings use `stage_display_name()`: the historical `balance` name is shown
as **Stance**, and the six current stance configs record `name = "stance"`.
This applies to CLI logs, text summaries, gate and probe reports, plot
legends, notebook tables, the public catalog, and W&B run titles.

| Species ID | Embodiment | Model display name | Stance gate |
|---|---|---|---|
| `velociraptor_mongoliensis` | `anatomical` | Velociraptor Mongoliensis (Anatomical) | `stance_quality/v2` |
| `tyrannosaurus_rex` | `anatomical` | Tyrannosaurus Rex (Anatomical) | `stance_quality_heading/v1` |
| `brachiosaurus_altithorax` | `anatomical` | Brachiosaurus Altithorax (Anatomical) | `reward_and_length/v1` |
| `dibothrosuchus_elaphros` | `anatomical` | Dibothrosuchus Elaphros (Anatomical) | `reward_and_length/v1` |
| `compsognathus_longipes` | `anatomical` | Compsognathus Longipes (Anatomical) | `stance_quality_heading/v1` |
| `compsognathus_longipes` | `robot` | Compsognathus Longipes (Robot) | `stance_quality/v1` |

Consistent presentation does not imply identical anatomy or certification
criteria. The gate kind remains visible in each report. `stance` and `recover`
are **recipes**: `stance`, the stance stage's id and its label, resolves to the
stance stage on every model, and `recover`, on the anatomical Tyrannosaurus Rex
and both Compsognathus embodiments, through stance and recovery (decision D-D35;
before it the stance label, then `stand`, resolved through both there). The
stance label was `stand` until decision D-D38, which refuses `stand` naming
`stance`. Their recovery outputs are labeled **Recovery**.

## Shared replay layout

Stance and recovery stages use the same replay outputs for every model.
When both checkpoint pairs are present and rendering dependencies are
available, each stage writes eight files under `replays/`:

| Checkpoint role | Files |
|---|---|
| Selected | `<species>_<embodiment>_<algorithm>_<stage>_selected.mp4`, `_selected_side.mp4`, `_selected_front.mp4`, `_selected_stance.csv` |
| Final | `<species>_<embodiment>_<algorithm>_<stage>_final.mp4`, `_final_side.mp4`, `_final_front.mp4`, `_final_stance.csv` |

`<species>` and `<embodiment>` are the model's two IDs
([SPECIES_NAMING.md](SPECIES_NAMING.md)) and `<stage>` is the existing artifact label,
such as `stage1` or `recovery`. The selected replay uses the same
`select_handoff_checkpoint()` choice and matched VecNormalize statistics
as the gate report and the next-stage handoff. The final replay uses the
final checkpoint and its own statistics. Missing pairs and rendering
failures are reported through the existing artifact-generation logging.
An unpaired final checkpoint is skipped, just like an unpaired selected
checkpoint, rather than silently replayed with unnormalized observations.

The default, side, and front cameras record **one rollout**, with one
policy prediction and physics step per frame. Side/front views share fixed
angles and preserve each species' tracking body and default camera distance.
Tyrannosaurus Rex keeps its historical 3.4 m side/front framing;
Brachiosaurus uses a 9 m front view to include its tall neck.

Playback uses `fps = 1 / env.dt`. The four 0.01 s models produce 100 fps
videos; both Compsognathus embodiments use 0.02 s steps and 50 fps. A recording
lasting 10 s in simulation lasts 10 s on playback. Historical videos used
50 fps universally, making 0.01 s recordings play at half speed. The W&B
callback's optional evaluation video uses the same `1 / dt`, read from its
env (30 fps when the env reports no `dt`). wandb encodes it as a GIF, whose
players clamp frame delays below 2 cs, so above 50 fps it keeps every n-th
frame (every second at 0.01 s) and still plays in real time. The training
entry points pass it no video env.

Each stance CSV has one row per frame, with `step`, elapsed `time_s`,
`reward`, `terminated`, `truncated`, and common `root_*` position/orientation
columns. It also preserves measured scalar environment info. The four bipeds
add the historical T. rex joint, foot, head, and tail diagnostics, measured
at each species' own foot sites and leg joints (below). Unavailable
measurements stay absent. In particular, the two front-foot contacts of a
quadruped are not converted into a fabricated whole-animal support verdict.

On Brachiosaurus and Dibothrosuchus, the raw `r_foot_contact` and
`l_foot_contact` columns are the **front**-right and front-left touch
forces: their envs report the `fr` and `fl` feet under these biped names,
beside `rr_foot_contact` and `rl_foot_contact` for the hind feet. The names
are kept because the diagnostics callback, episode metrics and plots read
them. Quadruped CSVs have no support, foot-position, or leg-home columns;
the Dibothrosuchus CSV has the pad columns below.

### Pad columns

Every model whose `FOOT_SUPPORT` sole is a box adds two columns per such
foot, named by the floor-truth recorder's foot label (`r`/`l`, and
`fr`/`fl`/`rr`/`rl` on the Dibothrosuchus):

| Column | Meaning |
|---|---|
| `<label>_sole_cop_fore_aft` | The step's signed fore-aft centre of pressure on the pad, in pad half-lengths, positive toward the toe (`+1` the toe edge, `-1` the heel edge) |
| `<label>_digit_share` | The step's floor force on the foot's digits (its `FOOT_SUPPORT` support geoms other than the sole) over the whole limb's floor force |

They come from the floor-truth recorder
(`environments/shared/gait/recorder.py`), attached to the replay env for
the replay only, and are reduced by
`environments/shared/gait/pad_telemetry.py`'s `pad_step_columns`. The CoP is
the stance gate's measurement, normalized as `stance_quality/v2`'s
`max_sole_cop_fore_aft` reads it. A cell is empty on a step the gate counts as
unloaded: no floor force on the pad, or no pad contact above 0.1 N. The gate
counts such a step as 1. So the gate's per-foot statistic over any window is
the mean of `|value|`, with an empty cell read as 1. The digit share is empty
on a step the limb carries nothing. The column is absent, not empty, on a
foot that declares no digits (the Compsognathus robot's sole and every
Dibothrosuchus foot).

The pad columns cover the four box-soled models: Tyrannosaurus Rex, both
Compsognathus embodiments and Dibothrosuchus. The Velociraptor's capsule feet
and the Brachiosaurus's ellipsoids have no box pad, so their CSVs have no pad
column, just as their gates have no fore-aft CoP.

A pad foot whose `FOOT_SUPPORT` entry declares its ankle and a support span
(`ankle`, `heel_line`, `toe_line`; only the Tyrannosaurus Rex, whose span is
its pad's own heel and toe edges, 0.070 and 0.210 m ahead of the ankle) also
has the whole-foot columns, and every pad foot with digits has per-digit
columns, named by the digit's geom less the foot label and `_geom`
(`r_toe_d2_geom` is the digit `toe_d2`, with the columns
`r_toe_d2_load_share` and `r_toe_d2_load_pos`):

| Column | Meaning |
|---|---|
| `<label>_foot_cop_fore_aft` | The step's signed WHOLE-FOOT centre of pressure: every floor contact of the limb (pad, digits, any other leg geom), force-weighted along the pad's forward axis from the ankle axis, in half-lengths of the declared span (`-1` the heel line, `+1` the toe line, beyond `+1` past it on the digits) |
| `<label>_toe_margin_nm`, `<label>_heel_margin_nm` | The floor-load moment left before that CoP reaches the toe line, `F (toe_line - x)`, and the heel line, `F (x - heel_line)`, in N m, with `F` the limb's floor force; negative past the line |
| `<label>_<digit>_load_share` | The step's floor force on that digit over the whole limb's |
| `<label>_<digit>_load_pos` | Where along the digit its floor load sits, force-weighted: 0 at the MTP joint's axis, 1 at the tip. Only on a digit with its own MTP joint (the Tyrannosaurus Rex's; the anatomical Compsognathus's digits share the pad's body) |

The whole-foot cells are empty on a step the limb carries nothing above
0.1 N, so the per-foot whole-foot statistic over any window is again the
mean of `|value|` with an empty cell read as 1, and its maximum over the
feet is `stance_quality/v2`'s report-only `max_foot_cop_fore_aft`. The load
position is empty on a step its digit carries nothing.

### Biped feet and leg joints

| Model | Foot site (`r_`/`l_` columns) | Leg joints |
|---|---|---|
| Velociraptor Mongoliensis (Anatomical), Tyrannosaurus Rex (Anatomical) | `r_foot`, `l_foot` | `r_*`, `l_*` |
| Compsognathus Longipes (Anatomical) | `r_foot_touch_volume`, `l_foot_touch_volume` | `r_*`, `l_*` |
| Compsognathus Longipes (Robot) | `right_foot_touch_volume`, `left_foot_touch_volume` | `right_*`, `left_*` |

The foot site is the `foot_site` of the embodiment class's `FOOT_SUPPORT`
(`environments/<species>/<embodiment>/env.py`, read through
`environments/shared/gait/morphology.py`): the reference point the
floor-truth recorder measures stance width and foot shift from, and the one
the T. rex and anatomical Compsognathus stance-width rewards read. The CSV's
foot positions, support midpoint and pelvis-over-support offset are
therefore measured where the gate (and, on those two species, the reward)
measures the feet. The leg joints share the site's side spelling. Both
Compsognathus volumes enclose the load-bearing sole. The anatomical one also
covers the digits; its centre lies 10 mm ahead of the `r_sole` site, which
marks the centre of the plantar pad's underside, and 5 mm above it. The
robot's volume shares the `right_sole` site's x and y and sits 4 mm higher.
The sole sites are not used: T. rex and Velociraptor have none, so the same
columns would name different points on different species, and neither the
gate nor any reward reads them.

On every biped, `stance_width` is the world-frame lateral offset `|Δy|` of
the two foot sites, as on T. rex before. T. rex and the anatomical
Compsognathus also report an env `stance_width`: the planar distance between
the same two sites, which their stance-width rewards read and which the
floor-truth gate's stance width (`max_settle_stance_width_change_m`)
measures on every species whose gate bounds it (T. rex, Velociraptor and the
anatomical Compsognathus). The CSV column replaces it, and
`hypot(stance_width, foot_fore_aft_offset)` recovers it exactly. The env's
`stance_width_target`, `stance_width_error` and `stance_width_quality`
columns stay in that planar distance, so compare them with the hypot, not
with the CSV's `stance_width`. Leg home
errors compare each side's hip pitch, hip roll, knee, and ankle with the
`home` keyframe, with per-leg and per-joint means; the robot's hip yaw and
ankle roll are not included. The robot's env has no leg-home-pose reward
term, so its CSV has these errors but no env `leg_home_pose_error`.

## Evaluation telemetry: `gate_progress.npz`

Each training stage directory holds `gate_progress.npz` beside
`diagnostics.npz` and `evaluations.npz`. `StageGatePlateauCallback`
(`environments/shared/eval_diagnostics.py`) rewrites it after every periodic
evaluation (every `eval_freq` steps, 50,000 by default). It is measured on the
deterministic policy over the evaluation panel: `max(30, min_eval_episodes)`
episodes, 40 on the stance stages that declare it. Every array has one
value (or one row) per evaluation, aligned with `timesteps`. A key a run
cannot supply on every evaluation is dropped rather than written misaligned.
The file is report-only. No gate verdict, checkpoint selection or reward
reads it. The opt-in hop watch reads only its `unsupported_duty`. The
posture-drift advisory (below) reads its fore-aft over-bar fractions and,
for the line it prints, the latest row's signed per-foot centre of
pressure. The posture-trend advisory reads the same fractions and the
signed per-foot centre-of-pressure series.

| Keys | Meaning |
|---|---|
| `timesteps` | The training step of each evaluation |
| `full_horizon_fraction`, `duty_episodes`, `unsupported_duty`, `unsupported_duty_ucb`, `bilateral_support_duty`, `mean_reward` | The stance panel's gate quantities, as `curriculum/stance_gate.py` reduces them |
| `action_dc_rms`, `action_ac_rms`, `action_delta`, `action_jerk`, `action_freq_hz`; `action_dc_per_actuator`, `action_ac_rms_per_actuator` | The deterministic action's static offset from the home control and its tremor, over the post-settle window, as scalars and as (evaluations, actuators) matrices |
| `term_*` | Each reward term's per-episode mean |
| `full_horizon_term_*`, `fall_term_*` | Each reward term's per-episode mean over the evaluation's full-horizon episodes, and over its fallen ones (shorter than the horizon); NaN when it has no episode of that kind. With `n_fh` full-horizon and `n_fall` fallen episodes of `n`, `n_fh × full_horizon_term_x + n_fall × fall_term_x = n × term_x`. TensorBoard gets only the totals, `reward_terms/eval_full_horizon_total` and `reward_terms/eval_fall_total` |
| `<label>_sole_cop_fore_aft` | Per box-soled foot: the evaluation's mean, over its full-horizon episodes, of the foot's signed fore-aft pad centre of pressure, in pad half-lengths, positive toward the toe. An episode's value is the mean over the post-settle steps the gate counts as loaded |
| `<label>_sole_cop_fore_aft_gate` | Per box-soled foot: the evaluation's mean, over its full-horizon episodes, of the foot's per-episode gate statistic: the post-settle mean of `\|CoP\|`, with an unloaded step counted as 1. In each episode, its maximum over the feet is that episode's `max_sole_cop_fore_aft`. The maximum over the feet of these per-evaluation means is not the panel's statistic, only a lower bound of the next key |
| `max_sole_cop_fore_aft_mean` | A model with a box pad: the evaluation's mean, over its full-horizon episodes, of each episode's own `max_sole_cop_fore_aft` (the maximum over the feet of `<label>_sole_cop_fore_aft_gate`), the number the gate's pad rail reads |
| `max_sole_cop_fore_aft_over_bar_fraction` | A model with a box pad, on a stage that declares the `max_sole_cop_fore_aft` bar (the T. rex 0.80, the anatomical Compsognathus 0.70): the fraction of the full-horizon episodes whose `max_sole_cop_fore_aft` exceeds the bar, the rail's own criterion on this panel |
| `<label>_digit_share` | Per box-soled foot with digits: the evaluation's mean, over its full-horizon episodes, of the post-settle floor force on its digits (its `FOOT_SUPPORT` support geoms other than the sole) over the whole limb's |
| `<label>_foot_cop_fore_aft`, `<label>_foot_cop_fore_aft_gate` | Per foot that declares a support span (the T. rex): the same pair for the whole-foot centre of pressure (every floor contact of the limb, from the ankle axis, in half-lengths of the declared span; an unloaded step counts 1 in the second). In each episode the maximum over the feet of the second is that episode's `max_foot_cop_fore_aft`. On the T. rex's span, its pad's own edges, it reads in pad half-lengths |
| `max_foot_cop_fore_aft_mean` | A model whose feet declare a span: the evaluation's mean, over its full-horizon episodes, of each episode's own `max_foot_cop_fore_aft`. Report-only: no stage declares that bar, so there is no `max_foot_cop_fore_aft_over_bar_fraction` (it would follow a declared bar as the pad rail's does) |
| `<label>_foot_cop_std_m`, `<label>_foot_cop_dominant_hz` | Per foot that declares a span: the standard deviation (m) of the per-step whole-foot CoP over the post-settle loaded steps, and the strongest non-zero frequency of that series (unloaded steps dropped), means over the full-horizon episodes. The window means cannot see a foot rocking heel to toe; these can |
| `<label>_toe_margin_nm`, `<label>_heel_margin_nm` | Per foot that declares a span: the post-settle mean, over loaded steps, of the floor-load moment left before the whole-foot CoP reaches the toe line and the heel line, in N m; means over the full-horizon episodes |
| `<label>_<digit>_load_share`, `<label>_<digit>_load_pos` | Per digit of a box-soled foot: its share of the limb's post-settle floor force, and (a digit with an MTP joint only) where along it that load sits, force-weighted, 0 at the MTP axis and 1 at the tip; means over the full-horizon episodes |
| `leg_dc_<left>_minus_<right>_<joint>_deg` | Per left/right foot pair and sagittal joint (`hip_pitch`, `knee`, `ankle`): the left joint's commanded offset from home minus the right one's, in degrees, each signed as a rotation about the animal's leftward axis. A mirror-symmetric posture reads 0 |
| `leg_dc_<left>_<right>_mean_<joint>_deg` | The same pair's mirror-even mean, `(left + right) / 2` of the two signed offsets, in degrees: a symmetric crouch or lean, which the left-minus-right key reads as 0 |
| `<label>_sole_tilt_deg`, `<label>_sole_corner_lift_m` | Per box-soled foot: the evaluation's mean, over its full-horizon episodes, of the sole's post-settle mean tilt from level (degrees) and mean corner lift (the height of its highest bottom corner over its lowest, m). In each episode, the maximum over the feet of each is that episode's `max_sole_tilt_deg` and `max_sole_corner_lift_m`, the gate's two roll rails |
| `max_sole_tilt_deg_mean`, `max_sole_corner_lift_m_mean` | A model with a box sole: the evaluation's mean, over its full-horizon episodes, of each episode's own `max_sole_tilt_deg` and `max_sole_corner_lift_m` |
| `max_sole_tilt_deg_over_bar_fraction`, `max_sole_corner_lift_m_over_bar_fraction` | A model with a box sole, on a stage that declares the bar (the T. rex 2.0° and 0.004 m, the anatomical Compsognathus 2.0° and 0.003 m): the fraction of the full-horizon episodes whose value exceeds the bar |
| `termination_<cause>` | Per evaluation, the number of its episodes that ended by each cause: the env's `termination_reason` for an episode it terminated (`nosedive`, `head_contact`, `fallen`, ...; `terminated` when the env gave none) and `truncated` for one that reached its time limit. Every cause the file has seen is written on every evaluation, so the counts of one evaluation sum to its episodes |

The pad and leg DC keys exist so that an asymmetric pad posture is visible
when it forms. In the 2026-10-11 T. rex stance run, the left-right leg offset
that put its right pad on the toe edge had formed about 11M steps before
`stance_quality/v2` refused it at the end. Its artifacts kept only the
per-episode maximum over the feet.

A pad value is the mean of its per-episode values over the full-horizon
episodes, the set the duty is averaged over. The per-episode values are not
kept. The worse pad can change from episode to episode: one 1M-step probe
policy failed the rail on its right pad in 23 episodes and on its left in 7.
Then every per-foot mean can sit under the bar while most episodes fail it.
The two `max_sole_cop_fore_aft_*` keys keep the rail's own per-episode
number for that reason. They are telemetry on the training evaluation panel,
not the gate's verdict. The gate judges its own certification panel, counts
a fallen episode as failed whatever its pad reads, and decides on its bound
of clean episodes. An episode measured on no foot is left out of both keys.

The pad values use the floor-truth recorder
(`environments/shared/gait/recorder.py`), reduced by
`environments/shared/gait/pad_telemetry.py`. The recorder is attached to the
evaluation env for the evaluation only and never to the training envs. It
reads the physics without writing it, so the evaluation itself is unchanged.
It adds about a third to each evaluation's simulation time on the four
box-soled models (+34 % measured on the T. rex); the other two attach no
recorder. The pad keys follow one rule, the same as the replay CSV's pad
columns:

- A field exists for a foot only if the foot has what the field measures.
  The CoP keys exist for every foot whose `FOOT_SUPPORT` sole is a box: the
  Tyrannosaurus Rex, both Compsognathus embodiments and the Dibothrosuchus.
- The digit share exists only on a box-soled foot with digits: the Tyrannosaurus
  Rex and the anatomical Compsognathus.
- The Velociraptor and the Brachiosaurus have no pad key: they have no box pad.
- The `max_sole_cop_fore_aft_*` keys follow the pad keys, and the bar
  fraction exists only where the stage declares the bar.
- The whole-foot keys, and `max_foot_cop_fore_aft_mean`, exist only on a foot
  whose `FOOT_SUPPORT` entry declares its ankle and support span: the
  Tyrannosaurus Rex. The per-digit load share exists on every box-soled foot
  with digits; the load position only on a digit with its own MTP joint (the
  Tyrannosaurus Rex's three per foot).
- NaN in a key that exists means the value could not be measured at that
  evaluation. That happens when no episode reached the horizon or the
  evaluation could not be recorded. It also happens when no full-horizon
  episode measured the key: the signed pad CoP when the pad was never loaded;
  the signed whole-foot CoP, its spread, its dominant frequency and both
  moment margins when the limb was never loaded (both gate statistics then
  read 1); the dominant frequency also on a still series (a range of at most
  1 nm) or one of fewer than 3 loaded steps; the digit shares when the limb
  carried nothing; and a digit's load position when that digit carried
  nothing. The frequency drops unloaded steps from its series, so its time
  base is approximate when a foot lifts within the window.

The leg DC is computed from `action_dc_per_actuator`, which is in action
units. Each actuator's DC is mapped through the env's own action map
(`_scale_action`), and the control of action zero, the home control, is
subtracted. The result is the joint target offset the DC commands, in
radians. Each side's offset is multiplied by the sign of its joint axis
along the leftward axis: the horizontal direction from the right foot's
reference site to the left one's, in the `home` keyframe. The right side is
then subtracted from the left, and the difference is given in degrees. So
the zero action reads 0, and so does any mirror-symmetric offset. Action
units alone would not do this. The Compsognathus robot's left hip pitch axis
points right and its right one left, with mirrored control ranges. There, a
symmetric 0.03 rad hip pitch offset read -0.036 or +0.61 action units,
depending on its direction.

Positive is a rotation about the leftward axis by the right-hand rule. On the
T. rex, the Velociraptor, the anatomical Compsognathus and both quadrupeds,
that is each joint's own positive direction. On the Compsognathus robot it is
the negative direction of every leg joint except the right hip pitch. The
2026-10-11 T. rex run's mean action reads -3.2° at the ankle and -2.1° at
the knee: its left ankle and left knee were commanded below the right ones.

Under the anatomical Compsognathus's softcubic leg map, this is the target of
the mean action rather than the mean target. The two differ by the map's
curvature times the tremor's variance. A pair is read only when both joints
are unit-gear hinge position servos, so that a control value is an angle, and
when both axes lie within about 45° of the leftward axis. Every model's hip
pitch, knee and ankle pass, so the keys exist on all six. Each side's joints
use the prefix of its `FOOT_SUPPORT` foot site (`r_knee`, or `right_knee` on
the Compsognathus robot).

The roll-rail keys follow the pad keys' rule: they exist on the box-soled
feet only, so they add no recorder to any model. The Brachiosaurus's
ellipsoid soles have a tilt the gate would read, which includes the pitch
the feet are authored at (16 to 19°), and no corners; recording it would
attach the recorder to every Brachiosaurus evaluation (45% more evaluation
time, measured) for a bar no Brachiosaurus stage declares, so it is left
out. A roll-rail key is NaN where no full-horizon episode measured it; the
corner lift also when a sole had no finite corner reading in the window. An
episode whose roll rails cannot be reduced reads NaN on them alone (logged
once); its pad and whole-foot values stand.

A termination count is 0 for an evaluation in which no episode ended by
that cause. A cause first seen at a later evaluation reads 0 on the earlier
rows, except on rows a same-stage resume carried over from a file recorded
before these counts existed, which read NaN, as does every count of an
evaluation whose causes were not captured. Before the first cause there is
no `termination_` key at all.

A training run that names no model (`model_id=None`) records neither the pad
and roll-rail keys nor the leg DC keys. TensorBoard carries the finite values
of the same keys as `diagnostics/eval_<key>`, and the termination counts as
`terminations/eval_<cause>` (the training rollouts' termination fractions are
`terminations/<cause>`).

### Whole-body balance and floor forces

The pad and whole-foot centres of pressure say where each foot carries its
load. They cannot say how close the animal stands to falling over its toes,
or how it holds a lean: the T. rex stance runs that nosedived reached the
toe line with every foot rail still inside its bar. So on a model whose
recorder runs (the box-soled models), each evaluation episode is also
reduced to whole-body balance and force numbers. For the evaluation only,
the recorder is built with its balance capture on. At the last physics
substep of every control step, and once at the episode's first substep, it
copies the whole animal's centre of mass (`subtree_com` of the free root),
the applied actuator forces, the torque at each declared ankle
(`qfrc_actuator + qfrc_passive`), and each floor contact's frame, tangential
constraint forces and sliding friction. It reads the physics and writes
nothing, so the evaluation is unchanged.
`environments/shared/gait/balance_telemetry.py` reduces each episode over
the gate's post-settle window. The heading of every sample is the one
`heading_observation` uses: forward is the horizontal projection of the
root's forward axis. Each key below is the evaluation's mean, over its
full-horizon episodes, of the per-episode value, except where noted.

| Keys | Meaning |
|---|---|
| `com_fore_aft_mean_m` | The whole animal's centre of mass (CoM) ahead of the mean of the feet's reference sites, along the heading (m), window mean |
| `root_pitch_mean_deg` | The root's forward axis above horizontal (degrees, + nose up), window mean |
| `xcom_fore_aft_max_m` | The extrapolated centre of mass (XcoM, Hof 2008: `xi = c + v / omega0`, `omega0 = sqrt(g / h)`, with `h` the CoM's height above the floor plane and `v` from the differences of successive samples) ahead of the feet's reference point, window maximum (m) |
| `hof_cop_term_mean_mps2`, `hof_counter_rotation_mean_mps2`, `hof_counter_rotation_rms_mps2` | Hof's split (2007) of the CoM's forward acceleration `a = f . F / m`, with `F` the net floor force on the animal, the only horizontal external force on a stage without pushes (see below). The centre-of-pressure term `F_z f . (c - CoP) / (m h)` is the part the floor centre of pressure's offset from the CoM explains; the counter-rotation term is the rest, body segments rotating about the CoM. The two sum to `a` at every sample. Window means of both, and the counter-rotation term's root mean square (m/s²) |
| `leg_q_<left>_minus_<right>_<joint>_deg`, `leg_q_<left>_<right>_mean_<joint>_deg` | The achieved counterpart of the commanded leg DC: each sagittal joint's angle offset from the home pose (the env's reset keyframe), each side signed about the leftward axis as the `leg_dc_` keys are; left minus right, and the mirror-even mean; window means in degrees |
| `xcom_toe_margin_min_m`, `xcom_toe_margin_mean_m`, `xcom_heel_margin_min_m` | Feet that declare a support span (the T. rex): how far the XcoM is inside the declared toe line and heel line, each line the mean over those feet along the heading (m, + inside): the toe margin's window minimum and mean, and the heel margin's window minimum. In Hof's inverted-pendulum model, once the XcoM is past the toe line, moving the centre of pressure within the feet can no longer stop the CoM |
| `xcom_past_toe_fraction` | The same feet: the window fraction of steps with the XcoM past the toe line |
| `time_to_toe_min_s` | The same feet: the shortest time for the CoM to reach the toe line at its forward speed, over the window steps on which it moves toward a line still ahead of it; NaN when it never does |
| `fall_xcom_toe_lead_s` | The same feet: over the evaluation's fallen episodes, the mean time from the XcoM's first step past the toe line in the window to the episode's end (s). Only a fallen episode has the value, so there is no full-horizon mean of it; NaN when no fallen episode passed the toe line |
| `xcom_tip_margin_min_m` | Hinged digits (the T. rex): the XcoM's margin to the digit tips (the forward-most tip of each foot along the heading, averaged over the feet), window minimum (m) |
| `actuator_force_frac_max` | The highest, over the force-limited actuators, of the window mean of `\|F\| / limit` (the upper limit for a positive force, the lower one negated for a negative force) |
| `actuator_force_limited_fraction` | The window fraction of steps on which some force-limited actuator is at 0.99 of its limit or more |
| `actuator_force_frac_per_actuator` | An (evaluations, actuators) matrix: each actuator's window mean of `\|F\| / limit`, averaged over the evaluation's episodes that have a window, fallen ones included, as the action DC is; NaN for an actuator without a force limit |
| `shear_over_normal`, `<label>_shear_over_normal` | The floor's forward force over its normal force, each summed over the loaded window steps, for the whole animal and per limb (+ the floor pushes the animal forward) |
| `<label>_cone_use_mean`, `<label>_cone_use_max` | Per limb: the horizontal floor force over `mu F_z`, with `mu` the limb's load-weighted sliding friction, the fraction of the friction cone used; mean and maximum over the loaded window steps |
| `<label>_ankle_moment_arm_m` | Per foot with a declared ankle (the T. rex): the ankle torque about the leftward axis over the normal force on the limb, each summed over the loaded window steps (m, + ahead of the ankle). It estimates the whole-foot centre of pressure's lever about the ankle from the joint side; it also carries the shear times the ankle's height and the foot's own weight |

The positions are those of the control step's last physics substep, one
physics step (2 ms on the T. rex) before the control boundary, as for every
other recorder field. The CoM's height is measured from the z = 0 plane:
the stance floors are flat.

Two limits of the Hof keys (`hof_*`):

- **Pushes.** The split assumes the floor is the only horizontal external
  force. A recovery stage's env also pushes the animal on a schedule,
  through `xfrc_applied` on its root, and the split does not include the
  push: while one acts, `a` is the floor's share of the CoM's acceleration
  only, not the CoM's acceleration, and the counter-rotation term and its
  RMS take in the push's effect on the floor forces. In one zero-action T.
  rex recovery episode (seed 3042, 561 steps, 40 of them pushed), the
  root mean square of the CoM's second difference less `F/m` was
  0.84 m/s² on the pushed steps (correlation -0.47) and 0.024 m/s² on the
  others (correlation 0.999). On a stage with pushes, read the `hof_*`
  keys as the floor's share only.
- **One substep per control step.** `F` is decoded exactly, but at one
  substep, the control step's last. Where the contacts chatter between
  substeps, that one sample can stand far from the step's mean. On the T.
  rex (`frame_skip` 5) `F/m` and the CoM's second difference agree with
  correlation 1.000 even in a nosedive, and every model's statue gives
  0.998 or more. On the Compsognathus robot (`frame_skip` 10) holding a
  -0.05 ankle offset, the correlation was 0.42 forward and 0.03 vertical,
  and the window mean of `F/m` was -0.0099 m/s² forward against the CoM's
  -0.0001: on a quiet `frame_skip`-10 stance the `hof_*` window means are
  mostly the sampling phase of that substep, not the body's balance.

The force keys decode the elliptic friction cone, which every model uses;
under a pyramidal cone every force key but the two actuator ones would
read NaN.

The keys follow the pad keys' rule: a key exists only where the model has
what it measures, so no model attaches a recorder it did not attach
before.

- `com_fore_aft_mean_m`, `root_pitch_mean_deg`, `xcom_fore_aft_max_m`, the
  Hof terms, the achieved leg offsets and the force keys but the ankle's:
  the four box-soled models (the Tyrannosaurus Rex, both Compsognathus
  embodiments and the Dibothrosuchus).
- The toe and heel margins, `xcom_past_toe_fraction`, `time_to_toe_min_s`,
  `fall_xcom_toe_lead_s` and the ankle moment arm need a declared support
  span and ankle, and the tip margin needs hinged digits: the
  Tyrannosaurus Rex only.
- The Velociraptor and the Brachiosaurus have none of them: no box pad, no
  recorder.
- `leg_dc_<left>_<right>_mean_<joint>_deg` follows the left-minus-right leg
  DC keys: every model.

NaN in a key that exists means it was not measured at that evaluation: no
full-horizon episode (no episode with a window, for the matrix), an empty
window, a limb never loaded in it, or a failed capture. The balance capture
and its reduction each have their own guard: a failure reads NaN on the
balance and force keys alone, logged once, and leaves the evaluation, the
run, and the pad, whole-foot and roll-rail fields as they were. If the
recorder cannot be built with the balance capture, it is built without it,
and those keys read NaN for the rest of the run. The capture copies at
most a few hundred numbers per control step, and the reductions run once
per episode, vectorised. On the T. rex, a four-episode evaluation of
1,000-step zero-action episodes takes about 3% longer with them: over ten
interleaved rounds on one CPU, the median of each round's best of three
was 7.33 s on main before this change and 7.58 s with it. TensorBoard
carries the finite scalars as `diagnostics/eval_<key>`, and never the
console.

### The in-training `stance_quality/v2` screen

On a stage whose gate is `stance_quality/v2` or the six-heading certificate
`stance_quality_heading/v1` (which applies v2's decision) and whose
recorder runs, every evaluation is also judged by the stage's own v2 bars:
`evaluate_stance_v2_gate(..., in_training=True)` on the evaluation's
episodes, at yaw 0, on the evaluation env's own resets, without the
statue-relative criteria (on the T. rex, `min_avg_reward_statue_ratio`; on
the anatomical Compsognathus, that and `min_foot_load_share_statue_ratio`).
Today that is the T. rex and the anatomical Compsognathus stances. The
Velociraptor's `stance_quality/v2` stance has no box pad and no recorder,
so it has no screen. The screen shows during training whether an
evaluation's episodes clear the stage's clean-episode bound at yaw 0. It is
never a verdict: the certificate applies the full decision at six headings
on its own seeded panel after the stage, and no gate, verdict or checkpoint
selection reads these keys.

| Keys | Meaning |
|---|---|
| `screen_v2_episodes` | The episodes judged, the evaluation's |
| `screen_v2_clean_count` | Those clean on every per-episode bar the screen applies |
| `screen_v2_clean_lcb` | The lower confidence bound on the clean fraction the gate certifies with (`binomial_lcb`) |
| `screen_v2_yaw0_bars_met` | 1 when the screen's decision passes, else 0 |
| `screen_v2_hop_or_fall_episodes` | Where the stage declares `max_hop_or_fall_episodes`: the episodes that hopped or fell |
| `screen_v2_fail_<criterion>_count` | Per criterion the gate counts (`horizon`, then each declared episode criterion): the episodes failing it; NaN for a criterion the screen skips (`min_foot_load_share_statue_ratio`), which it never judges |

The key set is fixed by the stage's declaration, and every key is NaN on an
evaluation the screen could not judge (a failed capture, an episode without
metrics). A skipped criterion keeps its key, but its count is NaN on every
evaluation: the gate's own tally starts it at 0, which would read as "no
episode failed it" when the screen never applied it (`progress.json` lists
it under `evaluation.screen_v2.skipped`). The names keep the screen apart
from a verdict: `screen_v2_yaw0_bars_met` says what it is; no key is the
recorded-history gate's `stance_clean_count` (`reporting/gates.py`), so
nothing reads the screen as one; the per-criterion counts end in `_count`,
so that no key ends in a foot rail's suffix (`..._sole_cop_fore_aft`),
which the posture readers would take for a foot; and TensorBoard files them
under their own namespace, `screen_v2/`, apart from
`diagnostics/eval_gate_met`.

### The posture-drift advisory

On a stage that declares a fore-aft centre-of-pressure bar
(`max_sole_cop_fore_aft` or `max_foot_cop_fore_aft`), a warning line starting
`PostureDrift` appears in the training output (in the notebook, the training
cell's output) when that rail's `_over_bar_fraction` is at least 0.5 on 2
evaluations in a row. It names the rail, the fractions, the steps and each
foot's latest signed centre of pressure (+ toward the toe), and repeats at
most once per 10 evaluations while the drift lasts. It never stops or
changes training (`curriculum/posture_watch.py`). Three `[curriculum]` keys
tune it, like `baseline_warn_after_budget_fraction`, and enter no gate, task
or hyperparameter digest: `posture_drift_warn_over_bar_fraction` (0.5),
`posture_drift_warn_consecutive_evals` (2) and
`posture_drift_warn_repeat_evals` (10). No committed stage sets them.

Two evaluations in a row, not three, because two warn earlier, and on no
series that three would not warn on. Replayed on the T. rex physics-r10
screen's own whole-foot series (bar 0.80, one evaluation per 131,072 steps,
to 6.16M steps), two first warn on the drifting seeds 52 and 60 at 917,504
steps and never warn on seed 42 (through 6.03M). Three first warn on seed 52
at 2,359,296 steps, 1.44M steps later, and on seed 60 at 1,048,576, one
evaluation later. Over the 69 pad-rail series of 3M steps or more from the
earlier T. rex stance screens, every series two warn on, three warn on too,
1 to 10 evaluations later, and none warns under two only.

An evaluation with no full-horizon episode has a NaN over-bar fraction
(nothing measured the posture), and NaN ends the streak of drifting
evaluations. A seed whose evaluation episodes all fall therefore goes quiet
in this advisory exactly when it is worst. The falls still show in
`progress.json`'s `evaluation.falls` and in `gate_progress.npz`'s
`full_horizon_fraction`.

### The posture-trend advisory

`PostureDrift` warns once half the panel stands over a fore-aft bar. A line
starting `PostureTrend` (`PostureTrendWatchCallback`,
`curriculum/posture_watch.py`) reads earlier. On a stage that declares a
fore-aft centre-of-pressure bar, after every evaluation and for each such
rail, it takes:

- the rail's over-bar fraction;
- the mean over the feet of their signed per-foot centre of pressure
  (`<label>_sole_cop_fore_aft` for the pad rail, `<label>_foot_cop_fore_aft`
  for the whole-foot rail; + toward the toe);
- that mean's least-squares slope against the steps, in half-lengths per
  million steps, over the finite values among the last 5 evaluations (at
  least 3).

It prints one line when the last 2 evaluations both have an over-bar
fraction of 0.1 or more and the signed mean moves toward the edge it is on
(slope times mean above 0). The line names the rail, its fractions and
steps, the signed mean and its slope, and each foot's signed centre of
pressure, and says that training continues unchanged. It repeats at most
once per 10 evaluations, and a NaN fraction (no full-horizon episode) ends
the streak, as for `PostureDrift`. It is listed in `progress.json`'s
warnings (`source: "posture_trend"`). TensorBoard gets
`posture/<rail>_signed_mean` and `posture/<rail>_trend_per_mstep` at every
evaluation. It never stops or changes training.

Why 0.1: on the T. rex's 40-episode panel, 4 episodes of 40 over a bar
already fail the 37-of-40 bound on that rail alone. Four `[curriculum]`
keys tune it, registered and validated like the drift watch's, set by no
committed stage and in no digest: `posture_trend_warn_over_bar_fraction`
(0.1), `posture_trend_warn_consecutive_evals` (2),
`posture_trend_window_evals` (5, at least 3) and
`posture_trend_warn_repeat_evals` (10).

Replayed on the arm L screens' own series (10-episode evaluations every
131,072 steps; r9 on the pad rail to 6,291,456 steps, r10 on the
whole-foot rail to 11,927,552; the first four lines of each r10 series),
against `PostureDrift` at its defaults:

| Series | `PostureTrend` lines | `PostureDrift` lines |
|---|---|---|
| r9 arm L seed 60, pad rail | 4,325,376; 5,636,096 | 4,849,664 |
| r9 arm L seed 52, pad rail | 3,407,872 | none |
| r9 arm L seed 42, pad rail | 2,621,440; 4,063,232; 5,505,024 | none |
| r10 arm L seed 60, whole-foot rail | 393,216; 1,703,936; 3,014,656; 5,111,808 | 917,504; 2,228,224; 3,538,944; 4,849,664 |
| r10 arm L seed 52, whole-foot rail | 786,432; 2,097,152; 3,801,088; 5,111,808 | 917,504; 2,228,224; 3,538,944; 4,849,664 |
| r10 arm L seed 42, whole-foot rail | 655,360; 3,407,872; 4,718,592; 6,553,600 | none |

On r9 seed 60 the first line came at 4,325,376 steps, one evaluation into
the drift that lasted (from 4,194,304) and four evaluations before
`PostureDrift`. In these 10-episode series 0.1 is a single episode, so the
rule fires more readily there than it will on a 40-episode panel: it
printed on all six series, r10 seed 42 included, the one r10 seed that
passed `stance_quality/v2` at 6M. A `PostureTrend` line says "look at this
run's posture", not that the stage will fail.
`test_curriculum_posture_watch.py::TestTheReplays` holds the series and
pins these steps.

### The gate progress figure

When a stage ends, `generate_stage_artifacts` draws
`figures/gate_progress.png` from `gate_progress.npz`
(`visualization.plot_gate_progress`): one panel per group the file holds,
against millions of steps.

1. The mean reward, with the zero-action statue's reward when the run
   captured one, and the full-horizon fraction.
2. The signed per-foot pad and whole-foot centres of pressure, with each
   declared bar at plus and minus.
3. Each rail's over-bar fraction, with `PostureDrift`'s 0.5 and
   `PostureTrend`'s 0.1 dashed.
4. The screen's `screen_v2_clean_lcb` against the stage's
   `min_clean_stance_lcb`.
5. `com_fore_aft_mean_m` and `xcom_toe_margin_min_m`, with 0.
6. The leg DC, left minus right and mirror-even: commanded (`leg_dc_`,
   solid) and achieved (`leg_q_`, dashed).

It is drawn after the other figures in its own guard, so a failure costs
this figure only. To plot a run while it trains, from a copy of its stage
directory (a Drive download): `python
environments/shared/scripts/plot_gate_progress.py <stage_dir> [--out
PNG]`. It reads the bars from the copy's `stage_config.json` and writes
`gate_progress_<stage dir name>.png` in the current directory. It refuses
an `--out` inside the run directory (the nearest ancestor holding
`provenance.json`): a result bundle's manifest hashes every file there, and
verification rejects a file it did not list.

## Evaluation episodes: `eval_episodes.csv`

`gate_progress.npz` keeps one value per evaluation, a mean over its
episodes. Each stage directory also holds `eval_episodes.csv`, one row per
evaluation episode (`StageAwareEvalCallback` with
`environments/shared/episode_ledger.py`), on every model. It is rewritten
atomically after every evaluation and when training ends, so every row up
to a checkpoint's step is on disk before a run can resume from that
checkpoint (SB3's `best_model`, saved inside the evaluation just before
the rewrite, is the one exception). A same-stage resume keeps the previous session's rows at or
before the resumed step. It does so where it seeds `gate_progress.npz`:
when the resume can seed the evaluation history from the stage's
`evaluations.npz`. When that file is missing, unreadable or holds another
panel size, this file, `train_episodes.csv` and `policy_probe.npz` start
again with the resumed session's rows. Floats are written in full
(`repr`), `nan` where not measured, and a column whose values the
evaluation could not align with its episodes reads empty.

| Columns | Meaning |
|---|---|
| `evaluation`, `timesteps`, `episode`, `env` | The evaluation's index and training step, the episode's index in it, and the evaluation env it ran on |
| `return`, `length`, `full_horizon`, `cause` | The episode's return and length (in `evaluations.npz`'s order), 1 when it reached the horizon, and how it ended: `truncated`, the env's `termination_reason`, or `terminated` |
| `screen_v2_clean`, `screen_v2_reasons` | Where the screen runs: 1 when the episode met every bar the screen applies, and the bars it failed, `;`-joined |
| `spawn_stance_width_change_m`, `<label>_spawn_tilt_deg` | Where the recorder runs: at the episode's first substep, the largest change of a foot pair's horizontal distance from the home pose (m), and each box sole's tilt (degrees). In this file only |
| `<label>_sole_cop_fore_aft`, `<label>_sole_cop_fore_aft_gate`, `<label>_digit_share`, `<label>_foot_cop_fore_aft`, `<label>_foot_cop_fore_aft_gate`, `<label>_toe_margin_nm`, `<label>_sole_tilt_deg`, `<label>_sole_corner_lift_m` | Per foot, where it has them: the episode values whose full-horizon means `gate_progress.npz` holds |
| `max_sole_cop_fore_aft`, `max_foot_cop_fore_aft`, `max_sole_tilt_deg`, `max_sole_corner_lift_m` | The episode's maxima over the feet, the per-episode numbers the gate's rails read |
| The balance and force keys | The episode's own values of the keys in the balance table above, with `xcom_toe_lead_s` in place of its fall mean (NaN on an episode that did not fall) |

The per-digit load fields and the whole-foot CoP spread and frequency stay
in `gate_progress.npz` only. On the T. rex a row has 63 columns and about
1.06 KB (measured), so the file of a 12M stance (40 episodes every 50,000
steps, 9,600 rows) ends at about 10 MB, and rewriting it at every
evaluation writes about 1.2 GB to Drive over the run.

## Training episodes: `train_episodes.csv` and the rollout counts

The evaluation episodes are the deterministic policy's. The falls a
stance run learns from are its training episodes: 15 to 17% of one T. rex
seed's training episodes ended early at 4.7 to 5.2M steps while its
evaluations stood. `TrainingEpisodeLedgerCallback`
(`environments/shared/rollout_ledger.py`) writes one row per training
episode to `train_episodes.csv`, on every model trained by `train()` or
the command-line curriculum:

| Columns | Meaning |
|---|---|
| `timesteps`, `env`, `length`, `return`, `cause` | The run's step count when the episode ended, its training env, Monitor's length and return, and how it ended, as in `eval_episodes.csv` |
| `com_fore_aft_mean_m`, `root_pitch_mean_deg` | The evaluation keys' definitions, over the post-settle window |
| `com_fore_aft_last_m`, `root_pitch_last_deg` | The same at the last reading before the episode ended |
| `actuator_force_limited_fraction` | The evaluation key's definition (one function serves both), over the window |
| `<label>_touch_share_pad`, `<label>_touch_share_<stem>`, `<label>_touch_share_digits` | On a foot with a sole, one touch sensor on the sole's body and at least one on a hinged digit (the T. rex: `toe_d2`, `toe_d3` and `toe_d4`): each sensor's share of the foot's summed touch readings over the window, and the digits' sum. Measured against floor truth, these four shares predict the whole-foot centre of pressure with R² 0.91. No other model has such a foot |

When an env reports `done` it has already reset, so its data belong to the
next episode: the terminal state is never read, and the window is the
post-step readings past `settle_steps`, the terminal one excluded. An
episode that ends inside the settle window has NaN means and finite last
values. Per step the callback only copies raw `mjData` reads of the envs
that are not done (the root's CoM and orientation, the feet's reference
sites, those touch sensors and the actuator forces); the frames and
reductions run once per episode. A `SubprocVecEnv`'s envs live in other
processes: their rows keep the outcome columns and read NaN in the rest
(one warning). A failure costs that env's episode (NaN, one warning), never
the run. The file is rewritten on the checkpoint cadence (in the calls a
periodic checkpoint is saved in) and at training end, and a same-stage
resume keeps the previous session's rows at or before the resumed step.
On four T. rex envs the per-step copies cost 44 to 48 µs per vector step
in a micro-benchmark, against a 4.8 to 5.2 ms physics step (about 0.9%);
the short T. rex proof run clocked the whole callback, row closes and
checkpoint-cadence writes included, at about 59 µs per vector step (0.85 s
over 14,336 vector steps, about 1.2% of the physics step), with the main
run beside it on the same machine. TensorBoard gets, per
rollout, over the episodes that ended in it: `rollout_balance/episodes`,
`falls`, `com_fore_aft_mean_m`, `com_fore_aft_last_falls_m` and
`force_limited_fraction_falls`.

A fall here is any episode that did not reach the time limit (`cause` not
`truncated`). On a stance stage no termination is a success. On a task stage
the env also terminates on success (`bite_success`, `strike_success`,
`food_reached`, `target_reached`, `snap_success`), and those episodes
count as falls in the ledger's summaries (`rollout_balance/falls`,
`training_rollouts.falls`, `fall_fraction`, `falls_total` and the `_falls`
means) and in `rollout_falls` below; the ledger's `cause` column and
`rollout_term_count_<reason>` name them.

`diagnostics.npz` gains the counts behind its termination fractions:
`rollout_episodes` (episodes that ended in each rollout) and
`rollout_falls` (those that terminated, success terminations included
on a task stage), aligned with `timesteps`, and
`rollout_term_count_<reason>`, aligned with `term_timesteps` like the
`term_<reason>` fractions and back-filled with 0 the same way. A fraction
alone cannot say whether a rollout ended two episodes or forty. They are
deliberately not `term_count_<reason>`: every `term_` key but
`term_timesteps` is read as a termination fraction
(`visualization.termination_fraction_keys`,
`scripts/compare_run_diagnostics.py`). TensorBoard gets
`terminations/rollout_episodes` and `terminations/rollout_falls`. Like the
rest of `diagnostics.npz`, the counts start again on a same-stage resume
(`docs/KNOWN_ISSUES.md`).
`diagnostics.npz` also gains `reward_action_jerk`, the action-jerk reward
term every stance stage pays (`action_jerk_weight`): the env emitted it,
but nothing recorded it. It reaches TensorBoard
(`diagnostics/reward_action_jerk`), W&B and the post-training results CSV
(`eval_reward_action_jerk`) as the other reward terms do.

## The policy probe: `policy_probe.npz`

TensorBoard's `train/std` is the exploration noise averaged over the
actuators; on the r9 arm L runs it hid a 9- to 13-fold spread between
them. Nothing recorded what the policy does at a known posture as it
trains, or what its critic makes of a lean. `PolicyProbeCallback`
(`environments/shared/policy_probe.py`) records, at the end of every PPO
rollout (so for the policy that collected it, before `train()` updates
it), the policy's state and its outputs on a fixed bank of states.

The bank is ten states of the stage's own env, on reset seeds 3042 and
3043: the settled statue at step 199, the statue at step 599,
mirror-symmetric ankle leans of +0.03 and -0.03 action units (ramped over
steps 200 to 300, then held) at step 599, and a -0.05 lean at step 400.
The lean is applied as `sign × x` to each side of every sagittal ankle
pair, with each side's leftward-axis sign, so it is the same posture on
every model. The bank is built once per session on a separate env instance
that nothing else steps, with a horizon of its own (the stage's, or 601
steps if that is longer), so a short stage horizon still yields all ten
states. On a stage with scheduled pushes (a recovery stage's
`perturbation_capture_velocity_multiple`), the bank env runs with the
pushes off (`policy_probe.bank_env_kwargs`): a push topples the statue and
the leans before their kept steps, and with the pushes on the T. rex
recovery stage kept 4 of the 10 states (no `statue` row) and the anatomical
Compsognathus recovery stage 6. With the multiple at 0 the env's reset
draws exactly what a push-free reset of the same seed draws, and the
observation does not depend on it. A row whose episode still ends at or
before its kept step is dropped, with one warning naming every dropped row, and
the scalars that compare with it read NaN (`null` in `progress.json`). A
guard snapshots and restores the global random generators around the
build, which draws from none of them. The build costs about 5.6 s per
session on the T. rex. SAC gets no bank and no rows.

Per update the probe reads parameters and runs `torch.no_grad()` forward
passes of the constant bank: no sampling, no gradient, and no parameter,
optimizer, buffer or training-mode change. It cost 1.6 to 2.3 ms per update
on the T. rex. The file is rewritten atomically on the checkpoint cadence
and at training end. A same-stage resume keeps the previous session's rows
at or before the resumed step when its bank equals this session's (the
labels, seeds, steps, offsets and raw observations); otherwise it drops
them with one warning.

| Key | Shape | Meaning |
|---|---|---|
| `update_timesteps` | (U,) | The step count at each rollout's end |
| `std_per_actuator` | (U, actuators) | The exploration standard deviation, `exp(log_std)` |
| `obs_rms_mean`, `obs_rms_var`, `obs_rms_count`, `ret_rms_var` | (U, observations), (U,) | VecNormalize's statistics (NaN without VecNormalize) |
| `ent_coef`, `learning_rate` | (U,) | The entropy coefficient and the learning rate the coming `train()` applies (the rate from SB3's own schedule, not the optimizer's current value, which is the previous update's) |
| `bank_value_now`, `bank_value_frozen` | (U, B) | The critic's value at each bank state, under the current observation normalization and under the one frozen at the session's start, so that a change of value can be told apart from a change of the normalizer |
| `bank_action_now`, `bank_action_frozen` | (U, B, actuators) | The deterministic mean action at each bank state, under both normalizations, before clipping |
| `frozen_reference_timesteps` | (U,) | The step the frozen statistics were taken at (the session's start) |
| `frozen_obs_mean`, `frozen_obs_var` | (observations,) | The frozen statistics |
| `bank_raw_obs` | (B, observations) | The bank's observations, before normalization |
| `bank_labels`, `bank_seeds`, `bank_steps`, `bank_ankle_offset`, `bank_com_fore_aft_m` | (B,) | The bank: each state's label, reset seed, step, lean, and its CoM ahead of the feet (m) |
| `bank_horizon`, `bank_build_seconds` | () | The bank env's horizon and the build's wall time |
| `ankle_actuators`, `ankle_signs` | (P, 2) | Each ankle pair's left and right actuators and leftward-axis signs: one pair on a biped, two on the Dibothrosuchus and the Brachiosaurus |

The per-update arrays are float32, but the step counts (int64) and
`obs_rms_count` (float64): about 1.9 KB per update on the T. rex
(15 actuators, 63 observations, 10 states), about 2.8 MB over the 1,465
updates of a 12M run. TensorBoard gets, at every update and never on the
console:

| Tag | Meaning |
|---|---|
| `policy_probe/std_min`, `std_max`, `std_max_over_min` | The spread of the per-actuator exploration noise that the mean `train/std` hides |
| `policy_probe/ent_coef` | The entropy coefficient the next `train()` uses, logged nowhere else |
| `policy_probe/obs_var_min`, `obs_var_max` | The normalizer's per-dimension extremes |
| `policy_probe/value_statue` | The critic's value at the two `statue` states |
| `policy_probe/value_ankle_plus_minus_statue`, `value_ankle_minus_minus_statue`, `value_ankle_minus_0.05_minus_statue` | Each lean's value less the statue's on the same seed, averaged over the seeds |
| `policy_probe/value_normalization_shift_statue` | The statue's value under the current normalization less its value under the frozen one (on a fresh normalizer, against the initial statistics: see below) |
| `policy_probe/statue_ankle_action_mean` | The mirror-even ankle action at the statue states, averaged over the ankle pairs |
| `policy_probe/ankle_gain_per_m` | The difference of the mirror-even ankle action between the +0.03 and -0.03 leans over the difference of their CoM positions, averaged over the seeds: the deterministic policy's ankle feedback gain to a CoM lean, in action units per metre |

The frozen statistics are VecNormalize's at the session's start. On a
stage that loads a normalizer (a later stage, or a resume) they are that
normalizer's. On a stage that starts with a fresh one they hold only the
initial reset observations (a count of about the number of envs), so some
variances are tiny and the frozen reference is near-degenerate: on the
short T. rex proof run the smallest frozen variance was 2.5e-5 (median
7.5e-4), 1.4% of the bank's inputs clipped at ±10 under it and none under
the first update's statistics, and the first row's `bank_value_frozen`
read 0.63 at the settled statue against 0.16 for `bank_value_now`. There
`bank_*_frozen` and `value_normalization_shift_statue` measure against the
initial statistics, not a meaningful earlier normalizer; compare rows of
`bank_value_now` with `obs_rms_*` instead.

## Training progress: `progress.json`

Each training stage directory also holds `progress.json`, a small
human-readable snapshot of the stage, so a run can be checked mid-run
without opening the binary `.npz` files. On the T. rex it is about 8 KB
before any warning (7.8 KB measured on a short stance run, which wrote
4.6 KB without the balance, force, policy and training-rollout blocks
below); each warning it keeps adds 0.6 to 0.9 KB, and it keeps the
newest ten, so it stays under about 17 KB. On Colab, open it from
Drive (`MyDrive/mesozoic-labs/logs/<species>/<embodiment>/<algorithm>/<run>/<stage_dir>/progress.json`).
`TrainingProgressCallback` (`environments/shared/curriculum/progress_report.py`)
rewrites it atomically (a temporary file in the stage directory, then
`os.replace`, safe on a Drive mount) when training starts, after every
periodic evaluation and when training ends. It reads what the evaluation
callbacks already recorded (the evaluation history, the latest
`gate_progress.npz` row, the policy probe's latest row and the training
ledger's summary) and recomputes nothing. Nothing reads it back, and
a failure to write it is logged once and never stops a run. It is strict
JSON: a value that could not be measured is `null`, never NaN. Numbers keep
six significant digits, and the posture, balance, force and probe
statistics four. A block whose source a run does not have is left out,
never filled with `null`.

The schema string changes only when a key is removed or renamed, or changes
its meaning, units or type. Adding a key or a block does not change it:
the balance, force, policy and training-rollout blocks, the screen and the
achieved leg offsets were added under `mesozoic.training-progress/v1`, and
every key and type the file had before is kept.

| Key | Meaning |
|---|---|
| `schema` | `mesozoic.training-progress/v1` |
| `status` | `training`, or `ended` once `learn()` returns. A run killed mid-stage (a Colab disconnect) is left at `training` with an old `written_at` |
| `written_at` | When the file was written, UTC (`2026-10-16T20:20:01Z`) |
| `species`, `embodiment`, `model` | The model's two IDs and its display name (`null` for a run that names no model) |
| `stage` | The stage id (`stance`) |
| `run_id`, `run_dir`, `stage_dir` | The run's `provenance.json` `run_id` (`null` when the run directory has none) and the two directory names |
| `seed` | The run's seed |
| `timesteps`, `budget_timesteps`, `fraction_done` | Steps trained (cumulative across a same-stage resume), the cumulative step count this session trains to, and their ratio |
| `session_started_at`, `wall_clock_s`, `steps_per_s`, `projected_finish` | When this session's training started (UTC), the seconds since, the steps per second observed over them (evaluations included), and the UTC time the budget is reached at that rate (`null` before the first step) |
| `zero_action_reward` | The zero-action statue's mean reward from the run's `zero_action_baseline.json`, when the run captured one |
| `evaluation` | The latest evaluation: `timesteps`, `episodes`, `mean_reward`, `std_reward`, `mean_episode_length`, `falls` (episodes shorter than the horizon), `full_horizon_fraction` and `termination_causes` (each cause's count, causes no episode ended by left out); `null` before the first. Where the screen runs, `screen_v2`: `clean`, `episodes`, `clean_lcb`, `bar` (the stage's `min_clean_stance_lcb`), `yaw0_bars_met` (never a verdict), `hop_or_fall` and `hop_or_fall_bar` where the stage declares that bar, `failing` (each criterion some episode failed, with its count) and `skipped` (the statue-relative criteria the screen leaves out) |
| `posture` | Per rail the model measures (`max_sole_cop_fore_aft`, `max_foot_cop_fore_aft`, `max_sole_tilt_deg`, `max_sole_corner_lift_m`): `bar` (the stage's, `null` when it declares none), `mean` (the `_mean` key), `over_bar_fraction` (with a bar only) and `per_foot`. Per foot, the CoP rails give `gate` (the per-foot statistic's mean) with the signed CoP and the digit share or the moment margins where the foot has them; the roll rails a number. Then `leg_dc_deg`, the commanded leg DC per joint pair, left minus right (`l_minus_r_ankle`) and mirror-even (`l_r_mean_ankle`), and, where the recorder runs, `leg_q_deg`, the achieved joint offsets in the same form. `{}` before the first evaluation |
| `balance` | Where the recorder runs: the latest row's balance keys (`com_fore_aft_mean_m`, `root_pitch_mean_deg`, `xcom_fore_aft_max_m`, the Hof terms, and on the T. rex the XcoM margins, `xcom_past_toe_fraction`, `time_to_toe_min_s`, `fall_xcom_toe_lead_s` and `xcom_tip_margin_min_m`) |
| `forces` | Where the recorder runs: `actuator_force_frac_max`, `actuator_force_limited_fraction`, `shear_over_normal`, and `per_foot` with `shear_over_normal`, `cone_use_mean`, `cone_use_max` and, on a declared ankle, `ankle_moment_arm_m` |
| `policy` | PPO stages: the policy probe's latest row, its `policy_probe/*` scalars (without the prefix) and `update_timesteps` |
| `training_rollouts` | The training episodes that ended in the last 262,144 steps (`window_timesteps`): `episodes`, `falls` (episodes that did not reach the time limit, a task stage's success terminations included), `fall_fraction`, `com_fore_aft_mean_m` over them, `com_fore_aft_last_falls_m` and `force_limited_fraction_falls` over the falls, and per foot with touch shares `touch_shares` (`pad`, each digit, `digits`; each a mean of the episodes' shares); then this session's `episodes_total` and `falls_total` |
| `history` | The last 5 evaluations, oldest first: the `evaluation` fields but the causes and the screen, with each rail's `_mean` and `_over_bar_fraction`, and the screen's `screen_v2_clean_count` and `screen_v2_clean_lcb`, where recorded |
| `checkpoints` | `latest_periodic_timesteps` (the newest periodic checkpoint in `models/`, by its name), `periodic_timesteps` (every periodic checkpoint kept, sorted: the newest few and, under the archive rule below, one per interval), `best_model_timesteps` and `best_mean_reward` (SB3's mean-reward best), `robust_best_model_timesteps` and `robust_best_score` (mean minus one standard deviation). A best checkpoint whose evaluation a resume dropped from the history reads `null`, while its score (`best_mean_reward`, `robust_best_score`) keeps the value the resumed tracker carried over |
| `warnings_total`, `warnings` | How many advisories this session has raised, and the newest 10 in step order, each `{"timesteps", "source", "message"}`: `posture_drift` and `posture_trend` (each with `rails`), `evaluation_plateau`, `zero_action_baseline` and `hop_watch`. Each also appears as a warning line in the training output. A resumed session counts its own |

## Training history pages: `plot_training_history.py`

`gate_progress.npz` and `progress.json` hold per-evaluation means and the
latest row. To see how one run's feet, balance, asymmetry and saturation
moved over its training, or to overlay several runs, draw them from the
ledgers above:

    python environments/shared/scripts/plot_training_history.py STAGE_DIR [STAGE_DIR ...] --out DIR \
        [--labels ...] [--groups ...] [--mark-steps 4194304 ... final] [--title ...]

One stage directory gives that run's pages; several give a comparison, one
colour per run (at most eight) and one line style per `--groups` value, each
series a rolling mean over 5 evaluations (`--smooth`), except the saturation
counts. `--out` receives one PNG per page, `training_history.html`, which
embeds them (base64) beside tables of the values at the marked, best and
last checkpoints, and `training_history.json`
(`mesozoic.training-history/v1`: every reduced series, the declared bars,
the band settings, the alarms and what was absent). An `--out` inside a run
directory is refused, as `plot_gate_progress.py` refuses it; for a copy
without `provenance.json` the run directory is the stage directory's parent.
It reads the files and writes nothing else (`reporting/training_history.py`).

| Page | What it draws, against the training step |
|---|---|
| Outcomes | Clean episodes (`screen_v2_clean`), falls (shorter than the horizon), the mean return per evaluation, and the training falls per 262,144-step window of `train_episodes.csv` (`cause` not `truncated`) |
| Saturation | The episodes per evaluation listing `max_actuator_saturation_fraction` in `screen_v2_reasons`, and `actuator_force_frac_max` |
| Whole-foot and pad CoP | Per foot, `<label>_foot_cop_fore_aft` and `<label>_sole_cop_fore_aft`, with `max_sole_cop_fore_aft` at plus and minus and, where the run pays it, the reward band's edges (`stance_cop_band_*`) |
| Digits and margins | Per foot, `<label>_digit_share` and `<label>_toe_margin_nm` |
| Balance | `xcom_toe_margin_min_m`, `xcom_heel_margin_min_m`, `com_fore_aft_mean_m`, `root_pitch_mean_deg` |
| Sole roll | `max_sole_tilt_deg` and `max_sole_corner_lift_m` against their bars |
| Floor forces | Per foot, `<label>_shear_over_normal` and `<label>_cone_use_mean` |
| Asymmetry | `leg_q_<left>_minus_<right>_<joint>_deg` for hip pitch, knee and ankle. `eval_episodes.csv` records no hip roll difference, so there is none to draw |
| Policy std | `exp(log_std)` per actuator at each checkpoint zip in `models/` (`robust_best`, `best`, final, and at most 12 periodic ones evenly spaced), named in the env's actuator order from the checkout's MJCF |

Each per-episode value is reduced over the evaluation's full-horizon
episodes, as `gate_progress.npz` reduces it: one run's pages draw the mean
with one standard deviation shaded, or with the worst episode dashed for a
maximum or minimum. The counts are over all of the evaluation's episodes.
An evaluation is flagged when at least 5 of its episodes list
`max_actuator_saturation_fraction`, and two flagged evaluations in a row
are an alarm, at the second (the stance4 and stance5 scorers' report-only
rule); every page marks the alarms and the `--mark-steps` checkpoints, and
one run's pages also mark `robust_best` and `best`. From a checkpoint zip
the report reads only `policy.pth`'s `log_std` (`torch.load(...,
weights_only=True)`) and, for a named checkpoint, `num_timesteps` from the
JSON `data` member; it unpickles nothing else, and the reads stop after
10 s. A missing file or column is an absent series, listed in the JSON and
on the page, never an error.

## The handoff's in-training score: `handoff_in_training.json`

A stance certificate scores the handoff checkpoint (`robust_best_model`
first) on its own panel, but nothing beside it said at which in-training
evaluation that checkpoint was saved, or how it scored there. Whenever a
stance gate (any of its kinds) scores the handoff checkpoint, after that
attempt has returned, whatever it returned, the stage writes
`handoff_in_training.json` into its directory and logs one line, for
example `Handoff in training: robust_best_model, saved at
655,360 steps: evaluation mean 3959.1, std 0.0 over 40 episodes (mean -
1.0 x std = 3959.1)`.

| Key | Meaning |
|---|---|
| `selected` | The handoff checkpoint's name |
| `timesteps` | Its own `num_timesteps`, from the zip's `data` member (no torch load) |
| `mean_reward`, `std_reward`, `episodes` | The evaluation at exactly that step in `evaluations.npz` (the standard deviation with ddof 0, as the robust selector computes it) |
| `selection_rule`, `selection_score` | `mean - 1.0 x std` for `robust_best_model` and `mean` for `best_model`, and its value at that evaluation |
| `first_scoring_timesteps`, `step_agrees` | The first evaluation whose score equals the history's best (`progress.json`'s reading) and whether it is the checkpoint's own step |
| `error` | Only when a value could not be found (a truncated zip, no `evaluations.npz`, no evaluation at that step, as for `stage1_final`): the reason, with the values not found `null` |

It is a sidecar, not a key of the certificate: a failure inside the
certificate's own guard fails the stage closed, so nothing new runs there.
It never raises, nothing reads it back, and the certificate's bytes, its
verdict and `gate_verdict.json` are the same with or without it. It sits in
the stage directory before the result bundle is built, so the bundle's
manifest lists it with the other artifacts.

## Onset checkpoints: the archive rule

`CheckpointRetentionCallback` keeps the newest `max_checkpoints` periodic
checkpoints (5 by default) and deletes older ones. By the end of a 12M
stance, nothing more than about 0.5M steps old remains, so the checkpoints
from when a posture set in cannot be replayed. A retention key adds an
archive: with `checkpoint_archive_every_timesteps = A`, a periodic
checkpoint outside the newest `max_checkpoints` is also kept when it is
the newest one present in its interval `(k A, (k + 1) A]`. The kept set
depends on the checkpoint names alone, so a restart or a resume prunes the
same way. `best_model`, `robust_best_model` and `stage1_final` are never
periodic checkpoints and are never pruned, and a zip's VecNormalize pickle
is kept or deleted with it. The key must be a whole number of steps, 0 or
more (0 or absent: no archive). Like `max_checkpoints`, it enters no gate,
task or hyperparameter digest; only the stage's two
`stage_config_view_sha256` lines of the digest snapshot move with it.

The T. rex stance sets 500,000. At the notebook's stance cadence (a
checkpoint every 100,000 steps) a 12M run keeps, at 4.04 MB per pair (a
policy zip and its VecNormalize pickle, measured on the T. rex):

| Retention | Pairs kept at 12M | Drive | Onset resolution |
|---|---|---|---|
| The newest 5 only (before) | 5 | 20.2 MB | none older than 0.5M steps before the end |
| `max_checkpoints = 0` (every one) | 120 | 484.5 MB | 0.1M steps |
| Archive every 1,048,576 steps | 16 | 64.6 MB | about 1.05M steps |
| **Archive every 500,000 steps (the T. rex stance)** | **28** | **113.1 MB** | **0.5M steps** |
| Archive every 250,000 steps | 51 | 205.9 MB | 0.25M steps |

The result bundle's manifest reads each kept pair once when the bundle is
written. At a cadence of 1,048,576 steps the rule keeps every checkpoint,
so there the cadence sets the resolution. `progress.json`'s
`checkpoints.periodic_timesteps` lists the steps kept.

## Which model records what

Every model whose stage is trained by `train()` or the command-line
curriculum records `train_episodes.csv` (its outcome, posture and
limited-force columns), the rollout counts and `reward_action_jerk` in
`diagnostics.npz`, `eval_episodes.csv` (its outcome columns), the reward
terms split by outcome, the mirror-even leg DC, and, on PPO,
`policy_probe.npz`. The rest follows each model's feet:

| Model | Balance and force keys | Span, tip and ankle keys | In-training screen | Touch shares | Probe ankle pairs |
|---|---|---|---|---|---|
| Tyrannosaurus Rex | yes | yes | yes | `pad`, `toe_d2`, `toe_d3`, `toe_d4`, `digits` per foot | 1 |
| Compsognathus Longipes (Anatomical) | yes | no | yes | no | 1 |
| Compsognathus Longipes (Robot) | yes | no | no (`stance_quality/v1`) | no | 1 |
| Dibothrosuchus Elaphros | yes | no | no (`reward_and_length/v1`) | no | 2 |
| Velociraptor Mongoliensis | no (no box pad) | no | no (no recorder) | no | 1 |
| Brachiosaurus Altithorax | no (no box pad) | no | no | no | 2 |

Behaviour stages trained by `train_behaviors.py` build their own callbacks
and get none of these, nor `progress.json`.

## Compatibility and verification

The display-name change kept run directories, replay filename IDs, semantic
stage IDs, recipe labels, W&B resume IDs and filter tags, and checkpoint
loading contracts stable. Decision D-D33 later renamed the species IDs and
added the embodiment to run directories, replay names and W&B IDs, so runs
recorded under the old IDs no longer resolve. The Drive summary formats copies
of its dataframes: raw records and CSV exports retain their species and
embodiment IDs and recorded stage names. Existing
completed bundles are not rewritten; re-rendering historical reports maps
`balance` to Stance without changing their payloads or verdicts. A stage
resumed after this change records the new stage name, `stance`, in its
rewritten `stage_config.json` and in its W&B config's `stage_name`, which
nothing functional reads.

The config-name change deliberately changes twelve config-view golden
hashes: PPO and SAC for each of the six stance nodes. All other 637 lines of
the 649-line digest snapshot remain equal, including plant identities,
policy interfaces, task and gate digests, hyperparameters, behavior recipes,
and reward captures. No physics or policy-interface revision is required.

Regression coverage checks the display names through the
summary writers, notebook displays/exports and W&B resume path; all six
stance replay producers plus the three recovery nodes; snapshot collection
without changing simulation state or fabricating quadruped support; on each
biped, the stance width, foot position, pelvis-over-support and leg
home-error columns at its own foot sites and leg joints; and, on T. rex, that
the shared snapshot keeps every column
`capture_tyrannosaurus_rex_stance_snapshot` records, with its value.

The rendering check serializes and reloads a tiny untrained PPO and matched
VecNormalize pair for each model, with deterministic zero actions and a
six-step horizon. It produces six videos and two CSVs per model, checks
the encoded frame counts and durations with `ffprobe`, and inspects all
eighteen camera views. These short runs verify artifact production and
playback; they are not evidence that a trained policy passes its stance gate.
