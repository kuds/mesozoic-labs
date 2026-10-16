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
criteria. The gate kind remains visible in each report. `stand` remains a
**recipe**, rather than an alias to rename: the anatomical Tyrannosaurus Rex
and both Compsognathus embodiments resolve it through stance and recovery. Their recovery
outputs are labeled **Recovery**.

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
(`r_toe_d2_geom` gives `r_toe_d2`):

| Column | Meaning |
|---|---|
| `<label>_foot_cop_fore_aft` | The step's signed WHOLE-FOOT centre of pressure: every floor contact of the limb (pad, digits, any other leg geom), force-weighted along the pad's forward axis from the ankle axis, in half-lengths of the declared span (`-1` the heel line, `+1` the toe line, beyond `+1` past it on the digits) |
| `<label>_toe_margin_nm`, `<label>_heel_margin_nm` | The floor-load moment left before that CoP reaches the toe line, `F (toe_line - x)`, and the heel line, `F (x - heel_line)`, in N m, with `F` the limb's floor force; negative past the line |
| `<label>_<digit>_load_share` | The step's floor force on that digit over the whole limb's |
| `<label>_<digit>_load_pos` | Where along the digit its floor load sits, force-weighted: 0 at the MTP joint's axis, 1 at the tip. Only on a digit with its own MTP joint (the Tyrannosaurus Rex's; the anatomical Compsognathus's digits share the pad's body) |

The whole-foot cells are empty on a step the limb carries nothing above
0.1 N, so `stance_quality/v2`'s report-only `max_foot_cop_fore_aft` over any
window is again the mean of `|value|` with an empty cell read as 1. The load
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
reads it. The opt-in hop watch reads only its `unsupported_duty`.

| Keys | Meaning |
|---|---|
| `timesteps` | The training step of each evaluation |
| `full_horizon_fraction`, `duty_episodes`, `unsupported_duty`, `unsupported_duty_ucb`, `bilateral_support_duty`, `mean_reward` | The stance panel's gate quantities, as `curriculum/stance_gate.py` reduces them |
| `action_dc_rms`, `action_ac_rms`, `action_delta`, `action_jerk`, `action_freq_hz`; `action_dc_per_actuator`, `action_ac_rms_per_actuator` | The deterministic action's static offset from the home control and its tremor, over the post-settle window, as scalars and as (evaluations, actuators) matrices |
| `term_*` | Each reward term's per-episode mean |
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
  evaluation could not be recorded, and for the signed CoP alone when the pad
  was never loaded (the gate statistic then reads 1).

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

A training run that names no model (`model_id=None`) records neither the pad
keys nor the leg DC keys. TensorBoard carries the finite values of the same
keys as `diagnostics/eval_<key>`.

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
