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
them. Quadruped CSVs have no support, foot-position, or leg-home columns.

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
