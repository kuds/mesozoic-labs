# Training artifact names and stance replays

Human-facing training output uses the full species labels from
`configs/species_manifest.toml`, through `species_display_name()`. Stage
headings use `stage_display_name()`: the historical `balance` name is shown
as **Stance**, and the six current stance configs record `name = "stance"`.
This applies to CLI logs, text summaries, gate and probe reports, plot
legends, notebook tables, the public catalog, and W&B run titles.

| Stable species ID | Display name | Stance gate |
|---|---|---|
| `velociraptor` | Velociraptor Mongoliensis | `stance_quality/v2` |
| `trex` | Tyrannosaurus Rex | `stance_quality/v2` |
| `brachiosaurus` | Brachiosaurus Altithorax | `reward_and_length/v1` |
| `dibothrosuchus` | Dibothrosuchus Elaphros | `reward_and_length/v1` |
| `compsognathus` | Compsognathus Longipes | `stance_quality/v2` |
| `compsognathus_robot` | Compsognathus Longipes (Robot) | `stance_quality/v1` |

Consistent presentation does not imply identical anatomy or certification
criteria. The gate kind remains visible in each report. `stand` remains a
**recipe**, rather than an alias to rename: Tyrannosaurus Rex and both
Compsognathus variants resolve it through stance and recovery. Their recovery
outputs are labeled **Recovery**.

## Shared replay layout

Stance and recovery stages use the same replay outputs for every species.
When both checkpoint pairs are present and rendering dependencies are
available, each stage writes eight files under `replays/`:

| Checkpoint role | Files |
|---|---|
| Selected | `<species>_<algorithm>_<stage>_selected.mp4`, `_selected_side.mp4`, `_selected_front.mp4`, `_selected_stance.csv` |
| Final | `<species>_<algorithm>_<stage>_final.mp4`, `_final_side.mp4`, `_final_front.mp4`, `_final_stance.csv` |

`<species>` is the stable ID and `<stage>` is the existing artifact label,
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

Playback uses `fps = 1 / env.dt`. The four 0.01 s species produce 100 fps
videos; both Compsognathus variants use 0.02 s steps and 50 fps. A recording
lasting 10 s in simulation lasts 10 s on playback. Historical videos used
50 fps universally, making 0.01 s recordings play at half speed.

Each stance CSV has one row per frame, with `step`, elapsed `time_s`,
`reward`, `terminated`, `truncated`, and common `root_*` position/orientation
columns. It also preserves measured scalar environment info. Instrumented
bipeds retain the historical joint, foot, head, and tail diagnostics where
their models expose them. Unavailable measurements stay absent. In
particular, the two front-foot contacts of a quadruped are not converted
into a fabricated whole-animal support verdict.

## Compatibility and verification

Run directories, replay filename IDs, semantic stage IDs, recipe labels,
W&B resume IDs and filter tags, and checkpoint loading contracts remain
stable. The Drive summary formats copies of its dataframes: raw records
and CSV exports retain their species IDs and recorded stage names. Existing
completed bundles are not rewritten; re-rendering historical reports maps
`balance` to Stance without changing their payloads or verdicts.

The config-name change deliberately changes twelve config-view golden
hashes: PPO and SAC for each of the six stance nodes. All other 637 lines of
the 649-line digest snapshot remain equal, including plant identities,
policy interfaces, task and gate digests, hyperparameters, behavior recipes,
and reward captures. No physics or policy-interface revision is required.

Regression coverage checks full names and historical aliases through the
summary writers, notebook displays/exports and W&B resume path; all six
stance replay producers plus the three recovery nodes; and snapshot collection
without changing simulation state or fabricating quadruped support.

The rendering check serializes and reloads a tiny untrained PPO and matched
VecNormalize pair for each species, with deterministic zero actions and a
six-step horizon. It produces six videos and two CSVs per species, checks
the encoded frame counts and durations with `ffprobe`, and inspects all
eighteen camera views. These short runs verify artifact production and
playback; they are not evidence that a trained policy passes its stance gate.
