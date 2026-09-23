# Behavior Recipes Plan — from a stage ladder to a deliverable DAG

**Date**: 2026-09-05. **Baselined on**: `main` @ 640d2ce (the tree after the
six gap-review phases, PRs #514–#519, merged). **Companions**:
`docs/STAGE1_SPLIT_PLAN.md` (rev 5, the stance/recovery split and the
semantic stage manifest), `docs/STAGE1B_IMPLEMENTATION_PLAN.md` (the recovery
stage as a warm-started, post-gated mini-stage — the pattern this plan
generalises), and `docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md` (whose
behavior-gate and seed-multiplicity decisions, CF2/CF3/SS2 and SS1, this plan
absorbs). Statements about current code cite `file:line` in the baseline
tree; they will drift and should be read as anchors, not contracts.
Statements tagged `[probe]` were measured by throwaway scripts during the
2026-09-05 assessment and are re-measured by the tests §8 pins before the
corresponding phase lands.

---

## TL;DR

The maintainer's goal: every species learns a named set of behaviors —
**stand/balance**, **walk**, **hunt/eat**, **follow direction** (move the way
a controller stick says) — each as its own policy, each built by a short
curriculum of mini-stages. The question was whether to move away from the
linear stage curriculum to "recipes" that target behaviors.

The answer is yes, and the tree is closer than the phrasing suggests. Three
of the four behaviors already exist as separately trained, separately gated,
separately warm-started checkpoints with their own TOMLs: stance and recovery
(standing), locomotion (walking), behavior (hunting). What the tree cannot do
is (a) declare which node warm-starts from which — parentage is inferred as
"the previous numbered stage" in four places, (b) publish more than one policy
per run — a failed bite gate suppresses the walking and standing policies
entirely, and (c) name a new behavior — the stage vocabulary is a closed set
of four ids. Follow-direction is the one behavior the environments cannot
express: there is no command input, only a static prey target.

The adopted design is a **DAG on one shared certified trunk**, not four
independent ladders:

- **Manifest v2** adds three optional keys per stage — `warm_start_from`,
  `deliverable` and a `recipe` label — and opens the id vocabulary. Recipes
  are *derived* from the edges (a deliverable plus its ancestor chain), never
  declared in a second table. Legacy numbers, config files, stage ids and
  task fingerprints do not move, so the certified stance evidence and any
  run's frozen recovery resolution stay valid.
- **Publication becomes per deliverable**: a bundle is complete for a
  deliverable iff that node and its ancestors passed; a run publishes every
  certified deliverable it contains; a failed leaf no longer hides the trunk.
- **Every deliverable is gated on a measured capability against a null with
  a confidence bound**, never on episode-mean reward. The hunting gate's
  unattainable 2.0 m/s term goes (speed is the walking deliverable's claim,
  carried by lineage), bite success moves to a Clopper-Pearson lower bound,
  and the hunting stage gets a measured collapse floor. Seed replication
  becomes a provenance field; single-seed deliverables are labelled
  provisional.
- **Follow-direction** is a two-node leaf chain warm-started from the
  certified walker. It needs one policy-interface revision that reserves three body-relative
  command dims (forward speed, lateral speed, yaw rate) at the end of every
  species' observation. That revision invalidates every existing checkpoint
  at three independent layers, so it is done **once, early, for all species
  at the same time**, and the certified stance checkpoint is *widened*
  (zero columns, re-paneled) rather than retrained. The follower starts as
  exactly the walker (zeroed command columns) and learns command dependence.
- **The SB3 notebook** gains a `BEHAVIOR` knob and one chain loop: it
  resolves the deliverable's ancestor chain from the manifest, reuses
  certified ancestors (from this run or an earlier one), trains only the
  missing nodes, and publishes whatever certified.

Sequencing (§5): the manifest edges and per-deliverable publication first
(no env change, no checkpoint invalidation), then the hunting-gate and
seed-provenance fixes, then the one interface bump, then the follow leaf
(pilots landed 2026-09-15 as a separate pipeline; consolidation pending,
§6.2).

---

## 1. Motivation

### 1.1 The proposal

Each species should learn four behaviors, each its own policy for now:
standing/balancing, walking, eating/hunting, and follow-direction ("move in a
certain direction, like a video-game controller would do"). Training should be
organised as *recipes* that target a behavior, with mini stages inside a
recipe for curriculum learning, instead of one linear ladder whose last rung
is the only product.

### 1.2 Why the ladder is the wrong shape for that goal

The ladder makes every stage a stepping stone and only the last a product.
Under the maintainer's goal every behavior is a product. Three consequences of
the ladder shape are load-bearing today:

1. **One published model per run.** Provenance's `selected_model_path` must
   equal the *terminal advancing* stage's checkpoint
   (`environments/shared/result_schema.py:557-566`), the bundle is
   `complete` only when every advancing stage passed
   (`environments/shared/reporting/bundles.py:158-161`), and `summary.json`
   is not written otherwise (`bundles.py:164`). A T-Rex run whose bite gate
   fails publishes neither its walking policy nor its stance policy.
2. **Parentage is inferred from position.** `train_curriculum` carries one
   `load_path` forward through a single loop
   (`environments/shared/train_base.py:1656-1660`), the stage-entry warm-up
   and reward ramp trigger on manifest position > 1
   (`train_base.py:857`), the notebook infers the load mode the same way
   (`notebooks/sb3_training.ipynb` cell 14, lines 239 and 245), and the JAX
   runner carries `params` and `obs_stats` forward through one integer loop
   (`environments/shared/jax_curriculum.py:503, 548-570`). A second leaf off the
   same parent cannot be expressed.
3. **Closed vocabulary.** `KNOWN_STAGE_IDS` is four ids
   (`environments/shared/stage_manifest.py:38`) and an unknown id is fatal in
   the reader and in `stage_label` (`:206`, `:253-254`). Integer references
   1/2/3 are pinned to stance/locomotion/behavior forever (`:43`,
   `:226-237`). "Follow-direction" has no identity anywhere.

### 1.3 Why not four independent ladders

Separate *policies* per behavior is right for now: SB3 has no multi-task
machinery, each behavior wants its own reward tuning and its own gate, and
the 2026-08-23 lineage rule already forbids an uncertified policy from feeding
another (`docs/STAGE1B_IMPLEMENTATION_PLAN.md` §4: feeding an ungated policy
forward "would launder uncertified robustness into the curriculum").
Separate *ladders* would be wrong: the stance stage alone costs 11M steps
(11h 30m per 10M-step seed run; 13h 20m for the one recorded 11M leg) and is
seed-sensitive (2 of 3 seeds certify,
`docs/KNOWN_ISSUES.md:59-97`), so every behavior must branch from one
certified trunk rather than re-roll it.

---

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Node** (mini-stage) | One stage TOML with `[env]`, `[curriculum]`, `[ppo]`, `[sac]`, `[jax]` and a `gate_kind`; one training run with one checkpoint and one gate verdict. Today's "stage". |
| **Edge** (`warm_start_from`) | The declared parent a node initialises from, under `task_load_mode = initialize_next_stage`, with lineage recorded. |
| **Trunk** | The shared certified chain every behavior builds on: stance → (recovery) → locomotion. |
| **Leaf** | A node with no children: today `behavior` (hunt); new `follow_direction_speed`. |
| **Deliverable** | A node whose certified checkpoint is a published policy (`deliverable = true`). |
| **Recipe** | A deliverable plus its ancestor chain, *derived* from the edges; the optional per-entry `recipe` label groups nodes under a behavior name. "walk" = stance → locomotion. "hunt" = stance → locomotion → behavior. "follow" = stance → locomotion → follow_direction → follow_direction_speed. "stand" = stance → recovery. A label resolves to its deepest deliverable in manifest order. |
| **Certified** | A property of one checkpoint: the node's own gate passed *and* every ancestor's gate passed, all evidence hash-bound to the checkpoints it describes. **Replication** (how many distinct-seed runs certified the same task) is a separate count, §4.5. |

The internal id of the hunting node stays `behavior`: the manifest pins
legacy number 3 to that id and refuses a rewrite
(`environments/shared/stage_manifest.py:43, 226-231`), and every run names
its directory `04_behavior`. (For a numbered stage the task fingerprint
records the legacy integer, `environments/shared/task_fingerprint.py:173`, so
the id itself is not what the fingerprint binds.) Its label in the manifest,
catalog and website is "hunt". Species-specific task
names (bite, strike, food reach, snap) come from each stage TOML's `[stage]
name`.

---

## 3. Where the tree stands (verified 2026-09-05)

### 3.1 The manifest is already a recipe system in embryo

`configs/trex/stages.toml` (schema `mesozoic.stage-manifest/v1`) orders four
stages by semantic id — stance/1, recovery/–, locomotion/2, behavior/3 — and
the three other species get a synthesized manifest from their
`stage{1,2,3}_*.toml` filenames (`stage_manifest.py:153-172`). Each entry may
carry exactly `id`, `config`, `legacy_number` (`:202-204`, pinned by
`test_stage_manifest.py:123`). "Advancing" is *defined* as "has a legacy
number" (`:136-150`, pinned by `:326`), and every consumer derives "terminal"
as "the last advancing entry" (`result_schema.py:557`, `bundles.py:314`,
`environments/shared/reporting/summaries.py:166-171`). The split plan sketched
a per-stage `terminal:` field that never shipped
(`docs/STAGE1_SPLIT_PLAN.md:856-866`).

Recovery is the maintainer's "mini stage" already built: its `[env]` mirrors
stance's plus exactly the perturbation block (pinned by
`test_stage_manifest.py:146`), it warm-starts from stance under
`initialize_next_stage` with lineage recorded
(`environments/shared/config.py:334-368`), its gate is judged once,
post-stage, against a frozen record (`environments/shared/reporting/gates.py`
dispatch on `recovery_quality/v1`), and it is non-advancing in the ladder.
The notebook already runs a de-facto fork from stance (stance → recovery and
stance → locomotion both load `path_1`); it works only because recovery is
excluded from the linear machinery.

Everything below the orchestration layer is parent-agnostic today: any
checkpoint can be warm-started into any stage with
`train --stage <ref> --load <ckpt> --load-mode initialize_next_stage`
(`environments/shared/cli.py:219-243`); `select_handoff_checkpoint` reads one
stage directory (`environments/shared/curriculum/checkpoints.py:95`); the
task fingerprint records whichever parent was used
(`task_fingerprint.py:76`, `config.py:334`); gate kinds are declared per
TOML. A DAG can be driven by hand today. The code that must change is the
orchestration and publication layers, not training or gating.

### 3.2 What the environments can express

Every species' observation is `[joint_pos, joint_vel, root_quat(4),
root_gyro(3), root_linvel(3), root_accel(3), foot_contact(2|4),
target_direction(3), target_distance(1), command(3)]` — 64 / 70 / 86 / 80 /
56 / 46 dims for trex / velociraptor / brachiosaurus / dibothrosuchus /
compsognathus / compsognathus_robot since the Phase C interface revision
(`configs/plant_manifest.generated.json`; segments at
`environments/shared/plant_contract/policy_layer.py` `observation_segments`).
The trailing `command(3)` segment is the body-relative `(v_x_cmd, v_y_cmd,
yaw_rate_cmd)` frame of §4.6, pre-scaled to `[-1, 1]` and constant zero
under `command_mode = "none"` — the only mode implemented in Phase C, so
the *only* live goal signal is still a world-frame unit vector from the
pelvis to a mocap target placed once at reset
(`environments/trex/envs/trex_env.py`, `_get_obs`). At the time this
section was written (before Phase C) there was no command conditioning of
any kind: no target heading, no desired speed, no moving goal; the slot now
exists, and Phase D fills it.

"Walking" is `dot(qvel[0:2], initial_target_direction)` with the direction
frozen at reset from the world origin to the target spawn
(`trex_env.py:965-968`, `environments/shared/base_env.py:983-997`);
`heading_weight` rewards facing that same fixed vector. The MJX backend
recomputes the reference live each step (`environments/shared/mjx_env.py:1034-1036`),
a documented divergence (`docs/KNOWN_ISSUES.md`). In the trex, velociraptor
and brachiosaurus locomotion stages the target is 8–12 m ahead within ±2 m
lateral (dibothrosuchus: 6–10 m within ±1 m), and the SB3 reset never
randomises yaw, so the animal always starts facing +X with the target within
about ±14° (±9.5° for dibothrosuchus).

Eating/hunting is a one-shot episodic event with a static target: trex bite
= head geom contacts prey; velociraptor strike = a sickle-claw geom contacts
prey; brachiosaurus food reach = head-tip site within a threshold of an
elevated food body; dibothrosuchus snap = snout geom contacts prey. All four
pay a 1000-point bonus, set `info["success"]`, and terminate the episode
(`trex_env.py:936-947` and the species equivalents).

Adding command dims breaks warm-starting at three independent layers: the
plant contract hashes `_get_obs`, the shared observation builders and
`build_mjx_observation` into every species' `policy_interface_sha256`
(`policy_layer.py:352-366`) and refuses a mismatched checkpoint; SB3's
`PPO.load(env=...)` raises on an observation-space mismatch; and
`VecNormalize.set_venv` asserts equal shapes before `obs_rms` is copied.
`build_bipedal_obs` and `build_mjx_observation` are in every species' hash
(and `build_quadruped_obs` in both quadrupeds'), so an edit to those shared
builders — which the command segment requires — is an all-species interface
revision; an edit confined to one species' `_get_obs` bumps that species
alone.

### 3.3 Gates and publication

Gate kinds are a versioned, fail-closed registry
(`environments/shared/curriculum/gate_schema.py`): `stance_quality/v1`
(full-horizon ≥ 0.95, unsupported-duty one-sided 95% UCB ≤ 0.02, 40-episode
panel on seeds 3042–3081), `recovery_quality/v1` (episode-level success — full horizon *and* every
scheduled push recovered, where "recovered" is re-entering the calibrated
safe set and dwelling there — on an exact Clopper-Pearson LCB, plus a paired
per-seed policy-minus-null bound against a frozen null panel; frozen
2026-08-28 at LCB 0.30 / paired 0.20),
`reward_and_length/v1` (episode-mean reward, length, forward velocity, raw
success mean), `none/v1`. Publication re-derives every verdict from
per-episode evidence files bound to checkpoint and normalization hashes
(`environments/shared/result_bundle/evidence.py`).

The trex hunting stage is gated by `reward_and_length/v1` on a 30-episode
panel: reward ≥ 100, episode-mean forward velocity ≥ 2.0 m/s, bite success
≥ 0.5, three consecutive passes (`configs/trex/behavior.toml:88-107`). The
gap review's findings against it stand: bite-terminated episodes structurally
cannot average 2.0 m/s from a standing start (CF2); the raw success mean at
n=30 blocks a genuinely 50% policy 43% of the time (SS2); the absolute
collapse floor of 100 sits far below the stage's measured do-nothing reward
of 557 ± 142 and certifies nothing (CF3).

The catalog and website hard-code a "stage 3" headline
(`environments/shared/species_catalog.py:533, 549`), one success metric per
backend per species, and integer-keyed stage videos
(`configs/species_manifest.toml:79-96`). The website already tolerates a
semantic-id row (schema v3 shipped with recovery), and the Python catalog
renderer already describes the stance and recovery gates.

Provenance records a scalar `training_seed` plus role-labelled evaluation
seeds; nothing records replication (gap review SS1). The stance/recovery
certification panel block 3042–3081 is not bound to any provenance role.

**Phase B (2026-09-13; #533, #534 and #535).** The paragraphs
above describe the 2026-09-05 baseline. Since Phase B the registry also
holds `task_success/v1` (`curriculum/task_success_gate.py`: the exact
one-sided 95% Clopper-Pearson lower bound on the selected checkpoint's
per-episode `task_success` over `evaluation_selected.csv`, hash-bound to
the handoff pair, at a declared panel size; `min_avg_reward` a collapse
rail only), and the trex hunting stage is gated on it — the velocity term
and the raw success mean are retired from `configs/trex/behavior.toml`
(D1; §4.4 as implemented). `gate_verdict.json` records the gate it was
judged under (`gate` / `gate_sha256`, D-A22) and reuse rule 7 checks it.
Provenance records replication per deliverable (§4.5 as implemented), and
the 3042–3081 panel block is bound as the `certification_panel` seed role,
per evidence file (D-B17).

### 3.4 The SB3 notebook

`notebooks/sb3_training.ipynb` is a hand-threaded ladder: cells 18, 21, 23
and 26 run stance, recovery (behind `RUN_RECOVERY_STAGE`), locomotion and
behavior, each passing `path_N` / `vecnorm_N` to the next by hand. A numbered
stage that fails its gate disconnects the runtime and raises. The final
evaluation cell hard-codes stage 3; the replay cell iterates
`completed_stages`. Cell 29 resumes an interrupted stage inside the same run.
There is no way to reuse a certified checkpoint from an earlier run, so every
"Run all" pays the stance stage again.

### 3.5 Lessons from the record that bind this design

The T-Rex curriculum was reshaped four times in August 2026, each time for a
measured reason (`docs/STAGE1_SPLIT_PLAN.md`,
`docs/STAGE1B_IMPLEMENTATION_PLAN.md`, `docs/KNOWN_ISSUES.md`, the
investigations they cite). The recipe design keeps the machinery each lesson
produced:

- **Gate on a capability against a null, never on return.** The zero-action
  statue is the reward optimum of the undisturbed stance task; every
  species' stage-1 reward gate was cleared by its statue. Hence
  `stance_quality/v1` and the frozen-null recovery gate.
- **Semantic ids, never renumbering.** Renumbering silently changes what
  "stage 2" means in every historical artifact. Recipes are layered on top
  of the existing ids and legacy numbers.
- **Declare the load mode on every launch path.** Two stance runs died at
  the recovery boundary on a fingerprint bug (a cwd-relative model path,
  PR #508); the gap review found (OP1) that cross-stage sweep chaining would
  crash at worker startup on the missing load mode; the 20260821 recovery
  pilot trained its first 500k
  steps on a forward-velocity ramp its own fingerprint denied. Entry shaping
  must key on "has a parent", not on position.
- **Derive each node's schedule from its own budget.** Recovery mirrored
  stance's 7M entropy horizon against a 5M budget and peaked at ~1.6M of 5M;
  the 5M checkpoint was indistinguishable from the 3M one on every panel, and
  the budget was cut to 3M. On the velociraptor, 8M-step stage-2 runs did no
  better than 2M. Warm-started leaves should be short (3–4M).
- **Do not carry forgetting-mitigation folklore.** The stance→recovery
  transfer needed no ramp; the velociraptor stage-2 configs tuned to "match
  stage 1 to prevent forgetting" (high alive and posture weights) collapsed
  catastrophically, while low alive/posture weights were what let the agent
  leave the stand-still optimum (`docs/investigations/TRAINING_REVIEW.md`).
  Forgetting is measured per recipe against the warm-start parent, not
  assumed.
- **Relative collapse floors with a physics-revision pin and an arming
  delay.** Two absolute floors never armed (they sat above the runs' peaks);
  the statue-relative floor then armed on an initialisation spike and killed
  run 20260803_012355 at 14.5% of budget, which is why stance and recovery
  carry `collapse_peak_warmup_timesteps` beside the reference/fraction pair
  (`configs/trex/stance.toml:449-462`).
- **One seed is not a result.** Stance replicates 2 of 3.
- **Fail closed wherever absence can read as a pass**, and pin every gate with
  a test that answers "what code would have to be deleted for it to stop
  being consulted".

---

## 4. Adopted design

### 4.1 Manifest v2

Schema `mesozoic.stage-manifest/v2`. Per `[[stages]]` entry the allowed keys
become `id`, `config`, `legacy_number`, and three new optional keys:

- `warm_start_from = "<id of an EARLIER entry>"` — this node initialises from
  that node's handoff checkpoint (`select_handoff_checkpoint`:
  `robust_best_model`, then `best_model`) under `initialize_next_stage`.
  Absent means root, trained from scratch. "Earlier entry" is enforced by the
  loader, so list order is a valid topological order by construction and
  every existing in-order walker stays correct without a toposort.
- `deliverable = true` — this node's certified checkpoint is a published
  policy. Recipes are derived: a deliverable plus its `warm_start_from`
  chain.
- `recipe = "<label>"` — the behavior name a node belongs to (`stand`,
  `walk`, `hunt`, `follow`), used by the notebook's `BEHAVIOR` knob and the
  catalog. A label resolves to its deepest deliverable in manifest order.
  Optional; a node without one is addressed by its id.

Ids become an open vocabulary matching `^[a-z][a-z0-9_]*$`; the four existing
ids stay reserved and `LEGACY_STAGE_IDS` with its no-rewrite / no-reorder
rules (`stage_manifest.py:43, 226-237`) is unchanged. Reading a v1 manifest,
or a synthesized one, derives `warm_start_from` = the previous advancing entry
and `deliverable` = the last advancing entry only — bit-identical to today's
behaviour. Phase A therefore commits a small v2 `stages.toml` for every
species (§4.8), since stand and walk become deliverables only when declared.

The T-Rex manifest under v2:

```toml
schema = "mesozoic.stage-manifest/v2"

[[stages]]                       # quiet stance
id = "stance"
config = "stance.toml"
legacy_number = 1
recipe = "stand"
deliverable = true               # stance_quality/v1; certified 2/3 seeds

[[stages]]                       # balance under pushes
id = "recovery"
config = "recovery.toml"
warm_start_from = "stance"
recipe = "stand"
deliverable = true               # recovery_quality/v1, judged against a per-run frozen resolution

[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "stance"       # the 2026-08-23 lineage rule, now declared;
recipe = "walk"                  # flips to "recovery" once a recovery run passes
deliverable = true

[[stages]]                       # id stays "behavior"; the label is the recipe
id = "behavior"
config = "behavior.toml"
legacy_number = 3
warm_start_from = "locomotion"
recipe = "hunt"
deliverable = true

[[stages]]                       # heading commands at cruise — lands with Phase D
id = "follow_direction"
config = "follow_direction.toml"
warm_start_from = "locomotion"
recipe = "follow"
deliverable = true               # a usable turn-to-heading policy in its own right

[[stages]]                       # + commanded speed and mid-episode switches
id = "follow_direction_speed"
config = "follow_direction_speed.toml"
warm_start_from = "follow_direction"
recipe = "follow"
deliverable = true               # "follow" resolves here (deepest deliverable)
```

Stage directories keep their `{position:02d}_{id}` names, file labels keep
`stage{N}` for legacy ints and the bare id otherwise, and the task
fingerprint's `stage` field is unchanged, so every existing artifact key,
CSV row, `summary.json` key, the certified stance evidence and any run's
frozen recovery resolution remain valid.

### 4.2 Orchestration: parents by edge, ancestors reused

Three derivations are retargeted from position to the edge:

1. `_stage_entry_shaping_callbacks` triggers on *has a parent* (an edge and
   an `initialize_next_stage` load) instead of `stage_position > 1`
   (`train_base.py:857`); the notebook's inference at cell 14 lines 239 and
   245 follows.
2. `train_curriculum` walks the manifest in order and resolves each node's
   parent from its edge — the parent's stage directory, its handoff
   checkpoint, its VecNormalize sidecar — instead of carrying one
   `load_path` (`train_base.py:1656-1660`). A node whose parent has no
   certified checkpoint in this run is skipped with a warning naming the
   missing ancestor, never trained from scratch silently. The JAX runner's
   chain (`jax_curriculum.py:503, 548-570`) gets the same treatment when it is
   next touched; SB3 remains the evidence backend (§4.9).
3. `train()` refuses an `initialize_next_stage` load whose recorded
   `parent_stage` is not the manifest's declared parent for that node, with
   the existing warn-only valve for unfingerprinted parents kept.

**Reuse of certified ancestors across runs.** A node may be satisfied by an
existing certified checkpoint instead of trained. The rule, applied by the
notebook loop (§4.7) and by a new `--trunk-from <run_dir>` on
`train_curriculum`: the candidate's stage directory must carry a recorded
gate verdict that passed — a new per-node `gate_verdict.json` written by
`generate_stage_artifacts` beside the handoff checkpoint and hash-bound to
it (today the verdict lives only in run-level `collected_results.csv` /
`summary.json`) — its plant identity must validate against the
current plant (the existing `validate_model_plant`), and its recorded
`task_sha256` must equal the fingerprint derived from the current stage
config (the same check `resume_same_stage` applies). On reuse the child run
copies the ancestor's `stage_config.json`, gate record and checkpoint hashes
into `ancestors/<stage_id>/` (small files, never the checkpoint), and the
lineage gains `parent_run_id`, so the bundle audit can verify a cross-run
parent it cannot verify today (`environments/shared/result_bundle/audit.py`
checks parents only inside one bundle).

Two rules adopted while implementing Phase A sharpen this. *The chain is
checked by digest, root-first*: a non-root candidate is reused only when
its recorded `parent_checkpoint_sha256` equals the digest of the checkpoint
resolved for its declared parent in the child run — two runs that both
certified stance produced two different checkpoints, and a walk descends
from exactly one of them — so a child is reusable only once its parent is,
and once a node is trained in the child run none of its descendants are
looked up. *The target is never reused*: the node a run exists to certify
is always trained; an earlier run's certified target is that run's
deliverable, published from there. The notebook loop (§4.7) codes against
the same two rules.

**Curriculum manager.** `CurriculumManager` stays integer-keyed through
Phase A (`environments/shared/curriculum/manager.py:109-149`): semantic-id
nodes are judged post-stage, exactly as recovery is today, and the leaf's
in-training advancement callback is not needed because a leaf has no child.
Re-keying the manager by manifest entry is deferred until a semantic node
needs in-training advancement.

### 4.3 Publication per deliverable

`RESULT_SCHEMA_VERSION` 3 → 4, with v3 still readable. Changes:

- `provenance.deliverables`: a map from deliverable stage key to
  `{model_path, model_hash, normalization_hash, gate_kind, certified,
  replication}`; `selected_model_path` / `model_hash` point at the *primary*
  deliverable — the deliverable the run targeted, or else the deepest
  certified deliverable present — so v3 readers still see one model. The
  v3 rules "a checkpoint for every advancing stage" and "the primary is the
  manifest's last advancing entry" (`result_schema.py:487-492, 557-566`,
  `bundles.py:314`) are replaced by "a checkpoint for every present
  deliverable": under the old rules a walk-only run could not write a bundle
  and a failed-hunt run would headline the uncertified hunt checkpoint.
- Bundle status: `complete` iff *every* deliverable present in the run is
  certified; `partial` iff at least one is; `failed` iff none. `summary.json`
  is written whenever at least one deliverable is certified. Completeness
  per deliverable = its node and every `warm_start_from` ancestor present
  (in the bundle or as an `ancestors/` record) and passed
  (`bundles.py:129-164`, `result_schema.py:479-497, 554-566`).
- Results stay at `results/<species>/<algo>/summary.json`; the two-level
  path (`result_schema.py:222`) is unchanged, and the summary carries the
  deliverables map.
- Catalog: stage rows gain `deliverable` and `warm_start_from`; result rows
  gain a `deliverables` list with a per-behavior headline metric chosen by
  gate kind (stance: duty UCB and full-horizon fraction; recovery: recovery
  LCB; walk: certified velocity; hunt: success LCB; follow: tracking-success
  LCB). `stage3_success_rate` stays for the historical ladder rows.
  `species_manifest.toml` gains per-deliverable success metrics beside the
  existing per-backend ones, and stage videos are keyed by stage id with
  integer aliases. The TypeScript `AdvancementGate` type and `formatGate`
  learn the non-reward kinds the Python renderer already describes.

### 4.4 Gates per deliverable

| Deliverable | Gate | Change from today |
|---|---|---|
| stand — stance | `stance_quality/v1` | none |
| stand — recovery | `recovery_quality/v1` (frozen) | none |
| walk — locomotion | `reward_and_length/v1`, `min_avg_forward_vel = 1.0`, `min_avg_episode_length = 750` | none now; a cruise-window velocity kind is a later refinement (CF2's suggested metric), not a prerequisite |
| hunt — behavior | `task_success/v1` (new) | **delete `min_avg_forward_vel = 2.0`** (`behavior.toml:97`) — this plan's decision, chosen over the gap review's suggested windowed/peak velocity metric or no-prey probe episodes, which move to the walk gate as a later refinement: speed is the walking deliverable's claim, carried by lineage; gate success on `recovery_gate.binomial_lcb` at a declared n; replace the absolute `collapse_peak_floor = 100.0` with the `collapse_peak_floor_reference` / `fraction` pair plus `statue_constants_physics_revision`, measured with `zero_action_baseline.py trex:3`, together with `collapse_peak_warmup_timesteps` (the pair without the arming delay is what killed run 20260803_012355); set the collapse patience keys explicitly, no tighter than locomotion's 20/10/0.5. **Landed 2026-09-13 as the note below records** |
| follow — follow_direction | `command_tracking/v1` (new, §4.6) | new |

`task_success/v1` keys: `min_success_lcb`, `min_eval_episodes`,
`min_avg_reward` (collapse rail only), `min_avg_episode_length` (optional).
Sizing: a one-sided 95% Clopper-Pearson bound ≥ 0.5 needs 20/30 or 26/40
successes; at n=40 a true 70% policy passes 81% of the time, a true 80%
policy 99%, a true 50% policy 4%. The one committed hunting result (29/30,
LCB 0.85) passes comfortably. Thresholds are frozen attainable-not-
aspirational from the first pilot, as recovery's were. Each new kind is
registered in `GATE_KINDS` and `_REQUIRED_THRESHOLD_KEYS`, dispatched in
`reporting/gates.py`, re-derived from per-episode evidence at publication,
listed in the sweep's offline-evaluable set, rendered by the catalog and the
TSX, and pinned by a fail-closed dispatch test.

**As implemented (2026-09-13; #534, WS-B1
+ WS-B2).** `configs/trex/behavior.toml` declares `gate_kind =
"task_success/v1"` with `min_eval_episodes = 30` (D-B1: the panel size
every existing panel uses; the notebook's selected panel and the
provenance `evaluation_episodes` are coupled to it and pinned) and
`min_success_lcb = 0.5` (D-B2) — **provisional**: frozen before any
Phase-B pilot on the strength of the committed 2026-03 row alone, to be
re-frozen attainable-not-aspirational from the first pilot's
`evaluation_selected.csv` (§10). Sizing at the bar: 20/30 → 0.5006 clears
and 19/30 → 0.4669 does not; 26/40 → 0.5081 and 25/40 → 0.4828; the
committed 29/30 → 0.8514 (a recomputation from a rounded mean, not a
re-judgement — that row holds no per-episode evidence and was judged at
`min_success_rate` 0.25 with no velocity term). The collapse floor is the
measured pair: the hunting statue is **602.13 ± 175.35 over 40 episodes**
(seed 3042, 40/40 full horizon, `zero_action_baseline.py trex:3 --episodes
40 --seed 3042`, physics revision 7), so `collapse_peak_floor_reference =
602.0`, `collapse_peak_floor_fraction = 0.45` (floor 270.9) and
`min_avg_reward = 361` = round(0.6 × 602.13) as the rail (D-B4), pinned by
`statue_constants_physics_revision = 7`; `collapse_peak_warmup_timesteps =
1_000_000` is a judgment (D-B5, §10), and the patience keys are explicit
at 20 / 10 / 0.5. `required_consecutive = 3` stays as in-training
hysteresis only (D-B3). The certifying verdict is one post-stage panel:
`generate_stage_artifacts` judges `evaluation_selected.csv` hash-bound to
the handoff pair (rolling a panel on the publication seed when none is
bound), publication re-derives the pass from the same rows bound to the
certified checkpoint, and the sweep judges a row from the trial's recorded
`success_count` / `n_success_episodes` (D-B12). The CLI curriculum's hunt
verdict is the in-training manager's bound over the last EvalCallback
panel (recorded as `success_count` / `n_success_samples` in the verdict's
`stage_result`); it writes no evidence CSV and is not backfillable, so a
CLI-certified hunt is a training-time signal, not a certification. The
JAX path refuses the kind and, per D-B13, refuses a final stage declaring
it before any training — on the declared kind alone. Every other species'
hunt stays on `reward_and_length/v1` (D-B14).

### 4.5 Seed replication as provenance

A run trains one seed, so replication is a property of a *set* of runs,
not of a checkpoint. Provenance keeps its scalar `training_seed` and gains,
per deliverable, a `replication` record: the count of published runs at the
same `task_sha256` and plant identity whose gate passed, with their run ids
and seeds, aggregated by the catalog from the published bundles (a run may
also list known `replicates` explicitly). The 3042–3081 panel block is
declared as a `certification_panel` seed role so the auditor binds it the
way it binds the publication seed. Stage configs gain
`certification_seeds = N` (default 1 = today's behaviour); a deliverable
whose replication is below its config's `N` is labelled **provisional** in
the summary and the catalog, and the catalog's per-species headline names
the count. Enforcement bites only at publication; training and checkpoint
advancement are unchanged, since advancing a checkpoint that itself passed
is valid at n=1. Initial setting: 2 for trex stance, 1 elsewhere. Four
deliverables per species at one seed each is four draws of the seed lottery,
which is why this lands before any deliverable is called more than
provisional (Phase B).

**As implemented (2026-09-13; #535,
WS-B4).** The count is writer-recorded, not catalog-aggregated (D-B10):
`environments/shared/replication.py` discovers a deliverable's replicates
among the run's `LOG_BASE/<species>/<algo>/` siblings when the notebook's
publication cell runs, and `save_result_bundle(replicates=...)` records
them in each deliverable's `replication` record — this run first, then its
replicates, with distinct run ids and seeds. A replicate is a sibling of
the SAME recipe (D-B16, tighter than "same task and plant, gate passed"
above): equal `task_sha256`, plant identity, `gate_sha256` (the verdict's
own record of the gate it was judged under) and `hyperparameters_sha256`,
a passed reusable verdict, and a different training seed — two seeds of
different gates or algorithm blocks are not replication of one
deliverable. A sibling whose run block predates D-A21 has its
hyperparameters digest DERIVED from its recorded blocks
(`config.recorded_hyperparameters_sha256`) and is never skipped for that
alone; a sibling whose verdict predates D-A22 (no `gate_sha256`) is skipped
until re-backfilled, consistent with rule 7 and D-B6. Discovery never
raises on a malformed neighbour (every skip is logged with its reason);
the record is what fails closed (schema and audit). `certification_seeds`
is a `[curriculum]` publication key (D-B9: positive int, default 1, in no
digest; trex stance declares 2 on the seed 42 PASS / 43 FAIL / 44 PASS
record), written into the record beside `provisional = count <
certification_seeds` (D-B11); the catalog validates the record and
re-derives only `provisional` from the CURRENT declaration, never
aggregating across bundles, and renders `N run(s) of M seed(s)` plus
`provisional` identically to the website. A replicate that finishes later
is counted by re-running the counting run's publication cell with the
sibling present: a `partial` bundle is rebuilt, and a verified `complete`
bundle regenerates its derived artifacts when the replication record is
the only change (the writer masks `replication` and `provisional` — not
`certification_seeds` — when comparing the existing and prospective summaries) and is refused as immutable on any other
difference. The `certification_panel` role is a default provenance role
(`PUBLICATION_SEED_START`, 3042, for every SB3 bundle) bound per evidence
(D-B17): stance panel row *i* must carry `panel_seed == role + i`, a
recovery `gate_resolution.json` must register `panel_seed_start == role`,
a declared role must equal the registered block, and a recorded stance
PASS in a bundle without the role is refused at publication. The JAX saver
passes no replicates, so its records count the run alone.

### 4.6 Follow-direction

**Command frame (decision D2).** Three body-relative dims appended at the
end of the observation: `v_x_cmd` (forward speed), `v_y_cmd` (lateral speed),
`yaw_rate_cmd`, each pre-scaled to [-1, 1] by the stage's declared ranges.
Body-relative matches the controller analogy (stick = velocity, triggers =
turn), the sim-to-real plan's objection to anchoring on absolute yaw
(`docs/hardware/SIM_TO_REAL_PLAN.md:119-123, 265`), and the convention in
legged-robot velocity-command work. Walking is the fixed-command special case
of following *as a task*: a follower holding `v_x_cmd` at cruise with the
others zero is asked to walk. It is not the special case *at warm-start*: the
trunk walker is trained with the command channel all-zero, so its zero is a
placeholder it cannot read, not a "stop" command, and the freshly
warm-started follower walks at the walker's natural speed for every command
until the leaf learns the dependence, stop included. "Cruise" is defined as
the parent walker's mean forward velocity measured on its certification
panel (locomotion declares no target speed, only the 1.0 m/s gate floor and
the 2.5 m/s reward cap), recorded in the follow node's config as the fixed
point of `command_speed_range`.

**The one interface revision (Phase C).** Appending the segment at the end
keeps every existing slice offset; dims become 64 / 70 / 86 / 80. Edits:
each species' `_get_obs`, `obs_functions.build_bipedal_obs` /
`build_quadruped_obs` (a `command` argument defaulting to zeros),
`mjx_env.build_mjx_observation` (zeros), `policy_layer.observation_segments`
(a `command` segment of width 3), `configs/plant_versions.toml`
`policy_interface_revision` for all four species (trex 12 → 13,
velociraptor 9 → 10, brachiosaurus 7 → 8, dibothrosuchus 6 → 7), and the
regenerated plant manifest. **The queued height-channel removal
(`base_env.py:1410-1411`) is deliberately *not* batched with this bump**:
removing that reset draw shifts every subsequent seeded draw, which would
turn the stance re-panel below from a near-reproduction into a fresh roll of
a 2-of-3 seed-sensitive certificate. It stays queued for a revision that
already plans a trunk re-certification.

**Widening instead of retraining.** A tool `widen_checkpoint` maps an SB3
checkpoint plus VecNormalize sidecar from revision r to r+1: it pads the
first `Linear` of `policy_net` and `value_net` with zero columns (net_arch
`[512, 256]`, separate first layers, so the padded network computes the same
function on zero-padded observations), pads `obs_rms` mean/var, updates the
saved observation space, pads the Adam moments of those two weight tensors
in the saved optimizer state (or strips the optimizer state so it
re-initialises; the archive carries `policy.optimizer`, which the repo's
warm-start path restores through `alg_cls.load`), and re-stamps the plant
identity **with the parent checkpoint hash recorded as lineage**, never
silently. Pinned by two tests: the padded columns are exactly zero and
actions on zero-padded observations are allclose to the parent's over a
seeded rollout, and the widened checkpoint completes at least one PPO update
under `initialize_next_stage`. The widened stance checkpoint is then
re-paneled (40 episodes, seeds 3042–3081): a real roll whose pass/fail is
recorded; if it fails, the widened checkpoint is not certified and the trunk
is retrained under the new interface with seed replicates. There is no
committed recovery resolution to re-freeze — the resolution is per-run
pre-registration written into that run's stage directory (§4.7) — but
because `task_sha256` carries `policy_interface_sha256`
(`task_fingerprint.py:164`), the next recovery run freezes fresh null panels
under the new task hash; Phase C runs one such freeze from the widened
checkpoint to record how the statue and brace nulls re-roll.

**As implemented (Phase C, 2026-09-13/14; decisions D-C1–D-C17, §6.1).**
The scope is all six plants, not the four above: compsognathus (53 → 56,
policy interface r1 → 2) and compsognathus_robot (43 → 46, r1 → 2) took the
segment beside trex 61 → 64 (r12 → 13), velociraptor 67 → 70 (r9 → 10),
brachiosaurus 83 → 86 (r7 → 8) and dibothrosuchus 77 → 80 (r6 → 7); physics
and visual revisions, the `observation_schema` strings and the Box bounds are
unchanged, and the height-channel removal stayed queued (D3). The
plant-contract probes inject the non-zero `COMMAND_PROBE_VECTOR =
(0.25, -0.5, 0.75)` on both backends so the SB3/MJX parity of invariant 9 is
asserted on the new slot rather than on zeros (the SB3-only compsognathus
pair reports parity `None`). The six `command_*` kwargs landed on
`BaseDinoEnv` / `MJXEnvConfig` with inert defaults, and while the effective
`command_mode` is `"none"` they are carved out of the task-fingerprint env
section for every species (the compsognathus perturbation carve-out's
pattern, amendment A1), so a stage's `task_sha256` moved only through the
plant's `policy_interface_sha256` and the payload carries a `command` section
only when a `command_manifest` is passed (Phase D). `widen_checkpoint`
(`environments/shared/scripts/widen_checkpoint.py`) implements the tool above
with these differences from the wording: the Adam moments are padded, never
stripped (D-C10); SAC is handled too (the critics' observation-action inputs
get the columns INSERTED after the observation block); `num_timesteps` is
kept (D-C12); the identity gate is the parent's recorded identity one
interface-only revision behind the current plant (never hand-edited; a
parent without one needs `--allow-legacy-plant`); the run block records the
eight `config.WIDEN_LINEAGE_KEYS` (`widened_from_path`,
`widened_from_checkpoint_sha256`, `widened_from_normalization_sha256`,
`widened_from_task_sha256`, `widened_from_policy_interface_sha256`,
`widened_from_policy_interface_revision`, `widened_from_run_id`,
`widened_by`) and never the `LOAD_LINEAGE_KEYS` — a widened node is a root
(D-C8) — with the same parent hashes in the archive's
`mesozoic_widen_lineage` attribute; and both artifacts are re-stamped with
the current plant identity AND the stage's CURRENT task fingerprint
(`derive_stage_task_fingerprint` under `stable-baselines3` with the stage's
current `[env]` and the current identity; the parent's `mesozoic_task_lineage`
is kept), which is what lets `validate_declared_parent` and the recovery
harness's `resume_same_stage` check accept the widened pair. The output is
the parent's own handoff name plus byte-identical `<stage_label>_final.*`
copies (D-C9) so the notebook's JUDGE branch fires; no verdict, provenance,
resolution, evaluation or periodic checkpoint is written. The pins are the
two above plus the probe: exact-zero padded columns, actions equal to the
parent's over a seeded 200-step rollout (seed 3042, `allclose(atol=1e-6)`,
the measured delta recorded in `widen_report.json`) both with the command
slice zero and with the probe vector in it, and one PPO update under
`initialize_next_stage` (`test_widen_checkpoint.py`); the tool runs the
first two as self-verification before it returns. The reseed rule is
implemented as `load_vecnorm_stats(reseed_command_slice=True)` on both the
train and eval destinations, with `train_base._load_vecnorm_into_envs`
deriving the flag from the stage's `command_mode` at both call sites and
never on a `resume_same_stage` load (whose parent is the same task and
already trained under the live slice); the widen tool applies the same
values to the appended slice at creation (`pad_running_stats`), and in Phase
C every stage is `"none"`, so the flag is always false. The SB3 notebook
gained a `WIDEN_FROM` knob (a run id or absolute run directory) and one widen
cell after the RESOLVE cell that widens the parent's certified ROOT handoff
into this run's root stage directory in a NEW run id — never by pointing
`RUN_ID` at the old run (D-C13) — and refuses `SEED != ` the parent's
recorded `run.seed` (D-C14); the chain loop then refuses the verdict-less
directory and judges it. The re-panel of the two certified stance parents,
the recovery-freeze re-roll and the C½ walker were the maintainer sessions
owed at that point (WS-C4;
`docs/investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md` is their
template, and KNOWN_ISSUES lists what the bump strands). As of 2026-09-19
the C½ walker exists (`20260914_123816`, a fresh r13 stance, §5), the
seed-42 widen is superseded by it, the recovery freeze is covered by the
recovery node the seed-44 session trains, and only that seed-44 widen +
re-panel of `20260815_205206` was still owed (`docs/NEXT_STEPS.md`
session 1; it ran 2026-09-20 as `20260920_010912` and passed). The two
certified parents themselves are r11 archives (trex
r11 → r12 landed 2026-08-16 after they trained, a fingerprint-only bump
with `observation_dim` 61 on both sides), two revisions behind r13: the tool's
gate admits a bounded revision gap (D-C17 — `max_revision_gap`, default 1,
with every other identity field still checked so only fingerprint-only
bumps can be crossed, and the gap recorded in the report), the notebook
threads its `WIDEN_MAX_REVISION_GAP` knob into the call, and the seed-44
session sets it to 2.

**Normalization of the command slice.** `[probe]` A constant-zero dim
accumulates running variance ≈ 1e-11 over 8M samples; a live command of 1.0
would normalise to ≈ 1e4 and clip at `clip_obs = 10`, and un-reseeded
statistics need on the order of the trunk's own sample count to adapt. So on
every load into a node whose `command_mode != "none"`,
`load_vecnorm_stats` (`checkpoints.py:418`) reseeds the command slice to
mean 0 / variance 1 at the carried count; commands are emitted pre-scaled, so
normalised values stay O(1) from step 0. Pinned by a test.

**Exact transfer.** `[probe]` With the command columns of both first layers
zeroed, walker and follower produce a 0.0 action delta on every observation,
live commands included; random columns also give a 0.0 delta while the
command dims are zero (so an allclose test on zero-padded observations
cannot tell the two apart) and a measurable perturbation once commands are
live. Hence the two pins above: the columns must be *exactly* zero, and the
follow leaf's warm-start asserts a zero action delta on non-zero commands
before its first update.

**Env changes (Phase D).** `[env]` keys `command_mode` (`none` |
`heading` | `heading_and_speed`), `command_speed_range`,
`command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`
and `_jitter`. Constructor kwargs join the task fingerprint automatically
(`task_fingerprint.py:109`); the switch schedule gets its own fingerprint
block like `perturbation`. The per-episode command is drawn in `reset()`
*after* every existing draw and *only when* `command_mode != "none"`, so the
trunk's seeded reset sequence is unchanged; mid-episode switches are applied
in `step()` from a seeded schedule using the push-schedule template
(`base_env.py:241-255`), which keeps policy and null panels paired.
`info` gains `command_v_x`, `command_v_y`, `command_yaw_rate`,
`tracking_error_v`, `tracking_error_yaw`. Reward: a Gaussian tracking term
on body-frame velocity error and one on yaw-rate error (the idiom of
`reward_target_centered_height`), with `forward_vel_weight = 0` so the
hard-wired ramp is skipped and the ramp callback's attribute comes from the
TOML. MJX: `EnvState.command`, the same tracking terms, and **`command_mode
!= "none"` raises on MJX until implemented** — unknown `[env]` keys only warn
there today (`mjx_env.py:136-153`), which would otherwise train a different
task silently.

**Gate `command_tracking/v1`.** Per command *event* (episode start and each
switch): success iff the velocity error is within
`tracking_velocity_tolerance` and the yaw-rate error within
`tracking_yaw_tolerance` inside `tracking_settle_steps` and held for
`tracking_dwell_steps` — the recovery gate's re-enter-and-dwell event
(`environments/shared/curriculum/recovery_gate.py:125-165`) applied to
commands. Required keys: `min_tracking_success_lcb` (Clopper-Pearson over
events), the tolerances, settle and dwell, `min_eval_episodes`, and a
worst-of-eight-heading-bins floor so a front-hemisphere-only policy cannot
pass on the mean. The panel **includes the walk command** (forward at cruise)
and is paired against the **command-blind walker** as a frozen null via
`paired_difference_lcb`, so forgetting of walking is measured, not assumed.
Evidence CSVs gain the command and error columns. Thresholds are frozen from
the first pilot.

**Two short mini-stages.** `follow_direction` trains heading commands at a
fixed cruise speed (3M, warm-started from the certified walker); a second
node `follow_direction_speed` adds commanded speed and mid-episode switches
(3M, warm-started from the first). Each derives its entropy and LR schedule
from its own budget (the recovery lesson).

**Zero-interface pilot (optional, before Phase C).** A
`heading_follow_pilot.toml` that mirrors locomotion's `[env]` and spawns the
existing target on the full circle far away (`prey_distance_range` and
`prey_lateral_range` spanning ±40 m; `_spawn_target_2d` has no positivity
check, `base_env.py:973-974`) turns the walker's fixed initial-direction
reward into direction-following with no observation change. It gives no
speed channel, its heading distribution is square-uniform rather than
angle-uniform, and the target distance leaves the normaliser's learned range,
so it is a measurement, not the deliverable. It runs through the notebook's
manual single-node cell under `none/v1` (verdict recorded, not enforced, as
the recovery cell does today; `none/v1` refuses to pass by design and the
chain loop would stop on it), and what it yields is qualitative —
heading-following videos plus the existing velocity and success columns —
not a tracking-gate baseline, since the command and error evidence columns
only exist after Phase D.

### 4.7 The SB3 notebook

Configuration gains:

```python
BEHAVIOR = "follow"             # a recipe label ("stand" | "walk" | "hunt" | "follow") or a deliverable's stage id
TRUNK_FROM = "auto"             # "auto" (D-A25): the sibling run covering the most of the chain; a run id pins one; "" trains every node here
```

Cells 18, 21, 23 and 26 collapse into one chain loop (decision D5), and
`RUN_RECOVERY_STAGE` goes away — recovery runs when the chosen behavior's
chain includes it. The loop resolves `BEHAVIOR` through the manifest (a
label to its deepest deliverable, an id to itself), walks the deliverable's
ancestor chain in order, and for each node:

1. **Reuse** if a certified checkpoint for the node exists in `RUN_DIR` or
   in `TRUNK_FROM` under the §4.2 rule; record it under `ancestors/` and
   continue. Under `TRUNK_FROM = "auto"` (the default, decision D-A25) the
   resolve cell first selects the trunk run: the run beside this one under
   `LOG_BASE/<species>/<algo>/` whose certified ancestors cover the most of
   the chain root-first, newest on a tie, with every refusal printed.
2. **Train** otherwise, warm-started from the parent's handoff checkpoint
   and sidecar (`task_load_mode = "initialize_next_stage"` when it has a
   parent, `resume_same_stage` never inferred from position), generate the
   node's artifacts, judge its gate through the shared
   `reporting.gates.evaluate_stage_gate`, and, for a frozen-null kind,
   freeze the resolution before training and roll the panel after, as the
   recovery cell does today.
3. **Judge a trained-but-unjudged node** — one whose budget was spent by
   cell 29's resume but whose `gate_verdict.json` is absent: generate its
   artifacts, judge the gate, record the verdict, continue. Today the
   verdict is produced only by the per-stage artifact cells the loop
   absorbs, so without this branch a resumed node could never be reused.
4. **On a gate failure**, stop the chain, write the bundle (which now
   publishes every certified deliverable above the failure), and disconnect
   as today. A leaf failure no longer suppresses the trunk's deliverables.

The loop uses `_stage_entry_shaping_callbacks` unfiltered, so SAC gets the
same stage-entry warm-up the CLI gives it (this closes the open question from
gap-review DU1). One manual single-node cell stays as a debugging escape
hatch, and cell 29 (resume an interrupted node in the same run) is unchanged.
The evaluation and replay cells become deliverable-aware: evaluate
`BEHAVIOR`'s policy, replay every node in its chain. For the follow
deliverable the video cell scripts a command sequence (forward, turn, stop)
and renders it, which is the controller-stick demonstration. The chain loop's structure
is pinned at the AST level like `test_jax_notebook_pins.py` pins the JAX
notebook.

### 4.8 Other species

Velociraptor, brachiosaurus and dibothrosuchus get stand (stance only), walk
and hunt when Phase A commits a small v2 `stages.toml` for each (three
entries, the implicit edges made explicit, `deliverable = true` on all
three); a synthesized manifest alone would yield only hunt as a deliverable
(§4.1). Their stage TOMLs, ids and legacy numbers do not change. Their follow-direction
leaves wait on plant preflight — the velociraptor's single toe site reads
about 55% of true load (`docs/STAGE1B_IMPLEMENTATION_PLAN.md` §5,
`configs/plant_versions.toml` note 8), the brachiosaurus statue has stood
40/40 since the physics-r4 repair (note 7) but its shins are uninstrumented,
so a kneeling pose reads identically to airborne (note 8), and the
dibothrosuchus has had no stance-quality or perturbation preflight at all —
and their stance gates remain reward-cleared by their statues, so "stand" for them is labelled by gate kind
in the catalog rather than claimed as certified stance quality.

### 4.9 What does not change

The four trex stage ids, config files, `[env]`/`[ppo]` blocks, legacy
numbers, manifest positions and directory names; every existing
`task_sha256` outside the Phase C bump; `stance_quality/v1`,
`recovery_quality/v1`, the resolver and `freeze_recovery_gate.py`;
`load_stage_config` / `load_all_stages` / `build_env`; the load modes and
lineage record; `select_handoff_checkpoint`; the CLI `train` path; the
published v2/v3 bundles (rendered as historical ladder results); SB3 as the
evidence backend (the JAX path reads v2 manifests and fails closed on a
command-mode config until Phase D reaches it); the sweep tooling (trunk-only
and integer-keyed until a recipe sweep is needed).

---

## 5. Phases and PRs

| Phase | PR | Content | Exit criteria | Effort |
|---|---|---|---|---|
| **Doc** | this | This plan; docs index and changelog entries | Reviewed; decisions §6 confirmed or vetoed | — |
| **A** | manifest edges + publication + notebook | §4.1 manifest v2 (loader, validators, v1/synthesized compatibility); §4.2 retargets, `gate_verdict.json`, `--trunk-from`; §4.3 schema v4, per-deliverable status, catalog and TSX; §4.7 chain loop | v1 manifests read bit-identically (pinned); v2 manifests committed for all four species; a run with a failed leaf publishes its certified trunk (pinned); notebook AST pins; full suite green | ~1.5 weeks |
| **B** | **landed 2026-09-13** — #533 (gate-configuration digest, reuse rule 7, backfill `--gate`; WS-B3), #534 (the `task_success/v1` hunting gate and `behavior.toml`; WS-B1 + WS-B2), #535 (seed replication; WS-B4); this docs pass is WS-B5 | §4.4 `task_success/v1`, `behavior.toml` edits, measured statue floor; §4.5 provenance fields, `certification_panel` role, provisional labels; decisions D-B1–D-B17 (§6.1) | Met: fail-closed dispatch test for the new kind (`test_gate_dispatch_fail_closed.py`); `load_all_stages("trex")` accepts `behavior.toml` under `task_success/v1` and the gate-schema tests pass; the catalog and the site render `N run(s) of M seed(s); provisional` identically (`test_species_catalog.py::test_website_replication_formatter_mirrors_python`); trex stance read `1 run of 2 seeds; provisional` on the r13 run `20260914_123816` (one certified seed against the two-seed bar) until the seed-44 parent was widened and re-paneled as `20260920_010912` on 2026-09-20 (§7, `docs/NEXT_STEPS.md` session 1), whose bundle records `2 runs of 2 seeds` (2026-09-21; the seed-42 run's own run-level records still read 1); the pre-Phase-C stance bundles are never republished and none is committed (§10, KNOWN_ISSUES) | 2–3 days |
| **C** | **implemented 2026-09-13/14** in three PRs — PR-C1 (the one interface revision on all six species, `command_frame.py`, the command-slice reseed plumbing, the non-zero probes, the restamped compsognathus calibrations; WS-C0 + WS-C1), PR-C2 (`widen_checkpoint`; WS-C2), PR-C3 (the notebook `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs and widen cell, the AST pins, this docs pass; WS-C3); of the maintainer Colab sessions (WS-C4: widen + re-panel both stance parents, one recovery freeze re-rolled, the C½ walker) only the seed-44 widen + re-panel of `20260815_205206` was still owed as of 2026-09-19, with the recovery node trained in the same run (`docs/NEXT_STEPS.md` session 1; template note `investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md`), and it ran 2026-09-20 as `20260920_010912` and passed (recovery 2026-09-21); the seed-42 widen and the C½ walker are superseded by the fresh r13 run `20260914_123816` (the C½ row) | §4.6 command dims across SB3 and MJX, plant revisions, `widen_checkpoint`, command-slice reseed, MJX fail-closed; decisions D-C1–D-C17 (§6.1) | Met: zero-column and action-equality pins (on zero and probe commands) and one PPO update from the widened checkpoint (`test_widen_checkpoint.py`); the regenerated manifests pass `plant_contract --check` with SB3/MJX parity on the non-zero probe; the seeded reset stream is pinned by the golden fixture. Met 2026-09-20/21: the seed-44 widen + re-panel of `20260815_205206` (an r11 archive, two revisions behind r13, so `WIDEN_MAX_REVISION_GAP = 2`, D-C17) with the recovery node trained in the same session, as `20260920_010912`, outcome recorded in the investigation note (§7, §8); the seed-42 parent no longer needs widening (`docs/NEXT_STEPS.md` session 1; KNOWN_ISSUES lists what the bump strands) | ~1 week |
| **C½** | **landed 2026-09-15** as run `20260914_123816` (a maintainer Colab session on the merged Phase C tree, commit 35dd44c; no PR): the certified r13 walker exists | A from-scratch r13 stance (seed 42, 11M steps, 13h13m, final eval 3460.6 ± 19.2, `stance_quality/v1` PASS) followed by locomotion (8M, 8h46m, 1.07 m/s mean forward velocity, 1000-step episodes, mean reward 1940.8, `reward_and_length/v1` PASS judged 2026-09-15): the walker came from a fresh r13 stance, not from the widened r11 stance this row planned (no widen session had run by then) | Met: both nodes carry a passed `gate_verdict.json` whose `task_sha256` (stance 82528a2e…, locomotion 31383192…) and `gate_sha256` equal the digests derived from the current stage TOMLs, verified at 22c1fc8, so `TRUNK_FROM = "auto"` selects this run and reuses BOTH nodes for a hunt session; a stand or walk session never reuses its target (D-A18), so it considers stance alone, and since 2026-09-20 the newer seed-44 run `20260920_010912` wins that tie. Cosmetic gap: the run-level `summary.json`, `provenance.json` and `artifact_manifest.json` were written after the stance and never refreshed after the locomotion verdict (reuse reads per-node files, so nothing breaks; re-run the bundle cell to refresh them). The second stance seed (§7) is `20260920_010912` (seed 44, certified 2026-09-20) | 13 h + 9 h of Colab (measured) |
| **D** | **landed 2026-09-15 as a separate pipeline** (#540 T. rex pilots, #541 all species + certified library); consolidation into the reserved hook pending (`docs/CONSOLIDATION_PLAN_2026_09.md`, PR-9..PR-13; the sequence was released 2026-09-20 in the notebook-first order of D-D13) | Designed: §4.6 env sampler, tracking reward, `command_tracking/v1`, the two mini-stage TOMLs, notebook command video; T-Rex pilot. Shipped: a direction controller, tracking reward, heightfield terrain generator and replay recorder (kept), built beside the canonical machinery: two behavior env classes that refuse the reserved `command_mode` and write `self._command` directly instead of filling `_draw_episode_command` (D-C5), a second trainer (`environments/shared/train_behaviors.py`) with its own recipe dialect, 66 behavior TOMLs (11 templates × 6 species) plus 8 trex pilot TOMLs, a certificate outside `GATE_KINDS`, a checkpoint identity keyed on source-file hashes, a certified library and a notebook mode switch; §6.2 (D-D1..D-D10, G1..G4) folds them back into manifest nodes under `locomotion` | Not met in the canonical machinery: the pilots write no `gate_verdict.json`, are not reusable through `find_certified_ancestor`, and their Drive bundles are evaluation-only (D-D9). The original criteria (pilot from the C½ walker with frozen thresholds recorded; MJX fail-closed raise test; gate consulted-test; live-command SB3/MJX parity a Phase E criterion) transfer to PR-9..PR-13 with G4's first thresholds | ~1.5–2 weeks + one pilot run (as designed); the consolidation PRs are sized individually in the consolidation plan |
| **E** | follow-on | Other species' follow leaves after preflight; cruise-window walk gate; manager re-keying; sweep re-keying | as needed | — |

Two PRs landed 2026-09-16 outside the phase table: #542 flipped the
notebook's `PUBLISH_CERTIFIED` default to `False` (the consolidation plan's
PR-1: library publication required a training-origin stamp that a widened
root handoff lacks, so a `WIDEN_FROM` session would have disconnected the
Colab runtime right after the stance passed its gate), and #543 landed
automatic trunk selection, D-A25 (`ancestors.select_trunk`, the notebook
default `TRUNK_FROM = "auto"`, CLI `curriculum --trunk-from auto`;
canonical chains no longer consult the certified library, and a widen
session selects no trunk).

Each PR follows the established process: adversarial pre-PR review with
lensed finders, three refuters per finding, a completeness audit; ruff, mypy
and the full suite in chunks; the notebook round-trips through `json.dump`
with indent 1.

Session order after Phase C follows the maintainer's priority (2026-09-13,
decision D-C15): **stance → recovery → walking before hunting** — widen and
re-panel the certified stance (both seeds), re-freeze one recovery
resolution from the widened handoff, then the C½ walker; the hunting set-up
comes after a walker is certified under the new interface. As of
2026-09-19 the walker exists (the C½ row) and the seed-42 widen is
superseded by it; what remained of that order was the seed-44 widen +
re-panel with the recovery node trained in the same session
(`docs/NEXT_STEPS.md` session 1; it ran 2026-09-20/21 as
`20260920_010912` and passed), and hunting stays after it.

---

## 6. Decisions and assumptions

Decisions taken 2026-09-05 (the maintainer approved the program with these
as recommendations; each may be vetoed at review of this document):

| # | Decision | Veto changes |
|---|---|---|
| D1 | The gap review's held decisions are settled here: CF2 by dropping the 2.0 m/s hunting term outright (this plan's choice; the review itself suggested a windowed/peak metric or probe episodes), SS2 (success LCB), CF3 (measured collapse floor with its arming delay), SS1 (replication as provenance, default 1, provisional labelling) | Phase B scope |
| D2 | Body-relative command frame (`v_x`, `v_y`, `yaw_rate`), three dims, pre-scaled | Phase C interface; cannot change after the bump without another revision |
| D3 | The interface bump is done early, once, for all species, **without** the height-channel removal; the certified stance checkpoint is widened and re-paneled, and one recovery freeze is re-rolled from it | Phase C ordering and which checkpoints survive |
| D4 | "Stand" has two certifiable nodes: stance publishes as quiet stance, recovery as balance under pushes once a run passes | Manifest `deliverable` flags; catalog rows |
| D5 | The notebook's four stage cells are replaced by the chain loop, with one manual single-node escape hatch | Notebook shape |

Assumptions (state an objection and the plan changes):

- A1 The hunting node keeps id `behavior` (recipe label "hunt"); the new
  nodes are `follow_direction` and `follow_direction_speed` (recipe
  "follow").
- A2 Locomotion's parent stays `stance` until a recovery checkpoint passes
  the frozen gate; the flip is a manifest edit.
- A3 A speed command dim is reserved even though the first follow node
  trains at fixed cruise speed.
- A4 The follow leaf is T-Rex only until other species pass plant preflight.
- A5 Results stay at `results/<species>/<algo>/` with deliverables inside.
- A6 Sweeps stay trunk-only and integer-keyed for now.
- A7 SB3 is the evidence backend; MJX fails closed on command-mode configs.
- A8 `certification_seeds` defaults to 1; n=1 deliverables are labelled
  provisional; trex stance declares 2.
- A9 The notebook loop applies stage-entry warm-up to SAC as the CLI does.
- A10 Ancestor reuse copies records, never checkpoints, into the child run.

### 6.1 Decisions taken during Phases A, B and C (the D-A, D-B and D-C series)

Taken while implementing Phase A (2026-09-06 to 2026-09-12, D-A1–D-A24;
D-A25 landed 2026-09-16 in #543), Phase B (2026-09-13, D-B1–D-B17) and
Phase C (2026-09-13/14, D-C1–D-C17);
the pull requests, CHANGELOG and code comments cite them by number. Each
D-B and D-C row states the decision as taken and as implemented; where the
implementation deviated from the design's wording, the row says what the
code does (the D-C rows fold in the amendments adopted at the Phase C
critique, numbered A1–A15 in that record — distinct from the §6 assumptions
A1–A10 above — which override the original wording where they conflict).

| # | Decision |
|---|---|
| D-A1 | `save_result_bundle(target_deliverable=...)` (default the manifest's last deliverable; the notebook passes `BEHAVIOR`'s node). `complete` = target present and certified and every present deliverable certified; `partial` = at least one certified deliverable; `failed` = none. A stance-only run targeting hunt is partial, never complete, so the bundle stays mutable. |
| D-A2 | `validate_result_summary` / `validate_result_bundle` gain `require_publishable` (at least one certified deliverable); `require_complete` keeps meaning `complete`. The audit, the bundle writer and the catalog use `require_publishable`; the notebook completion cell keeps `require_complete`, its failure branch `require_publishable`. |
| D-A3 | A root node under `initialize_next_stage` accepts a parent whose recorded stage is the node itself, refuses any other fingerprinted parent, and warns on an unfingerprinted one. |
| D-A4 | Two v2-only manifest validators: a numbered reserved id must declare its legacy number; a v2 manifest with no deliverable is fatal. |
| D-A5 | `gate_verdict.json` is written by `generate_stage_artifacts` (post-stage, evidence-backed) and by `train_curriculum`'s in-training manager verdict, with `judged_by` recorded; reuse accepts any well-formed passed verdict. `train()` and the JAX saver do not write it in Phase A. |
| D-A6 | `scripts/backfill_gate_verdict.py` re-derives a verdict for a pre-Phase-A stage directory from its evidence through `evaluate_stage_gate`; it refuses when evidence is missing. |
| D-A7 | Reused nodes write no `curriculum_results.csv` row; `train_curriculum` stops (does not skip past) a node whose declared ancestor has no certified checkpoint; non-advancing nodes are still skipped by the CLI curriculum in Phase A. |
| D-A8 | Catalog `schema_version` 3 → 4; `species_manifest.toml` `schema_version` 1 → 2. |
| D-A9 | Stance and recovery deliverable headlines render the metric name with a null value in Phase A; exporting per-stage gate metrics into the summary is Phase B (deferred to a later phase by D-B15; the hunt's `selected_model_success_lcb` is the exception). |
| D-A10 | The README's generated SPECIES table gains Recipe and Warm-start-from columns; the generated RESULTS block and the website's published run summaries stay byte-identical (golden regression). |
| D-A11 | The notebook's `BEHAVIOR` defaults to `"hunt"`; the recovery gate is enforced by the chain under `BEHAVIOR="stand"`; the manual single-node cell never swallows a `ResultBundleError` silently. |
| D-A12 | Species-free readers (`detect_stage_from_path`, the sweep collector) stay reserved-id only; species-aware readers accept any declared id. |
| D-A13 | The website `RawStage` / index-page defect fix ships inside the catalog workstream. |
| D-A14 | CSV deliverable columns, the training summary's deliverables block and `species_registry.describe_stages` are out of Phase A. |
| D-A15 | `duration_seconds` is recorded into `stage_config.json`'s run block on every `train_stage` exit so a resumed-then-judged node reports its real duration. |
| D-A16 | `target_deliverable` / `primary_deliverable` are finalization fields written by `save_result_bundle`, not identity fields. |
| D-A17 | Reuse is chain-aware by digest (§4.2 rule 4): a non-root candidate's recorded `parent_checkpoint_sha256` must equal the digest resolved for its declared parent; a root candidate must not have entered from a parent. Reuse is root-first and a child of a node trained in the same run is never looked up. |
| D-A18 | The target node is never reused across runs; an earlier run's certified target is that run's deliverable, published from there. |
| D-A19 | A retrain-from knob (`curriculum --retrain-from <stage_id>`, the notebook's `RETRAIN_FROM`) reuses certified ancestors strictly above the named node and trains it and every descendant; it generalises D-A18. Landed with the notebook loop (#530). |
| D-A20 | Training refuses to write into a stage directory that already holds a verdict or stage config unless the load is an explicit same-stage resume, so a fresh run directory per variant is enforced. Landed with the notebook loop (#530). |
| D-A21 | Every trained node records a `hyperparameters_sha256` (its algorithm block and stage-entry shaping keys) and an optional label, propagated to `provenance.deliverables` and the W&B run; at reuse time the loop compares the current digest against the ancestor's copied config and warns, naming the differing keys, when an edit is being ignored. The task fingerprint and the reuse rule are unchanged. Landed with the notebook loop (#530). |
| D-A22 | `gate_verdict.json` records the gate configuration it was judged under (kind, schema version, every threshold, and a `gate_sha256` over them) and reuse gains rule 7: a candidate judged under a different gate configuration is refused, naming the differing thresholds. Landed in #533 (the payload is D-B7's threshold projection, not literally every key). |
| D-A23 | Trunks compose: reuse rule 1 prefers the candidate's stage directory and otherwise follows its `ancestors/<stage_id>/ancestor.json` to the `source_run_dir` it names (as recorded, else the sibling of the same name beside the candidate; else refused naming both paths), applying every rule at the source and requiring the source's handoff pair to hash to the record's `handoff` digests. At most 8 records are followed, a cycle is refused, and refusals on the followed path are prefixed with the hop taken. The result describes the source (`via` lists the followed runs; logged, never persisted), so a child's `parent_run_id` names the run that certified the node. Following is opt-in (`follow_records=True`): `train_curriculum --trunk-from` passes it (the trunk is another run by construction); the notebook loop does not, because it tries its own `RUN_DIR` first and reads `same_run` off the candidate — a followed record there would present the trunk's node as this run's own on a re-run — so the loop passes `follow_records=candidate is not RUN_DIR`: `TRUNK_FROM` composes, this run's own directory never follows (pinned). `ancestor.json`'s schema is unchanged; `source_run_dir` / `source_stage_dir` are written as absolute paths so a record made from a relative `--trunk-from` follows from any working directory. |
| D-A24 | `curriculum --target BEHAVIOR` (a recipe label, a deliverable's stage id or a legacy number; `train_curriculum(target=)`) walks the target's chain (`chain_for`) and stops at the target, so a walk-only certified run exists on the command line as through the notebook's `BEHAVIOR`. A label or id resolves as the notebook's `BEHAVIOR` does; a legacy number as `--stage` does. Every node of an explicit target's chain must be advancing and the chain a prefix of the advancing ladder — a chain through a non-advancing node (`stand` on T-Rex / Compsognathus) or one that skips a ladder node (an edge rewired past it, which the loader accepts) is refused before any directory is written, naming the notebook, since the integer-keyed manager would judge the node after the gap against the skipped stage's thresholds — and `--retrain-from` must name a chain node. The default (the last advancing stage) walks the whole advancing ladder unchanged. The manager stays integer-keyed over the full ladder and is never advanced past the target. |
| D-A25 | Automatic trunk selection (landed 2026-09-16 in #543). `TRUNK_FROM = "auto"` (the notebook default) and `curriculum --trunk-from auto` call `ancestors.select_trunk`: every run beside the new one under `LOG_BASE/<species>/<algo>/`, newest first by directory name, is tried against the target's ancestors root-first (above `RETRAIN_FROM`; the target is never reused, D-A18) under the §4.2 rule with `follow_records=True`, each child chained onto the ancestor found for its parent; the run covering the most consecutive nodes wins, the newest on a tie — one coherent trunk, never one node from one run and its child from another (rule 4 refuses that anyway). The selection copies and writes nothing; the loop then reuses from the selected `TRUNK_DIR` exactly as from a hand-picked run, and the chain loop re-derives `TRUNK_DIR` from that selection (a storage-cell rerun resets nothing it relies on; a widen session uses no trunk). A run whose `provenance.json` or root `stage_config.json` names another species, algorithm or backend is refused before the rules run (the identity check a pinned trunk makes; the rules never read the algorithm), and no unreadable neighbour aborts the scan. It prints the choice, the replication each covered node rests on (informative: a provisional ancestor is reused and labelled, never refused — §4.5 makes replication provenance, not a gate), the refused runs with the rule that refused them (the first 20; the selection holds every run scanned), and any run whose root passed under an older policy interface that the widen gate can bridge (a `WIDEN_FROM` candidate, D-C13/D-C17). A `WIDEN_FROM` session selects nothing: its widened root is judged in the new run and no earlier checkpoint descends from it. The certified library's recommendation (PRs #540/#541) no longer feeds canonical chains: it read a separate store none of the certified runs had been published into, refused a single-seed stance as provisional under trex's two-seed publication bar, and its publish step disconnected the runtime on a widened root. |
| D-B1 | The declared hunting panel is n = 30 (`min_eval_episodes = 30`), matching every existing panel; the coupled knobs (the notebook's selected panel, the provenance `evaluation_episodes`, `train_base`'s velocity episodes, the Ray Tune worker) are untouched and `test_trex_behavior_gates_on_task_success` pins the coupling. n was not sized up, so the bar separates ≥ 0.67 from 0.5, not 0.45 from 0.60. |
| D-B2 | `min_success_lcb = 0.5` (20/30 needed, LCB 0.5006; 19/30 → 0.4669 fails). PROVISIONAL: frozen before any Phase-B pilot on the committed 2026-03 row alone (29/30 → 0.8514, a recomputation from a rounded mean with no per-episode evidence), to be re-frozen from the first pilot's `evaluation_selected.csv`; the TOML comment, §4.4, KNOWN_ISSUES and the catalog say so. A verdict minted under the provisional bar is re-judged, never retrained, when the bar moves (D-B7/D-B8). |
| D-B3 | `required_consecutive` is an ALLOWED, not required, key of `task_success/v1`: scheduler hysteresis for the in-training manager only, in the digest like every advancing kind's copy of it. The certifying verdict is one post-stage panel judged once; re-running a deterministic panel is not replication. |
| D-B4 | The collapse rail is `min_avg_reward = round(0.6 × statue reference) = 361` from the measured hunting statue (602.13 ± 175.35, n = 40, seed 3042, 40/40 full horizon, physics revision 7 — the stance rail's statue ratio), pinned by `statue_constants_physics_revision = 7`; the test asserts rail < reference. Re-derived WITH `collapse_peak_floor_reference` whenever a reward weight or the plant moves; a moved rail is a re-judge of every certified trunk, never a retrain. |
| D-B5 | `collapse_peak_warmup_timesteps = 1_000_000`: a judgment (the stance/recovery value, bounded below by the 600k stage-entry window and above by the ≤ 0.5 × budget test pin), not a replay — no behavior-stage evaluation series under the current config exists. Re-measured from the first full hunting run's `evaluations.npz` (KNOWN_ISSUES follow-up; §10). |
| D-B6 | A verdict without `gate_sha256` (judged before D-A22) is REFUSED by reuse rule 7 until re-judged — through the notebook's JUDGE branch (`generate_stage_artifacts`) or `backfill_gate_verdict.py --force [--gate current]`; recovery verdicts need a notebook re-roll; never a retrain. The refusal names both paths, and a refused trunk is trained in the new run. Every pre-D-A22 verdict on the log tree (the certified stance run `20260810_145546` and the seed-44 replicate `20260815_205206` included) is on the KNOWN_ISSUES inventory. |
| D-B7 | The gate digest payload is thresholds only: `gate_config_view` projects the kind, the schema version and `GATE_KINDS[kind] ∩ declared` (every advancing kind's set carries `min_eval_episodes` / `required_consecutive`; `none/v1` is empty; `_ALL_THRESHOLD_KEYS` for a null or unregistered kind), numerics float-normalised (`100` and `100.0` are one gate); schedule, collapse, diagnostic, retention and publication keys and any `[curriculum.jax]` table are excluded; a recovery verdict hashes the declared block. `min_avg_reward` STAYS in the digest, so a moved rail (a statue re-measure) is a re-judge of every certified trunk under the new gate. |
| D-B8 | Judge-time WARNING adopted; rule 7b NOT adopted. `generate_stage_artifacts` logs a warning naming the `gate_config_differences` when the block it judges under is not the one the directory's `stage_config.json` recorded, and never refuses. The stricter 7b variant — refusing reuse when the candidate's own recorded block does not digest to the verdict's `gate_sha256` — was rejected because under D-A22 re-judging a directory under an edited gate IS the intended path (edit a threshold, re-judge, never retrain): 7b would refuse every legitimately re-judged directory and make retraining the only escape. Only the gate the verdict was judged under counts. Replacement checks: a verdict must agree with itself — when both `gate` and `gate_sha256` are present, `gate_sha256 == gate_config_sha256(gate)`, enforced in `read_gate_verdict` and hence by rule 7, `load_ancestor_records` and the audit — and the warning names the differing thresholds. `backfill_gate_verdict.py --gate recorded\|current` judges under the recorded block (the default) or the checkout's, digests whichever it used, and refuses a stance report scored under other thresholds under either. |
| D-B9 | `certification_seeds` is a `[curriculum]` key in `gate_schema`'s new `_PUBLICATION_KEYS` class (positive int, default 1; `declared_certification_seeds`), never in any digest — `gate_sha256`, `hyperparameters_sha256` or `task_sha256` — so raising it relabels published bundles and invalidates no verdict; trex stance declares 2, every other stage the default. |
| D-B10 | Replicates are writer-recorded: `replication.discover_replicates_for_run` finds them among the run's `LOG_BASE/<species>/<algo>/` siblings when the publication cell runs, `save_result_bundle(replicates=...)` records them, and the catalog validates the record and re-derives only `provisional` from the current config — never aggregating across bundles. The `certification_panel` role and its collision question are settled by D-B17. |
| D-B11 | `provisional` iff `count < certification_seeds`; count and N are always rendered — `N run(s) of M seed(s)`, plus `provisional`, and `headlineFor` appends ` (provisional, N of M seeds)` — pinned identically in `species_catalog._format_replication` and the site's `formatReplication`. |
| D-B12 | `task_success/v1` is offline-evaluable in the sweep from the trial's recorded `success_count` / `n_success_episodes` (new `metrics.json` keys and CSV columns beside `success_lcb_threshold`), railing `min_avg_reward` on the same panel's `selected_mean_reward`. One shared trainer panel (`train_base.run_success_panel`) writes the trial's `evaluation_selected.csv` hash-bound to the handoff pair and records the count only beside it, so a trial never carries a FAILED task_success verdict without evidence and the row verdict and the on-disk verdict agree by construction. |
| D-B13 | The JAX preflight (`jax_curriculum`) refuses a FINAL stage declaring a registered, non-none kind no JAX path can judge — `task_success/v1`; a final stage declaring `recovery_quality/v1` (the single-stage pilot's designed shape; `_FINAL_STAGE_PILOT_KINDS`, with no condition on chain length) keeps its allowance — before any training, on the declared `gate_kind` alone: the final stage's block is still not schema-validated there, so single-stage pilots train as before. |
| D-B14 | Every other species' hunt, compsognathus and compsognathus_robot included, keeps `reward_and_length/v1`; every reader keeps both paths. The committed-config test loops cover every species with a stage manifest, and the compsognathus pair's collapse-floor omission passes those loops by the species-level decision their stance TOMLs record ("Deliberately omit a collapse floor"), not by a per-stage statement (§10). |
| D-B15 | Exporting stance and recovery gate statistics into summary stage rows (D-A9's "Phase B" pointer) is DEFERRED to a later phase (Phase E); every pointer — `species.ts`, the catalog docstring, `species_manifest.toml`, recipes.md — defers it to a later phase citing D-B15. The hunt is the exception: `task_success/v1` exports `selected_model_success_lcb` with the count and panel size. |
| D-B16 | A replicate is a sibling run of the SAME recipe: equal `task_sha256`, plant identity, `gate_sha256` (the verdict's) and `hyperparameters_sha256`, a passed reusable verdict and a distinct training seed — tightening §4.5's "same task and plant, gate passed" (two seeds of different recipes are not replication of one deliverable). A sibling with no recorded `hyperparameters_sha256` (pre-D-A21) gets it DERIVED from its recorded blocks (`config.recorded_hyperparameters_sha256`), never skipped for that alone; one whose verdict lacks `gate_sha256` (pre-D-A22) IS skipped until re-backfilled (D-B6). Implemented reading of a complete bundle: a re-run of the publication cell that changes nothing but the replication fields (`replication`, `provisional`) of a VERIFIED-COMPLETE bundle regenerates its derived artifacts — the writer masks those two fields when comparing the existing and prospective summaries (`_without_replication` in `reporting/bundles.py`) — and any other difference, a changed `certification_seeds` included, is still refused as immutable; a raised bar reaches published bundles through the catalog's re-derivation of `provisional` from the current declaration (D-B9/D-B10), not through republication; a late replicate is therefore counted by re-running the counting run's publication cell with the sibling present. The audit reads an absent curriculum block as the default bar 1. |
| D-B17 | `certification_panel` is a default seed role: `initialize_result_bundle`'s default seed roles (`result_bundle/provenance.py`) carry `certification_panel = PUBLICATION_SEED_START` (3042) for every SB3 bundle, and a declared role must equal that block (refused otherwise). It is bound PER EVIDENCE, not through `evaluation_protocols` (the name deliberately lacks "evaluation"; the protocol family is untouched): stance panel row *i* must carry `panel_seed == role + i`, and a recovery `gate_resolution.json`'s `decision_procedure.panel_seed_start` must equal the role. A recorded stance PASS in a bundle whose provenance lacks the role is refused at publication (fail closed); pre-Phase-B stance bundles need republishing (none are committed; KNOWN_ISSUES). `seed_role_collisions` is unchanged — it checks training and selection roles only, so the role may equal the publication seed. |
| D-C1 | All six plants take the 3-dim command segment now, appended LAST: velociraptor r9 → 10 (67 → 70), trex r12 → 13 (61 → 64), brachiosaurus r7 → 8 (83 → 86), dibothrosuchus r6 → 7 (77 → 80), compsognathus r1 → 2 (53 → 56), compsognathus_robot r1 → 2 (43 → 46); physics and visual revisions, the `observation_schema` strings and the Box bounds unchanged (`plant_versions.toml` note 12). |
| D-C2 | The two compsognathus recovery calibrations are RESTAMPED (plant identity + task hash + a `restamp_history` entry) by `scripts/restamp_recovery_calibration.py`, not re-measured: fixed-command nulls never read the observation and the reset draw stream is pinned by the golden fixture (`tests/fixtures/phase_c_reset_golden.json`). The `profile_sha256` moves, so every pre-Phase-C compsognathus recovery freeze must be re-frozen (RECOVERY_CALIBRATION.md, CHANGELOG, note 12; amendment A10). |
| D-C3 | The six command kwargs (`command_mode`, `command_speed_range`, `command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`, `command_switch_jitter`) land on `BaseDinoEnv` / `MJXEnvConfig` now with inert defaults. As amended (A1): while the effective `command_mode == "none"` the six keys are carved out of the task-fingerprint env section for EVERY species (the compsognathus perturbation carve-out's pattern), so off-configs keep their pre-Phase-C env encoding, a stage's `task_sha256` moves only through the plant's `policy_interface_sha256`, and `test_compsognathus_task_compatibility` stays green. |
| D-C4 | A non-zero plant-contract probe: `COMMAND_PROBE_VECTOR = (0.25, -0.5, 0.75)` is injected on both backends (`env._command` in the SB3 probe, `command=` in the MJX probe) so SB3/MJX parity is non-vacuous on the new slot; SB3-only species keep `backend_observation_equal = None`. |
| D-C5 | The per-episode command comes from a `BaseDinoEnv._draw_episode_command()` hook called once in `reset()` after the push block and before `_get_obs()`; it draws nothing under `"none"`, and Phase D replaces the hook body, never `reset()`. (As found 2026-09-17: the #540/#541 pilots bypass this hook — their env classes refuse `command_mode != "none"` on `BaseDinoEnv` and assign `self._command` directly after `super().reset()`; decision D-D1 folds them back through it, PR-9.) |
| D-C6 | No Box bounds change (stays `(-inf, inf)` float32) and no `observation_schema` rename. |
| D-C7 | SB3 also refuses `command_mode != "none"` in Phase C (`validate_command_mode` raises the "reserved for … Phase D" text in `BaseDinoEnv.__init__`); MJX refuses every live mode in `canonicalize_env_kwargs` and `MJXDinoEnv.__init__` until an MJX command path lands (A7); an unknown mode is refused on every backend naming the valid set. |
| D-C8 | Widened checkpoints record the eight `config.WIDEN_LINEAGE_KEYS` in the run block (never the `LOAD_LINEAGE_KEYS` — a widened node is a root; `ancestors._check_chain` refuses a root that entered from a parent and the audit binds `parent_run_id` to an `ancestors/` record) and re-stamp the current plant identity and the stage's CURRENT task fingerprint on BOTH artifacts, with the parent hashes also in the archive's `mesozoic_widen_lineage` attribute. |
| D-C9 | Widen output = the parent's own handoff name (`robust_best_model` or `best_model`, exactly one) plus byte-identical `<stage_label>_final.*` copies, so the notebook JUDGE branch fires on the widened directory; never a verdict, provenance, resolution, evaluation, metrics or periodic checkpoint. |
| D-C10 | Adam `exp_avg` / `exp_avg_sq` (and `max_exp_avg_sq`) moments are padded identically to their weights, matched by parameter order with shape assertions; a missing optimizer member, a stale moment, or a non-Adam moment still shaped like the pre-pad weight is refused rather than carried (unpadded moments load but crash the first update). |
| D-C11 | SUPERSEDED by A1/A9: the task payload carries a `command` section only when a command manifest is passed (`command_manifest: Mapping \| None = None` on `compute_task_fingerprint` / `derive_stage_task_fingerprint`; both backends expose `command_manifest() -> None`); `"command"` is still added to the differing-sections tuple and the v1 valve list; the schema string stays `mesozoic.task-fingerprint/v2`. |
| D-C12 | `num_timesteps` is preserved on widen (recorded in `widen_report.json`). |
| D-C13 | Notebook knob `WIDEN_FROM` (configuration cell; a run id under `LOG_BASE/<species>/<algo>/` or an absolute run directory) plus ONE widen cell after the RESOLVE cell, which widens the parent's certified ROOT handoff into `RUN_DIR`'s root stage directory (refusing an occupied target, this run's own directory, a parent without `provenance.json`, and a parent of another species / algorithm / backend) and trains nothing; the chain loop then refuses the verdict-less directory and JUDGES its `<stage_label>_final.*` pair. Never via `RUN_ID` pointing at the old run — for a pre-Phase-C run the storage cell's `initialize_result_bundle` refuses the directory first (the recorded `plant_identity` is not this checkout's), and for a same-plant run the chain loop raises "mint a fresh RUN_ID" on a refused verdict. The cell imports every name it uses (it runs before the infra cell; A14a), refuses a parent stage without a `stage_config.json` or a recorded run seed before the tool runs, and its `SEED` refusal names the remedy (correct `SEED`, restart the runtime so a fresh `RUN_ID` is minted — the storage cell will not re-mint the directory under another seed). |
| D-C14 | Both certified stance parents (`20260810_145546` seed 42, `20260815_205206` seed 44) are widened and re-paneled in two sessions so trex stance reads `2 runs of 2 seeds` instead of `1 run of 2 seeds; provisional` (replicates need distinct `run.seed` values, which the copied run blocks provide, and the same `task_sha256`, which only widened siblings share); each session sets `SEED` to its parent's seed — the widen cell refuses `SEED != ` the parent's recorded `run.seed`, because the minted provenance publishes `training_seed = SEED` (A14b). **As found at implementation (2026-09-14):** both parents are policy-interface r11 archives — trex r11 → r12 (commit `8795e28`, 2026-08-16, fingerprint-only: `observation_dim` 61 on both sides) landed after they trained, and an archive's identity is stamped at training time — and `widen_checkpoint`'s default bound is one revision, so it refuses them at r13 (`policy_interface_revision: parent=11, current=13 (gap 2 exceeds max_revision_gap=1; pass --max-revision-gap 2 / max_revision_gap=2 …)`; `--allow-legacy-plant` covers only an archive with no identity). Remedy, D-C17: the gate takes a bounded gap — both sessions set `WIDEN_MAX_REVISION_GAP = 2` (the CLI's `--max-revision-gap 2`), under which every other gate field is still checked and the report records `revision_gap` 2; the r12 stance PASSes of 2026-08-16..18 widen under the default bound but are not the certified parents (KNOWN_ISSUES, the Phase C entry). **Status 2026-09-19:** seed 42 is certified at r13 by the fresh run `20260914_123816` (stance and locomotion PASS, selected by `TRUNK_FROM = "auto"`), which supersedes the Session 1 widen of `20260810_145546`; only the seed-44 widen of `20260815_205206` (`WIDEN_MAX_REVISION_GAP = 2`, `SEED = 44`) is still owed for the two-seed bar. Seed 43 (`20260915_160239`, a fresh r13 stance) failed the duty rail (mean 0.0323, UCB 0.0350 against 0.02). **Status 2026-09-21:** the seed-44 widen ran 2026-09-20 as `20260920_010912`; the re-panel reproduced the r11 certificate to every printed digit, its bundle records trex stance at replication 2 (with `20260914_123816`), and the two-seed bar is met. |
| D-C15 | Session order follows the maintainer's priority: stance (widen + re-panel) → recovery freeze re-roll → walk (C½); hunting later. |
| D-C16 | `validate_mjx_environment_plant` gains an observation-width check against the plant identity (`_deterministic_probe_data` imported lazily inside it, A8). |
| D-C17 | The widen gate takes a bounded revision gap (adopted 2026-09-14, from the PR-C3 review): `widen_checkpoint(..., max_revision_gap: int = 1)`, the CLI's `--max-revision-gap N` and the notebook's `WIDEN_MAX_REVISION_GAP` (threaded into the widen cell's call) admit a parent `1 <= current - parent.policy_interface_revision <= N` revisions behind; the default stays 1 (fail closed: the Phase C bump alone). Every other field is checked whatever the bound — same species, `physics_sha256`, `nq` / `nv` / `nu`, `action_dim`, `observation_dim + COMMAND_WIDTH == current` — so only fingerprint-only intermediate bumps can be crossed; a parent further behind than N is refused with both revisions, the measured gap, the bound and the flag named. `widen_report.json` records `revision_gap` / `max_revision_gap`, the archive's `mesozoic_widen_lineage` carries `revision_gap`, and the run block's `widened_from_policy_interface_revision` names the parent's revision. Opting in asserts, from `plant_versions.toml`'s numbered notes, that the intermediate bumps changed nothing the widening cannot bridge — for trex r11 → r13, note 11 recorded the perturbation engine and note 12 appended the command segment. The two r11 stance parents widen under `WIDEN_MAX_REVISION_GAP = 2` (WS-C4 Sessions 1 and the seed-44 repeat); a hand re-stamp is never the path. |

### 6.2 Decisions taken at the 2026-09-17 consolidation review (the D-D and G series)

Taken by the maintainer on 2026-09-17 at the review of the direction/terrain
pilots (#540/#541) against this design; the review's working labels D1..D12
map one-to-one onto D-D1..D-D12 (D-D13 was added on 2026-09-20 when the hold
lifted), and nothing in §6 or §6.1 is renumbered.
The sequence of consolidation PRs each row unblocks — fifteen in all, with
sizes, per-PR file lists and acceptance tests — is
`docs/CONSOLIDATION_PLAN_2026_09.md` (§3 the sequence, §6 and §7 the same
decisions with the questions they answered). Of that sequence PR-1 (#542)
and the auto-trunk PR (#543) landed 2026-09-16, PR-2 (#544) is the
2026-09-19 documentation pass that records these rows, and **PR-3..PR-15
were released on 2026-09-20 in the notebook-first order of D-D13** (PR-3,
bounding the SB3 CI job, landed first as #546, and PR-4..PR-6 as
#547..#549 the same day; the maintainer paused the sequence after PR-6
and lifted the pause on 2026-09-23; the notebook-only PR-12 slice is
next). Training runs in parallel
(G3): the walker sessions in `docs/NEXT_STEPS.md` use the current notebook.
D-D11 and D-D12 were recommended on 2026-09-17 and confirmed by the
maintainer on 2026-09-20, when the hold lifted (D-D13) and the widen path's
future was fixed (D-D14).

| # | Decision |
|---|---|
| D-D1 | Direction-following and terrain traversal are ordinary manifest nodes under `locomotion` (this plan's Phase D shape: stage TOMLs, registered gate kinds, reuse through `find_certified_ancestor`), not a standalone pipeline. Consequence accepted: a follow/terrain node needs a certified locomotion ancestor at the current policy interface (r13 for trex, each other species at its own revision) — trex has one (`20260914_123816`); the other five species get theirs from the walker sessions (G3); the manual single-node cell is the escape hatch until then. Unblocks PR-9 to PR-15. |
| D-D2 | One `command_config: DirectionCommandConfig \| None` kwarg replaces the five numeric reserved kwargs (`command_speed_range`, `command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`, `command_switch_jitter` — none has a reader anywhere, and the pilots' dataclass is already validated and JSON-able); the task-fingerprint carve-out is extended so no canonical `task_sha256` moves while `command_mode` is `"none"`. Amends D-C3. Unblocks PR-9. |
| D-D3 | Command-slice normalisation follows invariant 8 (§8: reseed to mean 0 / variance 1, statistics keep updating); the pilots' `BehaviorVecNormalize` passthrough subclass (a pickled class in every sidecar) is deleted; #540/#541-trained policies saw different inputs and are not continuations. Unblocks PR-8 and PR-10. |
| D-D4 | Automatic parent selection is kept and is `ancestors.select_trunk` (D-A25, landed as #543 before PR-4/PR-5 delete the library trio, so the capability never lapses); the certified library is deleted outright and no recommendation pointer survives. Unblocks PR-4 and PR-5. |
| D-D5 | Four base stage TOMLs per species (`follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain`, and the existing `locomotion` as the `extends` parent) with a ~20-line `extends` key in `load_stage_config`; the terrain templates become a `terrain_families` list inside `difficult_terrain`; single-template runs are documented `[env]` overrides, not files. Refined by G1: three new files per species, `follow_direction_speed` folded into `follow_direction`. Unblocks PR-11 and PR-12. |
| D-D6 | Honest gate name first: the pilots run `none/v1` (recorded, not enforced); then the certificate's per-episode statistic is registered as `terrain_command/v1` with one shared threshold block (first values from G4). This plan's `command_tracking/v1` (§4.6: per-event settle/dwell, the worst-of-heading-bins floor, the paired null against the command-blind walker) is a later second kind — new work outside the sequence; no implementation of it exists anywhere. Unblocks PR-13. |
| D-D7 | Notebook depth: only the `train_stage` wrapper over `train_base.train` now (PR-14; the §4.7 AST pins on the chain loop stay); whether to move the chain loop / widen / resume cells into a package module is decided after PR-14 has settled. Fixes PR-14's scope. |
| D-D8 | No interim behaviors notebook; the `COMMAND_TERRAIN_BEHAVIOR` mode switch (19 code cells at 22c1fc8) is tolerated until PR-12 deletes it — #542 already removed the dangerous default. Unblocks PR-12. |
| D-D9 | The #540/#541 behavior bundles on Drive are evaluation-only; none is carried forward as a training parent (their checkpoint identity hashes source files, so any edit to `behavior_env.py` strands exact resume anyway). Unblocks PR-6, PR-7 and PR-9, and is PR-12's acceptance basis. |
| D-D10 | Terrain stays an opt-in env subclass; no r14 interface bump to move the model swap into `reset()` (the reset source is fingerprinted; the queued height-channel removal stays queued, D3). Unblocks PR-7 and PR-9. |
| D-D11 | Confirmed 2026-09-20: CLI runs record stage duration and seed model construction like the notebook does (PR-14 aligns `train()` with the notebook's `alg_kwargs["seed"]` line and its duration recording). PR-14. |
| D-D12 | Confirmed 2026-09-20: the dead `lateral_speed_scale` field (it always divides a zero) is dropped when the TOMLs are rewritten. PR-11 / PR-12. |
| D-D14 | Taken 2026-09-20: the widen path stays for the two pending parents (NEXT_STEPS.md sessions 1 and 2: the trex seed-44 r11 stance and the compsognathus seed-42 r1 stance) and is then demoted to CLI-only — the notebook refactor (PR-14, or a PR right after it once both sessions are decided) deletes the widen cell and the `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs, while `widen_checkpoint` stays a command-line tool for the next interface bump. Amends the "knobs kept" list of the consolidation plan §4: those two knobs leave once sessions 1 and 2 are decided. The 2026-09-19 kernel deaths were the cross-interpreter archive fault (KNOWN_ISSUES), not the widening; the alternative (deleting the path now) would cost about a day of Colab retraining for the two parents. Both sessions decided PASS by 2026-09-21 (`20260920_010912`, `20260921_203149`: each re-panel reproduced its parent's report to every printed digit); the knobs leave with PR-14. |
| D-D13 | Taken 2026-09-20 (the maintainer released the consolidation hold): the sequence lands notebook-first — PR-3, PR-4, PR-5, PR-6, then a notebook-only slice of PR-12 (the `COMMAND_TERRAIN_BEHAVIOR` mode switch, the ten `BEHAVIOR_*` knobs, the direction/terrain cells and their guard sites, and `behavior_notebook.py` with its tests and pins; `train_behaviors.py` stays a CLI-only path) pulled ahead of PR-11, then PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13 and PR-15. Amends D-D8: the mode switch is tolerated only until that slice, and between the slice and PR-11 the direction/terrain pilots have no notebook path (they are evaluation-only under D-D9). |
| G1 | Chain shape `walk → follow_direction` (the full command set on flat ground) `→ follow_direction_difficult_terrain` (the target deliverable: every species follows a direction on difficult terrain); a commands-free `difficult_terrain` node stays as an optional diagnostic sibling; the recipe label `follow` resolves to the deepest deliverable; `follow_direction_speed` is folded into `follow_direction` (this supersedes the `follow_direction_speed` node in §4.1's manifest sketch and assumption A1); three new stage files per species; the final node is SB3-only (MJX fails closed on live commands and has no terrain). Shapes PR-11's node set. |
| G2 | Command set = heading, speed (half to full cruise), stops and restarts, switching every few seconds — the pilots' combined recipe: `follow_direction` carries the full set from the start (`command_config` with `speed_range = [0.5, 1.0]` of the cruise speed) and `follow_direction_difficult_terrain` inherits it. PR-11. |
| G3 | Walker sessions start now on the current notebook, trex first (its r13 chain `20260914_123816` is selected automatically under `TRUNK_FROM = "auto"`), the other five species one at a time, in parallel with the consolidation PRs; the sessions are `docs/NEXT_STEPS.md` §3, and every consolidation PR keeps the chain loop, `TRUNK_FROM`, `WIDEN_FROM`, `RETRAIN_FROM` and the resume cell working so those sessions run on whatever `main` is at the time. Changes nothing in the PR order. |
| G4 | The first `terrain_command/v1` gate adopts the pilots' certificate thresholds (20 episodes per terrain family, 20 s minimum horizon, survival LCB 0.80, success LCB 0.60, tracking and settle fractions 0.60; today `configs/behavior_certification.toml`), carried once into PR-13's `[curriculum]` block — never 66 copies — and tightened after the first certified species; a tightening is a gate-digest change (rule 7), so a node certified under the first values is re-judged, not silently reused. PR-13. |

---

## 7. Risks

- **Phase A leaves two vocabularies** ("advancing" for the legacy trio,
  "deliverable" for publication) until the manager and sweeps are re-keyed.
  Mitigation: semantic leaves are judged post-stage like recovery; the
  overlap is documented in `stage_manifest.py` and pinned.
- **The interface bump invalidates every checkpoint for warm-start.**
  Mitigation: D3 — once, early, with the widening tool and recorded
  lineage; the re-panel is budgeted as a real roll with a retrain fallback.
- **The certified walker exists, and the stance two-seed bar is met
  (2026-09-21).** The 20260821 locomotion leg was interrupted at 5.49M of
  8M with 0/109 gate passes at 0.55 m/s
  (`docs/reviews/TREX_REVIEW_2026_08_MERGES_AND_NEXT_STEPS.md`); the
  September run `20260914_123816` (a fresh r13 stance, seed 42, then
  locomotion 8M at 1.07 m/s over 1000-step episodes, `reward_and_length/v1`
  PASS) closed Phase C½ and is what `TRUNK_FROM = "auto"` selects, so
  the follow leaf can branch. The bar: trex stance declares
  `certification_seeds = 2` and seed 42 certifies at r13 — seed 43
  (`20260915_160239`) passed reward and full horizon but failed the
  unsupported-duty rail (mean 0.0323, UCB 0.0350 against 0.02), and seed
  44 certified on 2026-09-20 when the r11 parent `20260815_205206` was
  widened and re-paneled as `20260920_010912` (`WIDEN_MAX_REVISION_GAP = 2`,
  `SEED = 44`), whose bundle records trex stance as `2 runs of 2 seeds`.
  A failed panel is a measured deficit — re-rolling it does not help,
  another seed does — so the fallback was a fresh r13 stance at
  `SEED = 45` (about 13 h), not needed.
- **LCB gates change the meaning of thresholds** (20/30, not 15/30, for
  0.5). Mitigation: freeze thresholds attainable-not-aspirational from the
  first pilot, as recovery did.
- **Seed lottery multiplies** with four deliverables per species.
  Mitigation: §4.5 lands before any deliverable is called certified; the
  catalog says "provisional" out loud.
- **The publication layer is the largest change**; if it lags, leaves
  accumulate as unpublishable pilots. Mitigation: Phase A ships it before any
  new leaf trains.
- **MJX only warns on unknown `[env]` keys** (`mjx_env.py:136-153`).
  Mitigation: `command_mode != "none"` raises on MJX until implemented.
- **About fifteen tests pin the derivations being replaced** (advancing trio,
  carried load path, position-keyed shaping, terminal selection, notebook
  cells). Each is re-pinned to the new invariant, never deleted.
- **Naming overclaims**: "hunt" certifies reach-and-bite of a static target
  spawned 2–6 m ahead; pursuit of moving prey (ROADMAP Stage 4) is a future
  hunting mini-stage and is not implied by the label.

---

## 8. Invariants to pin

1. A v1 or synthesized manifest under the v2 reader yields the same entries,
   the same advancing trio, derived edges equal to "previous advancing
   entry", and one deliverable equal to the last advancing entry; the
   committed per-species v2 files are the only place stand and walk become
   deliverables.
2. `warm_start_from` must name an earlier entry; self and forward references
   are fatal; legacy rewrite and reorder rules unchanged.
3. Entry shaping fires iff the node has an edge and the load is
   `initialize_next_stage`; never on a same-stage resume; never on a root.
4. `train()` refuses an `initialize_next_stage` load whose recorded parent
   stage differs from the declared edge.
5. A run whose leaf gate fails writes a bundle whose certified trunk
   deliverables are published, whose status is `partial`, and whose
   `selected_model_path` is a certified deliverable, never the failed leaf;
   a walk-only run writes a valid bundle.
6. Ancestor reuse refuses a candidate whose gate did not pass, whose plant
   identity mismatches, or whose recorded task hash differs from the current
   stage config, or whose verdict's recorded gate-configuration digest
   differs from the current stage config's (rule 7, D-A22), or which lacks
   one (D-B6); a directory re-judged under the current gate is reusable
   although it trained under another (D-B8; `test_ancestors.py`).
7. `widen_checkpoint`: the padded columns are exactly zero, actions on
   zero-padded observations are allclose to the parent's over a seeded
   rollout, one PPO update completes from the widened checkpoint, and
   lineage records the parent hash.
8. Normalised command values are O(1) from the first step after a
   command-mode load.
9. `command_mode != "none"` raises on MJX until the reward path exists; the
   plant contract's SB3/MJX observation parity covers the widened (zero)
   command segment from Phase C on.
10. For every new gate kind: the fail-closed dispatch test and the
    "what code would have to be deleted for it to stop being consulted"
    test (`test_gate_dispatch_fail_closed.py`; for `task_success/v1`,
    `TestTaskSuccessGateIsConsulted`).
11. The follow panel contains the walk command and is paired against the
    command-blind walker; the evidence CSV carries the command columns.
12. The notebook chain loop: AST pins for the `BEHAVIOR` knob, the reuse
    rule, the no-position-inference of load mode, and the escape hatch.

---

## 9. Relationship to the gap review and the roadmap

- **CF2 / SS2 / CF3** (behavior-stage gate): resolved by §4.4. CF2 is
  resolved by *removing* the velocity term from the hunting gate rather than
  by the review's suggested windowed/peak metric — the recipe framing makes
  the term's misplacement structural rather than a tuning question — and
  the windowed metric becomes the walk gate's later refinement. Landed
  2026-09-13 in #534 (WS-B1 + WS-B2);
  the review carries dated Status lines under each finding.
- **SS1** (seed multiplicity): resolved by §4.5. Landed 2026-09-13 in the
  seed-replication PR #535 (WS-B4); the review carries its
  Status line. Trex stance published at 1 of 2 seeds — provisional — until
  the seed-44 parent `20260815_205206` was widened and re-paneled under r13
  (the r11 verdicts cannot be re-backfilled into reuse; `docs/KNOWN_ISSUES.md`,
  the Phase C entry) as `20260920_010912` on 2026-09-20; its bundle records
  trex stance at 2 runs of 2 seeds (2026-09-21).
- **DU1's open question** (SAC warm-up in the notebook): closed by §4.7.
- **ROADMAP "Turning and steering"** (`docs/ROADMAP.md:253`) is the
  follow-direction leaf; **"Hierarchical RL architecture"** (`:320`) and
  **"Stage 4: Prey Pursuit"** (`:259`) are future leaves and mini-stages on
  the same DAG, not part of this plan.
- **`docs/STAGE1_SPLIT_PLAN.md` §4's `terminal:` field** ships as
  `deliverable`.

---

## 10. Open questions to measure, not decide

- Does the widened stance checkpoint reproduce its 40-episode panel under
  the new interface? (Expected yes up to float rounding — the widen tool
  already pins a 0.0 max |padded column| and an action delta ≤ 1e-6 over a
  seeded rollout; the panel itself is verified by the WS-C4 re-panel of both
  stance parents, owed: `investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md`.)
  Answered 2026-09-20/21: yes, to every printed digit, for both parents that
  were widened — trex seed 44 (`20260920_010912`, r11 → r13: 3408.3 ± 88.5,
  full-horizon 1.0000, duty 0.0069 / UCB 0.0117, the r11 certificate's
  numbers) and compsognathus seed 42 (`20260921_203149`, r1 → r2: 2801.6 ±
  51.2, duty 0.0131 / UCB 0.0141); the seed-42 trex re-panel was not needed,
  since seed 42 certified from scratch at r13 in `20260914_123816`.
- Which direction/terrain thresholds survive contact with a certified
  species? G4 adopts the pilots' certificate thresholds (20 episodes per
  terrain family, 20 s minimum horizon, survival LCB 0.80, success LCB
  0.60, tracking and settle fractions 0.60) for the first
  `terrain_command/v1` gate and tightens after the first certified species;
  a tightening moves the gate digest (rule 7), so nodes certified under the
  first values are re-judged, never silently reused.
- What tracking tolerances and settle/dwell windows does the walker-derived
  follower reach at 3M steps? (Frozen from the Phase D pilot.)
- Does the command-blind walker's paired null show forgetting of straight
  walking in the follower? (The Phase D panel answers it.)
- Is a cruise-window velocity gate for walking worth its new evidence
  columns, or does the 1.0 m/s episode-mean gate with the 750-step length
  floor suffice? (Decide after the first certified walker under Phase C.)
- Where does the hunting bar settle? `min_success_lcb = 0.5` is
  provisional (D-B2): re-freeze it attainable-not-aspirational from the
  first Phase-B pilot's `evaluation_selected.csv` (20/30 clears at 0.5006,
  19/30 does not), then update the TOML comment and §4.4 and re-judge —
  never retrain — any verdict minted under the provisional bar.
- Is 1.0M the right arming delay for the hunting collapse detector?
  `collapse_peak_warmup_timesteps` is a judgment (D-B5): replay the first
  full hunting run's `evaluations.npz` through
  `EvalCollapseEarlyStopCallback` and re-derive it, and re-measure the
  statue (`zero_action_baseline.py trex:3 --episodes 40 --seed 3042`)
  whenever a behavior reward weight or the plant moves, re-deriving the
  361 rail and the 602.0 reference together.
- Should the compsognathus pair's locomotion and behavior TOMLs state
  their collapse-floor omission per stage? Today the committed-config
  loops exempt them on the species-level decision their stance TOMLs
  record (D-B14); a maintainer decision to add the relative pair or the
  unarmed statement to those stages would let the exemption narrow to
  per-stage statements.

**Phase B items owed, deferred by the maintainer on 2026-09-13** (recorded
here with the Phase C docs pass; the priority is stance → recovery → walking
before the hunting set-up, D-C15). To revisit later, in the maintainer's own
words: re-freeze the provisional `min_success_lcb` bar from the first hunt
pilot (D-B2, the bullet above); the replay warm-up delay re-measure
(`collapse_peak_warmup_timesteps`, D-B5, the bullet above); the
compsognathus collapse-floor statement (D-B14, the bullet above); and the
maintainer log-tree actions — re-backfill verdicts with `--force`, republish
stance bundles with the panel role. The log-tree actions changed shape under
Phase C: for every pre-Phase-C run the re-backfill and the republish are
dead (KNOWN_ISSUES), and their replacement is widening both stance parents
into new runs and re-paneling them there (WS-C4 Session 1 and the seed-44
repeat — as of 2026-09-19 only the seed-44 repeat was still needed, seed 42
having certified from scratch at r13 in `20260914_123816`; it ran 2026-09-20
as `20260920_010912` and passed), whose bundles
carry the `certification_panel` role from the start;
the pre-Phase-C directories stay on the log tree as history. Both certified
parents are r11 archives, two revisions behind r13, so those sessions set
`WIDEN_MAX_REVISION_GAP = 2` (D-C17: the gate's default bound of 1 refuses
them; under 2 it still checks every other identity field, and the crossed
r11 → r12 bump was fingerprint-only — KNOWN_ISSUES, the Phase C entry).
