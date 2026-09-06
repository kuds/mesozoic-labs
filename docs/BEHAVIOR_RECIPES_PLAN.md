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
seed-provenance fixes, then the one interface bump, then the follow leaf.

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
target_direction(3), target_distance(1)]` — 61 / 67 / 83 / 77 dims for
trex / velociraptor / brachiosaurus / dibothrosuchus
(`configs/plant_manifest.generated.json`; segments at
`environments/shared/plant_contract/policy_layer.py:325-334`). The *only*
goal signal is a world-frame unit vector from the pelvis to a mocap target
placed once at reset (`environments/trex/envs/trex_env.py:550-571`). There is
no command conditioning of any kind: no target heading, no desired speed, no
moving goal.

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
| hunt — behavior | `task_success/v1` (new) | **delete `min_avg_forward_vel = 2.0`** (`behavior.toml:97`) — this plan's decision, chosen over the gap review's suggested windowed/peak velocity metric or no-prey probe episodes, which move to the walk gate as a later refinement: speed is the walking deliverable's claim, carried by lineage; gate success on `recovery_gate.binomial_lcb` at a declared n; replace the absolute `collapse_peak_floor = 100.0` with the `collapse_peak_floor_reference` / `fraction` pair plus `statue_constants_physics_revision`, measured with `zero_action_baseline.py trex:3`, together with `collapse_peak_warmup_timesteps` (the pair without the arming delay is what killed run 20260803_012355); set the collapse patience keys explicitly, no tighter than locomotion's 20/10/0.5 |
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
TRUNK_FROM = ""                 # optional earlier run whose certified ancestors to reuse
```

Cells 18, 21, 23 and 26 collapse into one chain loop (decision D5), and
`RUN_RECOVERY_STAGE` goes away — recovery runs when the chosen behavior's
chain includes it. The loop resolves `BEHAVIOR` through the manifest (a
label to its deepest deliverable, an id to itself), walks the deliverable's
ancestor chain in order, and for each node:

1. **Reuse** if a certified checkpoint for the node exists in `RUN_DIR` or
   in `TRUNK_FROM` under the §4.2 rule; record it under `ancestors/` and
   continue.
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
| **B** | hunting gate + seeds | §4.4 `task_success/v1`, `behavior.toml` edits, measured statue floor; §4.5 provenance fields, `certification_panel` role, provisional labels | Fail-closed dispatch test for the new kind; `load_all_stages("trex")` accepts `behavior.toml` under `task_success/v1` and the gate-schema tests pass; catalog renders provisional | 2–3 days |
| **C** | interface bump | §4.6 reserved command dims across SB3 and MJX, plant revisions, `widen_checkpoint`, command-slice reseed, MJX fail-closed | Zero-column and action-equality pins; one PPO update from the widened checkpoint; regenerated plant manifest passes `plant_contract --check` with SB3/MJX parity over the widened probes; widened stance checkpoint re-paneled and one recovery freeze re-rolled from it, outcomes recorded in an investigation note | ~1 week |
| **C½** | walker under the new interface | Locomotion re-run warm-started from the widened stance checkpoint (8M, one seed), or the seed-replicate retrain fallback if the re-panel failed | A walking checkpoint certified by `reward_and_length/v1` (≥ 1.0 m/s, ≥ 750 steps) under the new interface; recorded in the same investigation note | ~9 h of Colab per attempt |
| **D** | follow leaf | §4.6 env sampler, tracking reward, `command_tracking/v1`, the two mini-stage TOMLs, notebook command video; T-Rex pilot | Pilot run from the C½ walker with frozen thresholds recorded; MJX fail-closed raise test; gate consulted-test (live-command SB3/MJX parity is a Phase E criterion, when MJX command mode lands) | ~1.5–2 weeks + one pilot run |
| **E** | follow-on | Other species' follow leaves after preflight; cruise-window walk gate; manager re-keying; sweep re-keying | as needed | — |

Each PR follows the established process: adversarial pre-PR review with
lensed finders, three refuters per finding, a completeness audit; ruff, mypy
and the full suite in chunks; the notebook round-trips through `json.dump`
with indent 1.

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

---

## 7. Risks

- **Phase A leaves two vocabularies** ("advancing" for the legacy trio,
  "deliverable" for publication) until the manager and sweeps are re-keyed.
  Mitigation: semantic leaves are judged post-stage like recovery; the
  overlap is documented in `stage_manifest.py` and pinned.
- **The interface bump invalidates every checkpoint for warm-start.**
  Mitigation: D3 — once, early, with the widening tool and recorded
  lineage; the re-panel is budgeted as a real roll with a retrain fallback.
- **No certified walking parent exists yet**: the 20260821 locomotion leg
  was interrupted at 5.49M of 8M with 0/109 gate passes at 0.55 m/s
  (`docs/reviews/TREX_REVIEW_2026_08_MERGES_AND_NEXT_STEPS.md`). The follow
  leaf cannot branch until a walker certifies under the new interface, which
  is why Phase C½ (the locomotion re-run from the widened checkpoint, with
  its own failure branch) sits between the bump and the leaf.
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
   stage config.
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
    test.
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
  the windowed metric becomes the walk gate's later refinement.
- **SS1** (seed multiplicity): resolved by §4.5.
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
  the new interface? (Expected yes up to float rounding; verified by the
  re-panel, Phase C.)
- What tracking tolerances and settle/dwell windows does the walker-derived
  follower reach at 3M steps? (Frozen from the Phase D pilot.)
- Does the command-blind walker's paired null show forgetting of straight
  walking in the follower? (The Phase D panel answers it.)
- Is a cruise-window velocity gate for walking worth its new evidence
  columns, or does the 1.0 m/s episode-mean gate with the 750-step length
  floor suffice? (Decide after the first certified walker under Phase C.)
