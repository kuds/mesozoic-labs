# Stage 2: training the learned controller with the owner's Colab notebook

*Pipeline map for the Stage 2 plan, 10 Nov 2026. Read-only research: nothing was trained, fitted, evaluated or swept;
files were read and records parsed. Paths are relative to `/home/user/mesozoic-labs`; `FS/` is
`hardware/trex_form_study/`. A notebook citation is "cell N, line M": cell index as nbformat stores it (0-based), line
within that cell's source. Labels: **[measured]** read from a record; **[derived]** arithmetic on recorded numbers;
**[inferred]**; **[unverified]** not checked here.*

## Short answers

1. **What a new robot embodiment needs in the main pipeline.** A `BaseDinoEnv` subclass on the species base: five
   hooks; actions are normalised residuals around the MJCF `home` keyframe's controls; the observation is privileged
   simulator state; the reward is a sum of named terms; episodes end on height, tilt or body contact with the floor.
   It also needs stage TOMLs of five tables with a registered `gate_kind`, a `stages.toml` recipe graph, a manifest
   table with success and deliverable metrics, and a plant-versions table with the plant manifest and digest snapshot
   regenerated. A recovery calibration, per-model test-table entries and doc rows complete it. The Compsognathus robot
   shows that bookkeeping end to end. It shows none of the physics the form study relies on: no servo delay, no
   torque-speed envelope, no linkage, no sensor noise and no controller under the policy. It has also produced no
   converged result.
2. **Options.** (a) Porting the form study into the main pipeline would reuse the most tested infrastructure: the
   chain loop, gates, bundles, resume, trunk reuse and the Drive summary. That reuse comes only after four things:
   re-implementing and re-validating the plant; replacing `BaseDinoEnv.step`, which holds one control for a whole step
   while R3's tracker sets servo targets every 2 ms; changing shared code (a gate kind for the form study's panels and
   a privileged critic); and fitting four plants (2 bodies × 2 heads) into a model slot that allows one robot per
   species. That is weeks of work, and Stage 1's records would no longer carry over bit for bit. (b) A separate Stage 2
   notebook, built on the `sb3_training.ipynb` scaffold, can run the form study's own `designlab.stage1.train`. It keeps
   the plant, the records and the gates intact. It reuses the scaffold's tested Colab parts by import, plus the form
   study's own tested PPO stack (36 tests). **Recommend (b).**
3. **Throughput.** Here the form study's R3 env runs about 200 env steps/s per core: one process, 4 envs, PPO
   included [measured]. MuJoCo is only about 14–20 % of an env step on hard floor; the Python tracker and estimator
   are most of the rest. On Colab the same code would run at about 150–185/s per process [derived]. A 24 h runtime
   would then give about 13–16M steps per job run Stage 1's way, 5–6× Stage 1's longest run (2.5M). Running one env process per vCPU gives several times that.
   **The bottleneck is the CPU, not the GPU.** The main pipeline's own Colab runs trained the T. rex stance no faster
   on an L4 runtime (231–276 steps/s) than on a CPU runtime (about 290–306) [measured/derived]. So pick runtimes for
   their vCPUs. MJX/JAX would first need the tracker, estimator, drive model and floors rewritten in JAX.
4. **Results.** The main pipeline writes each run to `MyDrive/mesozoic-labs/logs/<species>/<embodiment>/<algo>/<run_id>/`.
   The lead has reviewed past Colab runs by reading `progress.json`, `stage_config.json` and the gate verdicts there.
   For Stage 2, put a run tree outside `logs/`. It would hold the form study's own `run.json` (G6 results, throughput,
   sessions, provenance), the checkpoints, and a small STATUS.json that the notebook rewrites. The lead reads the JSON
   directly with the Drive tools. The lead then downloads the judged checkpoint (about 2 MB) and runs the panels
   locally from the main checkout, as Stage 1 did.

---

## 1. What the main pipeline needs to train a new robot embodiment

### 1.1 The model slot and registration

- **Model slot.** A model is the pair (species, embodiment), and the embodiment is `anatomical` or `robot`
  (`environments/shared/model_id.py:33-36`; anything else is refused at `:72-73`). "A robot's hardware iterations are
  physics revisions of the same embodiment, not new embodiments" (`docs/SPECIES_NAMING.md:4-6`). Each embodiment has
  one MJCF and one plant table, at revisions 1 / 1 / 1 (`CONTRIBUTING.md:318-321`).
- **The T. rex has no robot.** `tyrannosaurus_rex` has only `anatomical` (`configs/species_manifest.toml:145-153`).
  The notebook refuses any (species, embodiment) pair the manifest does not list, before it writes anything
  (`sb3_training.ipynb` cell 6, lines 9-14; cell 5, line 3).
- **Registration.** It is driven by data. The manifest's `env_entrypoint` is imported
  (`environments/shared/species_registry.py:32-38`). The trainer's config is built from the class attributes
  `HEIGHT_LABEL`, `BEHAVIOR_SECTION_LABEL` and `SUCCESS_KEYS` (`:41-60`). A new embodiment therefore needs no
  shared-code edit to register.

### 1.2 The environment contract

| Part | Main-pipeline contract | Evidence |
|---|---|---|
| Hooks | `_cache_ids`, `_get_obs`, `_get_reward_info`, `_is_terminated` and `_spawn_target` are abstract. The rest of the lifecycle is shared. | `environments/shared/base_env.py:411-428` |
| Control structure | `step()` low-passes the action (optional), applies any scheduled push, calls `_scale_action` and writes `data.ctrl` **once**. It then runs `frame_skip` physics steps, aggregating per-substep foot forces and floor strikes for the gates. | `base_env.py:1244-1303` (ctrl at `:1274-1275`, loop at `:1302-1303`) |
| Action | **Yes, residuals around a home keyframe.** `action_mapping = "home-keyframe-residual/v1"`: a value in [-1, 1] is scaled piecewise so that 0 is the `home` keyframe's ctrl and ±1 reaches the actuator limits. | Compsognathus `environments/compsognathus_longipes/base_env.py:49, 204-211`; T. rex `environments/tyrannosaurus_rex/base_env.py:130, 939-954`; keyframe cache `base_env.py:391-409` |
| Observation | Privileged simulator state: joint angles and rates, pelvis orientation (a world quaternion on the Compsognathus robot; projected gravity on the heading-free schema), gyro, linear velocity, accelerometer, foot forces, target direction and distance, and a 3-value command. The Compsognathus robot has 46 values and 12 actions. | `compsognathus_longipes/base_env.py:213-233`; `environments/compsognathus_longipes/README.md:85-86`; T. rex `tyrannosaurus_rex/base_env.py:973-1028` |
| Reward | A sum of named `reward_*` terms (alive × support, forward, energy, posture, height, home pose, smoothness, spin, drift, lateral, heading, gait, approach, target), plus info keys that the metrics and gates read. A fall adds `fall_penalty`. | `compsognathus_longipes/base_env.py:235-291`; `base_env.py:1352-1359` |
| Termination | A non-finite state, height or tilt, a non-foot body on the floor, or the target reached (success). Time limits truncate. | `compsognathus_longipes/base_env.py:293-313`; `base_env.py:1402-1404` |
| Class attributes | `HEIGHT_LABEL`, `BEHAVIOR_SECTION_LABEL`, `SUCCESS_KEYS`, `supported_training_backends`, `action_mapping`; per embodiment: `embodiment`, `MODEL_PATH`, `nominal_pelvis_height`, `FOOT_SUPPORT`, `REWARD_PROBE` | `compsognathus_longipes/base_env.py:44-49`; `compsognathus_longipes/robot/env.py:17-36` |
| Shared base | A robot "shares the species' base class, task, rewards, stage graph and docs". The plant contract hashes the species base's methods: a robot that changes them moves the anatomical model's policy-interface digest. | `CONTRIBUTING.md:299-301, 185-187` |

### 1.3 Stage TOMLs, gates and budgets

- **The recipe graph.** `stages.toml` (schema v2) declares nodes with `id`, `config`, `legacy_number`,
  `warm_start_from`, `recipe` and `deliverable` (`configs/compsognathus_longipes/robot/stages.toml:16-46`). The
  notebook's `BEHAVIOR` setting resolves a chain through that graph (cell 9, lines 11-13).
- **Allowed tables.** A stage TOML may contain only `[stage]`, `[env]` (constructor kwargs), `[ppo]` / `[sac]` (SB3
  kwargs) and `[curriculum]` (`environments/shared/config.py:179`). Each file is written whole per embodiment; it never
  `extends` another embodiment's TOML (`CONTRIBUTING.md:315`).
- **Gate kinds.** `reward_and_length/v1`, `stance_quality/v1`, `/v2`, `stance_quality_heading/v1`,
  `recovery_quality/v1` (frozen null panels, set before training) and `task_success/v1`; `none/v1` never advances
  (`environments/shared/curriculum/gate_schema.py:66-167`). The notebook writes the artifacts and the bundle **before**
  it enforces a verdict, then halts and disconnects on a failure (cell 20, lines 287-315). None of these kinds can
  express the form study's verdict (20 episodes per condition, all stay up, at least 17 meet every limit;
  `FS/designlab/stage1/panels.py:75, 284`) or Stage 2's agility bar.
- **Budgets.** The T. rex anatomical stages are stance 12M (`configs/tyrannosaurus_rex/anatomical/stance.toml:574`),
  recovery 3M (`recovery.toml:133`), locomotion 8M (`locomotion.toml:89`) and behavior 8M (`behavior.toml:68`). The
  Compsognathus robot's are stance 11M and 3M for each of the others (`configs/compsognathus_longipes/robot/*.toml`).
  `QUICK_TEST` uses 50,000 steps a node (cell 15, lines 38-45).
- **Before training.** Two preflights run first: an SB3 archive load (cell 11) and the zero-action baseline against
  the stage-1 gate (cells 12-13).

### 1.4 Manifest, plant contract, tests and docs (`CONTRIBUTING.md:297-341`)

- **Manifest.** The table `[species.<species>.embodiments.robot]` holds `env_entrypoint`, `model_path`,
  `training_backends`, `training_notebooks`, `result_summary_paths`, success metrics, and a deliverable metric for
  every deliverable. For the Compsognathus robot see `configs/species_manifest.toml:374-416`; every deliverable needs a
  stable-baselines3 entry (`:27-29`).
- **Plant contract.** `[plants.<species>.robot]` in `configs/plant_versions.toml` (the Compsognathus robot is at
  `:857-861`). After it, regenerate `plant_manifest.generated.json`, `digest_snapshot.generated.txt` and the species
  catalog. A stage graph with recovery also needs a `recovery_calibration.json`.
- **Tests and docs.** Entries in the per-model test tables (`test_species_integration.py`, `test_mjcf_assets.py`,
  `test_plant_contract_digest_snapshot.py`) plus robot-specific tests. A website model page, the README's model
  column, and rows in `intro.md`, `SPECIES_NAMING.md` and `api/overview.md`. This publishes a public "Tyrannosaurus
  Rex (Robot)" model, which needs the owner (P24).

### 1.5 How much the Compsognathus robot shows

- **What it shows: the bookkeeping, end to end.** A 36-line embodiment module on the species base
  (`compsognathus_longipes/robot/env.py`); its own MJCF, meshes, configs, manifest and plant tables, and a recovery
  calibration. It trains in the notebook by setting `SPECIES` and `EMBODIMENT` (`compsognathus_longipes/README.md:65-73`).
  Its integration checks passed: "This validates the pipeline; it does not establish a converged walking controller"
  (`environments/compsognathus_longipes/TRAINING_VALIDATION.md:3-5`). No result is published:
  `result_summary_paths = []` (`configs/species_manifest.toml:382`).
- **What it does not show:**
  - a robot whose joints or task differ from its species' anatomical model: it keeps the species task, with 12 leg
    servos and a fixed head and tail;
  - realistic drives: its MJCF position servos have kp 25 and kv 0.2, a fixed `forcerange` and ctrl limits
    (`compsognathus_longipes/robot/assets/compsognathus_longipes_robot.xml:281-285`). These are "constant torque
    limits, not an identified torque-speed … Backlash, delay, voltage sag and heating remain open"
    (`compsognathus_longipes/README.md:421-424`);
  - deployable inputs: its policies read privileged state and "cannot be deployed as-is" (`README.md:122-124`);
  - a reference controller under the policy;
  - a privileged critic.

## 2. Integration options and their costs

### 2.1 Option (a): T. rex robot embodiments in the main pipeline, with the form study's plant ported

| Mismatch | Detail | Evidence |
|---|---|---|
| Control structure | The form study's env exists *because* "the repository's `BaseDinoEnv` holds one ctrl per step while the tracker sets the servo targets every 2 ms". R2/R3 write targets at 500 Hz, with feedback at 100 Hz through a first-order hold. A port must replace `step()` or add a shared hook. Replacing `step()` bypasses the substep aggregation that the floor-truth gates read. | `FS/designlab/stage1/env.py:3-4`; `FS/designlab/stage1/control.py:15-21`; `base_env.py:1274-1340` |
| Model slot | Four plants (60-N-T-10 and 75-M-H-10, each with the current and the expressive head) against one robot embodiment per species with one MJCF. They become sequential physics revisions, with one current at a time; certified nodes are reused only on the same plant (cell 19, line 5). The alternative is changing the closed vocabulary (`model_id.py:33-34`), a new decision on D-D33. | `docs/SPECIES_NAMING.md:4-6`; `CONTRIBUTING.md:318-321` |
| Species base | The T. rex base hashes `_get_obs`, `_cache_ids` and `_scale_action`, built on the anatomical joints: neck pitch and yaw, head pitch, 4 tail joints (`tyrannosaurus_rex/base_env.py:27-30, 721-737`). The form study's bodies have 14 drives: neck pitch, tail yaw, and per leg hip roll, pitch and yaw, knee, and two ankle motors on tendons (`FS/stage1/runs/2026-11-08/60-N-T-10/plant.json` `drives`). Either a second species base (against `CONTRIBUTING.md:299-301`) or edits that move the anatomical digests (`:185-187`). | as cited |
| Plant features to port | All of these, then re-validation against Stage 1's G1–G4/G7 records under a new plant identity: <br>• the torque-speed envelope applied every physics step (`FS/designlab/stage1/drive.py:18-38`);<br>• the two-motor ankle on tendons (`drive.py:4-16`; `FS/designlab/stage1/plant.py:12-21`);<br>• ctrl limits off, with feedforward in the target, and the hold-servo gain rule (`plant.py:31-40`);<br>• the settle procedure (`plant.py:51-60`);<br>• the carpet and probe-pile floors (`plant.py:45-49`; `FS/designlab/carpet.py`);<br>• the latency queue (`env.py:504-519`);<br>• mass and kp randomisation with `mj_setConst` (`env.py:125-135, 313-321`);<br>• the noisy sensors and the estimator (`FS/designlab/stage1/estimator.py`). | as cited |
| Policy | The privileged-critic policy (`FS/designlab/stage1/policy.py:69-141`) is not a `[ppo]` option. The main trainer builds `MlpPolicy` or the critic-phase policy (`environments/shared/train_base.py:916-919`). Shared-code change. | as cited |
| Judging | No gate kind judges the panels (above) or the agility bar. A new kind means changes to `gate_schema.py` and the evaluation code, and new digests. | `gate_schema.py:66-167` |
| Records | Stage 1's records (refs, transitions, `tune.json`, panel keys) are keyed to the form study's model hashes and sources. A port starts over. | `FS/designlab/stage1/policy_rows.py:16-21` |

**Cost:** weeks of shared-code and plant work before the first useful run, with a real risk that the ported plant
drifts from the validated one. **Gain:** the chain loop, verdict and bundle machinery, `TRUNK_FROM` reuse, replication
discovery, `progress.json` and `google_drive_summary.ipynb`, all of them tested and keyed by `ModelId`.

### 2.2 Option (b): a separate Stage 2 notebook on the `sb3_training.ipynb` scaffold

**What carries over unchanged, by copying or importing it:**

- **Setup.** `REPO_REF`, the pinned installs (`mujoco==3.10.0`, `gymnasium==1.3.0`, `stable-baselines3[extra]==2.9.0`
  and `scipy`, the form study's PPO venv versions; `FS/designlab/stage1/train.py:109`), the clone at a commit, and the
  refusal to switch code after import (cell 3, lines 6-58). Copy this cell: it is pinned by test
  (`test_sb3_notebook_pins.py`, per `google_drive_summary.ipynb` cell 3, line 20).
- **Drive mount and run directory** (cell 8, lines 8-21).
- **Auto-disconnect** with a Drive flush: `environments/shared/notebook_runtime.py:15-43`
  (`flush_and_unmount` at `:32`, `runtime.unassign` at `:37`).
- **Checkpoint-pair integrity check**: `environments/shared/curriculum/checkpoints.py:174-208`. It is generic for any
  SB3 zip with its pkl. `newest_intact_periodic_pair` (`:210-235`) assumes SB3's `CheckpointCallback` names, so it
  needs a 10-line walker for `model_<n>.zip`.
- **Safe archive loading**: `environments/shared/policy_loading.py:500-561`.
- **Atomic JSON writes**: `environments/shared/file_io.py:94`.

**The form study's training stack, as tested.** `designlab.stage1.train.train()`:

- checkpoints every 250k steps (`train.py:64`) into `<out>/<body>/(ppo|policy_only)/<seed>/` (`:158-159`);
- resumes with refusals when the settings, env configs, gains, libraries or sources changed (`:426-465`);
- records the provenance and throughput of every session, with `cores = os.cpu_count()` (`:481-485, 602-619`);
- runs G6a (zero residual), G6b and G6c (`:519-523, 567-594`);
- runs on the CPU with 1 torch thread (`:468, 497, 513`);
- takes `--vec subproc` with spawn (`:476`).

`FS/tests/test_stage1_ppo.py` holds 36 tests, among them checkpoint and resume (`:515`), subprocess envs (`:674`),
latency (`:212`) and the zero residual (`:89`).

**What must be built (small, all inside FS):**

1. **The inputs.** The MJCF models (`FS/round2/.gitignore:1-3`) and the reference and transition arrays
   (`FS/stage1/.gitignore:1-11`) are not committed, so a fresh clone lacks them. They are about 20 MB per body plus a
   3.5 MB model case. Either regenerate them on Colab with `python -m designlab.stage1 prepare` (about 0.21 core-h per
   body, `FS/stage1/runs/2026-11-08/STAGE1_PROBE.md` §5) and check them against the records, or ship them through
   Drive with a sha256 manifest. Shipping is preferred: copying the `round2/refine/*/models` files keeps the panel
   record keys equal to the main checkout's (`policy_rows.py:16-21`).
   - `train.env_configs` reads the references from `<out>/<body>/` (`train.py:289-302`).
   - Resume compares the recorded env configs, `run_dir` included (`:435-436`), so the path must stay the same in
     every session.
2. **Resume integrity.** `train.checkpoints()` checks only that both files exist (`train.py:186-194`), and resume
   loads the newest pair (`:487-497`). Add the integrity walk, because Drive writes can be cut short.
3. **Archive portability.** `policy.linear_schedule` is a closure (`policy.py:144-151`), and `train.py:497` and
   `policy_rows.py:601` load with a plain `PPO.load`. Across Python minor versions that load segfaults
   (`docs/KNOWN_ISSUES.md:562-616`). Use a schedule class (the pattern of `environments/shared/curriculum/schedules.py:1-9`)
   or `load_sb3_model`. The change moves the Stage 1 sources hash, which is acceptable for a new round.
4. **A launcher cell.** It runs one subprocess per job and writes logs to Drive. A monitor rewrites STATUS.json. When
   every job has ended it writes `DONE.json`, flushes Drive and disconnects.
5. **The notebook's location.** It must **not** go in `notebooks/`. Four main-repo tests glob `notebooks/*.ipynb`
   (`environments/shared/tests/test_record_fields.py:41`, `test_stage_config_reader.py:44`,
   `test_stage_manifest.py:737`, `test_retired_species_identifiers.py:247-248`). The retired-identifier regex
   (`test_retired_species_identifiers.py:66-81`) matches the string `hardware/trex_form_study` [checked: the regex
   matches it and does not match `tyrannosaurus_rex`]. Put the notebook under FS, for example `FS/stage2/`. Colab
   opens any path of a pushed branch (`docs/investigations/stance_2026_10/stance3/colab_steps.txt:25-27` uses the same
   URL form).

**Needed in either option:** the expressive head needs new plant models, registered like arm K
(`FS/designlab/stage1/__init__.py:41-60`). The env's widths are hard-coded to 14 joints
(`FS/designlab/stage1/env.py:137-142, 284-286`) and must follow the plant.

### 2.3 Comparison

| | (a) Main-pipeline embodiment | (b) Stage 2 notebook on the scaffold |
|---|---|---|
| Form study's validated plant, records and gates | Re-implemented and re-validated; Stage 1 records do not carry over | **Intact**; G6a re-proves R3 = R2 bit for bit in each new run |
| Tested infrastructure reused | **Most**: the chain loop, verdicts, bundles, trunk reuse, replication, `progress.json`, the summary notebook | The scaffold's Colab parts plus the form study's own tested PPO stack |
| Shared-code changes | `step` hook or override, a gate kind, a privileged critic, possibly the embodiment vocabulary | None |
| Four plants (bodies × heads) at once | No: one robot slot, sequential physics revisions | Yes: separate body entries |
| Time to first Colab run | Weeks [inferred] | Days [inferred] |
| Public-facing change | Adds a catalog and website model (needs the owner, P24) | None |

### 2.4 Recommendation

Choose **(b)**. Import the main pipeline's tested Colab pieces: setup, Drive, the integrity check, the safe loader
and the disconnect. Keep the form study's plant, `run.json` records, G6 checks and panels as they are. Revisit (a)
only once a hardware revision is chosen and a learned controller is worth publishing as a main-repository model; one
robot embodiment with physics revisions per hardware revision fits that moment.

## 3. Throughput

### 3.1 The form study's env on this CPU

| Measurement | Value | Source |
|---|---|---|
| R3 job (4 envs in one process: hard, hard, thin carpet, low pile), with PPO, idle machine / load 4.2 | 199 / 204 env steps/s [measured] | `FS/STAGE1_PLAN.md:321-322` |
| One env alone, no PPO | hard 358, carpet 234, low pile 135 env steps/s [measured] | `FS/STAGE1_PLAN.md:323` |
| R3, CPU-normalised over long windows / first rollouts | 195–214 / 188–209 env steps per CPU-second [measured] | `FS/stage1/runs/2026-11-08/*/ppo/*/run.json` `throughput` |
| R4, CPU-normalised (no tracker; estimator, envelope and latency remain) | 251–301 per CPU-second [measured] | `*/policy_only/42/run.json` |
| Nine jobs sharing 4 cores | G6a rate over the first rollouts: R3 74–83, R4 100–120 per job [measured]; the plan's later reading: R3 76–87, R4 105–121 | `run.json` `g6.G6a_rate`; `STAGE1_PROBE.md:129-131`; `STAGE1_PLAN.md:346-347` |
| Cost split | Before the faster tracker was merged: the tracker took 0.24 wall-s per simulated s per core, and MuJoCo's step was about 16 % of the cost [measured]. With the merged tracker (the one PPO used): an env alone on hard floor spends about 279 µs per 2 ms physics step (358 env steps/s), against 38–57 µs for MuJoCo alone (17.5–26.4k physics steps/s), so physics is about 14–20 % [derived] | `FS/STAGE1_PLAN.md:252-253, 274-277, 323`; `FS/STAGE1B_PLAN.md:154-158` |
| Physics alone | about 20k steps/s per core on hard floor (at most 2,000 env steps/s, G1); probe piles 102–437 env steps/s | `FS/STAGE1_PLAN.md:62`; `<b>/plant.json` `throughput` |
| PPO overhead in a job | The floor mix without PPO would run 4 / (2/358 + 1/234 + 1/135) ≈ 232 env steps/s; the measured 199–204 leaves about 12–14 % for inference, updates and VecNormalize [derived] | as above |
| Stage 1 reached | Budget 4M; R3 trained to 2.5M (75 cm) and 1.75–2.02M (60 cm); R4 to 1.25–1.51M. "The residual barely moved (about 1 % of its authority at 1M steps)" | `train.py:76-80`; `STAGE1_PROBE.md:129-131, 376` |

### 3.2 The main pipeline's Colab budgets and measured rates

These are T. rex anatomical stance runs with PPO and `N_ENVS = 4` (cell 6, line 18). The notebook never passes
`use_subproc` (cell 15, lines 139-159), so PPO steps every env in **one** process with `DummyVecEnv`
(`train_base.py:437-440, 1946`; only SAC switches to subprocesses, `:1558`). The model is built with no device
(`train_base.py:916-919`), so SB3's "auto" puts the small MLP on the L4.

| Runtime | Steps/s | 12M-step stance | Source |
|---|---|---|---|
| Colab L4 | 262.8–276.5 (stance4); 230.9–245.2 (stance5) [measured] | 12–12.5 h; 13.7–14.5 h | `docs/investigations/stance_2026_10/stance5/colab_steps.txt:216-224`; `stance5/results/README.md:54-61`; `stance5/replication/criteria.txt:113` |
| Colab CPU | about 290–306 [derived from 12M in 10.9–11.5 h] | 10.9–11.5 h | `stance3/colab_steps.txt:73-75` |
| Session container CPU (smoke runs) | 314.6–362 [measured] | — | `stance5/colab_steps.txt:216-218`; `stance3/colab_steps.txt:74-75` |

- **Colab against local.** Colab's per-process rate is 0.75–0.79× local on the L4 runtime and 0.82–0.87× on the CPU
  runtime [derived].
- **Session cap.** Colab ends a session at about 24 h, and the notebook resumes from periodic checkpoints (cell 17,
  line 3).
- **Parallel runtimes.** The maintainer has run three Colab runtimes at once (`stance5/colab_steps.txt:4`) and planned
  around "a fourth Colab slot" (`docs/BEHAVIOR_RECIPES_PLAN.md:1319`).

### 3.3 What a 24 h runtime could reach with SB3 [derived; to be measured]

- **The base rate.** Scaling this machine's R3 rate (199–214) by the Colab/local factor (0.75–0.87) gives about 150–185
  env steps/s per process.

| Setup on one runtime | Env steps/s | Per 24 h |
|---|---|---|
| One job run as in Stage 1 (`--vec dummy`, 4 envs, floor spread) | about 150–185 | about 13–16M (5–6× Stage 1's 2.5M) |
| k such jobs on k vCPUs (bodies, heads, seeds in parallel) | k × 150–185 in total | 13–16M per job |
| One job, `--vec subproc`, one env per vCPU, same floor spread (lockstep at the low pile's pace, about 100–117 per env on Colab, less about 10 % for PPO) | 4 vCPUs about 360–420; 8 about 730–840 | about 31–36M; about 63–73M |
| As above, hard floor and thin carpet only (lockstep at the carpet's pace, about 160–180 per env) | 4 vCPUs about 630–730; 8 about 1,260–1,460 | about 55–63M; about 110–125M |

- **Caveats.** Colab vCPUs may be hyperthreads, and RAM per worker was not measured. The vCPU counts of consumer
  Colab runtimes are not published [unverified]. The standard runtime is reported at 2 vCPUs, and Colab Enterprise's
  default L4 runtime is a `g2-standard-4` machine (sources at the end). Read the count on day one: `train.py:611`
  records it.
- **Floor choice.** The low-pile env sets the lockstep pace. Under P26 (hard floors, perhaps thin compact carpet),
  dropping or down-weighting low pile is the largest single lever, but that choice is the owner's.

### 3.4 GPU or CPU

**The bottleneck is CPU-bound env stepping, in both pipelines.**

- **Main pipeline.** An L4 runtime trained no faster than a CPU runtime (§3.2). All envs run in one process, and the
  GPU only speeds the minibatch updates of a two-layer MLP.
- **Form study.** MuJoCo is about 14–20 % of an env step on hard floor, more on pile; the Python tracker and
  estimator are most of the rest. PPO is about 12–14 % of a job. The form study already pins PPO to the CPU
  (`train.py:513`).

**What follows:**

1. Choose Colab runtimes for their vCPUs, not their GPU. An L4 runtime is worth it only if it brings more vCPUs; its
   GPU otherwise burns credits while idle, which is why `AUTO_DISCONNECT` exists (cell 31).
2. Spend the vCPUs on more envs: `--vec subproc` or parallel jobs.
3. Per core, the biggest lever is a faster tracker and estimator: making the non-physics part 2× faster gives about
   1.7× per env [derived].
4. MJX/JAX helps only after the tracker, estimator, envelope, linkage, latency queue and pile floor are rewritten in
   JAX and the plant is re-validated. That is a separate project. The repository retired its own MJX trainer (D-D17;
   `environments/shared/mjx_env.py:1-18`), and the probe-pile floors are contact-heavy [inferred to suit MJX poorly].
5. Budgets of 10^7 to 10^8 steps fit Colab CPU work with SB3. Policy-only learning from scratch on GPU-parallel
   simulators usually uses far more samples [general knowledge, not measured here]. So R4-style training is the case
   that would need the rewrite, while residual-on-tracker (R3) and imitation-shaped training fit Colab.

## 4. How results come back for review

### 4.1 What the main pipeline writes, and how the lead read it before

- **Drive layout.** `MyDrive/mesozoic-labs/logs/<species>/<embodiment>/<algo>/<run_id>/` (cell 8, lines 12, 27).
  - Run level: `provenance.json`, `summary.json`, `artifact_manifest.json` (written last, the bundle's status),
    `collected_results.csv`, `training_summary.txt`, `trunk_run.json` and `ancestors/`.
  - Per stage, `NN_<id>/`: `stage_config.json` (with a `gpu` block naming the runtime; `config.py:983`),
    `gate_verdict.json` (hash-bound to the handoff pair), the evaluation CSVs, `models/`, `figures/` and `replays/`.
  - See `docs/RESULT_BUNDLES.md:61-87` and its Colab workflow, `:359-384`.
- **progress.json** is a human-readable snapshot, rewritten atomically after every evaluation: step against budget,
  projected finish, the latest evaluation, checkpoints and warnings (`environments/shared/curriculum/progress_report.py:1-33`).
- **The cross-run table.** `google_drive_summary.ipynb` writes `logs/runs_summary.csv`. It reads only the model
  directories the manifest lists (cell 1, lines 20-22; cell 23).
- **Past practice.** The lead read Colab runs straight from Drive: "There is nothing to send: I read the checkpoints and
  telemetry from Drive" (`stance3/colab_steps.txt:104-106`); "stance5 Drive check … progress.json, all three at
  2,621,440 …" (`stance5/results/run_log.txt:41`). The steps given to the owner also listed exact lines to check (the
  commit, "new run", the baseline row, the budget; `stance3/colab_steps.txt:58-71`) and a case table for resuming
  (`:108-150`).

### 4.2 Proposed Stage 2 layout (option b)

```
MyDrive/mesozoic-labs/stage2/                  outside logs/: neither the summary notebook nor TRUNK_FROM="auto" scans it
  inputs/<inputs_id>/                          uploaded once by the lead, read-only
    MANIFEST.json                              sha256 of every file, the source commit, how it was made
    round2/refine/<case>/models/*.xml, *.json  the git-ignored sizing models
    <body>/refs/, transitions/, *.json, tune.json (or the chosen gains file)
  runs/<run_id>/
    SESSION_<k>.json     runtime facts: os.cpu_count, RAM, nvidia-smi, Python/numpy/torch, REPO_REF commit, inputs sha
    STATUS.json          rewritten every few minutes: per job steps/budget, env steps/s, ETA, G6 so far, last error
    <body>/refs, transitions, tune.json        copied from inputs/ (train.env_configs reads <out>/<body>/)
    <body>/ppo/<seed>/run.json                 the form study's own record (settings, gains, sessions, throughput, g6)
    <body>/ppo/<seed>/checkpoints/model_<n>.zip, vecnormalize_<n>.pkl    about 2.1 MB per model
    logs/<job>.log                             each job's stdout
    DONE.json | HALT.json                      why the session ended
```

### 4.3 Verdicts: where each is judged

- **G6a, G6b and G6c** are judged inside the training job and written to `run.json` on Drive (`train.py:140-155,
  519-523, 567-594`), so the lead can read them mid-run.
- **The panels and the margin ledger** run on the lead's machine, from the main checkout, on the downloaded judged
  checkpoint, as in Stage 1: 20 episodes per condition, seeds 3042–3061, pass if all 20 stay up and at least 17 meet
  every limit (`panels.py:75, 284`). Point `STAGE1_POLICY_RUNS` at the downloaded run (`policy_rows.py:31-33, 70-71`).
  This keeps them comparable with Stage 1 and 1b, and keeps Colab's floating-point differences (another numpy and
  BLAS; locally numpy 2.5.4, per `run.json` `versions`) out of the verdicts.
- **The agility bar** (0.4 m/s or more, a 90° walking turn in about 2 s, a nudge while walking, a stop within one or
  two steps, all with servo delay) becomes new panel conditions in the form study's code. It is not a main-pipeline
  gate kind. Training tasks today go up to 0.35 rad/s walking turns and 0.5 rad/s in place (`env.py:78-99`), so the
  faster turns' references and transitions (made by `prepare`) join the inputs bundle.

### 4.4 What the lead's Drive tools can read

- **Directly:** JSON, CSV, text and Markdown. That covers `run.json`, STATUS.json, SESSION files, logs, MANIFEST.json
  and DONE/HALT.
- **After a download:** binaries (`.zip`, `.pkl`, `.npz`, `.mp4`), with local tools afterwards.
- **Liveness:** a stale STATUS.json `modifiedTime` shows a dead runtime. Order checkpoints by the step count in their
  names, never by mtime: on a Drive mount the mtime is the upload order (`checkpoints.py:224-225`).

## 5. Validate on this CPU first (owner item 1)

Before a long Colab run, check these locally. They are recommendations for the plan; none was run here.

1. **A short smoke run per mode.** It confirms G6a "identical", a rate record, and checkpoint, kill and `--resume`.
2. **Throughput.** `env.throughput` per floor (`env.py:879-900`) and `--vec subproc` scaling at 2, 3 and 4 workers,
   with the RSS of each worker.
3. **A fresh-clone dry run of the inputs bundle.** Copy it in, verify the hashes and build every env.
4. **An archive saved under the Colab stack loads locally** through the fixed loader.
5. **The notebook end to end with a tiny budget.** The scaffold runs locally too (cell 2, line 2; cell 3, line 58).

## 6. Risks and unverified items

1. **vCPU count and RAM** of the owner's Colab runtime are unknown. Size `N_ENVS` from day one's measurement.
2. **Colab image changes.** A resume is refused when library versions change (`train.py:457-460`), and Colab's image
   changes without notice (`docs/KNOWN_ISSUES.md:562-616`). Pin the installs as cell 3 does; plan budgets that end
   inside about 24 h, or accept that a run may need a fresh start.
3. **torch build.** Colab's torch is a CUDA build, not the venv's 2.9.0+cpu, so `matches_ppo_venv` is false. This is
   recorded only (`train.py:482`).
4. **The env layout** is fixed at 14 joints (`env.py:137-142, 284-286`). The expressive head needs it generalised,
   new plant models and, for character, new head and jaw references.
5. **Throughput estimates** in §3.3 are arithmetic on local records and Colab ratios measured on another env. Treat
   them as planning figures until `train.py`'s own rate record (`:602-619`) reports from Colab.

## Sources read

- `notebooks/sb3_training.ipynb` (all 33 cells) and `notebooks/google_drive_summary.ipynb` (all 25 cells).
- `environments/shared/`: `train_base.py`, `base_env.py`, `config.py`, `stage_manifest.py`, `species_registry.py`,
  `notebook_runtime.py`, `replication.py`, `reporting/`, `result_bundle/`, `curriculum/` (`gate_schema.py`,
  `checkpoints.py`, `progress_report.py`, `schedules.py`), `policy_loading.py`, `mjx_env.py`, `jax_setup.py`, and the
  notebook-scanning tests.
- `configs/species_manifest.toml`; `configs/compsognathus_longipes/robot/*.toml`; `configs/tyrannosaurus_rex/anatomical/*.toml`;
  `configs/plant_versions.toml`.
- `environments/compsognathus_longipes/` (`base_env.py`, `robot/`, `README.md`, `TRAINING_VALIDATION.md`) and
  `environments/tyrannosaurus_rex/` (`base_env.py`, `anatomical/env.py`).
- `docs/RESULT_BUNDLES.md`, `docs/SPECIES_NAMING.md`, `docs/KNOWN_ISSUES.md`, `CONTRIBUTING.md`, and the Colab steps,
  results and run logs under `docs/investigations/stance_2026_10/` (`stance3`, `stance5`, `stance5/replication`).
- `FS/designlab/stage1/` (`env.py`, `policy.py`, `train.py`, `plant.py`, `drive.py`, `control.py`, `policy_rows.py`,
  `panels.py`, `__init__.py`, `__main__.py`), `FS/STAGE1_PLAN.md`, the Stage 1 and 1b reports, and the Stage 1
  `run.json` and `plant.json` records.
- Web, for Colab runtime CPUs only:
  [Colab default runtimes with GPUs (L4 on `g2-standard-4`)](https://docs.cloud.google.com/colab/docs/default-runtimes-with-gpus),
  [Running LLMs in Google Colab (2 vCPUs on a default session)](https://datawookie.dev/blog/2026-07-31-running-llms-google-colab/),
  [Saturn Cloud: Colab hardware spec](https://saturncloud.io/blog/whats-the-hardware-spec-for-google-colaboratory/).
