# Map 08: Colab notebook scaffold, pipeline research and known issues

Read-only map, made 14 Nov 2026 on branch `claude/mesozoic-trex-stage2-refine-4jffre` at `167dce4`. Nothing in the
checkout was changed. `REPO/` is `/home/user/mesozoic-labs`; `FS/` is `REPO/hardware/trex_form_study`.

Labels: **[read]** read from the file cited; **[measured]** a short inspection run here; **[doc]** what a document
says, not re-verified against code; **[inferred]**; **not found** means searched for and absent.

Notebook citations are `cN:Lm (raw R)`: cell index N (0-based, as nbformat stores it), line m of that cell's source,
and line R of the raw `.ipynb` JSON. Raw first-source-line of each code cell of `notebooks/sb3_training.ipynb`:
c3 44, c4 112, c6 170, c8 251, c9 415, c11 525, c13 568, c15 621, c16 882, c18 974, c20 1139, c22 1482, c24 1603,
c26 1641, c28 1674, c30 1703, c32 1750 (raw R = start + m - 1).

Target of build item (H): `FS/stage2/stage2_training.ipynb`. **Not found** today (`FS/stage2/` holds `jetson_plem/`,
`look/`, `research/`, `x/` only). No steps sheet, Drive bundle manifest, launcher or STATUS/SESSION writer exists yet
anywhere in `FS/`.

---

## 1. The scaffold notebook, `REPO/notebooks/sb3_training.ipynb`

33 cells, nbformat 4.0, 107,834 bytes, LF endings, **no outputs and no execution counts stored** [measured].
Metadata asks Colab for a GPU: `"accelerator": "GPU"` (raw 1760), `"gpuType": "L4"` (raw 1762),
`"include_colab_link": true` (raw 1763), `language_info.version "3.11.0"` (raw 1773). A copy that keeps this
metadata opens on an L4 runtime; Stage 2 wants High-RAM CPU (`FS/STAGE2_PLAN.md:238-239`).

### 1.1 Cell by cell

| Cell | Type | What it does |
|---|---|---|
| 0 | md | "Open in Colab" badge, URL `https://colab.research.google.com/github/kuds/mesozoic-labs/blob/main/notebooks/sb3_training.ipynb` (raw 10) |
| 1 | md | Title and summary: one loop walks a behaviour chain from the model's stage manifest; budgets and hyperparameters come from stage TOMLs |
| 2 | md | "## 1. Setup & Installation": `REPO_REF` selects the code; setup never discards checkout edits and refuses to switch code after repository modules were imported |
| 3 | code | **Setup cell** (section 1.2 below) |
| 4 | code | Finds `repo_root` (Colab: `/content/mesozoic-labs`; local: first parent holding `configs/species_manifest.toml`, c4:L12-18), inserts it in `sys.path`, imports gymnasium, mujoco, `environments.shared.config.load_all_stages`, `evaluation.evaluate`, `stage_manifest.*`; prints MuJoCo and Gymnasium versions and the repo root (c4:L30-32, raw 141-143) |
| 5 | md | "## 2. Configuration": what `SPECIES`, `EMBODIMENT`, `BEHAVIOR`, `TRUNK_FROM`, `RUN_ID`, `QUICK_TEST` do |
| 6 | code | Configuration: `SPECIES`, `EMBODIMENT`, `MODEL_ID = require_model_id(...)`, `ALGORITHM = "ppo"`, `N_ENVS = 4`, `SEED = 42`, `VERBOSE`, `QUICK_TEST`, `USE_GOOGLE_DRIVE = True`, `AUTO_DISCONNECT = True`, `BEHAVIOR`, `TRUNK_FROM`, `RETRAIN_FROM`, `RUN_LABEL`, `RUN_ID`, `K3_PANEL*`, `CRITIC_PHASE` (c6:L6-55, raw 175-224) |
| 7 | md | "## 3. Open the run and resolve the chain" |
| 8 | code | **Storage cell**: Drive mount, `LOG_BASE`, run directory, provenance record, trunk run (section 1.3) |
| 9 | code | Resolve cell: `STAGE_CONFIGS = load_all_stages(MODEL_ID)`, `MANIFEST`, `TARGET_NODE`, `CHAIN`, `RETRAIN_NODE`; builds one env to print observation and action shapes; `select_trunk` under `TRUNK_FROM = "auto"`; `refuse_complete_run_session`, `record_trunk_run` |
| 10 | md | "## 4. Preflights" |
| 11 | code | **SB3 archive load preflight**: one call, `sb3_archive_load_preflight(MODEL_ID, CHAIN[0].reference, trunk_dir=TRUNK_DIR)` (c11:L13-15, raw 537-539) |
| 12 | md | Zero-action baseline: why `action = 0` is a real floor |
| 13 | code | `zero_action_baseline.preflight([MODEL_ID], stage=1, episodes=40, seed=PUBLICATION_SEED_START, model_id=..., log_base=..., run_dir=...)` (c13:L24-32) |
| 14 | md | "## Helpers (nothing to set)" |
| 15 | code | Infrastructure: imports `train_base`, `disconnect_runtime`, `display_stage_videos`, `halt`; chain state `completed_stages`, `NODE_RESULTS`, `NODE_HANDOFF`; `node_budget`, `k3_panel_config`, **`train_stage`**, `write_training_summary`, `save_run_bundle` (section 1.6) |
| 16 | code | Display wrappers `plot_training_curves`, `plot_diagnostics_graphs` |
| 17 | md | "## 5. Resume an Interrupted Node (Colab runtime cap)": the resume procedure in prose (section 1.5) |
| 18 | code | **RESUME cell**: `RESUME_STAGE = None`; when set, finds the newest intact periodic pair and trains the rest of the budget |
| 19 | md | "## 6. Train the behavior chain": reuse / judge / train |
| 20 | code | **Chain loop** `for NODE in CHAIN:`; writes artifacts and the bundle before enforcing the verdict; `halt(...)` on a failed gate (c20:L313-315, raw 1451-1453) |
| 21 | md | "## 7. Manual single node (debugging escape hatch)" |
| 22 | code | Manual node: `MANUAL_NODE`, `MANUAL_LOAD_PATH`, `MANUAL_VECNORM_PATH`, `MANUAL_LOAD_MODE`, `MANUAL_TIMESTEPS`; verdict recorded, never enforced |
| 23 | md | "## 8. Report" |
| 24 | code | `evaluate(species_cfg=..., model_path=NODE_HANDOFF[TARGET_NODE.id]["model"] + ".zip", n_episodes=30, ...)` |
| 25 | md | "### Training curves" |
| 26 | code | Replots every node of `completed_stages` (display only; a file written here would be undeclared in the bundle) |
| 27 | md | "### Replay the chain's videos" |
| 28 | code | `display_stage_videos(handoff["stage_dir"])` per chain node |
| 29 | md | "### Verify the result bundle" |
| 30 | code | `validate_result_bundle(RUN_DIR, require_complete=True)`, falling back to `require_publishable=True`; prints "Training complete!" and the per-node origin |
| 31 | md | "## 9. Auto-Disconnect" |
| 32 | code | `disconnect_runtime("Training finished — ...", in_colab=IN_COLAB, auto=AUTO_DISCONNECT, flush_drive=USE_GOOGLE_DRIVE)` (c32:L1-6, raw 1750-1755) |

Cells 6, 8 (from line 23 on), 9, 11-13 and 15-30 are bound to `ModelId`, the stage manifest, stage TOMLs, gate
verdicts and the result bundle. None of that exists for the form study (the T. rex has only `anatomical`,
`configs/species_manifest.toml:145-153` [doc: `research_pipeline.md:53`]). What carries over is cell 3, the first
21 lines of cell 8, the helpers in section 2, and the patterns.

### 1.2 Setup cell (cell 3, raw 44-101)

```python
REPO_REF = "main"  # @param {"type":"string"}                                    # c3:L6  (raw 49)
IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")   # L8 (51)
os.environ["MUJOCO_GL"] = "egl"                                                  # L12 (55), inside `if IN_COLAB:`
```

- **EGL.** Writes NVIDIA's ICD file `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` if missing (c3:L13-17, raw
  56-60). `MUJOCO_GL=egl` is set for the whole kernel, so every child process inherits it.
- **Legacy gym purge.** `pip uninstall -y -q gym gym-notices shimmy` (c3:L22, raw 65), before the first SB3 import
  ("segfaults under NumPy 2.x on some Colab images").
- **Pinned installs, conditional.** They run only if one of seven modules is missing:

  ```python
  if any(importlib.util.find_spec(name) is None for name in ("mujoco", "gymnasium", "stable_baselines3", "mediapy", "matplotlib", "scipy", "imageio_ffmpeg")):   # L29 (72)
      'pip install -q "mujoco==3.10.0" "gymnasium==1.3.0" "stable-baselines3[extra]==2.9.0" mediapy matplotlib "scipy>=1.10.0" "imageio-ffmpeg>=0.5.1"'       # L31 (74)
  ```

  Exact pins: mujoco 3.10.0, gymnasium 1.3.0, stable-baselines3 2.9.0. Ranges: scipy, imageio-ffmpeg. Unpinned:
  mediapy, matplotlib, **numpy, torch** (KNOWN_ISSUES confirms torch is unpinned, `docs/KNOWN_ISSUES.md:2318-2321`).
  Every requirement is quoted on purpose (an unquoted `>=` was a shell redirect, c3:L24-28).
- **Clone and checkout.** `repo_dir = /content/mesozoic-labs` (L36, raw 79); `git clone --no-checkout
  https://github.com/kuds/mesozoic-labs.git` (L41, raw 84); `git fetch origin REPO_REF` (L44); `requested_commit =
  git rev-parse FETCH_HEAD^{commit}` (L45); checkout `--detach` only when fresh, index missing, or HEAD differs (L47-52).
- **Two refusals before switching code** (L48-51, raw 91-94): if any `environments` / `environments.*` module is in
  `sys.modules` ("REPO_REF changed after repository modules were imported. Restart the runtime..."), and if `git
  status --porcelain` is non-empty ("The checkout has local edits..."). The import check names `environments` only;
  it does not know `designlab`.
- **The commit line** the owner checks (L53, raw 96):

  ```python
  print(f"Repository ref: {REPO_REF}; commit: {requested_commit}")
  ```

- **Package install.** `pip install -q -e /content/mesozoic-labs` if `environments` is not importable (L54-55, raw
  97-98). `pyproject.toml:66-67` packages only `environments*` and `configs*`, with dependencies `mujoco==3.10.0`,
  `gymnasium>=0.29.0`, `numpy>=1.24.0` (`pyproject.toml:24-28`). **`designlab` is not installed by it**: the form
  study runs with cwd `FS` and `PYTHONPATH=.`.
- Outside Colab the cell prints "Running locally; using this checkout. REPO_REF selects only the Colab checkout."
  (L58, raw 101) and installs nothing.

### 1.3 Drive mount and what is saved where (cell 8)

```python
if USE_GOOGLE_DRIVE and IN_COLAB:                                   # c8:L8  (raw 258)
    from google.colab import drive
    drive.mount("/content/drive")                                   # L11 (261)
    LOG_BASE = Path("/content/drive/MyDrive/mesozoic-labs/logs")    # L12 (262)
    LOG_BASE.mkdir(parents=True, exist_ok=True)
else: LOG_BASE = repo_root / "logs"                                 # L17 / L20
```

- Run tree: `_runs_root = MODEL_ID.path(LOG_BASE) / (ALGORITHM.lower() + ("_quick_test" if QUICK_TEST else ""))`
  (L27, raw 277), i.e. `MyDrive/mesozoic-labs/logs/<species>/<embodiment>/<algo>/<run_id>/`; `_run_id = RUN_ID or
  memo or datetime.now().strftime("%Y%m%d_%H%M%S")` (L51, raw 301).
- The line the owner checks: `print(f"Run directory: {_run_dir} ({_run_state})")` (L66, raw 316), where the state
  reads `new run`, `new run: RUN_ID names no existing run`, or `re-entering run '<id>': holds ...; bundle status ...`.
- `initialize_result_bundle(_run_dir, model_id=..., algorithm=..., backend="stable-baselines3", seed=SEED,
  evaluation_seeds=[SEED+1000, SEED+3000], seed_roles={...}, evaluation_episodes=30, parallel_envs=N_ENVS,
  hardware="Google Colab"|"local", plant_identity=..., run_id=..., repository_root=repo_root)` writes
  `provenance.json` (L79-101, raw 329-351).
- Written per run [doc: `research_pipeline.md:307-312`, `docs/RESULT_BUNDLES.md:61-87`]: run level `provenance.json`,
  `summary.json`, `artifact_manifest.json`, `collected_results.csv`, `training_summary.txt`, `trunk_run.json`,
  `ancestors/`, `zero_action_baseline.json`; per stage `NN_<id>/` with `stage_config.json`, `gate_verdict.json`,
  evaluation CSVs, `models/`, `figures/`, `replays/`, `progress.json`.
- Checkpoints inside a stage: periodic `models/<stage_label>_<steps>_steps.zip` with
  `<stage_label>_vecnormalize_<steps>_steps.pkl`; final `<stage_label>_final.zip` with `_final_vecnorm.pkl`
  (c18:L47, raw 1020; c15:L160, raw 780). Scaffold default cadence: `save_freq=100_000`, `eval_freq=50_000`
  (c15:L74-75, raw 694-695), "the newest five" kept (c17:L4).
- Everything is written **straight to the Drive mount**. There is no local-then-copy step in the scaffold.

### 1.4 Flushed disconnect

`REPO/environments/shared/notebook_runtime.py` (59 lines):

```python
def disconnect_runtime(reason: str, *, in_colab: bool, auto: bool, flush_drive: bool) -> None:   # :15
def halt(reason: str, *, in_colab: bool, auto: bool, flush_drive: bool) -> NoReturn:             # :40
def display_stage_videos(stage_dir) -> None:                                                     # :46
```

- `disconnect_runtime`: prints the reason; returns with "Auto-disconnect skipped (not in Colab or AUTO_DISCONNECT is
  False)." unless `in_colab and auto` (:22-24); if `flush_drive`, `drive.flush_and_unmount()` (:32) "otherwise the
  tail of files written this session ... can be lost or truncated"; prints "Disconnecting runtime in 5 seconds...",
  sleeps 5 s, `google.colab.runtime.unassign()` (:35-37).
- `halt`: `disconnect_runtime(...)` then `raise RuntimeError(reason)` (:42-43), so "Run all" stops.
- `google.colab` and IPython are imported inside the calls, so the module imports anywhere. **It imports
  `environments.shared.reporting.stage_layout` at module top (:12)**, so importing it pulls the reporting package.
- Callers pass the knobs by name every time: `in_colab=IN_COLAB, auto=AUTO_DISCONNECT, flush_drive=USE_GOOGLE_DRIVE`.

### 1.5 Checkpoint-pair integrity check and the resume cell

`REPO/environments/shared/curriculum/checkpoints.py`:

```python
def checkpoint_pair_problem(zip_path: Path, vecnorm_path: Path) -> str | None:                      # :174
def newest_intact_periodic_pair(model_dir, name_prefix: str) -> tuple[tuple[int, Path, Path] | None, list[str]]:   # :210
```

- `checkpoint_pair_problem` returns `None` or a reason string. Checks, in order (:184-207): the sidecar exists
  (`"missing matched VecNormalize sidecar <name>"`); `zipfile.testzip()` passes and the **outer** archive holds `data`
  and a member ending `policy.pth` (a truncated SB3 zip can still open on the nested torch archive's directory);
  the sidecar unpickles (`pickle.load`). It is generic for any SB3 zip with a pkl. **[measured]** on the committed
  `FS/stage1/runs/2026-11-08/75-M-H-10/ppo/42/checkpoints/model_2506752.zip` with `vecnormalize_2506752.pkl` it
  returns `None`.
- `newest_intact_periodic_pair` assumes SB3 `CheckpointCallback` names: glob `f"{name_prefix}_*_steps.zip"`,
  sidecar `f"{name_prefix}_vecnormalize_{steps}_steps.pkl"` (:259-265). Ordered by the step count parsed from the
  name, "never the mtime, which on a Drive mount is upload order" (:223-224). **It cannot walk the form study's
  `model_<n>.zip` / `vecnormalize_<n>.pkl`** (`FS/designlab/stage1/train.py:553-554`).
- RESUME cell (cell 18): `RESUME_STAGE = None` (c18:L2, raw 975). When set it
  1. refuses a node off the chain or covered by `RETRAIN_FROM`;
  2. trains nothing if the node holds a `gate_verdict.json` or an intact final pair (L46-55);
  3. `pair_res, skipped_res = newest_intact_periodic_pair(model_dir_res, label_res)` (L64, raw 1037), prints
     `WARNING: skipping <reason>` per skip, raises if none survives;
  4. prints `Newest intact periodic checkpoint: ...`, `Matched VecNormalize: ...`, `Checkpoint steps: N of B —
     remaining budget: R` (L85-89);
  5. refuses an early stop (final pair broken and more than one cadence short) and a spent budget without a final pair;
  6. `refuse_write_into_complete_run`, `refuse_trunk_other_than_recorded`, then `train_stage(stage=...,
     timesteps=remaining_res, load_path=str(ckpt_res), run_dir=RUN_DIR, vecnorm_path=str(vecnorm_ckpt_res),
     task_load_mode="resume_same_stage", label=..., evaluate=False)` (L133-142, raw 1106-1115).
- The chain loop also runs `checkpoint_pair_problem` on the final pair before its JUDGE branch (c20:L196, raw 1334).
- Known scaffold defect a steps sheet had to work around: the cell measures the shortfall against the notebook's
  default 100,000-step cadence (`inspect.signature(train_stage).parameters["save_freq"].default`, c18:L95), not a
  branch's own `checkpoint_freq` (`docs/investigations/stance_2026_10/stance3/colab_steps.txt:151-154`).

### 1.6 How training is launched

`train_stage(stage, timesteps, *, load_path=None, run_dir, vecnorm_path=None, task_load_mode, parent_run_id=None,
label=None, eval_freq=50_000, save_freq=100_000, evaluate=True)` (c15:L64-77, raw 684-697) calls, **in the kernel
process**:

```python
model = train_base.train(SPECIES_CFG, STAGE_CONFIGS, stage, timesteps, n_envs=N_ENVS, seed=SEED,
    load_path=load_path, vecnorm_path=vecnorm_path, eval_freq=eval_freq, save_freq=save_freq, verbose=VERBOSE,
    algorithm=ALGORITHM.lower(), output_dir=str(stage_dir), task_load_mode=task_load_mode,
    parent_run_id=parent_run_id, label=label, report_metrics=False, save_on_interrupt=False,
    k3_panel=k3_panel_config())                                           # c15:L139-159 (raw 759-779)
```

- No subprocess, no queue, no launcher: one node at a time, blocking the cell. `use_subproc` is never passed, so PPO
  steps all envs in one process with `DummyVecEnv`; no `device` is passed, so SB3's "auto" puts the MLP on the GPU
  [doc: `research_pipeline.md:242-245`, citing `train_base.py:437-440, 916-919, 1946`].
- `train_stage` then calls `evaluate_stage_checkpoints(...)` unless `evaluate=False`; returns `(model, handoff_stem,
  final_model_path, stage_dir, handoff_vecnorm, stage_results)`.
- The scaffold loads no archive itself (pinned by test, section 9); every load is inside the library through
  `load_sb3_model`.

---

## 2. `policy_loading.py`: the safe loader

`REPO/environments/shared/policy_loading.py` (1,039 lines). At import time it imports the standard library only
(:89-99); SB3, gymnasium, numpy and project modules are imported inside functions.

```python
SCHEDULE_MEMBERS = {"ppo": ("learning_rate", "lr_schedule", "clip_range", "clip_range_vf"),
                    "sac": ("learning_rate", "lr_schedule")}                                   # :113-116
INFERENCE_SCHEDULE_DEFAULTS = {"learning_rate": 0.0, "clip_range": 0.2, "clip_range_vf": None} # :123-127
class PolicyLoadError(RuntimeError)                                                            # :148
_PERIODIC_CHECKPOINT_RE = re.compile(r"(.+)_(\d+)_steps$")                                     # :174

@dataclasses.dataclass(frozen=True)
class SB3ArchiveInspection:            # :370   path: Path; saved_python: tuple[int,int] | None;
                                       #        serialized_members: frozenset[str]; bytecode_members: frozenset[str]
    saved_python_text -> str           # :382   "3.13" or "unknown"
    cross_interpreter -> bool | None   # :386   saved (major, minor) != sys.version_info[:2]; None when unrecorded

def inspect_sb3_archive(model_path) -> SB3ArchiveInspection                                    # :393
def schedule_custom_objects(inspection, algorithm, *, custom_objects=None, training_kwargs=None) -> dict   # :454
def load_sb3_model(model_path, *, algorithm=None, env=None, device=None, custom_objects=None, **load_kwargs) -> Any  # :501
def load_vecnormalize(path, venv, *, plant_identity=None, allow_legacy_plant=False, vec_normalize_cls=None) -> Any   # :565
def load_sb3_checkpoint(model_path, vecnorm_path, env_factory, *, guess_sidecar=True, allow_unnormalized=False,
                        plant_identity=None, allow_legacy_plant=False, reseed_command_slice=False)
                        -> tuple[model, normalizer, resolved_vecnorm_path]                     # :634
def sb3_archive_load_preflight(model_id, root_reference, *, trunk_dir) -> SB3ArchiveInspection # :785
def resolve_vecnorm_path(model_path, vecnorm_arg, allow_unnormalized) -> str | None            # :245
```

`load_sb3_model` behaviour (:529-562):

1. `algorithm` is `"ppo"`/`"sac"`, an SB3 class, or `None` to sniff: `_checkpoint_algorithm` reads the archive's `data`
   JSON and calls it PPO when it holds `clip_range` and `n_epochs`, SAC when `target_entropy` and
   `replay_buffer_class`; neither or both raises `PolicyLoadError` (:272-294).
2. `inspect_sb3_archive` reads `data` and `system_info.txt` **without unpickling**; the saving Python is parsed with
   `^-?\s*Python:\s*(\d+)\.(\d+)` (:145), minor version only. A member counts as bytecode when its pickle stream
   references cloudpickle's by-value reducers (:132-144, :356-366).
3. `schedule_custom_objects` replaces each schedule member that is bytecode (or all of them when the archive could
   not be read): from the caller's `custom_objects`, else the caller's keyword (`learning_rate`, `clip_range`,
   `clip_range_vf`; `lr_schedule` follows `learning_rate`, wrapped in `_ConstantSchedule` if a float), else
   `INFERENCE_SCHEDULE_DEFAULTS` (:484-498).
4. **Fail-closed:** bytecode in any member other than the replaced ones, from another Python minor version, raises
   `PolicyLoadError` ("... refusing to load it. Re-save the archive under the saving interpreter ...", :539-545).
   Unrecorded saving Python: a warning, and the load proceeds (:546-551).
5. `return alg_cls.load(model_path, env=env, custom_objects=custom, **load_kwargs)` (:562); `device` is passed only
   when not `None`.

A training (resume) load therefore passes the schedules as keywords: "A training load passes the stage config's own
values as keyword arguments, and SB3 applies those over the archive" (:121-122). An inference load needs none.

`sb3_archive_load_preflight` (:819-842): with `trunk_dir=None` it never reads `model_id` or `root_reference`; it
saves a throwaway 2-obs/1-action PPO (`_save_throwaway_ppo`, :761-782) into a temp directory, inspects it, prints
and **flushes** one line before the load, loads it with `load_sb3_model(archive, device="cpu")`, then prints the
pass line. Exact texts (:832-841):

```
SB3 archive load preflight: loading {source} (saved by Python {X.Y}; this runtime is Python {X.Y}; bytecode members: {...|none}) ... a kernel death HERE means this image cannot load SB3 archives
SB3 archive load preflight passed: archives load back on this runtime through load_sb3_model.
```

**[measured] The form study's committed archive under this loader's inspector** (`model_2506752.zip`, R3/42):
`saved_python (3, 13)`; serialized members `_last_episode_starts, _last_obs, _last_original_obs, action_space,
clip_range, ep_info_buffer, ep_success_buffer, learning_rate, lr_schedule, observation_space, policy_class,
rollout_buffer_class`; **bytecode members `learning_rate`, `lr_schedule`**; `clip_range` and `n_epochs` present, so
the sniff reads PPO. Its `system_info.txt`: Python 3.13.18, Stable-Baselines3 2.9.0, PyTorch 2.9.0+cpu, Numpy
2.5.4, Cloudpickle 3.1.2, Gymnasium 1.3.0. `policy_class` is stored by reference, so
`designlab.stage1.policy` must be importable in the loading process.

Picklable schedule pattern: `REPO/environments/shared/curriculum/schedules.py` `_ConstantSchedule(value)` (:39),
`LinearSchedule(initial, final)` (:57-79, "A module-level class rather than a closure on purpose"), `CosineSchedule`.
The form study's is a closure: `FS/designlab/stage1/policy.py:144-151` `linear_schedule(start, end)` returning the
nested `schedule`, used at `policy.py:162` (`learning_rate=linear_schedule(LR_START, LR_END)`).

Atomic writers: `REPO/environments/shared/file_io.py` `atomic_copy` (:43), `atomic_write_text` (:66),
`atomic_write_json(path, value, *, indent=2, sort_keys=False, ensure_ascii=True, allow_nan=True, default=None,
trailing_newline=True, encoding="utf-8") -> Path` (:94), `atomic_write_csv` (:142), `atomic_savez` (:165). The form
study's own: `train._write_json` (`FS/designlab/stage1/train.py:179-183`: `.tmp` then `os.replace`, `indent=1`).

---

## 3. `docs/KNOWN_ISSUES.md`: items that bear on Stage 2

1. **SB3 archives are bound to the interpreter that saved them** (`:595-649`, MEDIUM operational).
   - Mechanism: `learning_rate`, `lr_schedule`, `clip_range` are cloudpickled; a closure is pickled by value with its
     code object; `PPO.load` executes it (`_setup_model` -> `lr_schedule(1)`). "Bytecode compiled by Python 3.12 run
     by 3.13, or the reverse, segfaults the process with no Python traceback" (:603-604). Torch is not the cause
     (reproduced with torch held at 2.14, :605-607). "merely unpickling the data does not crash, calling the schedule
     does" (:610-611).
   - Incident: "Colab's L4 image moved from Python 3.12.13 / numpy 2.0.2 / jax 0.7.2 (... 2026-08-15) to Python
     3.13.15 / numpy 2.1.3 / jax 0.11.1 (already the image of the runs of 2026-09-14/15 ...)" (:611-615). The kernel
     died with `AsyncIOLoopKernelRestarter: restarting kernel` (:617).
   - Fix (2026-09-19): every archive load in the repository and both notebooks goes through `load_sb3_model`;
     archives saved since carry no bytecode; the notebook preflight loads a real archive (:622-634).
   - What stays true: "any path outside the loader (an ad-hoc `PPO.load` in a notebook cell, the SB3 CLI, a
     third-party tool, an older checkout) still dies on a foreign image with no traceback ... and the image will move
     again" (:638-642). `provenance.json` `python_version` names the interpreter a run's archives belong to.
   - The segfault is across Python **minor** versions. Colab 3.13.15 against the study's 3.13.18 is the same minor.
2. **Notebook pins** (`:2318-2321`): "The SB3 notebook pins the plant compiler (`mujoco==3.10.0`) and
   `stable-baselines3[extra]==2.9.0`, but torch is unpinned and the `pyproject.toml` extras stay ranges; pin complete
   lockfiles for reproducible training."
3. **Run-level records go stale when a run is continued in a later session** (`:650-680`): per-node files are
   written when a node is judged; run-level `summary.json`, `provenance.json`, `artifact_manifest.json` may lag.
4. **A re-entry that only reuses nodes stops at the bundle-verification cell** (`:681-713`): `initialize_result_bundle`
   appends a session to a `partial` run's `provenance.json`, the manifest then mismatches, `validate_result_bundle`
   raises and "Run all" stops before the auto-disconnect. Remedy: run the auto-disconnect cell by hand. Pattern to
   avoid: a verification cell that can raise between the work and the disconnect.
5. **A same-stage resume inside a stage's entry window drops the ramp** (`:1600-1619`, MEDIUM) and **does not
   continue `diagnostics.npz`** (`:1931-1947`, LOW): resume is not bit-for-bit the uninterrupted run in the main
   pipeline either.
6. **The final-checkpoint replay runs on raw observations when the final pair lost its sidecar** (`:1909-1930`): a
   policy without its normalisation statistics is a different policy; never load a zip without its pkl.
7. **Species `requirements.txt` take any MuJoCo from 3.0.0** (`:1976-1992`): a newer MuJoCo stops training in the
   plant-contract check. The form study pins 3.10.0 too.
8. **SAC in the notebook uses 4 envs where the CLI uses 8** (`:2322-2331`): not relevant to PPO runs.

Resume across **library** versions: KNOWN_ISSUES has no entry of its own beyond item 1 and the pins note. The
refusal lives in the form study's trainer (section 6.2).

---

## 4. `FS/stage2/research/research_pipeline.md` (10 Nov 2026, read-only research)

### 4.1 Recommendation: option (b)

"A separate Stage 2 notebook, built on the `sb3_training.ipynb` scaffold, can run the form study's own
`designlab.stage1.train`. It keeps the plant, the records and the gates intact. ... **Recommend (b).**" (:25-28;
comparison table :208-215; recommendation :217-222). Option (a), a main-pipeline embodiment, is "weeks of shared-code
and plant work" (:142).

Carries over unchanged, by copy or import (:148-161):

| Piece | Where |
|---|---|
| Setup: `REPO_REF`, pinned installs, clone at a commit, refusal to switch code after import | cell 3, lines 6-58 |
| Drive mount and run directory | cell 8, lines 8-21 |
| Auto-disconnect with a Drive flush | `environments/shared/notebook_runtime.py:15-43` |
| Checkpoint-pair integrity check | `environments/shared/curriculum/checkpoints.py:174-208`; `newest_intact_periodic_pair` (:210-235) "needs a 10-line walker for `model_<n>.zip`" |
| Safe archive loading | `environments/shared/policy_loading.py:500-561` (today :501-562) |
| Atomic JSON writes | `environments/shared/file_io.py:94` |

"What must be built (small, all inside FS)" (:175-200):

1. **The inputs.** MJCF models and reference/transition arrays are git-ignored, "about 20 MB per body plus a 3.5 MB
   model case". Regenerate on Colab (`python -m designlab.stage1 prepare`, about 0.21 core-h per body) and check
   against records, or ship through Drive with a sha256 manifest. "Shipping is preferred: copying the
   `round2/refine/*/models` files keeps the panel record keys equal to the main checkout's". `train.env_configs`
   reads references from `<out>/<body>/`; resume compares recorded env configs "`run_dir` included ..., so the path
   must stay the same in every session".
2. **Resume integrity.** `train.checkpoints()` checks only that both files exist; add the integrity walk.
3. **Archive portability.** The closure schedule and the plain `PPO.load` at `train.py:497` and
   `policy_rows.py:601`. "Use a schedule class ... or `load_sb3_model`. The change moves the Stage 1 sources hash,
   which is acceptable for a new round."
4. **A launcher cell.** "It runs one subprocess per job and writes logs to Drive. A monitor rewrites STATUS.json.
   When every job has ended it writes `DONE.json`, flushes Drive and disconnects."
5. **The notebook's location.** "It must **not** go in `notebooks/`" (section 9 below). "Put the notebook under FS,
   for example `FS/stage2/`."

Needed in either option: the expressive head's plant models; "The env's widths are hard-coded to 14 joints
(`FS/designlab/stage1/env.py:137-142, 284-286`)" (:202-204; verified today: `ACTOR_OBS` / `CRITIC_OBS` at `env.py:138-142`, `ACTOR_DIM`, `OBS_DIM`, `ACTION_DIM = 15` at `:284-286`).

### 4.2 Measured rates

Form study on this CPU (:228-238):

| Measurement | Value | Label |
|---|---|---|
| R3 job, 4 envs in one process (hard, hard, thin carpet, low pile), with PPO | 199 / 204 env steps/s (idle / load 4.2) | measured, `FS/STAGE1_PLAN.md:321-322` |
| One env alone, no PPO | hard 358, carpet 234, low pile 135 env steps/s | measured, `STAGE1_PLAN.md:323` |
| R3, CPU-normalised | 195-214 (long windows) / 188-209 (first rollouts) per CPU-second | measured, `run.json` `throughput` |
| R4, CPU-normalised | 251-301 per CPU-second | measured |
| Nine jobs sharing 4 cores, G6a rate | R3 74-83, R4 100-120 per job | measured, `run.json` `g6.G6a_rate` |
| Cost split | MuJoCo about 14-20 % of an env step on hard floor; PPO about 12-14 % of a job | derived |
| Stage 1 reached | budget 4M; R3 2.5M (75 cm), 1.75-2.02M (60 cm); R4 1.25-1.51M | read |

Main pipeline on Colab, T. rex stance, PPO, `N_ENVS = 4`, one process (:247-251): L4 262.8-276.5 steps/s (stance4)
and 230.9-245.2 (stance5) [measured]; CPU runtime about 290-306 [derived from 12M in 10.9-11.5 h]; session container
CPU smoke runs 314.6-362 [measured]. "Colab's per-process rate is 0.75-0.79x local on the L4 runtime and 0.82-0.87x
on the CPU runtime [derived]" (:253-254). **"The bottleneck is the CPU, not the GPU"** (:33, :281).

Projection for a 24 h runtime, "[derived; to be measured]" (:262-270): base 150-185 env steps/s per process.

| Setup on one runtime | Env steps/s | Per 24 h |
|---|---|---|
| One job as Stage 1 (`--vec dummy`, 4 envs, floor spread) | about 150-185 | about 13-16M |
| One job, `--vec subproc`, one env per vCPU, same floor spread | 4 vCPUs about 360-420; 8 about 730-840 | about 31-36M; 63-73M |
| As above, hard floor and thin carpet only | 4 vCPUs about 630-730; 8 about 1,260-1,460 | about 55-63M; 110-125M |

The plan's budget rule needs about 560 env steps/s to reach 40M in 20 h (`FS/STAGE2_PLAN.md:240-241`).

### 4.3 Colab runtime facts and unknowns

Facts the documents state:
- Session cap about 24 h; the scaffold resumes from periodic checkpoints (cell 17, line 3; `research_pipeline.md:255`).
- The maintainer has run three runtimes at once (`stance5/colab_steps.txt:4`; `research_pipeline.md:257`). P30: the
  owner confirmed 24 h sessions, background execution, High-RAM CPU runtimes, 3 runtimes at once
  (`FS/STAGE2_PLAN.md:17-18`).
- Colab opens any path of a pushed branch with the URL form
  `https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<branch>/<path>` (`stance3/colab_steps.txt:25-27`).
- An L4 runtime trained no faster than a CPU runtime (:283-284).
- Colab's torch is a CUDA build, so `matches_ppo_venv` is false; "recorded only" (:380-381).
- The lead reads JSON, CSV, text and Markdown on Drive directly; binaries (`.zip`, `.pkl`, `.npz`, `.mp4`) after a
  download (:357-359). A stale STATUS.json `modifiedTime` shows a dead runtime (:360).

Unknowns / unverified:
- **vCPU count and RAM** of the owner's runtime (:272-275, :376). The documents disagree on the L4 runtime:
  `research_pipeline.md:274` says Colab Enterprise's default L4 is `g2-standard-4`; `STAGE2_PLAN.md:238` says
  "High-RAM CPU first (8 vCPUs by a third-party table of October 2024, unverified; standard CPU 2, L4 12)".
- RAM per worker "was not measured"; vCPUs "may be hyperthreads" (:272).
- Colab's current Python patch version, numpy, scipy and torch versions (section 8).
- Rendering on a CPU runtime: "untested until the smoke" (`STAGE2_PLAN.md:307`).
- Moving binaries through the Drive connector: untested (`FS/LESSONS_LEARNED.md:470-481`): upload by
  `create_file` with `base64Content` set to `sha256:<hex>:<absolute path>`, size limit unknown; `download_file_content`
  returns base64 inline, a 2 MB result reaching disk is untested.
- Throughput in 4.2 above: "Treat them as planning figures until `train.py`'s own rate record (`:602-619`) reports
  from Colab" (:384-385).

### 4.4 Proposed Drive layout (`research_pipeline.md:325-339`, quoted)

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

No key schema is given for `MANIFEST.json`, `SESSION_<k>.json`, `STATUS.json`, `DONE.json` or `HALT.json` beyond the
phrases above: **field names not found**. `STAGE2_PLAN.md:234-236` repeats the list: "SESSION files (runtime type,
vCPUs, memory, versions, commit), STATUS.json, `run.json`, kept checkpoints, logs, a clip per check, DONE or HALT".

Where verdicts are judged (:343-353): G6a/b/c inside the job, written to `run.json`; panels and the margin ledger on
the lead's machine from the downloaded judged checkpoint (`STAGE1_POLICY_RUNS` points at the run,
`FS/designlab/stage1/policy_rows.py:71`), "keeps Colab's floating-point differences ... out of the verdicts".

### 4.5 Kill-and-resume

The research note gives no numbered procedure. What it and the plan state:
- Resume loads the newest pair; add the integrity walk first (`research_pipeline.md:186-187`).
- The run path must be the same in every session (:184-185).
- A resume is refused when library versions change (:377-379): "plan budgets that end inside about 24 h, or accept
  that a run may need a fresh start".
- Order checkpoints by step count in the name, never by mtime (:360-361).
- Validate locally first (:367-372): a short smoke run per mode confirming G6a "identical", a rate record, and
  "checkpoint, kill and `--resume`"; `--vec subproc` scaling at 2, 3, 4 workers with RSS; a fresh-clone dry run of the
  inputs bundle; an archive saved under the Colab stack loads locally through the fixed loader; the notebook end to
  end with a tiny budget.
- Plan: "after a disconnect, the same with RESUME on" (`STAGE2_PLAN.md:232`); "Checkpoints: each 2.5M check's and the
  newest two, written locally, copied to Drive in pairs" (`:150`); "8 envs in every run and session ... (`train.py`
  refuses a resume with other settings); the vCPUs set only the worker processes; a queue starts each next job"
  (`:143-144`).

### 4.6 Bundle contents and sizes

| Source | Statement |
|---|---|
| `research_pipeline.md:177-179` | models plus arrays "about 20 MB per body plus a 3.5 MB model case" |
| `research_pipeline.md:336` | a checkpoint "about 2.1 MB per model" |
| `FS/LESSONS_LEARNED.md:482` | "The inputs for one body and its `-X` come to about 40-65 MB, and for all seven registry bodies about 185 MB." |
| `FS/tools/README.md:57-61` | `restore_inputs.py` groups: `picks-models` 6 files, `picks-arrays` 84, `k` 60, `x` 60, `look` 73 (about 50 MB); 283 files in all (`:95`) |
| `FS/tools/README.md:124-136` | the six picks' model files pack to "a 2.96 MB archive" (`tar cf picks_models.tar ...`) |
| [measured] on disk now (the restore job is still running, so this is partial) | `stage1/runs/2026-11-08/<body>/refs` 1.6 MB each; `transitions` 16 MB each; `round2/refine/narrow_carpet_60/models` 2.0 MB; `feet_75/models` 956 KB; `model_2506752.zip` 2,123,104 bytes, `vecnormalize_2506752.pkl` 5,759 bytes |

Existing sha256 manifests that are precedents:
- `FS/stage1/runs/2026-11-08/stage1_inputs.json`: `{"rule": str, "bodies": [3 ids], "cases": {body: case},
  "files": {path relative to FS: sha256 hex}}` (bare hex, no `sha256:` prefix), written by
  `tools/restore_inputs.py manifest` (`restore_inputs.py:468-487`, `indent=1` plus newline).
- `FS/stage1/runs/2026-11-08/judged_checkpoints.json`: `{"what": str, "environment": str, "files": [{"path", "mode",
  "body", "seed", "steps", "bytes", "sha256"}]}`.
- `FS/stage1/runs/2026-11-10-speed/arm_k_inputs.json`, `FS/stage2/x/x_inputs.json`, `FS/stage2/look/look_refs.json`
  (per-group lists the restore tool reads; structure not opened here).
- `FS/r0_posture_study_2026-11-07/SHA256SUMS` with its own `.gitattributes` (`* -text`).

`tools/restore_inputs.py` already does the bundle's checking (`FS/tools/restore_inputs.py`):
`check [--group G ...]` (:394), `restore [--group G ...] [--procs 2] [--scratch DIR]` (:725),
`copy-from DIR [--group G ...]` (:783; DIR is another checkout or "a directory holding the files at their study
paths"; each file is copied through a `.part-` name and put in place only if its sha256 is the listed one, :171-179),
`not-restored` (:452), `manifest [--from DIR]` (:468). `check` ends with the line `all here with their listed sha256`
or `not all here: python tools/restore_inputs.py restore (or copy-from DIR)` (:417-418). Full restore measured at
151 min on 4 cores with `--procs 2` (`tools/README.md:93-95`); `copy-from` of all 283 files 2.4 s. One-CPU processes
cannot reproduce the picks' ledger JSON (`SIZE_BLAS_THREADS = 2`, `restore_inputs.py:115`; `tools/README.md:114-123`).
Rebuilding on Colab's numpy and BLAS is "unproven" (`LESSONS_LEARNED.md:479-481`).

---

## 5. The plan's notebook requirements (`FS/STAGE2_PLAN.md`)

S2-5 (`:134-150`), quoted in substance:
- at `stage2/stage2_training.ipynb`, "outside `notebooks/`, which main-repository tests scan";
- "reuses the scaffold's setup cell, also pinning numpy, scipy and torch by version, and its Drive mount, flushed
  disconnect, checkpoint-pair check and safe loader";
- "runs `designlab/stage1/train.py`, extended with R5 and R6, a picklable schedule, the pair check on resume, a
  flagged resume across library versions (as `--allow-source-change`), the G8b fix inside the running job, and
  fine-tunes from another run's checkpoint and normalisation statistics in a new run directory";
- inputs: "the lead's sha256-manifested Drive bundle, as rebuilt models change every record key ...; the smoke
  rebuilds them once, as a check";
- 8 envs in every run and session; a queue starts each next job;
- "Clips render in a guarded subprocess (the scaffold forces NVIDIA's EGL; the form study uses OSMesa)";
- "Training processes run with MUJOCO_GL=disable. With libOSMesa loaded ..., torch segfaults while building a
  stable-baselines3 policy ... Stage 1's `train.py` relies on that default, so the R5 and R6 entry points set it";
- checkpoints: each 2.5M check's and the newest two, written locally, copied to Drive in pairs.

Operation (`:230-236`): "The notebook runs a queue of jobs (body, head, seed, mode) per runtime. The owner opens it
from the branch, sets REPO_REF, the queue and RUN_ID, runs all and checks four printed lines (commit, inputs' hashes,
G8a, first rate); after a disconnect, the same with RESUME on; a steps sheet guides it, as for the stance runs
(`docs/investigations/stance_2026_10/stance3/colab_steps.txt`). It saves to
`MyDrive/mesozoic-labs/stage2/runs/<run_id>/`, outside `logs/`".

Day-one smoke (`:227-229`): vCPUs and memory per worker, the rate at 8 envs, a test clip, a kill and resume on Drive,
an archive that loads here, the inputs rebuilt once against the bundle, a launch's setup time.

Budget and stop rules the notebook must serve (`:240-264`): 40M env steps per run, or fewer, rounded down to 5M,
never under 15M without the owner; monitor every 2.5M steps (3 episodes each at 10 ms of the nominal and 0.40 m/s
walks, a 90 degree turn, the stop, two nudges); "G8a fails: no run"; G8b at 5M with one fix applied in the running
job; from 20M, three checks without a gain stop the run; cap 220 runtime-hours or 8 days.

### 5.1 The four printed lines

The documents name them **by topic only**: "four printed lines (commit, inputs' hashes, G8a, first rate)"
(`STAGE2_PLAN.md:231`; again "each checked by four printed lines", `:276`). **The exact text of the four lines is not
found** in any document or code. The nearest existing texts:

| Line | Nearest existing text | Where |
|---|---|---|
| commit | `Repository ref: {REPO_REF}; commit: {requested_commit}` | scaffold c3:L53 (raw 96); the steps sheets quote it with the expected sha (`stance3/colab_steps.txt:59-62`) |
| inputs' hashes | `all here with their listed sha256` (last-but-one line of `restore_inputs.py check`), after a per-group table `group files matching missing wrong records` | `FS/tools/restore_inputs.py:408-418`. No line prints a single bundle digest: **not found** |
| G8a | G8a does not exist in code. Stage 1's analogue: log line `{body} seed {seed}: G6a zero-residual check` and `run.json` `g6.G6a_zero_residual.identical` | `FS/designlab/stage1/train.py:520-523`; the CLI prints `{body} {mode} seed {seed}: {n} steps ({status}); G6: G6a_zero_residual pass; ...` only when the job **ends** (`FS/designlab/stage1/__main__.py:388-390`) |
| first rate | no print at all. The rate is written to `run.json` `throughput[0]` (`first_rollouts: true`, `env_steps_per_wall_s`, `env_steps_per_cpu_s`, `cores`, `env_processes`) and `g6.G6a_rate` once `G6A_MEASURE_STEPS = 16_384` steps have run | `train.py:602-619`, `:88`. `train.train` calls `log(...)` at `:520`, `:571`, `:586` only |

So a notebook monitor has to read `run.json` to print the first rate, and `G6A_ENV_STEPS_PER_S = 600`
(`train.py:87`) would mark `g6.G6a_rate.pass_` false at the projected 150-185 env steps/s per process.

---

## 6. What the notebook launches today: `FS/designlab/stage1/train.py`

(Another map covers `train.py`, `env.py`, `policy.py` in depth. Only what the notebook, bundle and resume touch.)

### 6.1 CLI

`python -m designlab.stage1 train` (`FS/designlab/stage1/__main__.py:528-552`), run from `FS`:
`--body` (choices `sorted(BODIES)`, required), `--seed` (required), `--mode {R3,R4}` (default R3), `--steps`
(default `train.BUDGET_ASKED` = 4,000,000), `--resume`, `--n-envs` (default design 2's 4), `--vec {dummy,subproc}`
(default dummy), `--checkpoint-every` (250,000), `--no-g6`, `--log-std-init`, `--allow-untuned`,
`--allow-source-change`, `--fix-task-weights`, `--fix-reason`, `--tune-json`, `--out` (default
`stage1/runs/2026-11-08`). Also `judge [--mode] [--budget] [--final] [--out]` and `noise`.

`train.train(body, seed, mode="R3", steps=BUDGET_STEPS, out=None, resume=False, n_envs=None, vec="dummy",
checkpoint_every=CHECKPOINT_EVERY, g6=True, config=None, log_std_init=None, ppo=None, allow_untuned=False,
allow_source_change=False, fix=None, fix_reason=None, log=print) -> dict` (`train.py:377-381`).

### 6.2 Records, checkpoints, resume

- Run directory: `run_dir_of(out, body, mode, seed) = out/body/("ppo" if mode == "R3" else "policy_only")/str(seed)`
  (`train.py:158-159`). Inputs are read from `out/body/` (`env_configs`, `train.py:289-302`:
  `base.setdefault("run_dir", str(Path(out ...) / body))`).
- `run.json` (schema 3, `train.py:514-517`), top-level keys: `schema, body, seed, mode, settings, env_configs, gains,
  final_rule, judged_rule, g6_rule, seed_rule, budget_rule, sessions[], checkpoints[], throughput[], g6{}, status,
  budget, budget_asked`, later `num_timesteps, judged_checkpoint, fix, stopped`. `status` is `running`, then `done`,
  `partial` or `stopped_g6b` (`:639`); a killed job stays `running` (`LESSONS_LEARNED.md:483-484`).
- Session entry (`:481-485`, `:533`): `index, started_at, provenance, versions, ppo_venv, matches_ppo_venv, vec,
  env_processes, n_envs, torch_threads, solve2, gains, env, other_envs, floors, seed_base, env_seeds`, on resume
  `resumed_from` and optionally `source_change`, `fix`; `episode_streams` written with every checkpoint; `ended_at`.
- `versions = _versions()` (`:162-172`): `python` (`platform.python_version()`, **patch included**), `numpy`,
  `mujoco`, `scipy`, `torch` (`torch.__version__`, with its `+cpu` / `+cu...` suffix), `stable_baselines3`,
  `gymnasium`. `PPO_VENV = {"torch": "2.9.0+cpu", "stable_baselines3": "2.9.0", "gymnasium": "1.3.0"}` (`:109`).
- `provenance` is `plant.provenance()` (`FS/designlab/stage1/plant.py:294-316`): `versions` (from
  `importlib.metadata`), `git` = `{commit, dirty, dirty_paths}` from `git -C FS status --porcelain
  --untracked-files=all -- .` (so **any untracked, non-ignored file under FS marks the session dirty**), and
  `stage1_sources` = `{sha256, rule, files}`.
- **The Stage 1 sources hash** (`plant.py:262-275`): `files = {p.name: p.read_bytes() for p in
  SOURCES.glob("*.py")}`, sha256 over each name, NUL, bytes, NUL in sorted order. It covers every
  `designlab/stage1/*.py`; **adding a new file to that directory changes it too**.
- Checkpoints: `checkpoints/model_<n>.zip` and `vecnormalize_<n>.pkl`, saved at the first rollout start at or past
  each multiple of `checkpoint_every` and after the last update (`:551-566`, `:629-630`). `checkpoints(run_dir)`
  returns the step counts whose two files **exist** (`:186-194`); no integrity check.
- Resume (`:426-498`). Refusals, each a `SystemExit`:
  - no `--resume` on an existing `run.json` (`:426-427`); nothing to resume (`:430`); a stopped run (`:431-432`);
  - `doc["settings"] != settings` (`:433-434`; settings hold `n_envs`, `checkpoint_every`, PPO overrides, `config`);
  - `doc["env_configs"]` differ (`:435-436`; **they hold the absolute `run_dir` string**);
  - tracker gains changed (`:451-456`);
  - **`first != versions`: "the libraries differ from the run's first session (...); refusing to resume"
    (`:457-460`). There is no flag to override it today**;
  - Stage 1 sources differ, unless `--allow-source-change` (`:461-465`), then flagged in the session (`:491-494`);
  - no complete checkpoint (`:487-490`).
  Then `VecNormalize.load(...)` and a **bare** `PPO.load(str(ck / f"model_{last}.zip"), env=vecnorm, device="cpu")`
  (`:496-497`). The other bare load is `FS/designlab/stage1/policy_rows.py:601`.
- Each session draws its own env seeds (`env_seeds`, `:280`), so "a resumed run is not the uninterrupted one bit for
  bit" (`:31-33`).
- CPU only: `torch.set_num_threads(1)` (`:468`), `device="cpu"` (`:497`, `:513`). `--vec subproc` uses
  `SubprocVecEnv(fns, start_method="spawn")` (`:476`).
- Policy rows refuse other library versions: `policy_rows.check_versions` (`policy_rows.py:412-419`) compares
  `importlib.metadata` versions with the run's recorded ones; `train.py` records `torch.__version__`. A run trained
  under PyPI's CUDA torch is refused by the policy rows even in its own venv (`LESSONS_LEARNED.md:580-589`).

### 6.3 The one import from the main repository

`FS/designlab/stage1/env.py:170-179` imports `environments.shared.direction_commands.gaussian_tracking_reward`,
adding `STUDY.parents[1]` to `sys.path` on `ImportError`. "A Colab bundle therefore needs the repository checkout,
not the study alone" (`LESSONS_LEARNED.md:570-572`).

`REPO/environments/shared/direction_commands.py` (skimmed): no simulator or training dependency; imports numpy and
`environments.shared.command_frame` (`COMMAND_COMPONENTS, COMMAND_RANGE, COMMAND_WIDTH`, :22). Units: radians,
m/s, rad/s, seconds (:5-6). Public names: `wrap_angle` (:42), `DirectionCommandConfig` (:48, defaults cruise
1.05 m/s, heading change at most 30 degrees every 4 s), `DirectionCommandState` (:117), `DirectionCommandController`
(:165; `reset(rng, initial_heading)`, `update(time_s, current_heading)`, `set_target(heading, speed, *, time_s=None)`),
`body_frame_velocity` (:343), `tracking_metrics` (:360), and

```python
def gaussian_tracking_reward(velocity_error: float, yaw_rate_error: float, *,
                             velocity_sigma: float = 0.25, yaw_rate_sigma: float = 0.15) -> float   # :421
    # exp(-((ev/sv)^2 + (ew/sw)^2)); errors are non-negative magnitudes, else ValueError
```

`import environments` registers every model with Gymnasium when gymnasium is installed
(`REPO/environments/__init__.py:25-34`); `environments/shared/__init__.py` re-exports nothing.

---

## 7. The steps-sheet precedent: `docs/investigations/stance_2026_10/stance3/colab_steps.txt`

Plain text, 160 lines, wrapped near 115 columns, addressed to the owner ("tell me"). Structure:

1. **Title and a line of `=`** (:1-2), then one sentence on what runs ("Three runs, one Colab runtime each, all
   started at the same time.").
2. **Runtime type** with the reason and the date of the owner's choice (:6-7).
3. **The runs**, one bullet each: branch, full 40-hex commit, SEED, RUN_ID (:9-17).
4. **The pre-fixed criteria**, by file name and sha256, with the check command (`sha256sum criteria.txt`) (:19-21).
5. **"FOR EACH RUN"**, numbered steps (:23-86):
   - 1: the Colab URL per run, a fallback URL on main, "File > Save a copy in Drive", named per run;
   - 2: Runtime > Change runtime type;
   - 3: `REPO_REF = "..."`;
   - 4: every configuration variable with its value, aligned, per-run differences in parentheses; "Leave everything
     else as it is.";
   - 5: Runtime > Run all, and allow Drive access;
   - 6: **"Check the output for:"** exact quoted lines with expected values, each with a stop rule ("A different
     commit: stop and tell me"): the `Repository ref: <branch>; commit: <sha>` line, the `Run directory: ... (new run:
     RUN_ID names no existing run)` line, the chain line, the zero-action baseline row with its expected number and
     what a wrong number means, `Timesteps: 12,000,000`, the first `Eval num_timesteps=...` line "within about 15
     minutes";
   - 7: how long to leave it, what the end looks like for PASS and FAIL, "the run is FINISHED: do not resume it";
   - 8: what to do if the last cell did not run.
6. **"IF COLAB DISCONNECTS"** (:88-100): first a file test that tells a finished run from a disconnect
   (`.../01_stance/gate_verdict.json` exists), then the resume step (same copy, fresh runtime, same settings and
   RUN_ID, `RESUME_STAGE = "stance"`, Run all), and the restart rule under `"<RUN_ID>_b"` when no checkpoint exists.
7. **"WHEN TO TELL ME"** (:102-106): milestones to report; "There is nothing to send: I read the checkpoints and
   telemetry from Drive (MyDrive/...)".
8. **"RESUMING AFTER A DISCONNECT"** (:108-159): what is on Drive and at what cadence; a lettered case table 0, a-d
   decided by which files exist; the expected lines on a resume, quoted (`re-entering run`, `Newest intact periodic
   checkpoint: ...`, `Checkpoint steps: <N> of 12,000,000 — remaining budget: ...`, `WARNING: skipping ...`); what to
   report; a list of error messages that mean "stop and tell me (do not restart, do not delete anything, and never
   start a new RUN_ID on the advice of an error message)"; closing don'ts.

The later sheet `stance5/colab_steps.txt` adds a section stating which expected lines were **measured** in a local
smoke and which are **inferred** (:205-224), and embeds the criteria text between `=== BEGIN` / `=== END` lines so
its hash can be checked.

---

## 8. Versions: Colab against the pins

Only what the documents and files say. "unknown" = not stated anywhere read.

| | Python | numpy | scipy | torch | SB3 | gymnasium | mujoco |
|---|---|---|---|---|---|---|---|
| Study venv, `FS/tools/requirements-study.txt:10-15` (records 8-14 Nov) | 3.13.18 (`:1`) | 2.5.4 | 1.18.1 | 2.15.0 | 2.9.2 | 1.4.0 | 3.10.0 |
| This container's `.venv` [measured] | 3.13.18 | 2.5.4 | 1.18.1 | 2.15.0+cpu | 2.9.2 | 1.4.0 | 3.10.0 |
| Stage 1 PPO venv, `FS/tools/requirements-stage1-ppo.txt` (all R3/R4 training and policy rows) | 3.13.18 (`:8`) | 2.5.4 | 1.18.1 | 2.9.0+cpu | 2.9.0 | 1.3.0 | 3.10.0 |
| Scaffold setup cell, c3:L31 | image's | **unpinned** | `>=1.10.0` | **unpinned** | 2.9.0 | 1.3.0 | 3.10.0 |
| Main CI SB3 job, `.github/workflows/python-ci.yml:282-291` | 3.12 | range | range | 2.13.0 (CPU wheel) | 2.9.0 | range | 3.10.0 |
| `pyproject.toml` | `>=3.11` (`:9`) | `>=1.24.0` | test extra `>=1.10.0` | `>=2.0.0` | `>=2.2.0` | `>=0.29.0` | `==3.10.0` |
| Colab L4 image, 2026-08-15 (`KNOWN_ISSUES.md:612`) | 3.12.13 | 2.0.2 | unknown | unknown | installed by cell | installed by cell | installed by cell |
| Colab L4 image, 2026-09-14/15 (`KNOWN_ISSUES.md:613`) | 3.13.15 | 2.1.3 | unknown | unknown ("a CUDA build", `research_pipeline.md:380`) | 2.9.0 | 1.3.0 | 3.10.0 |
| Colab today, High-RAM CPU runtime | **unknown** | **unknown** | **unknown** | **unknown** | by cell | by cell | by cell |

Other version facts: cloudpickle 3.1.2 in both study sets; Colab's is unknown. An older note attributed the crash to
"Python 3.13 / torch 2.11" (`KNOWN_ISSUES.md:610`). `setup_venv.sh` refuses any Python that is not 3.13
(`FS/tools/setup_venv.sh:38-45`). The scaffold's pins equal the Stage 1 PPO venv's SB3 and gymnasium, not the study
venv's (2.9.2 / 1.4.0). Which set R5 trains under is not decided in the documents read: `LESSONS_LEARNED.md:587-589`
lists it as open ("decide how Colab checkpoints are judged here: either in a local venv that matches Colab exactly,
Python's patch version and torch build included, or under a flagged cross-version rule").

---

## 9. Tests that pin behaviour

Main repository, `REPO/environments/shared/tests/` (not run here). `pyproject.toml:74-81` limits `testpaths` to
`environments/*/tests`; nothing under `hardware/` is collected by the main suite, and `norecursedirs` lists
`notebooks` (`:82-89`).

**Tests that glob `REPO/notebooks/*.ipynb`** (why a Stage 2 notebook must not live there):

| Test | Asserts |
|---|---|
| `test_record_fields.py:41` `NOTEBOOKS`; `test_the_64_hex_pattern_is_written_only_in_record_fields` (:307-314) | no string constant in any notebook code cell contains `{64` (a second sha256 regex) |
| `test_record_fields.py` `test_the_hex_alphabet_literal_is_only_the_hasher_check` (:317-320) | no string constant contains `0123456789abcdef` |
| `test_retired_species_identifiers.py:247-260` `test_no_notebook_code_cell_names_a_retired_identifier` | no code cell matches `RETIRED` (:66-81), which includes `trex`, `t-rex`, `t_rex`, `raptor`, `biological`, `TREX`. The path `hardware/trex_form_study` matches it [doc: `research_pipeline.md:196-198`] |
| `test_policy_loading.py:597-609` `test_every_sb3_archive_load_goes_through_the_loader` | no bare `PPO.load` / `SAC.load` / `alg_cls.load` in any `environments/**.py` (tests aside) or notebook cell |
| `test_policy_loading.py:615-638` `test_the_known_load_sites_call_the_loader` | the scaffold has 0 `load_sb3_model(` calls and exactly 1 `sb3_archive_load_preflight(` |
| `test_policy_loading.py:1624-1637` `test_no_notebook_cell_encodes_the_periodic_checkpoint_name` | no cell re-encodes `_(\d+)_steps` |
| `test_stage_manifest.py:736-737` `test_no_notebook_code_cell_names_a_retired_recipe_label` | no string constant in a code cell is a retired recipe label |
| `test_stage_config_reader.py:44, 214-223` (`_non_test_sources`); `test_task_fingerprint_stage.py:35, 69-77` (`_non_test_trees`) | every notebook code cell is parsed beside the non-test modules for those files' source pins (one stage-config reader; the fingerprint backend constant defined once) |
| `notebook_cells.py:109-119` (CI step "Validate notebooks", `python-ci.yml:100-109`) | every code cell of every `notebooks/*.ipynb` parses once `!`/`%` lines are replaced by `pass`; with arguments it checks the paths given |

**Scaffold pins**, `test_sb3_notebook_pins.py` (cells are found by marker text, not position, :48-62):
- markers: `"# ===== SPECIES SELECTION ====="`, `"# Storage Configuration"`, `"STAGE_CONFIGS = load_all_stages(MODEL_ID)"`,
  `"def train_stage("`, `"# ===== BEHAVIOR CHAIN LOOP ====="`, `"# ===== MANUAL SINGLE NODE"`,
  `"# ===== RESUME AN INTERRUPTED STAGE"`, `'print("Training complete!")'`, and the preflight cell's first line;
- `DISCONNECT_KNOBS` (:65): every `halt` / `disconnect_runtime` call passes `in_colab=IN_COLAB, auto=AUTO_DISCONNECT,
  flush_drive=USE_GOOGLE_DRIVE` (`test_one_disconnect_path`, :2646);
- `TestArchiveLoadPreflightCell` (:3608-3689): the preflight sits right after the resolve cell, is exactly
  `sb3_archive_load_preflight(MODEL_ID, CHAIN[0].reference, trunk_dir=TRUNK_DIR)`, calls no `load_sb3_model` or
  `print` itself, trains and writes nothing;
- `test_colab_ref_change_is_explicit_and_preserves_edits` (:4020-4082): executes the setup cell's real Git block with
  fake Git replies; "local edits" and "Restart" refusals; `commands[0] == ["git", "fetch", "origin", ...]`;
  `["git", "checkout", "--detach", "new-commit"]`;
- `test_the_drive_summary_checks_out_repo_ref_with_the_sb3_notebooks_git_block` (:4102-4119): the two notebooks' Git
  blocks are identical statement for statement and both declare `REPO_REF = "main"  # @param {"type":"string"}`.
- **No test asserts the pip pin strings** (`grep` for `stable-baselines3[extra]==`, `mujoco==3.10.0`,
  `gymnasium==1.3.0`, `uninstall -y`, `MUJOCO_GL` in the tests finds none in a pin role). The research note's "it is
  pinned by test" (`research_pipeline.md:152-153`) is true of the Git block, not of the install line.

`test_policy_loading.py` also executes the loader on committed fixture archives saved under Python 3.12 and 3.13
(`environments/shared/tests/fixtures/sb3_archives/{ppo,sac}_linear_py31{2,3}.zip`; `KNOWN_ISSUES.md:633-634`).

Form study, `FS/tests/test_stage1_ppo.py` (36 tests; needs restored inputs and SB3; not run here):
- `test_training_checkpoints_and_resumes` (:515-579): 30 asked steps become 32 (whole rollouts); `checkpoints(rd) ==
  [16, 32]`; `run.json` fields (`settings`, `sessions[0].env_seeds == [420000]`, `versions`, `ppo_venv`,
  `g6.G6a_rate.judged_on`, `g6.G6a_zero_residual.identical`); a second start without `--resume` raises; resume to 64
  gives `sessions[-1].resumed_from == 32` and `env_seeds == [420100]`; other settings refused (`match="settings"`);
  other sources refused (`match="Stage 1 sources"`) unless `allow_source_change`, then `source_change` recorded;
  changed gains refused. It calls a bare `PPO.load` itself (:535).
- `test_subproc_envs_report_their_processes_and_cpu` (:674-684); `test_cli_trains_and_records` (:687-705);
  `test_zero_residual_reproduces_r2_exactly` (:89); `test_latency_applies_the_command_of_steps_ago` (:212);
  `test_a_saved_model_keeps_its_actor_width` (:345); `test_the_budget_is_4m_steps_in_whole_rollouts` (:421).
- `FS/tests/test_restore_inputs.py` (24 tests, `tools/README.md:108`) pins the restore tool.

No test covers a Stage 2 notebook, a STATUS/SESSION writer, a Drive bundle manifest or a kill-and-resume on a mount:
**not found**.

---

## 10. Git rules that drop or rewrite outputs

Root `REPO/.gitignore` (all patterns unanchored, so they match at any depth):

| Line | Pattern | What it would silently drop for Stage 2 |
|---|---|---|
| 16 | `.venv/` | the venv |
| 34 | `logs/` | any `logs/` directory, e.g. `stage2/runs/<id>/logs/<job>.log` |
| 35 | `models/` | any directory named `models`, e.g. a bundle's `round2/refine/<case>/models/` (line 41 re-includes only `website/docs/models/`) |
| 42 | `*.zip` | every SB3 archive, and an inputs bundle packed as `.zip` (lines 45, 47 re-include two test fixture folders only) |
| 43 | `*.pkl` | every VecNormalize sidecar |
| 49 | `*.npz` | every reference, transition and look array |
| 50 | `tensorboard/` | |
| 53 | `.ipynb_checkpoints/` | |
| 70-71 | `*.mjb`, `MUJOCO_LOG.TXT` | |

Nested: `FS/stage1/.gitignore` adds `*.npy`, `*.pt`, **`*.xml`**, `refs/`, `traces/`, `checkpoints/` under
`stage1/`; `FS/round2/.gitignore:3` `models/`; `FS/stage2/look/.gitignore` `partial/`, `*.tmp`. There is no
`.gitignore` in `FS/stage2/` itself.

[measured] `git check-ignore -v` on hypothetical paths: ignored are `stage2/**/model_1.zip` (`*.zip`),
`vecnormalize_1.pkl`, `stage2/**/logs/job.log`, `stage2/**/refs/nominal.npz`, `stage2/models/a.xml`,
`stage2/bundle/inputs.zip`. **Not ignored** (would be tracked or show as untracked): `stage2/stage2_training.ipynb`,
`stage2/**/STATUS.json`, `MANIFEST.json`, `run.json`, `SHA256SUMS`, `colab_steps.txt`, a `.tar` or `.tar.gz` bundle,
`.mp4` clips.

Precedent for committing ignored binaries: the five judged Stage 1 checkpoints were force-added (`git add -f`) with
`judged_checkpoints.json` (commit `276935a`).

Root `REPO/.gitattributes`: `* text=auto` (:2), `*.json text` (:13), `*.txt text` (:10), `*.md text`, `*.xml text`
(:16), `*.svg text`; binary only for gif/png/jpg/jpeg. [measured] `git check-attr`: `stage2_training.ipynb`
`text: auto`; `MANIFEST.json` and `colab_steps.txt` `text: set`; `SHA256SUMS`, `.tar`, `.mp4` `text: auto`. So a
text-detected file is line-ending normalised on commit: a manifest, a steps sheet or a JSON record whose sha256 is
recorded elsewhere changes hash if it ever carries CRLF. The posture archive needed its own `.gitattributes` with
`* -text` "so SHA256SUMS verifies" (`FS/r0_posture_study_2026-11-07/.gitattributes`; `LESSONS_LEARNED.md:512-518`).

`REPO/.pre-commit-config.yaml` (only if hooks are installed; CI does not run pre-commit): `trailing-whitespace` and
`end-of-file-fixer` (:43-44) apply to every file outside the `exclude` pattern (:14), which does not cover `hardware/`;
they would rewrite a notebook, manifest or sums file. `check-added-large-files --maxkb=500` (:47-48) refuses a
notebook saved with outputs over 500 KB and any bundle. The ruff hooks are limited to `^environments/` (:27-29).
`pyproject.toml:128` has a ruff per-file ignore for `notebooks/*.ipynb` only.

CI path filters (`python-ci.yml:8-50`, one list for push and pull_request) include `notebooks/**`,
`environments/**`, `configs/**`, `docs/**`, `pyproject.toml`, `.gitattributes`; **there is no `hardware/**` entry**
(`grep hardware` finds only a comment at `:18`) [read], so a change under `FS/` alone triggers no main CI job.

Colab itself: "File > Save a copy in Drive" (`stance3/colab_steps.txt:31-32`) means the owner runs a Drive copy; its
outputs never reach git unless someone commits them. The scaffold is committed with outputs stripped [measured].

---

## 11. Hard-coded assumptions the build list would break

| # | Assumption | Where | Breaks |
|---|---|---|---|
| 1 | Notebook metadata requests a GPU, type L4 | `sb3_training.ipynb` raw 1760-1762 | (H): High-RAM CPU runtimes |
| 2 | `MUJOCO_GL=egl` for the whole kernel, inherited by children | c3:L12 (raw 55) | (H), (F): training must run with `MUJOCO_GL=disable`; `FS/designlab/__init__.py:15` only `setdefault`s (egl if no `DISPLAY` and `/dev/dri` exists, else osmesa); `train.py` sets nothing (`LESSONS_LEARNED.md:434-438`) |
| 3 | Pins install only if one of seven modules is missing; numpy and torch unpinned; scipy a range | c3:L29-32 (raw 72-75) | (H): the plan pins numpy, scipy, torch; an image that already has all seven skips the pins |
| 4 | Pins are gymnasium 1.3.0 / SB3 2.9.0 | c3:L31 | differs from the study venv (1.4.0 / 2.9.2); equals `train.PPO_VENV` (`train.py:109`) |
| 5 | `pip install -e` exposes `environments`, `configs` only | c3:L54-55; `pyproject.toml:66-67` | (H): `designlab` needs cwd `FS` and `PYTHONPATH=.` |
| 6 | Refusal to switch code checks `environments*` modules only | c3:L48 (raw 91) | (H): `designlab` imported in the kernel is not caught |
| 7 | Local repo root found by `configs/species_manifest.toml` | c4:L12-18 | fine for a notebook under `FS/stage2/` (parents are searched) |
| 8 | `LOG_BASE = MyDrive/mesozoic-labs/logs`, tree keyed by `ModelId` | c8:L12, L27 | (H): Stage 2 saves under `MyDrive/mesozoic-labs/stage2/`, outside `logs/` |
| 9 | Training runs in the kernel, one node, `DummyVecEnv`, device auto | c15:L139-159 | (H): a queue of jobs as subprocesses |
| 10 | Checkpoint names `{label}_{steps}_steps.zip` / `{label}_vecnormalize_{steps}_steps.pkl` | `checkpoints.py:259-265`; `policy_loading.py:174` | (H): the form study writes `model_<n>.zip` / `vecnormalize_<n>.pkl` (`train.py:553-554`) |
| 11 | Preflight takes a `ModelId` and a stage reference | `policy_loading.py:785-787` | usable with `trunk_dir=None` only (arguments then unread, :823-830) |
| 12 | `--mode` is R3 or R4 | `train.py:395-396`; `__main__.py:531` | (F): R5, R6 |
| 13 | Run directory name from mode: `ppo` or `policy_only` | `train.py:158-159` | (F): no slot for R5/R6 or a head variant |
| 14 | G6 checks only when `mode == "R3"`; G6a compares to R2 | `train.py:519-523`, `:568` | (F): G8a compares R5's zero action to the panels' R0 at 0 and 10 ms |
| 15 | `model_kwargs(env_mod.ACTOR_DIM, ...)`; widths built for 14 hinges, `ACTION_DIM = 15` | `train.py:511`; `env.py:138-142`, `:284-286` | (F): -X bodies have 16 hinges |
| 16 | Default 4 envs; `ROLLOUT_STEPS = 4 * 2048`; `BUDGET_ASKED = 4_000_000`; `CHECKPOINT_EVERY = 250_000` | `train.py:400`, `:66`, `:76`, `:64` | (H): 8 envs, 40M budget, checks every 2.5M, keep-newest-two |
| 17 | `G6A_ENV_STEPS_PER_S = 600` per job | `train.py:87` | (H): the "first rate" line; S2-8's rule is a different threshold |
| 18 | Rate is recorded, never printed; final summary line printed at job end | `train.py:602-619`; `__main__.py:388-390` | (H): the four printed lines at start |
| 19 | Resume refuses any library difference, no flag | `train.py:457-460` | (H): flagged resume across library versions |
| 20 | Resume refuses changed env configs, which hold absolute `run_dir` | `train.py:294`, `:435-436` | (H): local-then-Drive checkpoints, a different mount path, fine-tunes "in a new run directory" |
| 21 | `checkpoints()` checks existence only; resume takes `done[-1]` | `train.py:186-194`, `:495` | (H): Drive writes cut short |
| 22 | Bare `PPO.load`; closure `linear_schedule` | `train.py:497`; `policy_rows.py:601`; `policy.py:144-151`, `:162` | (H): archives across Python minor versions |
| 23 | No fine-tune entry (a new run from another run's checkpoint and statistics) | **not found** in `train.py` | (H) |
| 24 | `status` stays `running` after a kill | `train.py:516`, `:639` | (H): liveness must come from STATUS.json's modified time |
| 25 | Sources hash covers every `designlab/stage1/*.py`, new files included | `plant.py:262-275` | iron rule 2: any of rows 12-23 fixed inside `designlab/stage1/` moves it; resume of an older run then needs `--allow-source-change` |
| 26 | Session provenance is dirty on any untracked non-ignored file under `FS` | `plant.py:306-312` | (H): a notebook, STATUS file or bundle written into the checkout |
| 27 | `tools/memguard.sh` protects `designlab.stage1 train` by default | `tools/README.md:220-226` | (F): Stage 2 trainers with another module name |
| 28 | Main tests scan `notebooks/*.ipynb` | section 9 | (H): location |
| 29 | Policy rows compare `importlib.metadata` versions with the run's `torch.__version__` | `policy_rows.py:412-419`; `train.py:170-171` | (E)/(H): judging Colab checkpoints here |
| 30 | Delay works in whole 2 ms physics steps | `LESSONS_LEARNED.md:576-579` [doc] | (E): 0, 2, 4, 6, 8, 10 ms are all whole steps; the plan's 5, 15, 25 ms are not |

---

## 12. Extension points and hazards (short)

- **Reusable as they are:** `notebook_runtime.disconnect_runtime` / `halt`; `checkpoints.checkpoint_pair_problem`
  (measured to pass on a form-study pair); `policy_loading.inspect_sb3_archive`, `load_sb3_model`,
  `sb3_archive_load_preflight(..., trunk_dir=None)`; `file_io.atomic_write_json`; the setup cell's Git block;
  `tools/restore_inputs.py check` / `copy-from DIR` for a bundle unpacked at study-relative paths.
- **Loading a form-study archive through `load_sb3_model` for a resume** needs the learning-rate schedule passed as a
  keyword, since `learning_rate` and `lr_schedule` are bytecode and would otherwise be replaced by the inference
  default 0.0 (`policy_loading.py:123-127`, `:484-498`).
- **Two documents pull against each other on where edits go.** The plan says the notebook "runs
  `designlab/stage1/train.py`, extended with R5 and R6, a picklable schedule, ..." (`STAGE2_PLAN.md:138`) and the
  research note calls the moved sources hash "acceptable for a new round" (`research_pipeline.md:190-191`); iron rule
  2 says any edit to `designlab/stage1/*.py` makes earlier records read as made by other code. The sources hash is a
  glob over the directory, so only code outside `designlab/stage1/` leaves it unchanged.
- **Version sets are unsettled:** scaffold pins (PPO venv's SB3/gymnasium) against the study venv; Colab's Python
  patch version against 3.13.18; torch build suffix. Each one trips either `train.py:457-460` or
  `policy_rows.check_versions`.
- **`MUJOCO_GL`** must be assigned for child training processes, not inherited from the kernel.
- **`run_dir` is part of the resume contract**; decide local-versus-Drive paths before the first session.
- **Git hygiene:** check every new output with `git check-ignore -v` and `git check-attr`
  (`LESSONS_LEARNED.md:517-518`); keep run outputs outside the checkout so session provenance stays clean.
- **L4 vCPU count** differs between the two documents (4 against 12); only the smoke settles it.

## 13. Not found

- `FS/stage2/stage2_training.ipynb`, a Stage 2 steps sheet, a launcher, a STATUS/SESSION/DONE/HALT writer.
- The exact text of the four printed lines; a key schema for STATUS.json, SESSION files, MANIFEST.json, DONE/HALT.
- A numbered kill-and-resume procedure for Stage 2 (only the scaffold's section 5 and the stance sheets exist).
- Any G8a, G8b, G9, R5, R6 or R2L code.
- Colab's current Python patch, numpy, scipy, torch and cloudpickle versions; vCPUs and RAM of a High-RAM CPU runtime.
- A test asserting the scaffold's pip pin strings.
- A flag in `train.py` to resume across library versions, or a fine-tune entry point.
- KNOWN_ISSUES entries specific to resume across library versions (the refusal is the form study's own).
