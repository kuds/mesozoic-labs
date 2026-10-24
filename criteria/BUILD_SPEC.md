# Tail_3 bound screen: build spec (interfaces the builders implement)

Governs the build of the screen package for `criteria_tailscreen_v2.txt` (same folder; "the criteria"). Where this spec and the criteria differ, the criteria govern and the difference is reported. Nothing here is approved for training.

## 0. Hard rules for the build

1. **No optimizer step on any policy.** This means no `learn()`, no `train()` with a live optimizer, no continuation, and not even a short control run. Allowed:
   - collecting rollouts;
   - computing gradients with `torch.autograd.grad` or `.backward()` and discarding them;
   - `train()` with `policy.optimizer.step` replaced by a recorder that does not step (T1a);
   - setting a toy test model's parameter by hand under `no_grad`.

   Tests that take optimizer steps are written now and run only by `gate.sh` after approval (§5).
2. **Read only:**
   - the judge clones under `SP/fix/` (`scratch_copband`, `screen_s3_L`, `screen_s3_LT`, `screen_s4_LTS`);
   - the main checkout `/home/user/mesozoic-labs`;
   - `S2/tailround/fetch/`;
   - TAILEXP.

   Run `git --no-optional-locks -C <clone> status --porcelain --ignored` on every judge clone before and after the build, and save both outputs to `SCREEN/logs/judge_clean_{before,after}.txt`. Both must be empty. At 17:50 UTC they were empty, with HEADs cf80ecb1, 7e6b557d, 1af2d4d7 and c1e146a8.
3. **Nothing is pushed.** The scratch clone's push URL is `DISABLED-never-push`. `copy_durable.sh` runs only `--dry-run` in the build.
4. **Python and output location:**
   - Interpreter: `PY=S2/venv/bin/python`, always called with `-I` and with `PYTHONDONTWRITEBYTECODE=1`. Run the trace tool with `-I -B`.
   - Environment: set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, and call `torch.set_num_threads(1)`.
   - Outputs go under `SCREEN` only.
5. **Imports under `-I`.** `-I` implies `-P`, so the script's own directory is not on `sys.path`.
   - Tools load each other with `importlib.util.spec_from_file_location(<abs path>)`. `tools/screen_common.py` holds the constants.
   - Code that needs `environments` inserts the correct root at `sys.path[0]`: the scratch clone for training and calibration, the measuring copy for panels. It then asserts that `os.path.realpath(environments.__file__)` lies under that root.
6. **The build never writes:**
   - `SCREEN/APPROVED`;
   - `SCREEN/criteria.txt`;
   - anything under `SCREEN/runs/`.

   The coordinator writes the first two after the maintainer approves.
7. **Labels.** Mark every number reported back as [measured] or [inferred].

Short names follow the criteria: `S2`, `SP`, `SCREEN = S2/tailround/screen`, `FETCH = S2/tailround/fetch`, `DRAFT`.

## 1. Layout under SCREEN

```
SCREEN/
  criteria/criteria_tailscreen_v2.txt, BUILD_SPEC.md     (this text; inputs)
  criteria.txt, APPROVED                                  (after approval only; coordinator)
  scratch_tail3bound/                                     scratch clone (§2)
  measure_copy/                                           git archive cf80ecb1 + 3 overlays (§6)
  tools/  screen_common.py run_tailscreen.py calib_tailscreen.py measure_tailscreen.py
          score_tailscreen.py copy_durable.sh gate.sh run_waves.sh
          tests/test_tools.py tests/test_score.py tests/test_t7_screen.py
  calib/coef.json, calib/calib.log
  validate/V1_s42/ V1_s80/ v1.json
  tests_out/build_tier.txt build_tier.json  (gate_tier.txt gate_pass.json after approval)
  runs/<name>/                                            (after approval only)
  panels/<name>/<label>/                                  panel outputs, never inside runs/
  panels.csv
  scores/score.txt score.json
  durable_stage/                                          copy_durable.sh --dry-run tree
  logs/
```

`<name>` = `<state>_<arm>_r<seed>`, with:
- state ∈ {s42, s80};
- arm ∈ {control, b90, b60};
- seed ∈ {101, 202, 303}.

That gives 18 names. `<label>` is one of `4325376, 4456448, 4587520, 4718592, 4849664, 4980736, 5111808, 5242880, final`.

### tools/screen_common.py (constants only; importable without torch or environments)

- `SCREEN`, `SP`, `S2`, `PY`
- `CLONE = SCREEN/scratch_tail3bound`, `BRANCH = "scratch/tail3-upper-bound"`, `PARENT = "cf80ecb14d1630ce3b4dcdea28a05835d143470b"`
- `SCRATCH_COMMIT = None` (pin), `CHANGED_FILES = ("environments/shared/mirror_kl_ppo.py", "environments/shared/tests/test_mirror_kl_ppo.py")`
- `STATES`:

  | state | zip | zip sha256 | pkl sha256 | stage_config | own seed |
  |---|---|---|---|---|---|
  | s42 | `FETCH/stance5_LTSB_s42/A/stage1_4194304_steps.zip` | d3c90db8c107ce1fef6b63efa3e8cdd2271e6e437a3abdb5b30f57b3f46b8df2 | 3903cbcfe43559c8f5dccc0b2d6ed01f3c3989f039f966b12ed6aaa7ecbca118 | `SP/foot/stance5/runs/stance5_LTSB_s42/01_stance/stage_config.json` | 42 |
  | s80 | `FETCH/stance5_LTSB_s80/A/stage1_4194304_steps.zip` | 649db94c026db3925a43643dfcd58299877fbe7783bca25e7b7b6ca504a5a3cc | 98fe304328256720f59658da9a0c2b7ccae89dd0cb5ad5798a1768383adc06d9 | `SP/foot/stance5/replication/runs/stance5_LTSB_s80/01_stance/stage_config.json` | 80 |

  Each pkl sits beside its zip as `stage1_vecnormalize_4194304_steps.pkl`.
- `START_STEPS = 4194304`, `STEPS = 1048576`, `SAVE_EVERY = 131072`, `N_ENVS = 4`, `ROLLOUT = 8192`, `TAIL3 = 14`
- `ARMS = {"control": ("COEF0", 0.90), "b90": ("COEF_90", 0.90), "b60": ("COEF_60", 0.60)}`, `REP_SEEDS = (101, 202, 303)`
- `WAVES = [("s42",101),("s80",101),("s42",202),("s80",202),("s42",303),("s80",303)]`
- `R1 = 5111808`, `R2 = "final"`, `LABELS` as above, `PANEL_SEEDS = range(3042, 3062)`, `SETTLE = 200`
- `OVERLAYS`: the three paths and sha256 from criteria "HOW IT IS MEASURED"
- `STACK = {"python": "3.13.17", "torch": "2.13.0+cpu", "numpy": "2.5.4", "gymnasium": "1.4.0", "stable_baselines3": "2.9.0", "mujoco": "3.10.0"}`
- `V1_EXPECTED`: the criteria's V1 values
- `DURABLE_BRANCH = None` (pin)
- `TOOL_SHA256 = {}` (pins, filled after the reviews)
- `stack_versions()`: returns the running versions. `check_stack()` returns a list of differences from `STACK`.

## 2. Scratch clone

```
git clone --no-hardlinks --branch claude/screen-trex-r9-armlts-copband SP/fix/scratch_copband SCREEN/scratch_tail3bound
git -C SCREEN/scratch_tail3bound checkout -b scratch/tail3-upper-bound cf80ecb14d1630ce3b4dcdea28a05835d143470b
git -C SCREEN/scratch_tail3bound remote set-url --push origin DISABLED-never-push
```

If `scratch_copband` has no such local branch, clone without `--branch` and check out cf80ecb1 directly.

Required state:
- `git remote -v` shows `DISABLED-never-push` for every push URL.
- After the commit, `git diff --name-only cf80ecb1..HEAD` is exactly `CHANGED_FILES`.
- `git status --porcelain` is empty.

Commit message: `scratch: tail_3 upper mean bound (screen, never pushed)`. Add no attribution trailers to this scratch commit; it is never pushed.

## 3. The bound commit (environments/shared/mirror_kl_ppo.py)

Write the code fresh; P1's module is gone. Change nothing in `train()`. `test_train_is_stock_ppo_train_plus_four_marked_lines` must still pass.

```python
TAIL3_BOUND_ATTRIBUTE = "mesozoic_tail3_bound"

@dataclass(frozen=True)
class Tail3UpperBoundConfig:
    coef: float            # finite, >= 0
    bound: float           # finite, 0 < bound <= 1.0
    index: int = 14        # tail_3_pitch_act; any other value raises ValueError
    def checkpoint_record(self) -> dict:  # JSON-only
        return {"coef": self.coef, "bound": self.bound, "index": self.index, "side": "upper",
                "loss": "coef * mean_rows relu(mu[:, index] - bound)^2"}

def tail3_upper_bound_penalty(mu: th.Tensor, bound: float, index: int = 14) -> th.Tensor:
    return th.mean(F.relu(mu[:, index] - bound) ** 2)

def enable_tail3_upper_bound(model: MirrorKLPPO, config: Tail3UpperBoundConfig | None) -> None:
    """Off by default.

    TypeError unless type(model) is MirrorKLPPO.
    None: removes _tail3_bound and the JSON attribute.
    Otherwise: sets model._tail3_bound = config and model.mesozoic_tail3_bound = config.checkpoint_record().
    """
```

Inside `MirrorKLPPO`:
- Class attribute `_tail3_bound: Tail3UpperBoundConfig | None = None`. `_excluded_save_params` returns `[*super()..., "_mirror", "_tail3_bound"]`.
- `_mirror_begin_update`: when `_tail3_bound` is set, also returns `"tail3": {"penalty": [], "beyond": [], "mu_max": []}`.
- `_mirror_loss`: after the existing `loss + kappa * kl` is formed, and when `_tail3_bound` is set:
  - When coef > 0: `mu_g = self.policy.get_distribution(rollout_data.observations).distribution.mean` (with gradient), `pen = tail3_upper_bound_penalty(mu_g, b)`, `loss = loss + coef * pen`.
  - When coef == 0: `pen = tail3_upper_bound_penalty(mu, b)` from the hook's own no_grad `mu` (line 280). Add nothing to the loss, not even `0 * pen`, and make no extra forward pass.
  - Either way, append `pen.item()`, `(mu[:,14] > b).float().mean().item()` and `mu[:,14].max().item()` to the update record (`mu` is the no_grad mean).
- `_mirror_end_update`: when `_tail3_bound` is set, record:
  - `train/tail3_bound_penalty`: mean of the penalty list;
  - `train/tail3_bound_loss`: coef × that mean;
  - `train/tail3_bound_beyond_fraction`: mean;
  - `train/tail3_bound_mu_max`: max;
  - `train/tail3_bound_coef`;
  - `train/tail3_bound_b`.

  The mini-batches are those `_mirror_loss` ran on, including the one that triggers an early stop.
- When no bound is enabled, there are no new logger keys and no new checkpoint attribute. The existing `saves a stock checkpoint` test therefore still holds.
- Add no stage config key and change nothing in `train_base.py`.

## 4. tools/run_tailscreen.py (training entry point; never run in the build except `--preflight-only` in tests)

Derived from TAILEXP `run_continuation.py`.

```
PY -I SCREEN/tools/run_tailscreen.py --state {s42,s80} --arm {control,b90,b60} --rep-seed {101,202,303} --steps 1048576 [--preflight-only]
```

There are no `--coef`, `--bound`, `--out-dir`, `--save-every` or `--allow-dirty` options. Coef and b come only from `ARMS` and `calib/coef.json`:
- control: (0.0, 0.90);
- b90: (`COEF_90`, 0.90);
- b60: (`COEF_60`, 0.60).

Pure functions, importable without torch or environments (T6):
- `arm_config(arm: str, coef_json: dict) -> tuple[float, float]`
- `preflight(ns, root: Path) -> list[str]`: returns problems; empty means pass. It checks:
  1. `root/APPROVED` exists, parses, and its `criteria_sha256` equals sha256(`root/criteria.txt`).
  2. `coef.json` exists, follows the rule (§7), and its `COEF_90` and `COEF_60` equal the pins in `APPROVED`.
  3. arm, state and seed are in their sets, and `--steps == 1048576`.
  4. `root/runs/<name>` does not exist.
  5. The start zip and pkl sha256 equal `STATES`.
  6. The clone is clean, on `BRANCH`, HEAD equals `APPROVED.scratch_commit` (a child of `PARENT`), the push URL is disabled, and the diff file set equals `CHANGED_FILES`.
  7. `check_stack()` is empty.
  8. `tests_out/gate_pass.json` exists and names that HEAD.
  9. Free disk is at least 5 GB.

`--preflight-only` prints `{"preflight": "pass"|"refused", "problems": [...]}` and exits 0 or 2. It never builds a model or creates a folder.

`build(state, rep_seed, out_dir) -> dict` follows P1's `build()`, with these differences:
- The recipe check compares the stage_config's `run.seed` with the state's own seed (42 or 80), not with `rep_seed`.
- `create_vec_env(..., N_ENVS, rep_seed, ...)` and `eval_env_seed(rep_seed)`.
- `alg_kwargs["seed"] = rep_seed` (set, not setdefault).
- `device` is cpu.
- It asserts MirrorKLPPO, `num_timesteps == 4194304` and `_total_timesteps == 12000000`.

`calib_tailscreen.py` also uses this function.

Main flow (after preflight passes):
1. `os.makedirs(runs/<name>)`.
2. `enable_tail3_upper_bound(model, Tail3UpperBoundConfig(coef, b))`.
3. CSV logger to `runs/<name>/progress.csv`.
4. Callbacks: SB3 `CheckpointCallback(save_freq=131072//4, name_prefix="stage1", save_vecnormalize=True)`, the `EntCoefDecayCallback`, Telemetry and StopAndSave, as in P1.
5. `learn(total_timesteps=12000000-4194304, reset_num_timesteps=False)`, stopped after exactly 1,048,576 steps.

Outputs in `runs/<name>/`:
- `stage1_<N>_steps.zip` and `stage1_vecnormalize_<N>_steps.pkl` for N = 4325376 … 5242880 (8 pairs);
- `stage1_final.zip` and `stage1_final_vecnorm.pkl`;
- `progress.csv`, with SB3's `train/*` columns including `train/tail3_bound_*`;
- `telemetry.npz`: per rollout, P1's keys, plus:
  - `tail3_mu_gt_0p60`, `tail3_mu_gt_0p90` and `tail3_mu_gt_1`;
  - `mu_gt_0p60` and `mu_gt_0p90` (per actuator);
  - `tail3_bound_penalty_buffer` (at the arm's b);
  - `prev_train_tail3_bound_{penalty,loss,beyond_fraction,mu_max,coef,b}`.

  P1's `mean_bound_*` keys are dropped.
- `run_meta.json`, written with `"x"` at start and rewritten at the end. Keys:
  - identity and arguments: `schema "tailscreen-run/1"`, `name`, `state`, `arm`, `rep_seed`, `coef`, `b`, `bound_record`, `steps`, `target_num_timesteps 5242880`, `save_every`, `argv`;
  - code and stack: `clone {head, status, branch, push_url}`, `modules_root`, `stack`, `torch_threads`, `device`;
  - pinned inputs: `criteria_sha256`, `coef_json_sha256`, `approved_sha256`;
  - start state: `start {zip, zip_sha256, pkl, pkl_sha256, num_timesteps, total_timesteps_recorded, n_updates, digest}`;
  - checks and setup: `recipe_checks`, `ent_coef_decay`, `learn`;
  - progress: `status running|done`, `utc_start`;
  - end of run: `utc_end`, `final_digest`, `final_update_train`, `checkpoints`, `final_sha256`, `wall_seconds`.

  Digests use P1's `state_digest`: params, Adam, VecNormalize and all.

Exit codes:
- 0: done;
- 2: refused (nothing created);
- 1: crash (the folder stays for the crash rule).

## 5. Tests T1-T7 (adapted to two bounds), in two tiers

Clone tests live in `scratch_tail3bound/environments/shared/tests/test_mirror_kl_ppo.py`. Each new test name starts with `test_tail3_`. Gate-tier tests are decorated:

```python
@pytest.mark.skipif(os.environ.get("MESOZOIC_TAILSCREEN_GATE") != "1", reason="takes optimizer steps: gate only")
```

**Build tier.** No optimizer step on any policy. The build runs:

```
PYTHONDONTWRITEBYTECODE=1 PY -I -m pytest -p no:cacheprovider --rootdir SCREEN/scratch_tail3bound \
  SCREEN/scratch_tail3bound/environments/shared/tests/test_mirror_kl_ppo.py \
  -k "tail3 or train_is_stock_ppo_train_plus_four_marked_lines" -q
PYTHONDONTWRITEBYTECODE=1 PY -I -m pytest -p no:cacheprovider SCREEN/tools/tests -q
```

Write the output to `tests_out/build_tier.txt`, and per-test status plus HEAD to `build_tier.json`.

Do NOT run the rest of `test_mirror_kl_ppo.py` in the build: its existing tests call `learn()`.

**Gate tier.** Runs only after `APPROVED`, through `gate.sh`. It runs the whole `test_mirror_kl_ppo.py` with `MESOZOIC_TAILSCREEN_GATE=1`, then the tools tests again.

| Test | Tier | Where | What it asserts |
|---|---|---|---|
| T1a coef-0 identity, gradients | build | clone | Build two toy MirrorKLPPO models with the file's `_make(seed=13)` and `enable_mirror_kl(..., SymmetryConfig())`: one with no bound, one with `Tail3UpperBoundConfig(0.0, b)`. Run this for b = 0.90 and for b = 0.60. Fill each buffer with `_setup_learn` + `collect_rollouts` (no update). Replace `policy.optimizer.step` with a recorder that copies every `p.grad` and does not step. Call `train()`. Assert: (1) the recorded gradient sequences are byte-equal across the two models; (2) the parameters are unchanged after `train()`; (3) the bound model logs the six `train/tail3_bound_*` keys and the no-bound model logs none. Then repeat with coef 2.7 and `action_net.bias[14]` set to 1.0 under no_grad in both models: the gradients must differ, and only through rows with mu14 > b. |
| T1b coef-0 identity, training | gate | clone | `learn(256)` for no bound and for coef 0 at b = 0.90 and at b = 0.60: params, Adam and VecNormalize digests are bit-equal. |
| T2 same-seed reproducibility | gate | clone | Same seed, two runs each, at coef 0 and at coef 2.7 for b = 0.90 and for b = 0.60, with bias[14] = 1.0: equal digests. Coef 2.7 differs from coef 0. |
| T3 scope | build | clone | `tail3_upper_bound_penalty` for b ∈ {0.90, 0.60} on random mu (N × 15, values from −1.5 to 1.5): the gradient is zero outside column 14, zero on rows with mu14 ≤ b (lower side included), and equals `2*relu(mu14-b)/N` elsewhere. On a batch with every mu14 = b + 0.11, d/d bias[14] through a toy policy's action_net is 0.22 per unit coef (abs tol 1e-6) at both b. |
| T4 mirror | build | clone | With the T. rex `mirror_map`, `action.perm[14] == 14` and `action.sign[14] == +1`. The penalty is equal for mu and for its mirror image `mu[:, perm] * sign`, at both b. |
| T5 no stage key | build | clone | No file under `configs/` or `environments/` (`*.toml`, `*.json`, `*.py`, excluding `mirror_kl_ppo.py` and its test) contains `tail3_bound`, `tail3_upper_bound` or `Tail3UpperBound`. `train_base.py` is byte-identical to cf80ecb1's. |
| T6 arm mapping (two bounds) | build | tools/tests/test_tools.py | With a synthetic `coef.json` in `tmp_path`, `arm_config` maps control → (0.0, 0.90), b90 → (`COEF_90`, 0.90) and b60 → (`COEF_60`, 0.60). `preflight` refuses: a missing or mismatched `APPROVED`, a missing `coef.json`, a `coef.json` that breaks the rule, `COEF` ≠ the `APPROVED` pin, an unknown arm or state, a seed outside {101, 202, 303}, steps ≠ 1048576, an existing run folder, a wrong start sha256, and a missing `gate_pass.json`. The CLI with `--preflight-only` against the real `SCREEN` returns 2 now, since there is no `APPROVED`, and creates nothing. Importing the module does not import torch or environments. |
| T7a load, unit | build | clone | Enable the bound on a toy MirrorKLPPO without learning, save it, and read the zip's `data`. Its keys equal stock ∪ {`mesozoic_symmetry`, `mesozoic_tail3_bound`}; the record equals `checkpoint_record()`; there is no `_tail3_bound` key. `PPO.load` gives a `PPO` whose deterministic predictions equal the model's. Reloading through `enable_mirror_kl` without `enable_tail3_upper_bound` gives `_tail3_bound is None`, and its `_mirror_loss` adds no term. |
| T7a load, screen | build | tools/tests/test_t7_screen.py | Load the s42 start through `run_tailscreen.build(state="s42", rep_seed=101, out_dir=None)` without learning. Enable b60 with coef 2.7, then save to `SCREEN/tests_out/t7/stage1_t7.zip` and the vecnorm `.pkl`. To pass, all of these must hold: (1) the zip carries the `mesozoic_tail3_bound` record and no `_tail3_bound` key; (2) its policy parameters are bit-equal to the start zip's, since no step was taken; (3) in a subprocess on the measuring copy, `episode_trace_report.py --seed 3042 --episodes 1` succeeds with no warning on stderr. The test also reports, but does not pass or fail on, whether the trace's tail_3 window command equals the start panel's seed-3042 row (`DRAFT/trace_s42`, 6 decimals). Those two can differ only through the VecNormalize statistics that were saved. |
| T7b load after one update | gate | clone | The T7a unit test after one real `learn()` rollout at b = 0.60, coef 2.7. |

`tools/tests/test_score.py` is the scorer's synthetic-row suite; §9 lists its cases.

## 6. Measuring copy and tools/measure_tailscreen.py

**Measuring copy.** Build it with `git -C SP/fix/scratch_copband archive cf80ecb1 | tar -x -C SCREEN/measure_copy`, a read-only use of the judge clone. Then copy the three overlay files from `git show origin/main:<path>` in `/home/user/mesozoic-labs` (read only) and check them against `OVERLAYS`. At 17:50 UTC all three matched on origin/main 5f5dbf6b. Write `measure_copy/OVERLAY_SHA256.txt` and a tree digest (sha256 over the sorted `path\tsha256` lines of every file) to `measure_copy/TREE_DIGEST.txt`.

**Command line:**

```
PY -I SCREEN/tools/measure_tailscreen.py --run <name> --label <label>
PY -I SCREEN/tools/measure_tailscreen.py --all-pending [--follow]
PY -I SCREEN/tools/measure_tailscreen.py --validate-v1
```

- `--all-pending` measures, oldest first, every checkpoint that exists with no `panels.csv` row.
- `--follow` polls every 60 s until all 162 panels are measured.
- `--validate-v1` panels both start checkpoints into `validate/V1_<state>/`.

**Each panel.** The tool runs, through `nice -n 19`:

```
PY -I -B measure_copy/environments/shared/scripts/episode_trace_report.py \
  --species tyrannosaurus_rex --embodiment anatomical --model <zip> --vecnorm <pkl> \
  --seed 3042 --episodes 20 --out-dir SCREEN/panels/<name>/<label>
```

The subprocess gets `OMP_NUM_THREADS=1` and `PYTHONDONTWRITEBYTECODE=1`. The tool then:
1. Checks that the measuring copy's tree digest equals `TREE_DIGEST.txt`.
2. Reads `episode_trace.json` and the 20 per-step CSVs.
3. Writes `flags.csv`.
4. Appends one row to `SCREEN/panels.csv` under an `fcntl` lock.
5. Records the CSVs' sha256 in `csv_sha256.txt`.
6. Gzips the CSVs.

It never reads from or writes to `runs/` except to read the checkpoint pair and `run_meta.json`. It refuses below 5 GB free (exit 5, a pause message). A panel folder with no `panels.csv` row is a crashed panel: move it to `<label>_crashed<k>` and run it again.

**Per-episode `flags.csv`.** Window = steps 200..L−1. Columns:
- `seed`, `length`, `terminated`, `truncated`, `return`;
- `tail3_cmd_window_mean`, `tail3_share_cmd_ge_0p99`, `tailup_flag` (share > 0.10);
- `gate_stat`: max over actuators of the window share with |command| ≥ 0.99;
- `gate_actuator`, `gate_sign`;
- `other_flag`: not tailup, and gate_stat > 0.10 or length < 1000;
- `other_reason` ∈ {`saturation:<actuator><sign>`, `short:<length>`, `""`};
- `fall`: terminated and length < 1000.

**`panels.csv` (one row per panel). Columns in this order:**
- identity: `name, state, arm, rep_seed, coef, b, label, num_timesteps, zip_sha256, pkl_sha256, n_episodes, seeds`;
- tail_3 command and saturation: `frac99, frac90, cmd, cmd_episode_sd`;
- tail_3 raw mean: `raw_window_mean, share_raw_gt_0p60, share_raw_gt_0p90, share_raw_gt_1, raw_max_window, pinned_step_raw_mean`;
- posture: `q3_window_mean`;
- flags, falls and returns: `tailup_flags, other_flags, other_reasons, falls, short_episodes, return_mean`;
- other actuators and substitution: `others_max_frac90, others_max_frac99, others_max_abs_raw_mean, tail1_cmd, tail2_cmd, tail2_q_window_mean`;
- single-checkpoint class: `class_single` (RAIL if tailup_flags ≥ 1, else PARKED if cmd ≥ 0.60, else LOW);
- provenance: `trace_json_sha256, measure_tree_digest, stack, utc`.

For V1 panels, `name` is `start_<state>` and `label` is `4194304`. `cmd` is the mean over episodes of each episode's window-mean tail_3 command. The other formulas follow `DRAFT/agg_trace.py`.

**`--validate-v1` output.** It writes `validate/v1.json` with the expected and obtained values: frac values in full, others rounded to 6 decimals. It exits 0 only if every value is exactly equal.

## 7. tools/calib_tailscreen.py (V2; runs in the build; no optimizer step)

```
PY -I SCREEN/tools/calib_tailscreen.py            # writes SCREEN/calib/coef.json; refused if it exists or if any SCREEN/runs/* exists
```

The only option is `--out PATH`, for a dry run. Any `--out` other than the default must lie outside `SCREEN/calib`.

**Procedure** (P1's `lambda_calib.py` without `MK.mean_bound_penalty` and `rc.SEED`). For each state in (s42, s80):
1. `build(state, rep_seed=101, out_dir=None)`, with the logger set to `Logger(folder=None, output_formats=[])`.
2. `_setup_learn(12000000 - 4194304, ...)` and `_mirror_install()`.
3. `clip_range` at the current progress.
4. Record the start digest of the parameters and Adam state.
5. Four times: `collect_rollouts(...)` (assert ok). Then, for each of the 64 mini-batches from `rollout_buffer.get(128)`:
   - normalise the advantages as SB3 does;
   - `pg = -mean(min(A*r, A*clip(r)))`;
   - `G_pg = autograd.grad(pg, action_net.bias)[14]`;
   - for b in (0.90, 0.60): `G_pen[b] = autograd.grad(tail3_upper_bound_penalty(mu, b), action_net.bias)[14]` and `share_gt[b] = mean(mu14 > b)`;
   - also record `mu14` quantiles, `sigma14` and `grad_norm` as P1 does.
6. Assert the digest is unchanged.

**Rule:**
- `rms = sqrt(mean(G_pg^2))` over all 512 mini-batches;
- `COEF_b = float(format(rms / (2*0.11), ".2g"))` for each b;
- `realised_pull[state][b] = COEF_b * mean(G_pen[b] over that state's 256)`.

**`coef.json` schema** (`"tailscreen-coef/1"`):

```json
{"schema": "tailscreen-coef/1",
 "rule": "COEF_b = RMS(G_pg)/(2*0.11) over 512 pooled mini-batches, format '.2g'; G_pg = d clipped surrogate (ratio 1, per-minibatch normalised advantages) / d action_net.bias[14]",
 "reference_excess": 0.11, "unit_pull_per_coef": 0.22, "seed": 101, "rollouts_per_state": 4, "batch_size": 128,
 "states": {"s42": {"zip_sha256": "...", "pkl_sha256": "...", "minibatches": 256, "G_pg_rms": 0.0, "G_pg_mean": 0.0,
                    "G_pg_sem": 0.0, "G_pg_abs_median": 0.0, "tail3_sigma": 0.0, "theory_rms": 0.0, "grad_norm_median": 0.0,
                    "per_bound": {"0.90": {"share_rows_gt_b": 0.0, "G_pen_per_coef_mean": 0.0, "realised_pull": 0.0,
                                           "realised_pull_over_rms": 0.0, "realised_pull_over_abs_G_pg_mean": 0.0},
                                  "0.60": {"...": "same keys"}},
                    "digest_before": "...", "digest_after": "...", "per_rollout": []},
            "s80": {"...": "same keys"}},
 "pooled": {"minibatches": 512, "G_pg_rms": 0.0, "coef_unrounded": 0.0},
 "COEF_90": 0.0, "COEF_60": 0.0, "coef_equal_by_formula": true,
 "clone": {"head": "...", "status": "", "branch": "..."}, "stack": {}, "utc": "...", "seconds": 0.0}
```

Write the file with `open(..., "x")`. The scorer and `preflight` recompute `COEF_b` from `pooled.G_pg_rms` and check it.

Expected [inferred]: `COEF_90 = COEF_60 = 2.7`, within 2.6-2.8. The realised pull at s42 is about 0.064 at b = 0.90 and about 1.19 at b = 0.60.

## 8. Pre-training gate, waves and the durable copy (written now; run after approval)

**`tools/gate.sh`:**
1. Refuses without `APPROVED`, or if sha256(`criteria.txt`) ≠ `APPROVED.criteria_sha256`.
2. Checks the clone state and `check_stack`.
3. Runs the gate-tier pytest with `MESOZOIC_TAILSCREEN_GATE=1`, writing `tests_out/gate_tier.txt`.
4. Writes `tests_out/gate_pass.json` (`{"head", "passed": n, "failed": 0, "utc"}`) only if everything passed.
5. Prints the sha256 of `criteria.txt`, every tool and `coef.json` for RUNLOG.

**`APPROVED`** (written by the coordinator only) is JSON:

```json
{"criteria_sha256": "...", "runlog_line": "...", "approved_utc": "...", "scratch_commit": "...",
 "COEF_90": 0.0, "COEF_60": 0.0, "coef_json_sha256": "...", "durable_branch": "..."}
```

**`tools/run_waves.sh`:**
1. Refuses without `APPROVED` and `gate_pass.json`.
2. Runs `copy_durable.sh init`.
3. For each wave in `WAVES`: checks disk, launches the three arms (`control`, `b90`, `b60`) of that (state, seed) as background `run_tailscreen.py` processes with logs in `logs/<name>.log`, waits for all three, and then runs `copy_durable.sh replicate <name>` for each one that exited 0.
4. Applies the criteria's crash rule. It never reruns because of a readout.

Start `measure_tailscreen.py --all-pending --follow` beside it on the fourth core.

**`tools/copy_durable.sh {init | replicate <name> | panels} [--dry-run]`:**

`--dry-run` is the only mode the build runs. It:
- assembles `SCREEN/durable_stage/` with the files the criteria list;
- writes `FILELIST.txt` and `SHA256SUMS`;
- makes the patch as `git -C CLONE format-patch --stdout cf80ecb1..HEAD > patches/tail3-upper-bound.patch`;
- refuses any file over 5 MB;
- prints the git commands it would run.

Real mode refuses unless all of these hold:
- `APPROVED` exists;
- `DURABLE_BRANCH` is pinned;
- at least one `runs/*/run_meta.json` exists, so training has started.

In real mode it works in a separate repository `SCREEN/durable_repo`, cloned from the origin URL (never from the scratch clone, a judge clone or the main checkout):
1. `git checkout --orphan <DURABLE_BRANCH>` on the first run.
2. Commits the staged files.
3. `git push origin <DURABLE_BRANCH>`.

The branch is never merged and never opened as a PR.

## 9. tools/score_tailscreen.py

```
PY -I SCREEN/tools/score_tailscreen.py [--panels SCREEN/panels.csv] [--runs SCREEN/runs] [--out SCREEN/scores]
```

Writes `score.json` and `score.txt`. Exit codes: 0 final, 10 incomplete, 11 provisional, 2 input error.

**Logic, exactly the criteria's:**
- Per replicate, from the R1 (5111808) and R2 (final) rows:
  - `cmd_mean = (cmd_R1 + cmd_R2)/2`;
  - RAIL if `tailup_R1 + tailup_R2 ≥ 1`, else PARKED if `cmd_mean ≥ 0.60`, else CURED.
  - A missing or LOST replicate is RAIL for its own bound. For CONTROL it is CURED at s42 and "no rail" at s80.
- Per bound B ∈ {BOUND90 (`b90`), BOUND60 (`b60`)} and per state, the reading is the first line that applies:
  - s42: RAIL, PARKED, OTHER, NOT SEPARATED, PASS;
  - s80: RAIL, PARKED, OTHER, PASS.
  - OTHER compares `sum(other_flags)` at R1 + R2 over B's 3 replicates with CONTROL's. Report every line that applies.
- Onset note: INFORMATIVE if at least 2 of 3 s80 CONTROL replicates had `tailup_flags ≥ 1` at any of their 9 labels.
- Bound verdict: the first that applies of RAIL, PARKED, OTHER and NOT SEPARATED gives NOT ADOPTED (reason); otherwise PASSED.
- Screen outcome:
  - A: both PASSED;
  - B: exactly one PASSED;
  - C: neither PASSED, with the routing text per reason and the named combinations ("the lower bound cleared the rail but not the posture"; "neither bound moved deterministic play below the line").
- Report only: dose ordering of the arm-mean CMD per state, and the 0.5M readings.
- PROVISIONAL: every item in the criteria's list. Its inputs are:
  - pins in `screen_common`;
  - `criteria.txt` against `APPROVED`;
  - each `run_meta.json`'s HEAD, status, start sha256, (arm, coef, b), seed and stack;
  - the diff file set;
  - `build_tier.json` and `gate_pass.json`;
  - `coef.json`'s mtime and utc earlier than every `run_meta.utc_start`, plus its rule;
  - each panel's tree digest, seeds and stack;
  - `validate/v1.json`.
- Incomplete: any of the 18 names without both R1 and R2 rows that is not declared LOST. LOST is declared by the maintainer and recorded in `SCREEN/LOST.txt`, one name per line, with its RUNLOG line.

**`score.json` keys:**
- `status` and `provisional_reasons[]`;
- `replicates{name: {arm, state, seed, class, cmd_R1, cmd_R2, cmd_mean, tailup_R1R2, other_R1R2, onset_any_tailup, missing}}`;
- `states{s42|s80: {BOUND90|BOUND60: {reading, lines_applying[]}}}`;
- `onset_note`;
- `verdicts{BOUND90|BOUND60: {verdict, reason}}`;
- `outcome{line: "A"|"B"|"C", text, to_maintainer[]}`;
- `report_only{dose_ordering, half_point}`;
- `pins{}`.

**`tests/test_score.py` (synthetic rows; build tier) must cover:**
- every state-reading line, for each bound and state;
- every verdict line;
- outcomes A, B (both directions) and C, with each routing reason and both named combinations;
- missing-replicate scoping: a missing b60 replicate never changes BOUND90's verdict, and the reverse;
- monotonicity: over all 3^3 CONTROL class vectors × CONTROL OTHER counts, turning any CONTROL replicate from CURED to PARKED or RAIL, or raising CONTROL's OTHER count, never moves a bound's reading to an earlier line;
- bound independence: any change to b60 rows leaves BOUND90's reading and verdict unchanged, and the reverse;
- each PROVISIONAL trigger;
- incomplete.

## 10. What the build delivers (and stops at)

1. The scratch clone and the bound commit (§2-3), clean, with the push URL disabled.
2. The measuring copy with its digests (§6).
3. Tools and tests (§4-9). Build-tier tests pass, in `tests_out/build_tier.{txt,json}`.
4. V1 passes exactly (`validate/v1.json`).
5. V2: `calib/coef.json` is written, `COEF_90` and `COEF_60` follow the rule, and the digests before and after are equal.
6. `copy_durable.sh --dry-run init` succeeds. Nothing is pushed.
7. The judge clones are clean before and after (`logs/judge_clean_*.txt`), and `/home/user/mesozoic-labs` is unchanged.
8. The criteria with the pins filled go back to the maintainer for approval. Not done in the build: `APPROVED`, `gate.sh`, waves, V3, real durable pushes.
