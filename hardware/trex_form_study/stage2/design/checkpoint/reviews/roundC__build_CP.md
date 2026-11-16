# build:CP (roundC): the agent's final report

**WP CP: built and committed** on `wt/wpc-cp` (worktree `/home/user/mesozoic-wt/wpc-cp`), not pushed, not merged.

**Commits:** e09698b (merge of `wt/wp-r`, for `s2io`), 16e2f77, ff7f833, 77cb22c, 44c56c6, 8aa5ce3. Six files: `designlab/stage2/{tbenv,schedules,s2load}.py`, `tests/test_stage2_{tbenv,schedules,s2load}.py`.

**Acceptance** (study directory, the brief's environment):
1. pytest on the three files: exit 0, `54 passed in 16.21s`.
2. `s2load check <job> 64`: exit 0, `S2 CHECKPOINT 64: intact; fingerprint 0.0 (... tolerance 1e-05); cross_version False`.
3. `s2load pack`: exit 0 (`ckpt_64.tar`, 2,600,960 bytes). `s2load import`: exit 0, `imported ...; intact; fingerprint 0.0`.
4. `git status --porcelain`: exit 0, empty.

The scratch job comes from `python tests/test_stage2_s2load.py <dir>`.

**Numbers:**
- Stub layout 140/102/17: model 2,420,316 bytes, statistics 6,671, fingerprint file 164,776.
- A training load followed by the same boundary continues bit for bit from checkpoints 0, 32 and 64 (62 tensors: policy, optimizer, statistics).
- `load_sb3_model` forwards `force_reset` (tested).
- 16 seeded faults are all caught; one test had to be tightened first.

**Readings chosen (please confirm):**
- `keep`: step 0 is no check's; check k's checkpoint is the first in [k·M, (k+1)·M).
- Inference `clip_range` is `ConstantSchedule(0.0)` too.
- `pair_problem` runs the repository's `checkpoint_pair_problem` in every load, and that unpickles the `.pkl`. D14 says only a resume reads it. A `.pkl` that does not unpickle here makes the checkpoint "not intact".
- Commands "uniform on their ranges": 0 to 1.05 and ±1.05.
- `load_checkpoint(env=)` takes the unwrapped vec env; a `VecNormalize` is refused.
- `import_tar` returns `ok: false` with `reason`; it raises only for an unreadable record.
- Unknown actor block names are refused, so `r5core.layout` must use section 3.1's names exactly.

**Deviations and additions:**
- Merged `wt/wp-r` to use `s2io`.
- The record gains `sha256` (checked at load), `num_timesteps`, `layout.blocks`, `fingerprint.clipped_rows` and `fingerprint.rule`.
- A `run` dict with keys other than the four is refused.
- Added `remove_checkpoint`, `write_validation`, `tar_name`.
- `s2load.main` assigns `MUJOCO_GL=disable` itself (it is not in 2.1's list).
- SB3's load reseeds python, numpy and torch with the model's seed, in inference loads too.

**Not done:** a load under another Python or library set. There is no second torch environment here; that is the smoke's Y9.

**Needs:** package CB's `stage2/tb/.gitignore`. Without it `checkpoint_<n>.json`, `.tmp` names and `ckpt_<n>.tar` are not ignored; my test checks them once the file exists.
