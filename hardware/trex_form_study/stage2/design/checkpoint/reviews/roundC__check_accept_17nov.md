Verdict: sound

The step's report is accurate, and the step was done the way it was asked. The acceptance run itself failed, though, so V12 has not passed. `ACCEPTANCE.json` has `ok: false` and exit 1. 35 of 36 checks are true; `resumed_from_drive` is false. The run stopped on a real code fault in `nb acceptance`'s kill condition, and the report described that fault correctly. Launch stays blocked until that code is fixed and the acceptance is run again.

What I re-checked myself (I changed no tracked file, committed nothing, and pushed nothing):

- **Record.** At `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.json`: sha256 is `2d1578aaf1c85b8ac10e79a96621c46cbe2896acf4fe3dd6c049428f66ac117f` and size is 29,466 bytes, both as reported. I recomputed its own `sha256` stamp (f19ba607…) and it matches. Its log shows START 21:54:37Z, END 22:13:46Z, EXIT 1, WALL_S 1148.85, and the final line names `resumed_from_drive`.
- **Line 2 against the manifest.** C1's MANIFEST.json and the scratch copy have the same file sha256 (25ab5e41…). Recomputing `sha256_json(content)` gives `9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc`. The manifest lists 19 files and bodies `["75-M-H-10"]`. Line 2 in both sessions reads `bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK`, which agrees.
- **Parts.** `git ls-remote` shows the data branch at ecf95a81 and C1 as the head of `refine-4jffre`. The three parts give identical full sha256 in all four places: the blob on the data branch, the `--parts` copy, the stand-in Drive copy and the manifest entry. ecf95a81's parent b914088b is an ancestor of C1, and the manifest is the same at C1 and at ecf95a81.
- **Fresh clone at C1.** The record has `clone_source` set to the GitHub URL (not the local fallback), both clone heads at C1, and `fresh_clones_hold_nothing_ignored` true. `clone2` today is at C1 with nothing modified or untracked; its only ignored paths are the three unpacked input directories.
- **Main checkout.** It is at C1 and `git status --porcelain` is empty. The only files newer than 21:50 are from the C1 commit at 21:51:12, before the run started. `.git/FETCH_HEAD` holds the data branch fetch at 21:54:11.
- **Bit-identity.** It prints the expected pair.
- **Items in the report.** These all match the record and the logs:
  - the four lines and both line 4 rates
  - the kill: 188.7 s, `model_1024.zip.part`, progress 1536, `on_drive []`
  - `refused_live` (41 s) and `refused_finished`
  - "resuming session 1 … from 0 steps", with both R5 sessions at `resumed_from: None` and session 1 at `restarted_from_zero: True`; session 0's updates went to `updates.discarded.s1.jsonl`
  - the `.part` file swept
  - the monitor readings, G8b, both clips (I ran `ffprobe` on `clip_4096.mp4`: exit 0, 3.03 s), R6 done, DONE.json done, and all eight remedies checks true
- **Fault diagnosis.** The cited lines are right: `nb.py:3614` and `:3637` are the `resumed_from` checks, and `:3916` is `part_exists`.
  - The failure is built into the code, not bad luck. `copy_checkpoints` skips checkpoint 0. `_copy_checkpoint` copies the three binary files first, holding each as a `.part` for 0.5 s (`COPY_PAUSE_S`), and writes the record last. `_drive_checkpoints` counts only `checkpoint_<n>.json`, and `run_cells` polls every 0.05 s.
  - So the first `.part` the poll sees is always a binary of the first checkpoint, at a moment when no whole checkpoint is on Drive. A rerun of this code will fail the same way.
  - Design 3.10 step 4 asks for a kill while a copy is in progress, and step 5 asks for a checkpoint to be copied back. Together they need a whole checkpoint already on Drive when the kill fires; the code never waits for one.

What I added beyond the report:

- **Resuming from a Drive checkpoint works.** The remedies part of the same run killed after checkpoint 1024 was on the stand-in Drive and deleted the local run directory. Its last session has `resumed_from: 1024` and reached done. So resuming from a copied-back checkpoint is shown, just not inside the main kill test with the wiped clone.
- **The `git:` source works from a fresh clone.** The acceptance itself always uses `BUNDLE_SOURCE="drive"`, so it did not test the data branch route the owner chose (P46), and the report said so. To check it, I made my own fresh clone of origin at C1 in my scratchpad and called `nb.inputs("s2-b1", <empty drive>, "git:claude/mesozoic-trex-stage2-bundle-s2-b1", set_name="V6", local=…)`. It printed `S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK` and unpacked 19 files. That was a direct call from this container, not the notebook's cells and not Colab. On Colab, cell 2 clones `https://github.com/kuds/mesozoic-labs.git` as origin, which is the same remote this fetch uses. I then deleted that clone to free disk.

Things that could still fail on Colab, or that are left open:

- V12 has not passed. It needs the kill-condition fix (wait until a checkpoint above 0 is whole on the stand-in Drive and a `.part` exists), with that fix tested, and then a rerun.
- The steps sheet at C1 (`stage2/tb/colab_steps.txt`, lines 69 and 159) still sets `BUNDLE_SOURCE` to `drive` and lists `git:<<DATA_REF>>` only as a fallback (line 341). P46 says git, so the sheet step (V13) needs to change this.
- The tiny V6 episodes were not added to `stage2/tb/dev_episodes.json` (D23). The report says this and leaves it to the lead.

Things I did not do: rerun the acceptance, test a fix, or run anything on Colab.

My scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch/check_accept/`: `verify_manifest.py` and `git_inputs.py`.
