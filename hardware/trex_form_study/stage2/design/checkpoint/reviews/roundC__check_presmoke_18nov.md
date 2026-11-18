Verdict: merge; tip: 860755fdfedd790f3d5ae41364e4214ee62b7be9; git probe ok: True

## Verdict: merge. wt/c2's tip is 860755fd, one commit on top of 1b5686f6. The git-source probe passed.

## 1. (A) My own git-source probe from a fresh clone at C1

**Clone.** `df -h /` showed 12G free. I ran cell 2's own commands by hand into `/home/user/mesozoic-wt/tb-git-probe-check`:
- `git clone --no-checkout https://github.com/kuds/mesozoic-labs.git`
- `git fetch origin cc437b6a…`
- `rev-parse FETCH_HEAD^{commit}` gave cc437b6a44d575db63e89c63e2f9dc42c4424947
- `checkout --detach`

Facts about the clone:
- It took 28.7 s and is a full clone (not shallow); `.git` is 448M.
- The data branch came with the clone: origin/claude/mesozoic-trex-stage2-bundle-s2-b1 is at ecf95a81, before cell 4 ran.
- C1's bundle directory holds only MANIFEST.json (blob 99b0fcd2), which is the same blob as on the data branch.

**Run 1: the notebook's own cells, not just the inner call.** This goes further than the do step. I ran the clone's own `stage2_training.ipynb` code cells 1 to 4 as Python, in local mode (there is no /content here):
- Script: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/presmoke/check/run_cells.py`
- Values: S2_PARAMS set to section 3.4's values (RUN_ID smoke-1, SMOKE true, QUEUE empty, BUNDLE_ID s2-b1, BUNDLE_SOURCE `git:claude/mesozoic-trex-stage2-bundle-s2-b1`), with DRIVE_DIR and LOCAL_DIR under my scratch.
- How cell 4 ran: through `s2_step("inputs")`, which is `python -m designlab.stage2.nb inputs --params`. So it ran `_step_inputs` too: the LIVE.json ownership check, QUEUE.json and the session note.

What it printed:
- Exit 0, wall time 3.37 s; cell 4 alone took 2.34 s.
- Line 1: `commit: cc437b6a… (REPO_REF is a commit: yes; checkout clean: yes)`
- Run directory: `…/drive/runs/smoke-1 (new run)`
- Line 2: `S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files match their sha256: OK`. This is word for word sheet 3.6 (e).

What cell 4 left behind:
- **FETCH_HEAD** was C1 before cell 4 and `ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed branch 'claude/mesozoic-trex-stage2-bundle-s2-b1'` after it.
- **Parts:** the three parts landed in `<local>/inputs/s2-b1/parts`: 16281600, 16558080 and 972800 bytes. Each has the manifest's sha256, and its `git hash-object` equals the data branch's blob (66a5dff9, e6dc21dc, 38acab08).
- **Unpacked files:** 19 of 19 files match the manifest on disk: 2 in round2/refine/feet_75/models, 6 in stage2/sets/V6/75-M-H-10/refs and 11 in …/transitions. All are git-ignored, so `git status --porcelain` stayed empty.
- **Stand-in Drive:** it held only runs/smoke-1/{QUEUE,SESSION_0,LIVE,STATUS}.json and no inputs.
- **QUEUE.json** has bundle content_sha256 9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc and source `git:…`.
- **SESSION_0.json** has files_ok 19 of 19 and source `git:…`.

**Run 2: the inner call directly, as the do step ran it.** I called `nb.inputs("s2-b1", drive2, "git:…", set_name="V6", bodies=["75-M-H-10"], local=local2, expect=None)`. The signature is `inputs(bundle_id, drive, source='drive', *, set_name=None, bodies=(), local=None, expect=None)`. The values come from the code: `_bodies` gives [SMOKE_BODY] and the default SET is V6.
- Exit 0, 2.10 s wall, the same line 2.
- The dict: ok true, source `git:claude/mesozoic-trex-stage2-bundle-s2-b1`, content_sha256 9227d8a0…f9dc, files_ok 19 of 19, problems [], sets {V6: {75-M-H-10: true}}.
- unpacked: written 0, left 19. That is because this was a second pass in the same clone, so it also shows that re-running cell 4 works, which the do step did not try.

Clone, drives and local directories are removed. Bit-identity printed the expected pair before and after. The main checkout is clean.

**Do step's report (A):** it matches what I saw on every number: line 2, the dict, the part sizes and blobs, FETCH_HEAD ecf95a81, the full clone and the empty drive. Its choice not to use `nb local-run` is right: local-run uses the Stage 1 view with BUNDLE_ID "", so it never reaches cell 4's bundle or the `git:` path. It is also right that `nb inputs --params` needs cell 3's session first. My run 1 adds the wrapper it did not exercise, and that wrapper passed.

**What this probe cannot stand in for on Colab:**
- Cell 2's IN_COLAB branch running the clone itself, and the /content paths.
- Setup's pip and apt installs, and Colab's Python.
- The real Drive mount: cell 4 writes QUEUE.json onto Drive.
- Colab's git version (2.43.0 here) and its direct, unauthenticated route to github.com (here traffic went through a proxy).
- How long the 448M clone takes on Colab.
- How Colab shows git's stderr.
- The data branch is fetched by name, not pinned. If it moves before the smoke, cell 4 takes the new head. The sha256 check against C1's manifest would then turn line 2 into FAILED; it would not be silently accepted.

## 2. (B) The dev_episodes.json entry: correct, and counted the same way as C2

- **The diff against 1b5686f6** is the one entry, `V12-acceptance-b7527a66`. Its key order matches the other entries, and the file round-trips through `json.dumps(indent=1)` byte for byte: ASCII, no trailing newline. The step/set/body key does not collide with any other entry.
- **Against the first run's record** (sha256 2d1578aa…, ok false, 35 of 36, only resumed_from_drive false):
  - Kill at 21:58:11Z on `model_1024.zip.part`, progress 1536, on_drive [].
  - Session 0 printed line 4, and session 1 printed "from 0 steps".
  - The remedies run was killed on "checkpoint 1024 … is on the stand-in Drive"; its last session resumed from 1024.
  - Session 0's started_at is 21:55:03Z and the record's at is 22:13:46Z. These match the entry's from/done_at, picked the same way as C2's (23:29:57Z, which is that session 0's started_at).
  - The command is the do report's command. The commit is b7527a66.
- **The counting method** is C2's: steps up to each kill, plus the steps after each resume. 2048 + 4096 + 4096 + 2048 + 3072 = 15360 as the upper bound; the lower bound is 13824.
  - I confirmed train2.py is unchanged between b7527a66 and C1.
  - The ≥1536 bound holds: line 4 needs the trainer's `first_three` rate, written at the third 512-step rollout (TINY n_steps 64 × 8 envs).
  - The ≤2048 bounds rest on the trainer pausing for the monitor check at 2048. That is reasonable. The remedies kill came after 96.4 s here and 93.6 s in the second run, where the logs show the kill fell during the check at 2048.
  - Remedies session 0 printed no line 4 in this run. That does not tighten the bound, because the launcher prints line 4 later in its cycle than the copy step.
- **Check episodes:** 4 + 6 + 18 + 36 = 64, the same terms as C2's checks entry. The record confirms that line 3 printed in exactly two sessions, and the remedies sessions run cells 1-3 and 7-8 only (step6_note).
- **The unknowns are named honestly:** whether R5's restart from 0 reran G8a's short form or the baseline (at most 8 more episodes), and the removed logs.
- **Observation, not blocking:** as instructed, it is one entry, so `count` holds steps only and the 64 check episodes are only in the reason, whereas C2 split the second run into two entries. The do step offered the lead the split.

## 3. Banner commit 1b5686f6: right

- It changes only lines 4-7 of colab_steps.txt (one hunk, @@ -4,4 +4,4).
- It is ASCII, with line lengths 76, 112, 111 and 93. The only line over 115 in the whole sheet is line 56, the URL (169), which is older and not part of the banner.
- Its attribution lines are right.
- The text is plain:
  - day one is the smoke only (sections 1 to 3);
  - the sheet comes again after the smoke;
  - "Do not go past section 3 until then" agrees with 3.7's "Do not start a launch before I say so";
  - it still points to the closing list.
- Small nits, none blocking:
  - "the smoke measures it" is loose for TABLE OF LAUNCHES and BUDGET_STEPS, which are decided after the smoke from what it measured.
  - Section 8 and the closing heading still say "this draft".
- The remote working branch is now at 1b5686f6 (ls-remote), so it has been pushed since the task text was written.

## 4. wt/c2's head 860755fd

- After `git fetch origin claude/mesozoic-trex-stage2-bundle-s2-b1` (ecf95a81), `nb sheet-check … --before-smoke` exited 0 and printed `S2 SHEET CHECK colab_steps.txt at cc437b6a44d5: OK`, plus the list of fields left for the smoke.
- `tests/test_stage2_nb.py`: 73 passed, 2 skipped (66 s).
- The commit message ends with exactly the two attribution lines and names no model anywhere else.
- Its parent is 1b5686f6, the remote working branch's head, so it fast-forwards. The worktree is clean.

## Not done

I changed no tracked file and made no commit, push, tag or upload. Nothing ran on Colab. I did not run the full test suite.

My scripts and outputs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/presmoke/check/`: `run_cells.py`, `direct_inputs.py`, `cells_out.txt`, `direct_out.txt` and `params.json`.
