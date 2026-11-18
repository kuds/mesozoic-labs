## C2 is committed on `wt/c2` as `52b441d99a0e924892cddb1a083787080adddf4f` (parent C1 `cc437b6a`). The sheet check printed OK with exit 0, and `tests/test_stage2_nb.py` passed.

**Commit id:** `52b441d99a0e924892cddb1a083787080adddf4f` on `wt/c2`, in worktree `/home/user/mesozoic-wt/c2`. Nothing was pushed, tagged or uploaded.

**Steps 1 and 2 (start state and the record)**
- Before the reset, the worktree's uncommitted sheet was byte-equal (`cmp`) to the saved copy `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch2/v13_filled_sheet.txt` (sha256 d186f15e…8485).
- I ran `reset --hard cc437b6a…` and `clean -fd` in `/home/user/mesozoic-wt/c2` only. The clean removed the old untracked `ACCEPTANCE.json`.
- I copied the new `ACCEPTANCE.json`; its sha256 is `41d67f7f06f946ebfd18eea5a3c479ef266f2f5412a8ff501f176a1e9df2a2f8`, the same as the source. It is not git-ignored.

**Step 5, the sheet check (whole output; run from the worktree's study directory with the run environment)**
```
S2 SHEET CHECK: left for the smoke: <<LIKE THIS>> <<SMOKE_TOTAL_MIN>> <<MIN_SETUP>> <<MIN_Y1>> <<MIN_Y3>> <<MIN_Y5>> <<MIN_Y6>> <<MIN_Y8>> <<TABLE OF LAUNCHES>> <<BUDGET_STEPS>> <<MIN_LINE4>> <<RATE>> <<W>> <<VCPUS>> <<BUDGET_M>> <<HOURS>> <<FLOOR>> <<MIN_LINE4_LATE>> <<HOURS_PER_JOB>> <<BACKGROUND>> <<MIN_REBUILD>> <<...>>
S2 SHEET CHECK colab_steps.txt at cc437b6a44d5: OK
EXIT 0
```
Before it I ran `git fetch origin claude/mesozoic-trex-stage2-bundle-s2-b1`. FETCH_HEAD and `refs/remotes/origin/claude/mesozoic-trex-stage2-bundle-s2-b1` are both `ecf95a81…`, and ls-remote agrees. I also compared the three parts with `cmp` against that branch's blobs: they are byte-equal to the parts the acceptance run used.

**Test line**
- `73 passed, 2 skipped in 72.13s`, exit 0.
- Skips: "the Stage 1 arrays are not in this checkout" and "the real trainer on the S1 view; set STAGE2_SLOW=1".
- Log: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch2/c2fill/test_stage2_nb.log`

**Bit-identity:** both checkouts print `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. The main checkout is at C1 with `git status --porcelain` empty (0 lines), and the working branch's remote head is still C1.

**The filled lines (quoted from `stage2/tb/colab_steps.txt`, sha256 7601a443…5b94)**

Section 1:
```
   The code:    one fixed version of the repository, commit
                  cc437b6a44d575db63e89c63e2f9dc42c4424947
                This is REPO_REF in every step below. The address in 3.1 names the same 40 characters.
   The inputs:  bundle s2-b1, manifest 9227d8a0bf0b, 19 files in 3 parts. They are on a branch of the
                repository, claude/mesozoic-trex-stage2-bundle-s2-b1, and the notebook reads them from there.
   The rules:   fixed before any run, in three texts of the repository (in hardware/trex_form_study), each
                with its sha256 at the commit above:
                  STAGE2_TRACKB_PLAN.md
                    sha256 ba966378861e1b11664b338b1dae54e12d7da10e44bfb60260bde7c514e6d4a2
                  stage2/design/ROUND_C_DESIGN.md
                    sha256 93e041b9b33db2824e3045c7e28c51e0a7effbf1c033b6ef7c07d41adcf3540b
                  stage2/design/ROUND_C_RULINGS_17NOV.md
                    sha256 505991430a123c485cd5cf7f3e60cdd8dd5216d3397b91b9a8a6bf71c7a73764
```
- I computed the three sha256 values with `git cat-file blob C1:<path> | sha256sum`.
- The design and rulings hashes differ from the first try's; the plan's hash is unchanged.
- The template's sentence about the tag `s2-launch-1` is gone.

Section 2:
```
2. BEFORE DAY ONE (nothing for you to do)

   I have put the input files on a branch of the repository, claude/mesozoic-trex-stage2-bundle-s2-b1.
   The notebook reads them from there by itself. There is nothing to upload, and nothing for you to approve.
```

3.1:
```
          https://colab.research.google.com/github/kuds/mesozoic-labs/blob/cc437b6a44d575db63e89c63e2f9dc42c4424947/hardware/trex_form_study/stage2/stage2_training.ipynb
```
The notebook exists at that path at C1. This URL row is the only row over 115 columns.

Parameter tables (3.4 and 4.3), REPO_REF and the bundle:
- `REPO_REF cc437b6a44d575db63e89c63e2f9dc42c4424947`
- `BUNDLE_ID s2-b1`
- `BUNDLE_SOURCE git:claude/mesozoic-trex-stage2-bundle-s2-b1`

3.6 (a) is `Repository ref: cc437b6a…4947;` / `commit: cc437b6a…4947`. It is still marked "(expected, not yet seen)".

Lines 1 to 3, as the record printed them in session0 and session1:
```
              S2 LINE 1/4 commit: cc437b6a44d575db63e89c63e2f9dc42c4424947 (REPO_REF is a commit: yes;
              checkout clean: yes)
              S2 LINE 2/4 inputs: bundle s2-b1, manifest 9227d8a0bf0b, 19 of 19 files
              match their sha256: OK
              S2 LINE 3/4 G8a 75-M-H-10 set V6: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical
              (40368 physics steps): PASS
```
Lines 1 and 2 are each split over two rows to fit the width; the sheet says "on screen it is one line".

Section 8 (whole):
```
   Measured: the acceptance run of 17 Nov 2026 (23:29 to 23:48 UTC, on my machine). It made a fresh copy of the
   repository from GitHub at the commit of section 1, put the bundle's three files in a folder that stood in for
   your Drive, and ran the notebook's own boxes: two very short jobs (R5 and R6 on the 75 cm body, 4,096 steps
   each), a kill during a copy to that folder, a second fresh copy, and a resume to the end. All 36 of its
   checks passed. Its record is stage2/tb/bundle/s2-b1/ACCEPTANCE.json, sha256
   41d67f7f06f946ebfd18eea5a3c479ef266f2f5412a8ff501f176a1e9df2a2f8.
   Lines 1, 2 and 3 of 3.6 (b), (e) and (f) are quoted from that record: it printed them with these values, in
   both of its sessions. In those sessions line 4 read 463, 484 and 488 env steps/s (8 envs, 2 workers,
   4 vCPUs) on my machine: that is not Colab's rate, so the rate, the hours and the minutes of 4.5 and 4.6 come
   from the smoke. It did not measure the minutes of the smoke's items or of the setup (it ran no smoke, and its
   software was installed already), and it did not read the inputs from the branch: it took the same three
   files, byte for byte, from the folder that stood in for your Drive. The smoke reads them from the branch.
   Of the lines that begin "Run directory", the acceptance run printed "(new run)" in its first session. That
   session was then stopped by force while it copied a save to the folder that stood in for your Drive, after
   an earlier save had reached that folder whole. A second start made right after that printed "still running
   (LIVE.json written 29 s ago); nothing to do": nothing was running any more, but the stopped
   session's LIVE.json was that recent. The start after it printed "(resuming session 1: R5-75-M-H-10-s42 from
   1024 steps; 0 jobs done)" and went on from that save, copied back from the folder. That start
   had TAKEOVER set, which only I set: you never set it, and wait ten minutes instead (5.6). The line it
   printed does not depend on TAKEOVER: you see the same one.
```
- After that come the template's paragraph on the program's own wording and the "Expected, not yet seen" paragraph. As in the first try they now say:
  - "the commit, the bundle and the counts are the acceptance run's (above); the rate and the minutes are the <<...>> fields"
  - "The acceptance run ran on my machine and could not show them; I confirm each at the smoke".
- Where the values come from:
  - 23:29 is session 0's `started_at` (23:29:57Z) and 23:48 is the record's `at`.
  - 29 and 1024 are from `refused_live` and `session1`.
  - "installed already" is from the sessions' `setup.pip` being `[]`, with their versions equal to the pins.

**Other edits taken from the first try's wording**
- **Header:** the acceptance values are filled in and the rest come from the smoke. The "(expected, not yet seen)" note now says each is checked at the smoke.
- **3.6 (e) hint:** "could not be read from the branch".
- **7.1:** "I put that file right on the branch".
- **7.2:** keeps only `rebuild`, with "Before it, I put the bundle's one small model file on your Drive". I checked in `nb.inputs` that `rebuild` takes `models-*` from `<drive>/inputs/<id>/`. The `git:<<DATA_REF>>` fallback row is removed, because git is now the main source.
- **STILL OPEN:** now lists only the smoke's fields.
- Every other line is the template's text.

**Fields still open, and why**
- `<<LIKE THIS>>` and `<<...>>`: the template's own words for a field.
- `SMOKE_TOTAL_MIN`, `MIN_SETUP`, `MIN_Y1`/`Y3`/`Y5`/`Y6`/`Y8`: only the smoke measures them; the acceptance run ran no smoke and installed nothing.
- `BUDGET_STEPS`, `RATE`, `W`, `VCPUS`, `BUDGET_M`, `HOURS`, `FLOOR`, `MIN_LINE4`, `MIN_LINE4_LATE`, `HOURS_PER_JOB`: Colab's machine and the budget fixed after Y4. The local rates are not Colab's.
- `MIN_REBUILD`: from Y8.
- `BACKGROUND`: looked at during the smoke.
- `TABLE OF LAUNCHES`: see the queue below.

All of these are in `SHEET_SMOKE_FIELDS`, and none of them asks for a value known now.

**The queue: what I left out, and why**
- **The table of launches stays as the template's field.** The template marks it "FILLED IN AFTER THE SMOKE" and it is in `SHEET_SMOKE_FIELDS`, so the template does not ask for the queue now.
- **The first try's three rows are not carried over:** `launch-1a` R5:75-M-H-10:42, `launch-1b` R6:75-M-H-10:42, `launch-1c` R5:75-M-H-10:43.
- **The only queue text now is the template's own:** 3.4's "(leave it empty)" for the smoke, and 4.3's example `R5:75-M-H-10:42, R6:75-M-H-10:42`, which already names only 75-M-H-10.
- **When the table is filled, these stay out of launch series 1:**
  - 60-N-T-10, parked by the owner (P43);
  - 75-M-H-10-X, not in s2-b1 (ruling 44);
  - the fine-tunes, which need a parent checkpoint.
- **This conflicts with V13's "the queue per runtime".** You may want those rows in C2 after all.

**Step 4, `stage2/tb/dev_episodes.json` (two entries added; the six earlier ones are unchanged)**
- I followed the file's format: the same keys and key order as `gates2.dev_episode` writes, and `s2io.write_json`, which I confirmed rewrites the file byte for byte. I wrote the entries with a script, not with the `gates2 dev-episode` CLI. The CLI would stamp the run's done time as the listing time.
- Both entries:
  - set V6, body 75-M-H-10, state done, commit C1;
  - `done_at` is 2026-11-17T23:48:08Z (the record's `at`) and `listed_at` is 2026-11-18T00:01:31Z;
  - the reason says each was listed after the run under ruling 45, with the run's times.
  - The command is the acceptance command as the acceptance report gives it; I did not see it run.
- **`V12-acceptance-jobs`:** "R5 and R6 training env steps (tiny jobs)", count **14336**.
  - R5 in the run `acceptance`: 2048 steps in session 0 (killed during its check at 2048), then 3072 resuming from 1024.
  - R6: 4096.
  - R5 in the run `acceptance-remedies`: 2048 + 3072 in the same way.
  - Source: the trainer logs and `updates*.jsonl` under `/home/user/mesozoic-wt/tb-accept/drive/runs/`. The record does not hold these counts.
- **`V12-acceptance-checks`:** "check episodes: G8a (zero action and R0 at 0 and 10 ms), the R0L baseline, the monitor", count **64**:
  - line 3's G8a in the notebook's 2 sessions: 2 × 2 = 4;
  - the trainer's G8a short form at 3 job starts: 3 × 2 = 6;
  - the R0L baseline at 3 job starts: 3 × 6 = 18;
  - 6 whole monitor checks of 6 episodes each: 36.
  - Not counted: the two monitor checks at 2048 steps that the kills cut short. Clips are drawn from the monitor's traces, and G8b reads records; neither runs new episodes.
- **Not listed:** the first acceptance run on b7527a66 (21:54 to 22:13) also ran tiny episodes on V6. Its record is not committed, and I did not list it. You decide whether it belongs in the list.

**`git diff --stat cc437b6a44d575db63e89c63e2f9dc42c4424947 HEAD`**
```
 .../stage2/tb/bundle/s2-b1/ACCEPTANCE.json         | 797 +++++++++++++++++++++
 hardware/trex_form_study/stage2/tb/colab_steps.txt | 111 +--
 .../trex_form_study/stage2/tb/dev_episodes.json    |  26 +
 3 files changed, 885 insertions(+), 49 deletions(-)
```
- The commit message ends with exactly the two required lines.
- After the commit the worktree is clean.

**Open for your decision**
1. The sheet still opens with "*** DRAFT of 17 Nov 2026. DO NOT START FROM THIS DRAFT. ***", as the template and the first try do. It cannot go to the owner for the smoke as it stands. I did not change it.
2. The table of launches: see the queue above.
3. The `git:` source has not yet been run end to end through the notebook. The acceptance run read the parts from the stand-in Drive. `sheet-check` only verified that each part on the branch has the manifest's sha256.
4. 7.2's rebuild fallback depends on uploading about 1 MB (`models-75-M-H-10.tar`) to Drive, a size the route test has not tried.

**Not done:** no push, tag or upload. No change to code or tests. Nothing written in the main checkout. I did not delete the scratch directory `/home/user/mesozoic-wt/tb-accept`, and I ran no tests beyond `tests/test_stage2_nb.py`. My scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch2/c2fill/`: `fill.py`, `dev.py` and `commit_msg.txt`.
