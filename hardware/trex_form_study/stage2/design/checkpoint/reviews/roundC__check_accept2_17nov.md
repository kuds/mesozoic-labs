Verdict: not_sound; record ok: False

## Verdict: not_sound. The acceptance run did not happen and there is no record. The do-agent's report is accurate and it was right to stop.

The step's goal was an acceptance run on C1 472e8d31, and that run never started. There is no ACCEPTANCE.json, so `record_ok` is false: there is no `ok` field to read. Check 1 cannot pass. I checked every claim in the report and found none false, apart from one minor overstatement (point 5 under check 1). The refusal is the check working as designed, not a code fault. The cause is the lead's own handover commit 6abd75da ("Status: ... (handover)"), which changes only STATUS.md (+14 lines), is a child of C1, and was committed in the main checkout at 23:19:57Z according to the reflog. It was then pushed, so the working branch's tip moved past C1 about 90 s before the run started at 23:21:29Z.

## Check 1: the acceptance run

1. **Why it refused.** I ran `git ls-remote origin` at 23:22:35Z and again at the end of my review:
   - `refs/heads/claude/mesozoic-trex-stage2-refine-4jffre` = 6abd75da6a6bed54598633955292c29ff85e86af
   - `refs/heads/claude/mesozoic-trex-stage2-bundle-s2-b1` = ecf95a81a6ebb43e0331b5c1db74bdea7e74aaed
   - None of the 1253 remote refs has 472e8d31 at its head, and there is no `s2-launch` tag.
2. **The code matches the design.** In `nb.py` lines 3899-3907, `acceptance` refuses when `ls-remote` succeeds but does not list the ref as the head of any ref. It falls back to the remote-tracking ref only when `ls-remote` itself fails. Design 3.10 step 1 says "`git ls-remote origin` must list the ref's commit on the pushed branch", and ruling 45 says "the launch commit C1 is the new tip".
3. **I reproduced the refusal.**
   - I ran the same command, with `--scratch` and `--record` pointed at my own scratch directory: exit 4, about 1 s.
   - It printed the same line: `S2 REFUSED acceptance: 472e8d31... is not the head of a branch or tag of the remote https://github.com/kuds/mesozoic-labs: the acceptance run needs the pushed launch commit`.
   - It created neither the scratch directory nor the record; the refusal comes before the code deletes or creates anything.
4. **The log and the code the run used.**
   - `acceptance2.log` shows: START 23:21:29Z, the refusal line, EXIT 4, WALL_S .685253831, END 23:21:30Z.
   - The main checkout's `designlab/` is identical at 472e8d31 and at HEAD 6abd75da (empty diff), so the code that refused was C1's own.
   - `nb acceptance`'s options at C1 are `--notebook --ref --parts --scratch --record`, the same as at b7527a66, so the command was right as given.
5. **One small inaccuracy in the report.** It says commits in the main checkout could upset `main_checkout_unchanged`. That check compares `git status --porcelain` before and after the run (lines 3909 and 3996), and a clean commit leaves that empty. Only uncommitted edits or untracked files during the run would upset it.
6. **Steps 1 to 3 were done correctly.**
   - `/home/user/mesozoic-wt/tb-accept` is gone and is not in `git worktree list`.
   - All three parts are byte-equal (`cmp`) to `git cat-file blob ecf95a81:.../parts/<name>`.
   - Their sha256 and sizes equal C1's manifest:

     | Part | sha256 | Bytes |
     |---|---|---|
     | `models-75-M-H-10.tar` | 2e9a55fc...6917 | 972800 |
     | `V6-75-M-H-10-1.tar` | 11d7bd3e...2387 | 16281600 |
     | `V6-75-M-H-10-2.tar` | 6739c8b7...ba39 | 16558080 |

   - The MANIFEST.json beside the parts is byte-equal to C1's, to the data branch's and to 6abd75da's.
     - Its sha256 is 25ab5e41...3480, 10878 bytes.
     - I recomputed the content hash with `bundle.content_sha256`: 9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc. Bundle id s2-b1.
   - The first run's record is kept as `/home/user/mesozoic-wt/tb-accept-bundle/stage2/tb/bundle/s2-b1/ACCEPTANCE.v12-b7527a66.json`.
     - Its sha256 is 2d1578aa...117f, the same value the earlier check report gives.
     - It reads ok=false, ref b7527a66, with the one failed check `resumed_from_drive`.
7. **The house rules held.**
   - The bit-identity check prints 45719d03...528d c0027992...c1be1c1.
   - The main checkout's `git status --porcelain` is empty, and HEAD is 6abd75da.
   - The `<<LIVE_S>>` and `<<RESUME_STEPS>>` sources the report names are right: `nb.py` lines 4288-4290 read `attempts.refused_live.lines.live` and `attempts.session1.lines.resume`.
8. **What remains unverified because the run did not happen:**
   - ruling 45(1)'s new kill condition, end to end;
   - `resumed_from_drive` and `killed_during_a_copy`: the kill step, the checkpoint that was whole on the stand-in Drive, and the number of steps session 1 resumed from;
   - lines 1 to 3;
   - the run-directory lines;
   - the values for `<<LIVE_S>>` and `<<RESUME_STEPS>>`;
   - the record and its sha256;
   - `dev_episodes.json`.

   C2, which needs ACCEPTANCE.json, cannot be made from this step.
9. **Colab is not affected by the tip moving.** The notebook's cell 2 makes a full clone, runs `git fetch origin REPO_REF` and then `checkout --detach`. I ran `git fetch --depth 1 origin 472e8d31...` into an empty throwaway repo: exit 0, and FETCH_HEAD^{commit} is 472e8d31. So only the acceptance's own precondition breaks.

**For the lead.**
- **Option (a)** (take 6abd75da as C1) fits P46 and ruling 46, under which only the working branch and the data branch are pushed. 6abd75da differs from 472e8d31 only in STATUS.md: the code, notebook, template and manifest are identical. The sheet, record and status text would then name 6abd75da.
- **Option (b)** needs either a new pushed ref, which goes against ruling 46, or rewriting the working branch, which is forbidden.
- **Either way:** do not push to the working branch between naming C1 and the moment the acceptance starts. The tip is checked only at the start; the later clones check out C1 by SHA.

## Check 2: ruling 46's edit

1. **The notebook diff is one line.** `git diff 6b70887c 472e8d31 -- .../stage2_training.ipynb` changes one line in cell 0's intro.
   - It removes the link "[the file at the launch tag](.../blob/s2-launch-1/.../colab_steps.txt)".
   - It adds "filled in for the launch commit". That is a wording change beside the link, as ruling 46 describes; nothing else in the notebook changes.
   - The notebook at C1 parses as JSON (nbformat 4, 9 cells) and contains no `s2-launch` or `blob/`. The `nbformat` module is not installed, so I did not run a schema validation.
2. **Other files in C1 that mention `s2-launch` (`git grep`):**
   - **`colab_steps.template.txt`**, lines 28 and 54: expected.
   - **`stage2/tb/colab_steps.txt`, the one finding.** It is tracked, byte-identical to the template, and headed "DRAFT ... DO NOT START FROM THIS DRAFT". Its line 28 has the tag sentence and line 54 the Colab URL `blob/s2-launch-1`. The notebook's intro names this very file "in the repository", and at the launch commit it stays the draft with the dead tag URL; the filled sheet replaces it only from C2 onward. The banner and the intro's "the lead sends its final text" limit the risk, so this is low severity.
   - **`tests/test_stage2_nb.py`:**
     - line 860 makes its own tag in a scratch repo under `tmp_path`, and lines 894-930 use it;
     - line 628 asserts the template still holds the tag URL;
     - lines 2491 and 2949 pass `REPO_REF="s2-launch-1"` only as a test parameter.

     None of these sends a reader anywhere.
   - **`ROUND_C_DESIGN.md:445`** still says the sheet gives "the lightweight tag s2-launch-1 on it for the notebook's URL". This is stale design text that ruling 46 supersedes but did not amend.
   - **STATUS.md:749 and three review reports** mention the tag only as history.

## What I did and did not do

- **Done:**
  - ran `ls-remote` (read only);
  - ran the refused acceptance probe into `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch2/check_accept/`, which wrote nothing;
  - fetched C1 by SHA into a throwaway repo there, now deleted.
- **Not done:**
  - I changed no tracked file and made no commit or push.
  - I did not run the acceptance with another ref, because that is the lead's decision.
  - I ran no pytest.
  - I could not verify the do-agent's HEAD at its start, the 1.2G size of the removed directory, or that it pushed nothing. I can only see the remote as it stands now.
- **Copies kept in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/launch2/check_accept/`:** `lsremote.txt`, `MANIFEST.C1.json`, `MANIFEST.data.json`, `nb_C1.ipynb`, `sheet_C1.txt`, `template_C1.txt`, `grep_launch.txt`.
