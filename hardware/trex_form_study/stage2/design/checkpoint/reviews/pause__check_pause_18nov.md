Verdict: merge; tip: cd0c92f8c53da9602b786732337bda851f419f60

Verdict: merge. I checked the branch bundle, RESUME.md and the two pointer edits myself, from command output, and found no wrong number, hash, path or broken link. Four small points are listed at the end; none of them blocks the merge.

**What wt/pause holds (tip cd0c92f8)**
- It has two commits on top of b1d5e98e: 54d8afc1 (parent b1d5e98e), then cd0c92f8.
- `git diff --name-only b1d5e98e HEAD` lists exactly five files: the new bundle, its .md, RESUME.md, README.md and NEXT_SESSION_PROMPT.md. No file under designlab/, tools/ or tests/ changed.
- Both commit messages end with exactly the two required lines and name no model anywhere else.
- The worktree and the main checkout are both clean. The main checkout is at c5274905, the pushed P51 commit.
- `git merge-tree` of c5274905 and cd0c92f8 merges cleanly; c5274905 only adds one line to DECISION_LOG.md.

**1. The bundle (worktrees-2026-11-18.bundle)**
- Size 728,669 bytes, sha256 93f134d7…1fdd, the same as in HEAD's blob.
- I cloned origin fresh into the scratchpad (main, single branch) and fetched the working branch; it gave c5274905, which descends from b1d5e98e.
- `git bundle verify` printed "is okay", 4 refs, "requires these 22 refs", sha1, and exited 0. All 22 required commits are ancestors of b1d5e98e.
- I fetched `refs/heads/*` into restored/*. Each head equals the local branch: wp-l 8bc47346…, wpc-ck 5c9fe633…, wpc-cb f2e6aa7e…, wpc-ce cd32e19a…. `git fsck --connectivity-only` printed nothing.
- The four branches hold 77 commits and share none.
- My recount matches the .md table:

| Branch | Not on origin (merges) | `git cherry` + | `git cherry` - |
|---|---|---|---|
| wp-l | 45 (13) | 27 | 5 |
| wpc-ck | 15 (5) | 9 | 1 |
| wpc-cb | 14 (4) | 6 | 4 |
| wpc-ce | 3 (2) | 0 | 1 |

- wp-l's 27 split as stated:
  - 19 are inside 628d60b7, which is on wt/wp-l; the squash 036ec9d6 is on the working branch.
  - 4 have a commit with the same subject on the working branch.
  - 4 exist only on this branch: ebf2a85d, 5cff3e9f, f2c26b22, 8bc47346.
- The RB60 check reads "Verdict: merge_after_fixes": a test-only fix in test_stage2_moves_look.py, and a direct merge conflicts, so it must come in by cherry-pick. P43 parks the 60 cm sets.
- The superseded branches check out:
  - All 9 CK and 6 CB commits marked + have a commit with the same subject on the working branch.
  - The CK files and the five CB files show no diff against b1d5e98e.
  - CE's one commit is patch-equivalent to 67cb6f54. 9ecefa55 and 7b4df135 are merges on the working branch.
- The 7 commits without attribution are all "Merge branch …": 3 on wp-l, 2 on ck, 1 on cb, 1 on ce.
- The old worktrees.bundle is byte-identical at b1d5e98e and at HEAD (sha256 d7e77eec…8ec3) and verifies okay in the fresh clone.
  - On origin: wp-s, wp-t, wpc-ce 8d8c4508, wpc-cv and prereg-b.
  - Only in the old bundle: wp-l e2696b2a and wpc-ck 11a23b43. Each is an ancestor of the matching new tip.

**2. RESUME.md**
- **Links.**
  - My own script finds 63 of 63 relative links resolving, plus 23 of 23 in README.md and 1 of 1 in NEXT_SESSION_PROMPT.md.
  - Every link target is tracked at HEAD.
  - The archive README exists on be474190, which is the full sha the file gives.
- **Bit-identity check.** In the worktree, with the main checkout's venv, it printed exactly 45719d03…528d c0027992…c1be1c1. `regress check` printed "current (full tier, 256 of 256 …)" and exited 0.
- **Section 1, checked against the records:**
  - STAGE2_75_REPORT §1: 20 of 20 up to 6 ms; all fall at 8 ms; 0 to 3 of 20 at 10 ms; the stop passes at 0, 4 and 6 ms; no push or quick-turn test passes; the tail is over its limit from 0.35 m/s.
  - P44 / STATUS: 14 of 15 up, 25.04 s, N 0 against 6.
  - ACCEPTANCE.json: ok, 36 of 36 checks, killed while copying checkpoint 2048 and resumed from 1024.
  - Data branch: three parts totalling 33.8 MB.
  - Archive README: 380 files, 250,768,725 bytes, about 2.5 h to rebuild.
  - Gap arithmetic: 69 = 3 × 23, of which 17 are in s2-b1. I matched all 17 s2-b1 arrays (6 refs + 11 transitions) by sha256 against `restore_inputs`' frozen-set list, which leaves 52 missing.
  - The archive's part sha256s equal its MANIFEST.
  - P51 is in c5274905.
- **Section 2 (restart steps):**
  - The commands match tools/README (setup_venv, libosmesa6, 1 min 43 s and 1.4 GB, the 2.5 h restore of five groups) and the archive README.
  - `copy-from` and `restore` both take `--group`.
  - 679 passed, 107 skipped at e0ad284 is the last full-suite count on file; test_stage2_nb gave 74 passed, 1 skipped; tests/ holds 58 files.
  - Step 7 matches P50 and rulings 44 to 47; 1b5686f6 is the sheet's banner commit.
  - P29 caps Colab at 220 runtime-hours.
  - The 40M-step runs come from P44.
  - roundC__do_accept3_17nov.md holds the acceptance command.
- **Section 3 (open items):**
  - Every item STATUS 17 Nov 21:35 lists as parked is in the table: 60 cm (P43), S4, V0, add-on B and C, V10.
  - The other items are there and correct against their sources:
    - leftovers G1, J-vs-Pi, group j and A19: task list #10; group j has been in the tool since f5a0bf5e but was never timed in a fresh worktree;
    - A19's 18.0 W and 28.0 W;
    - buy gate: approved and proposed parts as stated in BUY_GATE_DRAFT;
    - carpet: P37 answer 5 replaces P36;
    - S4: SERVO_MODEL_DESIGN row S4;
    - V0, jaw cover and lr_probe: the -X head is probed at 1e-5;
    - r2l/index.json lists no fits;
    - owner questions: STATUS 16 Nov 21:13 point 5.
- **Section 4.** The carry-over paths exist, including servo_delay, --resume, acceptance, sheet-check and the runtime-hour tally.
- **README and NEXT_SESSION_PROMPT.** Both edits are small and right: one pause line plus the first "Read first" row, and one dated note with the rest of the prompt unchanged.

**Points, none blocking**
1. **The "only unpushed branches" sentence is out of date.** The bundle .md and commit 54d8afc1 say these four were the only local branches with commits not on origin. That was true at 03:55, but the archive branch `claude/mesozoic-trex-stage2-archive-a1` (be474190, one commit, about 251 MB, an orphan) was created at 03:58 and is local only; wt/pause is too. The archive belongs on its own push (P49), not in the bundle. The sentence could say "at 03:55 UTC, before the archive branch".
2. **One owner-facing phrase can mislead.** "V6noF5: … without the roll onto the middle claw" sits right after V6's list, which includes "body roll". WP_L.md defines it as "V6 without F5's toe-off"; "without the toe-off" would be clearer.
3. **A few open items are missing from the table.** All are reachable through the linked files:
   - P30's late scores from 3 to 5 new viewers (L1);
   - JETSON_BODY §9's other hardware points (heat, regulator, power box);
   - the decision log's older open F rows and A16.
4. **Section 1 says "It is all in the GitHub repository". That holds only after the lead does three things:**
   - merge wt/pause onto the working branch and push;
   - push the archive branch;
   - deal with the 52 missing arrays. They are only on this machine: all 69 are in /home/user/mesozoic-wt/wp-p42 with their listed sha256, and wp-l has 65. Either add them to the archive before this machine is deleted, or accept that they will be rebuilt (not yet tried).

   RESUME.md says this itself (step 0 and "One gap").

**What I did not do**
- I ran no test suite.
- I did not try `copy-from` from the archive or s2-b1 in a checkout.
- I changed no tracked file, committed and pushed nothing, and fetched nothing in the main checkout.
- I removed my scratch clone, the extracted bundles and the archive copy. Only my two checker scripts remain in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/pause/check_pause/. 11 GB free.

**Files**
- /home/user/mesozoic-wt/pause/hardware/trex_form_study/RESUME.md
- /home/user/mesozoic-wt/pause/hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.md
- /home/user/mesozoic-wt/pause/hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.bundle
