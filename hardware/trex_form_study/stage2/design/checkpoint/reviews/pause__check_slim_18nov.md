Verdict: merge; archive tip: 9c2d2a52bd7d7c8c4661c49efaf76b23a44ff359; resume tip: aee38d0077191c6b1557b54144b67af3971bd4d8

## Independent review: the slim archive a1 and the RESUME.md fixes

**Verdict: merge.** Every claim I checked matches my own tool output, and I found no defect. I changed no tracked file and pushed nothing. The main checkout still shows 0 lines in `git status --porcelain --untracked-files=all`.

### (A) The archive branch, `claude/mesozoic-trex-stage2-archive-a1` at 9c2d2a52

**The branch itself**
- It is a root commit with no parents, and it is local only: `ls-remote origin` returns 0 lines.
- Its tree holds exactly five files under `hardware/trex_form_study/stage2/archive/a1/`: `MANIFEST.json`, `README.md` and `parts/frozen-01..03.tar`.
- The worktree `/home/user/mesozoic-wt/archive-a1` is clean, including ignored files. No files from the first build are left there.
- `wt/archive-a1-full` still points at be474190. Its reflog shows the rename at 04:22 UTC.
- The commit message ends with exactly the two attribution lines and names no other model.

**The 52 files** (my script `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/slim/check/verify_archive.py`, log in `verify_archive.log` there)
- The tool's `listed('sets')` gives 136 files. V6 and V6noF5 account for 69 of them.
- s2-b1's committed MANIFEST at `origin/...bundle-s2-b1` (ecf95a81) lists 17 sets files. All 17 are among the 69, with the same sha256. So 69 = 52 + 17.
- I extracted the commit with `git archive`, then the parts with tarfile's "data" filter:

  | part | bytes | files | sha256 |
  |---|---:|---:|---|
  | `frozen-01.tar` | 39,956,480 | 13 | `9baed6c2…` |
  | `frozen-02.tar` | 40,499,200 | 20 | `3ea1ab3b…` |
  | `frozen-03.tar` | 39,157,760 | 19 | `d2d3ca06…` |

  - Each part's size and sha256 equal the manifest's, and each is under 50 MB.
  - The raw headers are plain ustar (no pax headers): all regular files, mode 0644, uid and gid 0, mtime 0, empty user names, in path order.
  - Each part's members are exactly the manifest's files for that part.
- **The extracted files:** 52 files, 119,553,986 bytes. The set equals the 52 expected (none missing, none extra), and so does the manifest's file set.
  - sha256 mismatches: 0 against the manifest, 0 against the tool's list, 0 against the wp-p42 files. Size mismatches: 0.
  - By set: V6 on 75-M-H-10 has its 6 `look` files (17,227,350 bytes). V6 on 75-M-H-10-X has `look` 6, `refs` 6 and `transitions` 11 (52,293,386 bytes). V6noF5 on 75-M-H-10 has the same 23 (50,033,250 bytes).
- **The rest of the manifest:**
  - its source commit is 61ae6fa0;
  - its `in_bundle_s2_b1` section equals s2-b1's 17 sets files, at commit ecf95a81;
  - its 67 other sets files equal the tool's V0, F25F9 and F25F9cl files;
  - its group counts (6, 84, 60, 60, 30, 73, 67) equal the tool's;
  - the 380 other files total 250,768,725 bytes in the main checkout, as stated;
  - the main checkout's sets tally is 67 present, 69 missing, 0 wrong.
- f9dc89cc (wp-p42) is an ancestor of 61ae6fa0, with no diff in `stage2/sets` or in `restore_inputs.py`.

**The README restore, in my own fresh clone** (`fresh_clone.sh` and `fresh_clone.log` in the same check folder)
- I cloned from origin and detached at 61ae6fa0. I fetched the archive from the local repo (9c2d2a52) and s2-b1 from origin (ecf95a81). I symlinked `.venv`.
- `sha256sum` printed the three part hashes above, plus `11d7bd3e…` and `6739c8b7…` for s2-b1's two V6 parts. Extraction gave 69 files.
- Before copy-from, `check --group sets` printed `136 0 136 0 agree`.
- `copy-from X --group sets` printed "copied 69, already here 0". It listed the 67 V0, F25F9 and F25F9cl files as "not made" and exited 1.
- Then full `check` printed `sets 136 files, 69 matching, 67 missing, 0 wrong, agree`. The other six groups were all missing and all agreeing. Exit 1.
- My own hash pass found the 69 V6 arrays present and right, and 0 of the other 67 sets files present. Those 67 are expected to be missing in that clone.
- `git status` in the clone showed only `?? hardware/trex_form_study/.venv`, the symlink.
- I removed the clone and all extraction directories. 11 GB are free.

**The restore order:** I read the code. `restore_sets` skips any set whose files all match, so copying first and then running `restore` works as the README says.

### (B) `wt/resume-fix` at aee38d00

- **Branch:** one commit, whose parent is 61ae6fa0. Clean and not pushed. The message ends with exactly the two attribution lines.
- **diff --stat:** RESUME.md 99 lines changed and the worktrees note 6. 2 files, 63 insertions, 42 deletions. Every hunk belongs to the four requested items.
- **Archive statements** are updated:
  - the kept-table row: 52 files, about 120 MB, with s2-b1's 17;
  - "Not kept": 380 files, about 251 MB, rebuilt in about 2.5 h;
  - step 0 names 9c2d2a52;
  - step 3 gives 52 + 17 + 380 = 449 and the order (fetch both branches, `copy-from --group sets`, `restore`, `check`), and what it says to expect matches my reproduction;
  - the group-j row is corrected;
  - the "One gap" paragraph and the "archive's gap" row are removed.
- **03:55 UTC:** added in `worktrees-2026-11-18.md`. be474190 was committed at 03:58 UTC, so "came later" is consistent. At 61ae6fa0, RESUME.md had no sentence about the four branches being the only local ones with commits not on origin, as the report says.
- **V6noF5:** it now reads "V6 without F5's toe-off", matching WP_L.md line 16.
- **P51:** the new paragraph at the top matches the P51 row in DECISION_LOG at 61ae6fa0. The added words "museum outreach" also come from that row.
- **Links:** all 65 relative links in RESUME.md resolve, and the set of links is unchanged. The worktrees note has none.

### Points for the lead (not defects)
- RESUME.md step 0 pins 9c2d2a52. If the archive commit changes before the push, that line must change too.
- P49 says "so a restart needs no rebuild". The 67 frozen V0, F25F9 and F25F9cl arrays (about 42 MB) are not in s2-b1 and not in this archive, so they depend on `restore --group sets`, which has never been tried. The task asked for exactly 52 (about 120 MB, matching P49's size), and the README and RESUME.md state this caveat.
- The worktrees note's last paragraph ("They reach origin only after…") is out of date, and was left unchanged.

### Not done by me
- I did not run `restore`, a full `check` to exit 0, or any tests.
- I did not re-check the claim that two builds of the parts gave identical sha256.
- I did not check the 03:55 time against any source other than the commit times.

My scripts and logs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/slim/check/`. They hold no data copies.
