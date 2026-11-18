## Archive a1 (P49): result

The archive is committed on a new local orphan branch, `claude/mesozoic-trex-stage2-archive-a1`, at **be47419033b8705e8943072bbea1ced2ed98b32a** (a root commit, 8 files). The worktree is `/home/user/mesozoic-wt/archive-a1`. Nothing was pushed, tagged or uploaded. The main checkout was only read: its `git status --porcelain` printed 0 lines before and after.

The archive covers 380 of the 449 files the tool lists. **69 frozen set arrays are not in it, and 52 of those exist only on this machine** (details under "Not archived").

### 1. `check` in the main checkout (read-only, run with lowprio)

It exited 1. Verbatim:
```
group         files matching missing wrong  records
picks-models      6        6       0     0  agree
picks-arrays     84       84       0     0  agree
k                60       60       0     0  agree
x                60       60       0     0  agree
j                30       30       0     0  agree
look             73       73       0     0  agree
sets            136       67      69     0  agree
  sets missing: stage2/sets/V6/75-M-H-10-X/look/nominal.npz; ... 63 more
not all here: python tools/restore_inputs.py restore (or copy-from DIR)
```
Each line also ended with the group's title, cut here. The last line, "not restored by this tool: ...", is also left out.

### Files kept, by group (present in the main checkout, each with its listed sha256)

| group | files | bytes |
|---|---:|---:|
| picks-models | 6 | 2,954,007 |
| picks-arrays | 84 | 54,654,576 |
| k | 60 | 38,419,843 |
| x | 60 | 41,310,856 |
| j | 30 | 19,190,577 |
| look | 73 | 52,147,553 |
| sets | 67 of 136 | 42,091,313 |
| **total** | **380** | **250,768,725 (about 251 MB)** |

- The total is under the 400 MB limit, so the job went ahead.
- The 67 set arrays are from V0, F25F9 and F25F9cl on 75-M-H-10.
- The largest single file is 3.5 MB.
- Every file, with its group, size, sha256 and part, is in `MANIFEST.json`.

### Not archived

**The 69 set arrays missing from the main checkout:**
- They are three frozen sets, 23 files each: V6 on 75-M-H-10, V6 on 75-M-H-10-X, and V6noF5 on 75-M-H-10.
- 17 of them (V6 on 75-M-H-10, refs and transitions) are in bundle s2-b1 on the pushed data branch.
- I compared sha256 against the lists: all 69 exist on this machine in `/home/user/mesozoic-wt/wp-p42`, and 65 also in `/home/user/mesozoic-wt/wp-l`.
- So **52 frozen arrays (V6 look files, V6-X, V6noF5) are only in worktrees on this machine.** I did not archive them, because the task limits the archive to the main checkout.
- Two ways to keep them:
  - a second commit on this branch adding a part made from wp-p42's files, which match the lists;
  - or rebuild them later with `restore --group sets` (not tried).

**Other git-ignored files under the study, left out:**

| what | files | bytes |
|---|---:|---:|
| eval traces `stage2/runs/**/eval/traces/` | 12,620 | 2,080,952,168 |
| `.venv` | 35,959 | 1,350,226,829 |
| logs directories | 190 | 19,572,346 |
| caches | 127 | 16,477,939 |
| zero-byte `done/` markers | 108 | 0 |
| `MUJOCO_LOG.TXT` | 1 | 357 |

The traces split by run:
- 2026-11-15-75cm: F25F9 86.1 MB, F25F9cl 976.9 MB, V0 895.3 MB, controls 5.9 MB
- 2026-11-16-75cm-addon/A: 116.8 MB

`not-restored` printed 0 files here for PPO checkpoints, Stage 1 eval traces and the round-one/two models. It listed 12,728 other git-ignored files: the traces plus the markers.

### Parts (ustar, uncompressed, study-relative paths)

| part | bytes | files | groups | sha256 |
|---|---:|---:|---|---|
| inputs-01.tar | 47,872,000 | 79 | picks-models, picks-arrays | 9d7ebcebcd29dfa6ab6d5d4fd4b7d6a4092a03e9fdc5e90ca110aab3282f78c2 |
| inputs-02.tar | 47,462,400 | 70 | picks-arrays, k | 9f601c5ad5d293424639af682ea9cfe6e9ccde1e949b71f126ea958cf6ef5ee2 |
| inputs-03.tar | 47,656,960 | 76 | k, x, j | 7d19b961d5011d751e4e29241d2c289ce58dfb356de73596a89f143a25ecfedf |
| inputs-04.tar | 46,940,160 | 63 | j, look | efd8d4bad3b77a245b2be92b99e231bcda4afc78f1b62ec4b8d1847ba4956b2a |
| inputs-05.tar | 47,278,080 | 78 | look, sets | fdd61ab034c8a9943d7d84e5a46be2cb8d8f9dac55c8ccede963e76e10d9a74f |
| inputs-06.tar | 13,885,440 | 14 | sets | 3ec094c999907105a1c609dea576195e61fd936e4ed06ce2a9812290f3dcc449 |

### Restore proof

1. **Extraction.** I extracted all 6 parts into a fresh scratch directory. It held 380 files and no others. Every file's sha256 matched both the manifest and the main checkout's file (0 mismatches).
2. **Fresh clone.** I cloned from origin into scratch. The pushed branch tip was already c5274905, so I fetched b1d5e98e and checked it out.
3. **`copy-from`.** I ran `restore_inputs.py copy-from <extracted dir>` in that clone with the main checkout's venv:
   - it copied 6 / 84 / 60 / 60 / 30 / 73 / 67 files, group by group;
   - it reported the 69 set arrays as "not made";
   - it exited 1.
4. **`check` in the clone.** Its output was identical to the main checkout's (diffed), and it exited 1. The clone's `git status` was clean.
5. **Committed blobs.** After the commit, I pulled the parts back out with `git archive` and their sha256 matched the manifest.
6. **Clean-up.** All scratch copies (extraction, clone, archive check) are removed. The disk had 11 GB free.

### Notes and things not done

- The main checkout's HEAD moved to c5274905 (P51) during the run. That commit's only change from b1d5e98e is one line in DECISION_LOG.md, so the input lists are the same.
- Not pushed. P49 says you may push this branch.
- The README restore steps fetch this branch from origin, so they only work once it is pushed. I tested only the equivalent local steps (the extraction and the `copy-from` above).
- Not tried:
  - restoring this archive together with bundle s2-b1;
  - `restore --group sets`.
- No code was changed.

Files:
- /home/user/mesozoic-wt/archive-a1/hardware/trex_form_study/stage2/archive/a1/MANIFEST.json
- /home/user/mesozoic-wt/archive-a1/hardware/trex_form_study/stage2/archive/a1/README.md
- /home/user/mesozoic-wt/archive-a1/hardware/trex_form_study/stage2/archive/a1/parts/inputs-01.tar … inputs-06.tar
- Scratch scripts and listings: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/a1/ (pack.py, listing.json, check_main.txt, copyfrom.txt, check_clone.txt)
