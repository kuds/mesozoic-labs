P47 step (A) is done. The four unpushed branches are in a new bundle, it restores from a fresh clone of origin, and both files are committed on the local branch `wt/pause`. Nothing was pushed.

**Commit:** `54d8afc12e33673ccb283e95c0e4837ddb1ea9cc` on `wt/pause`. Its parent is `b1d5e98e`, and it is in the worktree `/home/user/mesozoic-wt/pause`. The message ends with the two required lines and names no model anywhere else. It adds two files:
- `/home/user/mesozoic-wt/pause/hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.bundle`, force-added. `git check-ignore` exits 1, so it was not ignored anyway.
- `/home/user/mesozoic-wt/pause/hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.md`

**Bundle:**
- Size 728,669 bytes, sha256 `93f134d78cb9906158d682059504e0195eaeb622d215965fc170b482281b1fdd`.
- Made with `git bundle create ... wt/wp-l wt/wpc-ck wt/wpc-cb wt/wpc-ce --not b1d5e98e`.
- It holds 4 heads and 77 commits (45 + 15 + 14 + 3; no commit is shared).
- It requires 22 commits, and each one is an ancestor of `b1d5e98e`.

**Branches.** "Not on origin" counts merges in brackets. "No equal patch" is the count of those commits that `git cherry <working branch> <b>` marks with `+`. Every other local branch had 0 commits not on origin.

| Branch | Tip | Not on origin | No equal patch | Fate |
|---|---|---|---|---|
| wt/wp-l | 8bc47346 | 45 (13) | 27 | **Parked by P43.** Holds the 60 cm look sets, rebuilt and not frozen, all stopped at nominal_to_v0.4. Its last 4 commits (ebf2a85d, 5cff3e9f, f2c26b22, 8bc47346) exist only on this branch. The check `roundC__check_RB60_17nov.md` says `merge_after_fixes`: a test-only fix, then cherry-pick, because a direct merge conflicts. The other 23 commits counted as unique are on the working branch in another form: 19 inside 628d60b7, squashed there as 036ec9d6, and 4 as cherry-picks with the same subject. |
| wt/wpc-ck | 5c9fe633 | 15 (5) | 9 | **Superseded.** The look reading CK came in by cherry-pick on 16 Nov. All 9 have a same-subject commit on the working branch, and both CK files are identical at the two tips. |
| wt/wpc-cb | f2e6aa7e | 14 (4) | 6 | **Superseded.** The Drive bundle tool CB came in by cherry-pick. All 6 have a same-subject commit on the working branch, and its 5 files are identical at the two tips. |
| wt/wpc-ce | cd32e19a | 3 (2) | 0 | **Superseded.** CE is merged (9ecefa55, 7b4df135). Its one non-merge commit has the same patch as 67cb6f54 on the working branch. |

Seven of the unpushed commits lack the two attribution lines. All seven are plain `Merge branch ...` commits.

**Older bundle.** `worktrees.bundle` is unchanged (sha256 `d7e77eec…78ec3`, 751,594 bytes).
- On origin: wt/wp-s f5a0bf5e, wt/wp-t d8f1a535, wt/wpc-ce 8d8c4508, wt/wpc-cv c51c0dbf and wt/prereg-b 984f81db.
- Only in that bundle: wt/wp-l e2696b2a and wt/wpc-ck 11a23b43. Both are ancestors of the new tips, so the new bundle holds them too.
- Its 12 prerequisites are all on origin. It also verified "is okay" in the fresh clone.

**Restore test.**
- Clone: `git clone --no-checkout --single-branch --branch main` of origin into the scratchpad, then a fetch of the working branch (it gave b1d5e98e).
- `git bundle verify` printed "is okay", listed the 4 refs, "requires these 22 refs", hash algorithm sha1, and exited 0.
- Fetching `refs/heads/*` from the bundle gave 4 new branches. Each sha matched the local branch's: 8bc47346…, 5c9fe633…, f2e6aa7e…, cd32e19a….
- `git fsck --connectivity-only` printed nothing.
- I removed the scratch clone and its copy of the bundle.

**The .md** has the table, the older bundle's heads, the restore commands and this test record. The restore is `git bundle verify "$B"` then `git fetch "$B" 'refs/heads/wt/wp-l:refs/heads/wt/wp-l'`, after `git fetch origin claude/mesozoic-trex-stage2-refine-4jffre` if the clone was single-branch.

**Not done, and side effects:**
- `wt/pause` is not merged into the working branch and not pushed, so the new bundle and note exist only on this machine until the lead brings them in and pushes.
- I ran `git fetch origin` in the main checkout (allowed). It added the remote-tracking ref `origin/codex/triceratops-quadruped-design`.
- The main checkout is clean at b1d5e98e. I changed no code and deleted no branch or worktree.
- I made one short-lived set of files in `/tmp` while counting commits and removed them at once.
- Disk: 11 GB free at the end.
