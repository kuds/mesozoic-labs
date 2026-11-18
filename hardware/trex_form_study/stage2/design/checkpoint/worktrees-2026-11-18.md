# The unpushed local branches, archived 18 Nov 2026 (P47)

The owner paused T. rex development on 18 Nov 2026 (`DECISION_LOG.md` P47 to P50). Four local branches of the cloud
machine held commits that were on no branch of `origin`. They are kept in one git bundle beside this note:

- File: `worktrees-2026-11-18.bundle`, 728,669 bytes,
  sha256 `93f134d78cb9906158d682059504e0195eaeb622d215965fc170b482281b1fdd`.
- Made with `git bundle create worktrees-2026-11-18.bundle wt/wp-l wt/wpc-ck wt/wpc-cb wt/wpc-ce --not b1d5e98e`.
- It holds 4 heads and 77 commits. Its 22 prerequisites are all ancestors of the working branch's pushed tip
  `b1d5e98e` (`origin/claude/mesozoic-trex-stage2-refine-4jffre`), so a clone that has the working branch can read it.
- These were the only local branches with commits not on `origin` at 03:55 UTC 18 Nov (`git rev-list --count <b>
  --not --remotes=origin` was 0 for every other branch under `refs/heads`). The archive branch
  `claude/mesozoic-trex-stage2-archive-a1` came later and is not in this bundle: it goes to `origin` by its own push
  (P49).

## What it holds

"Not on origin" counts every commit of the branch that no `origin` ref reaches (merges in brackets). "No equivalent
patch" counts those that `git cherry <working branch> <b>` marks `+`.

| Branch | Tip | Not on origin | No equivalent patch | Last subject | Fate |
|---|---|---|---|---|---|
| `wt/wp-l` | `8bc473469a71` | 45 (13 merges) | 27 | Stage 2 look sets (WP L): set V6 on 60-N-T-10-X rebuilt under rulings 18, 22, 23, 26, 28 (stopped at nominal_to_v0.4; not frozen) | **Parked (P43).** The 60 cm look sets (V6 and V6noF5 on 60-N-T-10, V6 on 60-N-T-10-X), rebuilt and unfrozen, all stopped by `nominal_to_v0.4`. Only here: the last four commits (`ebf2a85d`, `5cff3e9f`, `f2c26b22`, `8bc47346`); the working branch has older 60 cm records. Their check (`reviews/roundC__check_RB60_17nov.md`) says `merge_after_fixes`: a test-only fix in `tests/test_stage2_moves_look.py`, then bring in by cherry-pick (a direct merge conflicts). The rest is on the working branch in other form: 19 of the 27 are inside the reviewed tip `628d60b7`, brought in squashed as `036ec9d6`; 4 code commits (rulings 18, 23, 26 and the rule-text one) have same-subject cherry-picks; the 75 cm rebuild's 5 commits are patch-equivalent. |
| `wt/wpc-ck` | `5c9fe6330675` | 15 (5 merges) | 9 | Stage 2 round C (WP CK): lookread: the tail term fails when the robot's own tail amplitude is under 0.02 rad | **Superseded.** The look reading (CK) was built, reviewed and fixed (`reviews/roundC__{build,review,fix}_CK.md`) and brought in by cherry-pick on 16 Nov (STATUS, 16 Nov 21:13 UTC block). Each of the 9 has a same-subject commit on the working branch, and `lookread.py` and `test_stage2_lookread.py` are identical at both tips. Kept for its builder's history. |
| `wt/wpc-cb` | `f2e6aa7e1c6e` | 14 (4 merges) | 6 | Merge the working branch into wt/wpc-cb (the env's small fix round, the added rulings, the servo model note) | **Superseded.** The Drive bundle tool (CB) and its follow-ups (`reviews/roundC__{build,review,fix}_CB.md`) were brought in by cherry-pick on 16 Nov. Each of the 6 has a same-subject commit on the working branch, and its five files (`bundle.py`, `s2load.py`, `stage2/tb/.gitignore` and the two test files) are identical at both tips. |
| `wt/wpc-ce` | `cd32e19a7201` | 3 (2 merges) | 0 | Stage 2 round C (WP CE, third fix round): the wording file follows the review of the second fix round; a test for each point | **Superseded.** The training env (CE) is merged (`9ecefa55`, then `7b4df135`; `reviews/roundC__{build,fix,review_fix}_CE.md` and the `_CE_minor` reports). Its one unpushed non-merge commit is patch-equivalent to `67cb6f54` on the working branch. Only its two merge commits are not on origin. |

Seven of the unpushed commits lack the two attribution lines, all of them plain `Merge branch ...` commits: 3 on
`wt/wp-l`, 2 on `wt/wpc-ck`, and 1 each on `wt/wpc-cb` and `wt/wpc-ce`. The bundle keeps them as they are. By the
rule in STATUS (16 Nov, 21:13 UTC, point 3), such a branch's package is brought in by cherry-pick, not by a merge.

## The older bundle (15 Nov)

`worktrees.bundle` (751,594 bytes, sha256 `d7e77eecf6e33188d663080db49351429d4215bbbf7d9a39d6c3193fa0c78ec3`,
committed in `08051d5a`) is kept unchanged. It has 7 heads:

- **On origin** (each is an ancestor of the working branch): `wt/wp-s` `f5a0bf5e`, `wt/wp-t` `d8f1a535`,
  `wt/wpc-ce` `8d8c4508`, `wt/wpc-cv` `c51c0dbf`, `wt/prereg-b` `984f81db`.
- **Only in that bundle**, at those old tips: `wt/wp-l` `e2696b2a` and `wt/wpc-ck` `11a23b43`. Both are ancestors of
  the tips above, so the new bundle holds them too.
- Its 12 prerequisites are all on origin. `git bundle verify` said it "is okay" in a fresh clone of origin with the
  working branch fetched.

## Restoring a branch in a fresh clone

The bundle needs the working branch's commits. A plain `git clone` of origin fetches every branch, the working branch
included. After a `--single-branch` clone, first fetch it:
`git fetch origin claude/mesozoic-trex-stage2-refine-4jffre`.

The two commands, run from the clone's top directory, with `B` this file's path in it:

```
B=hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.bundle
git bundle verify "$B"
git fetch "$B" 'refs/heads/wt/wp-l:refs/heads/wt/wp-l'
```

Replace `wt/wp-l` with the branch you need, or fetch all four with `'refs/heads/*:refs/heads/*'`. If the clone's
checkout does not hold this file, take it from a commit that does:
`git show <commit>:hardware/trex_form_study/stage2/design/checkpoint/worktrees-2026-11-18.bundle > /tmp/w.bundle`.

## The check made on 18 Nov

This was done in a scratch directory that was then removed. The clone was
`git clone --no-checkout --single-branch --branch main https://github.com/kuds/mesozoic-labs`, then
`git fetch origin claude/mesozoic-trex-stage2-refine-4jffre` (it gave `b1d5e98e42604e87d635f4cd2f0fcb98063f18db`).

`git bundle verify` printed:

```
../w18.bundle is okay
The bundle contains these 4 refs:
8bc473469a7175e302e01ce4e4a16b3dd31371ce refs/heads/wt/wp-l
5c9fe6330675f53055fdb9412af0234142534a79 refs/heads/wt/wpc-ck
f2e6aa7e1c6e11158c8d9459381ff16d2b5715f9 refs/heads/wt/wpc-cb
cd32e19a7201582fa097b11a07080128ae06cf3c refs/heads/wt/wpc-ce
The bundle requires these 22 refs:
(22 commit ids, every one an ancestor of b1d5e98e)
The bundle uses this hash algorithm: sha1
```

Then `git fetch <bundle> 'refs/heads/*:refs/heads/restored/*'` fetched the four heads. Each fetched sha equalled the
local branch's sha:

| Branch | Fetched from the bundle | Local branch | Same |
|---|---|---|---|
| `wt/wp-l` | `8bc473469a7175e302e01ce4e4a16b3dd31371ce` | `8bc473469a7175e302e01ce4e4a16b3dd31371ce` | yes |
| `wt/wpc-ck` | `5c9fe6330675f53055fdb9412af0234142534a79` | `5c9fe6330675f53055fdb9412af0234142534a79` | yes |
| `wt/wpc-cb` | `f2e6aa7e1c6e11158c8d9459381ff16d2b5715f9` | `f2e6aa7e1c6e11158c8d9459381ff16d2b5715f9` | yes |
| `wt/wpc-ce` | `cd32e19a7201582fa097b11a07080128ae06cf3c` | `cd32e19a7201582fa097b11a07080128ae06cf3c` | yes |

`git fsck --connectivity-only` in that clone printed nothing.

This file and the bundle were committed on the local branch `wt/pause`. They reach origin only after that branch is
brought into the working branch and pushed. Until then they exist only on the cloud machine.
