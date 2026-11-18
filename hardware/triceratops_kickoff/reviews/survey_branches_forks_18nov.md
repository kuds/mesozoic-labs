## Branch, local-state and fork survey: kuds/mesozoic-labs (read only, 18 Nov 2026)

I ran `git fetch origin`, then local git, the GitHub MCP `list_pull_requests` call (state=all) and `gh api -X GET search/repositories`. I wrote, pushed, commented on and edited nothing, and left no files in the checkout. origin has 575 remote-tracking refs. origin/main is 0748ac12 (14 Nov).

**One change since your context was written:** the working branch `claude/mesozoic-trex-stage2-refine-4jffre` has moved on. It is now at c5274905, one commit past b1d5e98e: "P51: the T. rex stays on the roadmap as the flagship for end 2027 / early 2028; the Triceratops is the Q3 2027 demo". No PR has been opened for it yet. The P48 PR should be opened from c5274905, not b1d5e98e.

### (1) Remote branches: every branch with commits since 4 Nov, plus the four you named

The counts are taken against origin/main. "Merged" means the branch tip is an ancestor of main.

| Branch (tip, last commit) | Behind / ahead of main | Merged | PR | What it holds | Content kept elsewhere? |
|---|---|---|---|---|---|
| claude/mesozoic-trex-stage2-refine-4jffre (c5274905, 18 Nov) | 0 / 672 | no | none yet (P48 pending) | T. rex study: 1526 files, all under `hardware/trex_form_study/`, about 5.33M inserted lines; decision log up to P51 | It is the source. Its history also contains all of jetson-orin-plem-study. |
| codex/triceratops-quadruped-design (84a13e1e, 17 Nov) | 0 / 2 | no | **#669 open** (created 18 Nov 03:18Z) | 8 files under `hardware/triceratops_design_study/` | Only on this branch |
| claude/mesozoic-trex-stage2-bundle-s2-b1 (ecf95a81, 17 Nov) | 0 / 650 | no | none | Data branch: the s2-b1 bundle's three parts, force-added (P46), plus launch commit C1 | Its tip commit ecf95a81 is **not** on the refine branch (the refine branch is 23 ahead of it, it is 1 ahead). Its data exists only here. Must not move. |
| jetson-orin-plem-study (6162b455, 14 Nov) | 27 / 2 | no | **#668 closed, not merged** (closed 16 Nov 00:33Z) | Jetson Orin Nano Super + Partial PLEM, speed ceiling, `BUILD_BUDGET_75_JETSON.md` (6 files) | **Yes.** It is an ancestor of the refine branch. Four of its files are byte-identical there. COMPUTE.md and JETSON_PLEM_AND_SPEED.md were edited later on the refine branch (P37 commit 8ce0621a). It reaches main once the P48 PR merges. |
| claude/mesozoic-trex-design-iteration-azm2fs (4881d639, 14 Nov) | 11 / 0 | yes | #667 merged 14 Nov 21:08Z | T. rex design study framework | On main |
| claude/velociraptor-form-factor (75b7a817, 9 Nov) | 243 / 0 | yes | #666 merged 9 Nov | Life-size velociraptor rebuild | On main |
| claude/stance-cert-records (0bbbebde, 8 Nov) | 250 / 0 | yes | #665 merged 8 Nov | Records of the T. rex stance's two-seed certification | On main |
| codex/trex-body-shape-posture-study (b5c18bce, 7 Nov) | 375 / 36 | no | **#664 open** | Intermediate T. rex platform: `examples/mini_tyrannosaurus_rex`, `examples/tyrannosaurus_rex_50cm`, `environments/shared`; 421 files | Same tip as the branch below; contains all of intermediate-rex-platform |
| codex/trex-r3-simulation-findings (b5c18bce, 7 Nov) | 375 / 36 | no | **#663 open** | Same commit as #664: size, gait and navigation findings | Duplicate of #664 |
| codex/intermediate-rex-platform (e103f7ff, 6 Nov) | 375 / 35 | no | **#662 open, draft** | Platform source, design inputs, reproduction instructions | Ancestor of #663/#664 |
| claude/stance-ltsb-main (2602b311, 6 Nov) | 253 / 0 | yes | #661 merged | r10 LTSB stance recipe | On main |
| claude/velociraptor-trex-lessons (b24be941, 6 Nov) | 268 / 0 | yes | #660 merged | Velociraptor heading-free observation and tail terms | On main |
| claude/gait-ledger-page (d389c0d4, 6 Nov) | 276 / 0 | yes | #659 merged | Gait ledger page | On main |
| claude/pilot-walk-r10-p4 (0e6a57bb, 6 Nov) | 278 / 22 | no | none | Walking pilot 4 code (heading price, command tables, a stage that accepts its own checkpoint); 24 files in `environments/`; every commit labelled "pilot-only, never merged" | Only here. It contains all of pilot-walk-r10-p3. It is not on the refine branch. |
| claude/pilot-walk-r10-p3 (31d02e42, 4 Nov) | 278 / 8 | no | none | Walking pilot 3 code (R6 reward, warm-start log_std reset) | Ancestor of p4 |
| claude/handoff-walking-pilot4 (f90304fa, 6 Nov) | orphan (no shared history with main) | no | none | Text-only copy of pilot 4's frozen pre-registration and evidence. Its README says 644 binary or >5 MB files were left out and "stay in the session scratchpad". | Only here; the binaries are not in git |
| claude/handoff-walking-pilot3 (90e515c8, 4 Nov) | orphan | no | none | Text-only handoff for pilot 3 (reports, frozen pre-registration, evidence) | Only here |

Older branches that are still not merged but have open PRs from before 4 Nov: #655 (copilot stance-pop diagnosis), #651 draft (60 cm T. rex skeleton), #649 draft (mini T. rex M1), #648 draft (mini juvenile hardware fit), #646 (MicroDuck transfer design), #635 (T-Rex concepts with medium hardware). I did not inspect their content.

### (2) Local branches whose commits are not on origin

I only listed these; a separate job is bundling them.

| Local branch | Commits not on origin | Note |
|---|---|---|
| claude/mesozoic-trex-stage2-archive-a1 (be474190) | 1 | "Archive a1: the T. rex study's git-ignored inputs (P49)". **Not pushed yet**: `git ls-remote` shows no such branch on origin. |
| wt/pause (54d8afc1) | 1 | "Checkpoint: the unpushed local branches archived as a git bundle (P47)" |
| wt/wp-l (8bc47346) | 45 | Stage 2 look sets (WP L), set V6 rebuild |
| wt/wpc-cb (f2e6aa7e) | 14 | WP CB fix round, merged with the working branch |
| wt/wpc-ck (5c9fe633) | 15 | WP CK lookread tail term |
| wt/wpc-ce (cd32e19a) | 3 | WP CE third fix round |

The other 28 wt/* branches and the local main, bundle-s2-b1 and refine-4jffre branches have no commits missing from origin. The local refine and bundle branches match origin exactly.

### (3) Forks (metadata only)

- `fork:only` search: 1 result, **yunerious/mesozoic-labs**, owner yunerious, last pushed 2026-11-08T21:27:38Z.
- kuds/mesozoic-labs shows forks_count 1, is public, and was last pushed 2026-11-18T03:56:07Z.
- That fork's last push is 8 Nov, so it cannot hold the 14–18 Nov work (the refine branch, the bundle, #668, #669). I did not read its contents, as the rules require.

### Not done

- I did not read the content of the open PRs other than through branch diffstats and commit subjects.
- I did not inspect the pre-4 Nov branches beyond the PR list.
- I did not check CI or review state on any PR.

### Branches that need a decision

1. **Refine branch:** open the P48 PR from c5274905 (P51 is included). Merging it also brings in #668's Jetson/PLEM content, so jetson-orin-plem-study can then be deleted or kept as a record.
2. **archive-a1:** it exists only on this machine. It needs pushing (P49) before this container goes away.
3. **bundle-s2-b1:** keep it frozen. Its tip ecf95a81 is not contained in any other branch, so deleting it loses the T. rex launch inputs that resuming in 2027 depends on.
4. **#669 (Triceratops):** this is now the main line of work and is open. It needs review, then a merge or rework.
5. **Codex T. rex PRs #662 (draft), #663, #664:** #663 and #664 point at the same commit and duplicate each other. All three are 375 commits behind main. They need closing, or consolidating into one, now the T. rex is paused. Since the T. rex resumes for end 2027 / early 2028, keeping them as unmerged, labelled branches is an option.
6. **Walking pilots p3/p4 and the handoff-walking-pilot3/4 orphans:** these are deliberately never merged and exist only on these branches; pilot 4's binaries are only in a session scratchpad. They should be kept as archive branches or tags so the T. rex can resume from them.
7. **Older open PRs #635, #646, #648, #649, #651, #655:** stale relative to the new direction. They need a triage decision.
8. **Merged branches that can be deleted** (no content would be lost): design-iteration-azm2fs, velociraptor-form-factor, stance-cert-records, stance-ltsb-main, velociraptor-trex-lessons, gait-ledger-page.