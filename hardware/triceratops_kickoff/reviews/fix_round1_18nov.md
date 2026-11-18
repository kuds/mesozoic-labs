All 13 critic fixes and the PLEM and RESUME pointers are applied and committed on `wt/trike-kickoff`. The commit is **ba9498d9** (ba9498d94dcfa6c5f4083e8d1e8612df1d2e48a3), with parent 7cab604c. Nothing was pushed or posted.

**Merge.** The worktree head was already ce13e61a when I started, not f2b79632. ce13e61a is an earlier merge of `origin/claude/mesozoic-trex-stage2-refine-4jffre` at ceadab71, from 05:26 and not made by this run. After fetching, the working branch was at **a0d0041a**, which adds 6 review files under `hardware/triceratops_kickoff/reviews/`. `git merge --no-edit` merged it cleanly as **7cab604c**. So the branch now has two merge commits, then my one commit. I amended nothing.

**diff --stat (ba9498d9 against 7cab604c)**
```
 hardware/trex_form_study/RESUME.md                 |   1 +
 hardware/triceratops_kickoff/CONSOLIDATION_PLAN.md | 122 ++++++++++++++------
 hardware/triceratops_kickoff/NEW_SESSION_PROMPT.md | 126 ++++++++++++++-------
 hardware/triceratops_kickoff/PARTS_PREMIUM.md      |   2 +-
 4 files changed, 175 insertions(+), 76 deletions(-)
```

**Where each fix landed** (prompt = NEW_SESSION_PROMPT.md, plan = CONSOLIDATION_PLAN.md)
1. **PR #669 errata.**
   - Prompt task 1: the session drafts an errata section for #669's README, plus a list for the PDF's maintainer (there is no PDF source), on its own branch or as review text. The owner applies it and merges.
   - Prompt WHAT NOT TO DO: "Do not push to PR #669's branch or to any other pull request's branch, and do not merge, close or comment on any pull request, unless the owner says so in this session."
   - Plan: "In short" step 2, the #669 row of table 1.1, and section 2 item 2 (a new "Who does what" bullet).
2. **T. rex resume line.**
   - Prompt GOAL has a new "When the T. rex resumes" bullet: restart by 1 Mar 2027 from RESUME.md, drives (10 x X4 + 2 x X6) in Q2 2027 after the foreleg bench (by 15 Apr 2027), and A17, citing plan 5.3.
   - `hardware/trex_form_study/RESUME.md`: one line added under "## 2. How to restart", with a relative link to `../triceratops_kickoff/CONSOLIDATION_PLAN.md` section 5.3. To keep it to one line there is no blank line before "### Step 0"; Markdown still renders it as a heading.
   - Plan 5.3 "Staffing" now links RESUME.md.
3. **Decision rounds.**
   - Plan section 7: round 1 is decisions 1-5 by about 15 Dec 2026, asked in two calls (1, 2, 4, 5, then 3). This matches "In short" step 3 and T0.
   - Round 2 is decisions 6-8 by about 15 Jan 2027.
   - Also updated: plan "In short" step 4, the T0 row, and prompt task 2.
4. **Cost against the spend wish.**
   - Where: prompt OWNER'S PREFERENCES and KEY FACTS (now with the T. rex-way totals); plan "In short" step 3 (cost and price-flag bullets) and decision 1.
   - Figures: class $17k-$32k, from the T. rex 75 cm ($17.2k self-built, $20.0k turnkey) to the 100 cm flagship ($29-32k). T20 is $31.8k the T. rex way (top of the class) and $37.3k landed (above it). T15 is $28.3k / $32.8k. The first spend is the bench set, $7.1k of parts / $8.9k landed.
   - The reviewer's flag is included: third-party US drive prices run about 2x MyActuator's store.
5. **SUE.** Prompt GOAL and plan 5.1: "the direct tie-in with SUE, the Field Museum's T. rex specimen (the owner wrote "Sioux connection")".
6. **Museums.** Both prompt and plan 5.1 quote the owner: "the Field Museum or the Natural History Museum". No repo file sources Griffin MSI, so I kept it only as "a further candidate the owner did not name".
7. **PR list.** Prompt WHAT NOT TO DO now names #585, #596, #623, #635, #648, #649, #651, #655, #662, #663, #664 and #646, plus #669 and the study PR.
8. **Fetching #669.** Prompt WHERE THINGS ARE: `git fetch origin pull/669/head:pr-669`, then `git show pr-669:<path>`. The same command is in plan section 2.
9. **0.40 m/s.** Prompt KEY FACTS and plan decision 1: "91-93 % of the cap, with no margin under load".
10. **Belt and 18 mm.**
    - Belt: changed to the parts list's figure, 1.3-1.8 kN at the 20 mm radius (parts list, section 1).
    - 18 mm: marked as the PR #669 review's arithmetic (17.8 mm; `reviews/survey_pr669_18nov.md`, section 2).
11. **Renders.** Prompt task 9: commit the stills under the 500 KB limit; ask the owner where a clip goes before committing one.
12. **Parts list timing.** The PARTS_PREMIUM.md committing note now says section 7's "Suggested timing" is replaced by plan 5.3, which follows [F13]. No figure changed: the word-diff shows only the added sentence.
13. **preview.png.** Prompt now lists `models/preview.png`. I confirmed it is in 84a13e1e (303,765 bytes).

**PLEM and RESUME pointers**
- PLEM_FIT.md:
  - Prompt kickoff list: recommendation C and the three blockers.
  - Prompt task 2: drafting the questions to WIM and MyActuator, round 2, and the pilot only after their answers, go or no-go by end Feb 2027.
  - Prompt task 8: the pilot's bench steps, if the owner approves.
  - Prompt WHAT NOT TO DO and OWNER'S ANSWERS.
  - Plan: relative links in the preamble, "In short" step 8 and decision 7. Section 6 now has the two MyActuator questions.
- RESUME.md: named by path in the prompt; linked from plan section 4 and 5.3.

**Other updates I made**
- Plan preamble: the branch now holds PLEM_FIT.md, `reviews/` and RESUME.md, and the working branch moved to a0d0041a.
- Table 1.1, study-PR row: 684 commits ahead at a0d0041a, and 8 kickoff paths.
- Plan section 8: a short "Revised after the first critic round" note.

**Checks**
- Link checker: 111 links in the four files, 0 bad. GitHub links were checked against origin refs (working branch a0d0041a, main 0748ac12); relative links against the tree.
- The prompt has no AI-model or tool names. I used no `claude/` or `codex/` branch prefixes in it.
- The commit message ends with exactly the two required lines.
- `git merge-tree` is clean for this branch into the working branch and into main.
- The main checkout `/home/user/mesozoic-labs` has no changes and no untracked files.

Scratch link checker: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/checklinks.py
