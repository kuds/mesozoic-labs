Verdict: merge; tip: ba9498d94dcfa6c5f4083e8d1e8612df1d2e48a3

**Verdict: merge.** All 13 critic fixes and the PLEM and RESUME pointers are applied correctly. The figures match their sources, every link resolves, and the branch merges cleanly. I found nothing that matters to the owner or to the new session. I changed no tracked file. My only scratch file is /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/check2/links.py.

**State checked**
- **Commits.** `wt/trike-kickoff` is at ba9498d9 (parent 7cab604c). 7cab604c merges ce13e61a with a0d0041a. ce13e61a is an earlier merge of f2b79632 with ceadab71, as the fix report says.
- **What the merge added.** 7cab604c adds only the 6 `reviews/` files that a0d0041a brought.
- **diff --stat.** It matches the report: 4 files, 175 insertions, 76 deletions.
- **Untouched files.** PARTS_PREMIUM.md differs from f2b79632 by one line: one sentence added to the committing note, no figure changed. The T. rex folder differs from ceadab71 by one added RESUME.md line.
- **Clean trees.** The worktree is clean, and the main checkout has no changes and no untracked files.
- **Remote refs (ls-remote).**
  - working branch a0d0041a;
  - main 0748ac12;
  - archive branch f71cba4b;
  - data branch s2-b1 ecf95a81;
  - pull/669/head 84a13e1e.
- **Merging.** a0d0041a is an ancestor of HEAD, so merging into the working branch is a fast-forward. `git merge-tree` is clean against the working branch and against main.

**The 13 fixes**
1. **PR #669 errata.**
   - Prompt task 1: the session drafts the README errata section and a list for the PDF's maintainer, on its own branch or as review text; the owner applies them and merges.
   - WHAT NOT TO DO: no push to any PR branch, and no merge, close or comment on any PR, unless the owner says so in the session.
   - Plan: "In short" step 2, table row #669, and section 2 item 2.
   - #669 holds no PDF source; I checked its 8-file list and README.
2. **T. rex resume.** The prompt GOAL has the resume line: 1 Mar 2027, 10 x X4 + 2 x X6 in Q2 2027, the bench by 15 Apr 2027, A17, plan 5.3. The RESUME.md line is under "## 2. How to restart", with a relative link that resolves. There is no blank line before "### Step 0", but GFM still renders the heading.
3. **Decision rounds.** Round 1 is decisions 1-5 by 15 Dec 2026, in two calls. Round 2 is decisions 6-8 by 15 Jan 2027. "In short" steps 3 and 4, T0, section 7 and prompt task 2 agree.
4. **Cost.** All figures match PARTS_PREMIUM.md lines 18-23, 33 and 56:
   - T20: $31.8k the T. rex way, $37.3k landed.
   - T15: $28.3k / $32.8k.
   - Bench set: $7.1k / $8.9k.
   - $17.2k self-built and $20.0k turnkey match BUILD_BUDGET_75_JETSON.md ($17,216 / $20,030).
   - The about-2x RobotShop flag matches PLEM_FIT's review note.
   - $29-32k for the 100 cm flagship is in no committed file. It comes from the lead's brief and the critic, as the task gave it.
5. **SUE.** Named as the Field Museum's T. rex specimen, with "Sioux connection" quoted, in both the prompt and the plan.
6. **Museums.** Both the prompt and plan 5.1 quote "the Field Museum or the Natural History Museum". Griffin MSI is kept, marked as a candidate the owner did not name.
7. **PR list.** It is complete. GitHub's REST API lists 13 open PRs, and all are named.
8. **Fetching #669.** `git fetch origin pull/669/head:pr-669` and `git show pr-669:<path>` are in the prompt and in plan section 2.
9. **0.40 m/s.** "91-93 % of the cap, with no margin under load" matches the parts list's [F1] (0.91-0.93).
10. **Belt and 18 mm.**
    - Belt: 1.3-1.8 kN at 20 mm matches parts list section 1 (1.34 / 1.79 kN in Appendix A). The 20 mm radius matches the survey.
    - 18 mm: marked as the review's arithmetic, from survey_pr669_18nov.md section 2, line 124 (17.8 mm).
11. **Renders.** Stills under 500 KB; ask the owner before committing a clip.
12. **Parts list timing.** The committing note says section 7's timing is replaced by plan 5.3.
13. **preview.png.** It is listed, and it is on 84a13e1e (303,765 bytes).

**PLEM and RESUME pointers**
- The prompt carries recommendation C and the three blockers, and task 2's PLEM bullet (questions first, pilot only after the answers, owner's go or no-go by end Feb 2027).
- Task 8's pilot steps cite PLEM_FIT 5.3; the questions cite 5.4. Both sections exist and say that.
- The plan carries "In short" step 8, decision 7 and section 6, which now adds the two MyActuator questions.

**Links, model names, prompt**
- **Links.** My own checker read the plan, prompt, parts list, PLEM_FIT and RESUME at HEAD: 111 links, 0 bad. GitHub links were checked against origin/main and the working branch; relative links against HEAD's tree.
- **Plain paths in the prompt.** They exist on their branches: reviews/survey_pr669_18nov.md, PLEM_FIT.md and RESUME.md on a0d0041a; all 8 #669 files on 84a13e1e.
- **Study-PR row.** Its counts are right: 684 commits ahead of main at a0d0041a, 8 kickoff paths, 1,539 T. rex paths.
- **Model names.** The prompt names no AI model or vendor; the only hit is the rule itself.
- **Task 1.** It can be done within the prompt's own rules: fetch locally, run the check, draft on its own branch or as text, and send to the owner.

**Attribution**
- ba9498d9's message ends with exactly the two lines and names no model elsewhere.
- The two merge commits (7cab604c, ce13e61a) have git's default message without the trailer. That follows from the `--no-edit` the task asked for, and it cannot be fixed without amending.

**Small points, none blocking**
- Prompt line 74 says "once the lead has merged it there", but "the lead" is never defined for a standalone session. The `git log --all` search still finds the folder.
- The plan's "Links" paragraph says links to PLEM_FIT.md name the working branch. The three new ones are relative. All resolve.
- Plan table 1.2 still shows the working branch at ceadab71 and `wt/trike-kickoff` as "Three files"; the preamble gives a0d0041a.
- "A reseller purchase would raise every total" overstates slightly: the parts list's high case already uses RobotShop's X6 price.
- Already in PLEM_FIT and the original plan, not new here: PLEM_FIT's end-Feb date is for moving the Triceratops to PLEM, which needs pilot data, while the bench drives come at T4 (Mar-Apr). The plan reads it as the pilot's go or no-go.
