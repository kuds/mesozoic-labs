**Verdict: fix first.** The three files are accurate and well built. But the prompt's first task can't be done as written, and the prompt has no T. rex resume point. A few smaller fixes would also help the owner. I changed no tracked file. My only scratch file is the link checker at /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/critic/links.py.

## What I checked and found sound

- **Commit and branches.** `wt/trike-kickoff` is at f2b79632. Its parent is b1d5e98e. It adds 3 files and 1,565 lines under `hardware/triceratops_kickoff/`. Fresh refs read with `git ls-remote` and `fetch` on 18 Nov:
  - working branch ceadab71, 683 commits ahead of main;
  - main 0748ac12;
  - archive branch f71cba4b;
  - data branch s2-b1 ecf95a81;
  - PR #669 branch 84a13e1e.
- **Merges.** `git merge-tree` is clean for:
  - the kickoff branch into the working branch;
  - the kickoff branch into main;
  - the working branch into main;
  - #669 into main.

  The working branch touches only `hardware/trex_form_study/` (1,539 paths) and `hardware/triceratops_kickoff/` (2 paths: `PLEM_FIT.md` and `reviews/plem_fit_check_18nov.md`). #669 touches only `hardware/triceratops_design_study/`, so the merge order is safe. cc437b6a and b1d5e98e are both in the working branch's history. 8 added files are over 500 KB, as the plan says. The pre-commit limit is `--maxkb=500`.
- **(4) Links.** My own checker against freshly fetched refs found 39 links:
  - 36 blob or tree links, all OK (31 on the working branch, 5 on main);
  - 2 relative links, both OK;
  - 1 link to PR #669.

  Every plain path the prompt names exists on its branch: 7 files in #669, 8 on main, and 35 T. rex paths including `latency_bus.py`, `parts.json`, `keepalive.sh`, `memguard.sh`, `lowprio.sh` and `setup_venv.sh`. `pyproject.toml` on main pins `mujoco==3.10.0`. RAPTOR section 4 says 12S and three CAN buses; section 3 is Pupper V3. Row P51 exists on ceadab71, so "P47 to P51" is correct there. RESUME.md section 4 is the carry-over list.
- **(2) Pull requests and branches.**
  - GitHub lists 13 open PRs. Their numbers, head branches and stack bases all match table 1.1 and section 3.
  - All 19 remote branches with commits since 4 Nov are in table 1.2.
  - The six branches marked deletable are all already in main.
  - `jetson-orin-plem-study` is in the working branch.
  - #585's head is an ancestor of #596's head.
  - The local branches that the working branch doesn't hold are exactly those in the table. The bundle's four heads match `wt/wp-l`, `wt/wpc-cb`, `wt/wpc-ck` and `wt/wpc-ce`.
  - No step deletes or moves the two T. rex data branches, the working branch or the pilot branches.
  - The only fork is metadata-only and was not checked again.
- **(3) PARTS_PREMIUM.md.** Diffed against `parts_check/PARTS_PREMIUM_corrected.md`, it differs only in the note on committing, the two script references (sections 2 and 2.8) and the file list at the end. No figure changed.
- **Spot checks of the prompt against the parts list.** The T20 drives, the design loads, the 0.755 kg battery and 165 mm bay, and the Teensy with an MCP2518FD for the 4th bus all match. The 18 mm sideways shift is not in any committed file. I re-derived it from the 650 x 350 mm footprint with the centre of mass 358 mm ahead of the rear feet: 17.8 mm.
- **Model names.** The prompt names no AI model. It reaches #669 through `pull/669/head`, uses only branch-name endings, and line 150 only says "name no AI model".
- **Environment.** The "mesozoic-labs" environment exists and installs libosmesa6.
- **(5) Attribution.** The commit message ends with exactly the two required lines. No model is named elsewhere in it.
- **T. rex in the plan.** The plan keeps the T. rex on the roadmap and gives resume dates: section 5.3 (A17 bench during the foreleg bench; simulation by 1 Mar 2027; drives in Q2 2027; CAD by 1 Jul 2027), plus decision 6.

## Fixes that matter (the reason for fix_first)

1. **Prompt task 1 can't be done as written.**
   - There's no PDF source: #669 holds only the built PDF, and its README points at no source. So "fix the PDF's p.6 'Orin / MCU' label", the battery figure and the date stamp can't be edited in the PDF.
   - "Four follow-ups on its branch" means pushing to `codex/triceratops-quadruped-design`. That conflicts with WORKING RULES ("work and push on the branch this session was given"), and a cloud session may be refused that push.
   - "Merge it" conflicts with WHAT NOT TO DO ("Do not merge … other pull requests"). The plan, by contrast, says the owner merges.
   - **Fix:** in the prompt and in plan section 2, have the session draft an errata section for #669's README on its own branch (or as review text for the owner). The owner applies it and merges. If the session may push to #669 or merge it after the owner's yes, say so explicitly in WHAT NOT TO DO.
2. **The prompt has no T. rex resume point.** It only says "outside this session's scope". Add one line under "The T. rex", for example:
   > "If the owner picks a built T. rex for that milestone, T. rex simulation resumes by 1 Mar 2027 in a separate session started from RESUME.md, and its drives (10 x X4 + 2 x X6) are ordered in Q2 2027 once your foreleg bench (by 15 Apr 2027) confirms the shared drives. The foreleg bench and its delay rows (A17) are therefore on the T. rex's path too (CONSOLIDATION_PLAN.md 5.3)."

   Optionally, add a pointer to section 5.3 in RESUME.md when the kickoff branch is merged, since RESUME.md carries no resume-by dates.
3. **Decision rounds and dates disagree.**
   - "In short" step 3 and T0 say five decisions by 15 Dec, including compute.
   - Section 7 puts compute in round 2, and round 1 has only four decisions.
   - Round 2 has no date, yet decision 6 (what the T. rex milestone shows) sets the 1 Mar 2027 restart.
   - **Fix:** make the round 1 count match the "five decisions" claim. Give round 2 a date, for example 15 Jan 2027.
4. **Cost against the owner's spend wish is not surfaced.**
   - The prompt anchors on "about $20k turnkey" but recommends T20 at $37.3k landed, or $31.8k counted the T. rex way, with no comment.
   - "In short" step 3 recommends 20 kg without a cost.
   - **Fix:** state the class as $17k-$32k (T. rex 75 cm $17.2k / $20k turnkey up to the 100 cm flagship at $29-32k). Say that T20 is at the top of that class counted the T. rex way and above it landed, and that the first spend is the $7.1k bench set.

## Smaller fixes

5. **The owner's "Sioux connection"** is most likely SUE, the Field Museum's T. rex specimen, misheard by speech-to-text. This is my inference. Section 5.1's wording is right in substance, but naming SUE would help the pitch; confirm with the owner.
6. **Museums.** The owner named "the Field Museum or the Natural History Museum". The prompt and plan list only the Field Museum and Griffin MSI.
7. **Stale PR list in the prompt.** WHAT NOT TO DO omits #655 (T. rex stance pop) and #646.
8. **Reading #669's files.** `git fetch origin pull/669/head` only sets FETCH_HEAD, so the prompt's `git show origin/<branch>:<path>` won't work for it. Use `git fetch origin pull/669/head:pr-669`, then `git show pr-669:<path>`.
9. **The 0.40 m/s line.** "75-77 rpm against an 83 rpm cap" reads as if it fits. Add "91-93 % of the cap, no margin under load".
10. **Unsourced figures.** The belt "1.2-2.1 kN" is in no committed file; the parts list says a belt at 20 mm radius sees 1.3-1.8 kN. Either align with the parts list or mark it as the survey's arithmetic. The same goes for the 18 mm figure.
11. **Task 9 renders.** A video clip will likely exceed the 500 KB limit. Say to keep it under the limit or ask the owner where it goes.
12. **Parts list section 7.** It still says "restart T. rex simulation … Q2 2027; buy T. rex hardware after the Q3 2027 demo" above the [F13] correction. That is as in the corrected document, so leave it, but the committing note could say the plan's section 5.3 replaces that timing.
13. **Trivial:** #669 also holds `models/preview.png`, which the prompt doesn't list.

## Not done

I ran no CI, tests or simulation, and did not open the PR diffs beyond file lists. I did not read the surveys, so 18 mm and 1.2-2.1 kN were checked only by arithmetic. I did not recheck prices or lead times, did not read the fork, and posted, pushed or changed nothing.