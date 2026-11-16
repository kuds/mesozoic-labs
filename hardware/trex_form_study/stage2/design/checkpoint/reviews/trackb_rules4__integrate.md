<!-- Working paper, 16 Nov 2026: the integrator's report on writing the lead's fourth list of rulings and the env's rule wording into the Track B rules draft. Nothing here is a rule. -->

## Result

Both sets are written into `/home/user/mesozoic-wt/prereg-b/hardware/trex_form_study/STAGE2_TRACKB_PLAN.md`: the fourth list's fourteen rulings, and the wording file's entries under ruling 13 of the third list. Three points go back to the lead.

I ran no episode, fit, training, dry run or test. Every number added to the draft is a rule's own constant, a planning figure from the third list's dry runs, or a figure quoted from the wording file with its kind (kinematic playback or a few env steps, by the env's builders, no record). Nothing is a hardware measurement. Line numbers are the file's at `0ea2114`.

Two things moved while I worked:
- **`wt/wpc-ce` moved** from `cdc4762` to `cd32e19` (its third fix round). I followed the newer text: entries 3a, 3b and 8 gained words, and entry 9 was withdrawn. The draft names `cd32e19`.
- **The working branch moved** from `cd68f2c` to `67cb6f5` (the env's small fix round, with the same wording file text). I merged it a second time. Only the draft differs from the working branch.

## Fourth list, per ruling

1. **Caps' source: Q5 (1222-1284), brief Compute (214), Q6 Check, change list item 15.** Each cap is 1.5 times the dry run on the frozen set itself, 60 cm included, committed before the row's first panel record. The stand-in rows and commands are gone, recorded once in an account. The table is now "planning figures" that cap nothing.
2. **How a cap is spent: Q5 (same paragraph) and its Check (1288-1297).** Units in the order 10, 0, 4, 6 ms by H6's unit rule; attribution last and cut first; one run directory and one `budget.json` per cap, one task key.
3. **K5 (1383-1400).** The reviewer's clause is in; the caveat it answered is gone. M4's sentence now reads "a function of the monitor records and of M1's log of attempts alone, which a resume keeps".
4. **M1 (838-848, Check 849-858), M4 (917-934), K5.** The three sentences are in M1, with an author's note on what follows (N = 0, S = 0; every condition of M4's table fails).
5. **L8's Check (1647-1656).**
6. **L10 (1658-1672, Check 1697-1707), L11 (1757-1760).** "Ends after that instant and no later than the end of the move's last ramp"; the Check item on lift-off `foot_pitch` and `rolling`.
7. **L12 (new paragraph 1798-1802, Check), brief (203-204), change list item 30.**
8. **U4 (new paragraph 1038-1040, Check 1050-1054), change list item 29.**
9. **Q5 (1203-1205), Check.**
10. **Y10 (1457-1468), Y11 (1479-1481, Check).** `budget_set_on`; Y11's "(...)" stays.
11. **M5 (957-958).**
12. **L11 (1749-1751, Check 1769-1773).**
13. **Six places, each with a dated bracket quoting the old sentence:** Status (17-20), Words (254-258), Tags (309-311, 318-320), first-list section (about 2063), second-list section (2150-2152). Facts from git: the designs and both lists were first added by `a0314fc` (15 Nov, 03:56:06 UTC), after the draft's commit `103bf35` (03:34:49 UTC).
14. **One dated sentence closing the second list's account, with a pointer in "What had been run" (88-89).** Heading 03:50 UTC; the draft's 03:43 UTC; commit `984f81d` at 04:01:56 UTC.

The frame: a Status sentence, a bullet in "What had been run" (104-118), the Tags entry, the brief's bullet (237-245), and a second closing amendment (2327-2437) that lists all of this, including the note passed to the look work package.

## Ruling 13 of the third list, per wording entry

Each mark carries the entry's number and whether the env's code changed. A check script found every "New" text word for word in the draft.

- **1a, 1b: W1 (540-574).** 1b is the lead's ruling as the entry holds it: E4 stays in walking turns, the heading penalty is not zero there, the tracking figure is scoped to its two sets, "every penalty below 1e-4". Code unchanged.
- **2a, 2b: W2** (`duty`'s own row; `reward_rule` in the Check). Code unchanged.
- **3a, 3b, 3c: W3 (611-645).** Code changed with 3c only (`0dc39b8`).
- **4a, 4b, 4c: W5 (661-706).** Old `progress` and `heading` sentences are quoted in an account.
- **5a, 5b, 5c: E4 (388-404), T1's row and Check.** Code changed (`0dc39b8`).
- **6a, 6b, 6c: E5 (406-427), J4 (723-735), a pointer in E2.** Ranges changed in code (`ad64656`); 6c unchanged.
- **7a, 7b, 7c: T1's `stop`, `start_stop` and `stand` rows; J4's event times.** Code unchanged.
- **8: K4 (1351-1381).** Inserted after "both freeze indices are recorded", plus four keys in the Check. Code changed (`d1e667c`, `011f6e2`, `4523ed6`).
- **9: X1.** The entry is withdrawn at `cd32e19`; X1's mark says no further wording is asked.
- **10: T3 (523-536).** Code changed (`225f4b3`).

## Not placed

Nothing. Entry 9 asks for no text.

## Handed back to the lead

1. **Q5, when S2-9 asks late.** S2-9's condition (R5 fails G8b on both seeds) is known only after a `V6noF5` row's cap is committed. Options: (a) the dry run is made again with all four delays and 1.5 times its total is the cap from then on; (b) the three other delays get a cap of their own. Written in as (a), labelled the author's reading. I recommend (a): it keeps one cap and one unit order per row.
2. **Q5, the candidate's caps.** Does "no number is fixed now" cover the two `F25F9cl` caps (0.708 and 1.416 from 16 Nov, made on the frozen set)? Written in as yes: both dry runs are made again and committed before the first record. I recommend yes; the ruling says "each cap".
3. **K4, entry 8.** The wording file offers it "if the lead wants K4 to say how the env decides". I wrote it in because ruling 13 names the K rules. Options: keep, or strike and keep only the Check's four keys. I recommend keeping it.

## Points of wording I settled

- The marks: fourth-list wording names "ruling n of the fourth list"; wording entries name "wording entry n" and the code label; rulings 13 and 14 use "[dated 16 Nov 2026, ...]". The Tags entry explains all three.
- Ruling 13 is applied as one dated bracket at each of the six places.
- Y10: an author's note on what "queued" means when the owner's answer lets the smoke pass.
- U4's Check: on every item but a stop item the companion's count stands alone.
- K5: two cases in which a due check without a record is not run (counted; two attempts never closed).
- The paths of a cap's run directory and of its committed dry run, the task key, and where the lead's L12 decision is recorded are not named by the rulings. The draft says so and names none.
- Join rule J1 ("are the window resampled") is left as written and noted in the amendment; no ruling names it.
- Commit `4c26ecc`'s message names `cdc4762`; the later commits and the draft name `cd32e19`.

## Commits (branch `wt/prereg-b`, not pushed, no history rewritten)

- `0d10a94` merge of the working branch (at `cd68f2c`)
- `24a7052` rulings 1, 2, 8, 9
- `0f29dbf` rulings 3, 4, 10, 11
- `7a2bbb4` rulings 5, 6, 7, 12
- `4c26ecc` the reward's wording (W1, W2, W3, W5)
- `9baadea` the env's wording (E2, E4, E5, T1, T3, J4, K4, X1; W3 updated)
- `cb2fe38` rulings 13 and 14, and the frame
- `32bdd15` second merge of the working branch (at `67cb6f5`)
- `0ea2114` one clause on the working branch's copy of the wording file

`git status --porcelain` in the worktree is empty. I wrote nothing in the main checkout; it shows two untracked review files that are not mine. Helper scripts and the word-level diff against `c4a0ff8` are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rules4/` (the diff is `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rules4/wdiff_all.txt`, made before the last one-clause commit).
