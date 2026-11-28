# E4 forced air: fixes after the independent review (28 Nov 2026)

**DESK (web search summaries; no page opened; nothing measured).** This addendum answers review findings E4-R1 to
E4-R6. Everything in it was decided after E4's result was seen and is labelled **changed after a result**. Item 12's
test (`design/PREREG_CANDIDATES_2026_11_28.md`, section 3 item 12; section 4 "E4") is not changed. The review runner
applies the same `qualifies` function from `run_e4.py`; its sha256 `59514dc3a3da0392` was checked before and after
the run. `E4_SUMMARY.md`, `e4_sources.json` and `e4_sources.csv` are left as they were. Their sha256 values are in
`e4_review_addendum.json` under `files_left_unchanged`.

Runner: `/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E4/run_e4_review.py`
(sha256 `b217a52aafc04e3a`, recorded before it ran). Raw search-tool text, verbatim:
`/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E4/review_searches_raw.txt`
(sha256 `ec8f40cbc2cc4ae8`). The same text is copied into `e4_review_addendum.json` (`raw_tool_text`). Commit
`fb1e80c8`; `sim/candidate_checks.py` is unchanged (`008aa308debdf1b6`).

## Outcome (unchanged)

The runner printed `usable sources: [] | outcome: no usable figure was found | c1_computed: False`, over 16 records
(S01-S16). The checks of S01-S14 are identical to E4's. `forced_air_c1` was not called.

## Findings

| Finding | Disposition | What was done |
|---|---|---|
| E4-R1: the F queries broke the generic-terms rule | **Accepted** | 17 of the 18 F queries are a breach of the hard rule, not only a departure from the pre-registration. F14 is the one with generic terms only. The breach changed no outcome. The wording for the final report is below. |
| E4-R2: a missed forced-air actuator source | **Accepted** | Found again with generic query R01, and returned by R02 too. Added as **S15**: Energies 11(11):2947 (2018). Checks: `rth_pair_given` None, `flow_stated` False, `heat_path_accepted` False, `comparable_drive` None, so not usable. Its "about three times greater than the rated current" with forced air goes on the reported-not-converted list. Two other hits from R02 go in S16 (typical ranges and one unverified blog value), also not usable. |
| E4-R3: S01 understated; raw tool text not kept | **Accepted as a note** | See the S01 section below. S01's check stays at None. The raw tool text of these searches is now kept. |
| E4-R4: the "What it means" sentence predates E2 | **Accepted** | Reworded below against `../E2/E2_summary.md` ("Verdict: no judged (transmission, r, placement) PASSES; 0 of 45."). Checked on `e2a_grid.csv`: at hip pitch, r = 1.2-1.3, s0, s1 and s2 each pass 18 of 18 judged rows and s3 passes 0 of 18, every one on the speed check. |
| E4-R5: S11 has no source | **Accepted** | S11's two liquid / immersion figures are withdrawn. They are unsourced, not used, and must not be quoted as sourced values. |
| E4-R6: file sizes were disk blocks | **Accepted** | Byte sizes: `e4_sources.json` 29,900 B (29.2 KiB), `e4_sources.csv` 6,270 B (6.1 KiB), `E4_SUMMARY.md` 6,303 B (6.2 KiB). `du -k` gives 32 / 8 / 8, which are disk blocks. |

**S01 (E4-R3).**
- The reviewer's search of 28 Nov 2026 quoted this from arXiv 2202.12395: "the total thermal resistance R_th was 2.27
  °C/W. The 3.5× reduction in R_th with the fans nearly doubles the amount of continuous current that can be safely
  dissipated". No airflow was given.
- My rerun did not reproduce that sentence. R02 did not return S01. R03 returned it and said "The snippets I got don't
  include the thermal resistance value or the continuous current rating". It gave only "nearly doubles the thermally
  driven torque limits" and the fan layout.
- If the reviewer's reading holds, S01 states an R_th ratio, with rho = 1/3.5 = 0.286. That agrees with "nearly
  doubles": sqrt(3.5) = 1.87 (desk arithmetic, supporting).
- The check stays at None (not established), for three reasons: the sentence was not reproduced here, the tool text
  behind it is not archived, and E4's summaries disagree on which value goes with the fans.
- S01 fails on stated flow and on drive size either way.
- If the full text gives the fan flow and a motor of 40-100 mm and 0.2-1.5 kg, S01 would qualify with rho of about
  0.29. C1 at 35 C would then have to be computed under item 12 for the four fore128 policies on both seed blocks.
  That is not computed now.

## Review searches (generic terms only, standard mode, 28 Nov 2026)

- R01 "modular robot actuator forced air cooling thermal equivalent circuit maximum continuous current rated current"
- R02 "BLDC robot joint actuator winding to ambient thermal resistance K/W natural convection versus fan cooled airflow m/s measured" (the reviewer's query, rerun)
- R03 "3D printed quasi-direct-drive legged robot actuator total thermal resistance with cooling fans continuous current"

No page was fetched. E4's fetch failure told the agent not to try other hosts.

**Reported, not converted (item 12):**
- Moog AML against AM: up to 1.5x continuous torque (S05).
- ACSELL: +32 % continuous torque with a 4.2 W fan (S03).
- HEJ 70: 50 W against 140 W, across two variants (S04).
- BEAR: 7.8 against about 21 N m, liquid cooling (S10).
- **Added:** S15's maximum continuous current of about 3x its rated current with forced air.

**Full texts to read if the owner allows the hosts:**
- S01 on arxiv.org: the flow, the motor's size and mass, and which R_th goes with the fans.
- S02 on osti.gov: the flow, the motor's size, and the nominal H_m.
- S15 on mdpi.com: the R_th without and with forced air, the flow, and the drive's size.

Reading them needs those hosts allowed in the session's network settings, not more targeted queries.

## Wording for `design/CANDIDATES_2026_11_28.md` (not yet written; this addendum does not create it)

**E4-R1.** E4's follow-up queries F01-F18 broke the hard rule "Web searches use generic part and engineering terms
only". They also broke PREREG section 4 "E4": "web search only, generic engineering terms". 17 of the 18 (all but F14)
named at least one of the following:
- paper authors (F02, F05, F09, F11);
- makers or products (F07, F11, F13, F16, F18);
- institutions or named projects (F08, F09, F10, F17, F18);
- report or paper identifiers (F03, F06, F08);
- quoted paper titles (F01, F04, F07, F12, F15).

None named this project or anyone tied to it. This is a breach of a hard rule, not only a departure from the
pre-registration. It changed no outcome: no source qualifies under item 12, so none of these queries produced a usable
figure. Later E4 work stays with generic engineering terms. Reading S01, S02 or S15 in full needs the owner to allow
arxiv.org, osti.gov or mdpi.com, not more targeted queries.

**E4-R4 (replaces E4_SUMMARY's "A's heat case therefore rests on ..." sentence).** A's heat case cannot lean on forced
air, because E4 found no usable figure.
- **E2(a), the verdict.** Under the pre-registered worst-of-four rule no reduction passes (0 of 45). At hip pitch,
  r = 1.2-1.3, C1 and C2 pass, but the speed check fails on fore128-s3's right fore hip pitch. Its peak is 99.2-111.3
  rpm over both seed blocks (102.7-111.3 on the worse one), against 91.3 rpm.
- **E2(a), per policy (supporting, not the verdict).** A 1.2-1.3:1 hip-pitch reduction passes C1, C2 and the speed
  check for s0, s1 and s2 on both blocks with every transmission.
- **E2(b), a separate reading (simulation only).** At r = 1.25 with the 83 rpm cap, rod, belt and cable are compatible
  with today's four policies. It does not override E2(a).

So A's heat case rests on one of two things: a policy chosen or trained for heat (in E3, fore128-s0 and s1 meet the
rule on their own at 35 C), or the bench heat test. A reduction would want a policy other than s3, or one trained with
the reduction and the cap.

**E4-R5.** S11's figures are unsourced and not used.

## What it means

- **For A:** no conclusion on forced air. This is the pre-registered "No usable figure" row, and the bench heat test
  (the study's 3.2) remains the way to know. S15 and S01 show that forced air helps comparable actuators. Neither gives
  an R_th pair at a stated flow for a drive in the comparable range, so neither can be credited against s2's C1 (93.27
  / 91.04 C) or s3's (98.28 / 97.34 C) at 35 C (E3).
- **For B:** no reading. B's desk C1 of 69.4 C (study, supporting) is unchanged.

Files (new):
- `E4_REVIEW_ADDENDUM.md`: this file.
- `e4_review_addendum.json`: the dispositions, the query audit, the review queries and raw text, all 16 records with
  their checks, and provenance.
- `e4_sources_reviewed.csv`: 16 rows, with a `review_change` column.
