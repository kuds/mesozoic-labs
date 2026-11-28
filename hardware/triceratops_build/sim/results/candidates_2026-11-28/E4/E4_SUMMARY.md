# E4 forced air: summary

28 Nov 2026. **DESK (web search only).** Nothing is measured and no page was opened. A fetch of arxiv.org failed with
"getaddrinfo ENOTFOUND", because the host is outside this session's network. Per the tool's message, no other host was
tried. Every figure below is the search tool's summary of a page, read on 28 Nov 2026. Rule and reading:
`design/PREREG_CANDIDATES_2026_11_28.md` (section 2 "E4", section 3 item 12, section 4 "E4"). Runner:
`/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E4/run_e4.py`
(sha256 `59514dc3a3da0392`, recorded before it ran). It imports `sim/candidate_checks.py` (`008aa308debdf1b6`, the
frozen file) for `FORCED_AIR_PATHS` and `provenance`. Commit `fb1e80c8`. Files: `e4_sources.json` (the queries, every
source with its figures and item 12's checks, and provenance) and `e4_sources.csv`.

**Rule (pre-registered).** Compute C1 at 35 C with a forced-air R_th only if a source gives an R_th for a comparable
drive (a BLDC motor or geared actuator 40-100 mm across and 0.2-1.5 kg). The source must give it both without and with
forced air, at a stated flow, for the winding-to-ambient path or the drive's total resistance. Otherwise report that no
usable figure was found.

## Result

**No usable figure was found.** The runner printed `usable sources: [] | outcome: no usable figure was found |
c1_computed: False`. `forced_air_c1` was not called, no C1 with forced air was computed, and there is no `e4_c1.json`.

The sources that came closest, and the checks of item 12 that each one fails ("not shown" means the summaries do not
establish it):

| Source (read 28 Nov 2026, summary only) | What it gives | Pair without / with air | Stated flow | Path | Comparable drive |
|---|---|---|---|---|---|
| S01 Urs, Enninful Adu, Rouse, Moore, arXiv 2202.12395 (IROS 2022), 3D-printed QDD actuators | Total R_th 2.27 C/W (Table 3) and 1.23 C/W in the same table, under "Fan" / "No Fan" columns. "Approximately 3.5x cooling advantage" with fans. | not shown: which value goes with which case is not given, and 2.27 / 1.23 = 1.85 does not match the 3.5x | no: two 30 x 10 mm centrifugal fans at $5 each, no flow | total | not shown: motor, size and mass not found (built for 8-15 kg robots) |
| S02 Mazumdar et al. (Sandia), ASME DSCC2016-9738, OSTI 1369514 | H_m in W/K, coil to ambient. Forced air +79 % and pumped water +107 % (2.09 W/K) over a nominal design. Another summary gives 1.25 to 1.51 W/K without fans. | not shown: the nominal figure disagrees between summaries (2.09 / 2.07 = 1.01 W/K against 1.25 W/K; desk arithmetic, not used) | no: fan power mentioned, no flow | winding to ambient | not shown: an Allied Motion Megaflux frameless motor of unknown size, in a lab housing |
| S03 Sandia ACSELL deck, OSTI 1242124 (STEPPR) | 0.65 / 0.63 / 0.62 K/W with 1.4 / 2.6 / 4.2 W fans; "32% increase in cont. T with 4.2 W" | no: no passive R_th read | no: fans given in electrical watts | not stated | not shown |
| S04 maxon HEJ 70 / HEJ 90 data sheets | HEJ 70-48-60: +/-27 N m without forced air, about 50 W passive dissipation, R_th winding to housing 0.38 K/W. HEJ 70-48-50: 140 W "active". | no: a dissipation pair, and the two figures come from different variants | no | winding to housing | not shown |
| S07 Aerospace 11(5):401 (2024), drone outrunners in a wind tunnel | external heat-transfer coefficient against freestream speed | no | wind tunnel, not a fan | housing to air (refused by item 12) | not shown |

Nine other records (S05, S06, S08-S14 in the JSON) give no forced-air R_th pair:
- industrial servo lines;
- passive or still-air data sheets (the MF0127 is 127 mm across);
- liquid or immersion cooling;
- a vendor blog with ratios but no values;
- a 2026 paper on heat-aware policies for the Go2;
- qualitative notes.

**Reported, not used.** Item 12 does not allow a rating pair to be converted into an R_th:
- Moog AML (fan) against AM: continuous torque "up to 1.5 times".
- ACSELL: +32 % continuous torque with a 4.2 W fan.
- maxon HEJ 70: 50 W passive against 140 W "active", on two different variants.
- BEAR: 7.8 N m air-cooled against about 21 N m liquid-cooled. This one is liquid, not forced air.

## Departures from the pre-registration

All of these are labelled **changed after a result**. None of them changes item 12's test.
- **Query P01b.** The rule's "the same with 'planetary gear actuator'" was read as adding or substituting that
  phrase. P01b ("BLDC planetary gear actuator thermal resistance forced air") was added after P07-P12 had returned.
- **Eighteen follow-up queries, F01-F18.** They were chosen after the pre-registered results, to read more of the
  summaries of hits S01-S04 and S07. They name paper titles, the papers' public authors and makers' products. They do
  not name this project or anyone tied to it. This goes beyond "generic engineering terms", which the lead may want
  to note.
- **Conductances.** A conductance in W/K was read as the inverse of an R_th, which matters for S02. This reading decides
  nothing: S02 fails on its stated flow and on the drive's size regardless.
- **Search mode.** Some queries ran in the search tool's "extended" mode.

## What this means

- **For A.** There is no conclusion on forced air (the pre-registered outcome row "No usable figure"), and the bench heat
  test (the study's 3.2) remains the way to know. Forced air cannot be credited against today's worst fore128
  policies: in E3, at 35 C, C1 is 93.27 / 91.04 C for s2 and 98.28 / 97.34 C for s3 (`../E3/E3_SUMMARY.md`). A's heat
  case therefore rests on a policy chosen or trained for heat (in E3, fore128-s0 and s1 meet the rule on their own),
  on E2's reduction, or on the bench. The parts list's $90 of blowers (`design/PARTS_60CM.md`, a parts-list estimate)
  stays unsized.
- **For B.** There is no reading. B's desk C1 of 69.4 C (study, supporting) is unchanged by this search.
- **What would give a usable figure.** One of two things:
  - A bench rise test of one X4 with and without a blower at a measured flow (MEASURED; not done).
  - The full texts of S01 and S02. These need arxiv.org and osti.gov allowed in the session's network settings. Even
    then they would still need a stated flow and the drive's size.
