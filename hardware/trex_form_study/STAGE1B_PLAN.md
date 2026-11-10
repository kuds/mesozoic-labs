# Stage 1b: the speed round (simulation)

**Owner request (P27, 10 Nov 2026, the evening after the Stage 1 report):** "continue with the next round of testing
and experimentation this evening." This round follows the Stage 1 report's speed findings (`stage1/runs/2026-11-08/
STAGE1_PROBE.md`) and the discussion after it. It answers how fast the 60 cm bodies can walk once the controller is
fitted for speed, with today's drives and with the faster knee. It is simulation only, like Stage 1. It changes no
body, and adopts nothing without the owner (P24).

## 1. Questions

1. **Controller.** Stage 1's tracker (R2) was fitted only on the nominal walk and the turns (decision 22). With the
   drives as picked, can the same tracker, re-fitted with the brisk walk and the animal-like walk in its mix, walk
   them on the 60 cm bodies?
   - The brisk walk is 0.315 / 0.314 m/s.
   - The animal-like walk is Froude 0.05: 0.446 / 0.444 m/s.
2. **Faster knee.** With round two's faster knee (sizing with 15 % knee headroom selects a RobStride 06 knee;
   `ROUND2_REVIEW.md` §7.3) applied to the narrow carpet-ready 60 cm picks, does the same speed fit reach them? In
   round two the faster knee took the knee from 0.95 to 0.56 of its speed limit at Froude 0.05, leaving the hip pitch
   (0.90) to bind next.
3. **Servo delay.** Stage 1 assumed no servo delay, yet the 60 cm nominal walk misses a limit at 6 ms and falls at
   8–10 ms. Do controllers fitted with delay in their episodes keep walking at 10 ms?

## 2. Arms

| Arm | Bodies | Controller |
|---|---|---|
| S | `60-N-T-10`, `60-N-H-10` (as picked) | R2 re-fitted for speed (R2s) |
| K | `60-N-T-10-K`, `60-N-H-10-K` (the picks with the faster knee) | R2s, fitted on the K bodies |

The 75 cm body is not in this round. Its brisk walk already stays up and misses only the tail servo's planning limit
(Stage 1 report).

## 3. Method

- **The K bodies.** Round two's sizing pipeline (`designlab/refine.py`) builds the narrow carpet-ready case with 15 %
  knee headroom (`narrow_knee_60`). Stage 1's `prepare` (plant, references, transitions) and gates G1–G4 then run on
  each K body, as Stage 1 ran them on the picks.
  - Their models' `xml_sha256` begin `85d98c24` (`60-N-T-10-K`) and `e0d4c6b7` (`60-N-H-10-K`), from the case's
    records of 10 Nov. G1 checks each model against these prefixes (`SHA256_PREFIX`), as it checks the picks'.
  - Registering them changes the panels' and the tuner's source hashes. With the round's code, Stage 1's panel and
    ledger records therefore read as made by other code, and none of them is run again (section 6).
  - The build's notes are in `stage1/runs/2026-11-10-speed/ARM_K.md`. They cover the git-ignored inputs a checkout
    needs, the open question on `60-N-H-10-K` (its body fails round two's clearance gate), and how the K bodies differ
    from their picks.
- **The speed fit (R2s).** This is Stage 1's CMA-ES fit of the tracker's gains (`stage1-tune-v2`, decision 22) with a
  wider episode mix. It is otherwise unchanged; capture margin held at 0.0.
  - Each evaluation runs the nominal walk, the brisk walk, the Froude 0.05 walk, the 0.35 rad/s walking turn and the
    0.5 rad/s in-place turn.
  - The brisk and Froude 0.05 walks are built and judged as the panels judge them (decision 31): they end in the gait
    and are judged over the last 10 s.
  - Each episode draws a servo delay between 0 and 10 ms.
  - The fit starts from each body's G5 selection (a K body starts from its pick's) and runs 360 evaluations (30
    generations of 12), seed 42.
  - Selection follows tune's G5 rule on 20 validation episodes per task: fewest falls, then the lower survival
    penalty, then the lower performance penalty.
- **Validation panels.** Each selected controller runs the panels' conditions (decision 31) on episodes 3042–3061:
  nominal walk, brisk walk, Froude 0.05 walk, both 0.35 rad/s walking turns and both 0.5 rad/s in-place turns. They run
  at 0 ms and at 10 ms of servo delay, all on hard floor at μ 0.9. Stage 1's R2 rows on the picks are the baseline.

## 4. Pass criteria (set before the runs)

- A condition is **shown** on a body if 20/20 episodes stay up and at least 17/20 meet every limit (Stage 1 §4).
- Q1 and Q2 are answered per arm and body by the brisk and Froude 0.05 walks at 0 ms: shown, survives only (20/20
  up, fewer than 17 meet every limit), or fails. The failing limits are named.
- Q3 is answered by the same conditions at 10 ms.
- A speed fit that loses the nominal walk or a turn at 0 ms is reported as a trade, not a success.

## 5. Budget

Compute runs from about 04:30 to 11:00 UTC on 10 Nov, on the session's 4 cores, with the report by about 12:30 UTC.
If the fits run long, every arm is judged at the last generation all of them reached. The round is recorded in
`stage1/runs/2026-11-10-speed/`.

## 6. Decisions taken

(None yet.)
