# Claim audits of the round-two documents

Every round-two document was checked by an independent claim audit before delivery. One auditor per document part
checks every checkable claim (numbers, ranges, counts, comparisons, causes, "all" and "none" statements, references to
sections, files and IDs) against the records: the JSON outputs, the code and the git history. Each reported problem
then goes to two independent checkers. One tries to refute it; the other recomputes the quantity from the raw data.
A problem is **confirmed** when both checkers agree that the text is wrong, unsupported, misleading or stale,
**disputed** when one does, and **rejected** when neither does.

Confirmed problems were corrected. Disputed ones were corrected where the agreeing checker showed the text was
imprecise (usually a range that did not cover every body, or a missing qualifier). Rejected ones were left as written.

| Audit | Documents | Claims checked | Reported | Confirmed | Disputed | Rejected | Corrections |
|---|---|---|---|---|---|---|---|
| Round two (8 Nov, early) | the round-two review, `DECISION_LOG.md` (F9–F16, A13, A14), `SCREENING.md`, `COMPUTE.md`, `README.md`, the review page | 681 | 58 | 42 | 15 | 1 | commits `b1e3dd9`, `8e030a7` |
| Overnight, early (8 Nov) | `round2/refine/MATERIALS.md`, `DECISION_LOG.md` F17–F22 | 406 | 39 | 36 | 2 | 1 | commit `63f7888` |
| Final (8 Nov, morning) | `ROUND2_REVIEW.md` (overnight version), `MATERIALS.md`, `DECISION_LOG.md` F22–F24, the review page | 2,232 | 88 | 67 | 14 | 7 | the commit that adds this file |

The final audit's problems, with each checker's verdict and reason, are in `round2/audit_final.json`. By part:

| Part | Claims checked | Reported | Confirmed | Disputed | Rejected |
|---|---|---|---|---|---|
| Summary and method | 160 | 11 | 7 | 2 | 2 |
| Sections 1–5 (round-two results) | 245 | 6 | 5 | 1 | 0 |
| Sections 6–7 (scorecard and refinement) | 870 | 14 | 11 | 2 | 1 |
| Sections 8–14 (size study, materials, head and tail, questions) | 380 | 17 | 14 | 2 | 1 |
| Decision log F22–F24 | 109 | 13 | 10 | 3 | 0 |
| `MATERIALS.md` | 280 | 12 | 8 | 3 | 1 |
| Review page | 188 | 15 | 12 | 1 | 2 |

None of the final audit's problems changes a recommended body. The largest corrections: the narrow carpet-ready
bodies' target turns clear by 3.4 mm (the 3.0–3.1 mm figure belongs to the 0.5 rad/s stretch walk); their
turn friction need is 0.53–0.54, not 0.53–0.55; the materials note's tether and pad-pressure figures now give the
final picks as well as the bodies the material runs used; the scorecard's carpet score does not count standing on
medium pile; and the 2.54 mm aluminium plate saves under half of what 3 mm carbon saves.

The audits check the documents against the records. They do not re-run the simulations or validate the models;
that is the job of the closed-loop probe and the bench tests (`ROUND2_REVIEW.md` §13).
