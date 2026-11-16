# Carpet side-check (P36): what a first literature lookup found

Working paper of 15 Nov 2026. Nothing here is adopted and nothing here is a model yet.

**Status of every figure below: secondary.** A research agent searched with generic terms and could open none of the
primary documents (the session's network did not reach the journal and manufacturer hosts), so each number is what a
search result's summary said about a document, not what the document says. No number here may enter a rule or a model
file until someone has read its source.

The carpet in question (the owner, 15 Nov): the thin kind laid in schools and museums, with almost no height. In the
trade that is level-loop commercial carpet, glued down or as carpet tile, on concrete with no underlay.

## What was found

| Question | Finding | Grade |
|---|---|---|
| Pile height of loop-pile carpet tile | about 2.4 to 4.4 mm across several data sheets (ISO 1766) | secondary |
| Total thickness of such tile | about 4.5 to 9.3 mm (ISO 1765), tolerances of 5 to 10 % | secondary |
| Face weight | about 390 to 860 g/m² (ISO 8543) | secondary |
| Glued-down broadloom | no figures found; tile is the nearest | gap |
| Friction of a rubber or polyurethane sole on loop pile, dry | **no number found.** Three sources say only that carpet friction is high: above varnished wood, near vinyl, and little changed when wet | gap |
| Indentation at 7 to 40 kPa (the foot's pressure) | **no number found.** One maker claims under 15 % thickness loss (about 0.9 mm of 6.1 mm) after the ISO 3415 static and ISO 2094 dynamic tests, which load far harder (ISO 3416: 700 kPa for 24 h) and read what is left afterwards, not the sink under load | gap |
| Damping, energy return | no restitution or loss figure found. One force-plate paper (PLOS ONE 2025, doi 10.1371/journal.pone.0319304) compares two thin carpets without underlay (3.1 and 4.5 mm) with vinyl under a mechanical impactor: force reduction, loading rate, contact time. Its numbers were not retrieved | lead to read |
| Robots on thin carpet against hard floor | no paper found with carpet parameters. A patent text says a light robot cannot press a thick carpet down and walks as if on a mattress (thick carpet, qualitative) | gap |
| A flat foot pivoting on loop pile | nothing quantitative. A 1985 gait study on carpets reports no force-plate difference except turning time | gap |

## What follows for the side-check

1. **A sourced model cannot be written from this.** The two numbers a contact model needs most, the pad's friction on
   the carpet and how far the pad sinks under the robot's weight, are not in anything found.
2. **Geometry bounds the sink.** A pad cannot sink further than the pile is high: 2.4 to 4.4 mm, and less in practice.
   The existing surrogate floor (`robot_floor_carpet`, parameters unsourced; Stage 1: every walking start fell on it)
   should be compared with that bound before it is used again.
3. **Two honest ways on**, for the owner to choose when the 75 cm results are reported:
   - a *bracket*: walk the frozen sets on a small grid of floors (friction from the two already used, 0.9 and 0.54, and
     up; sink from 0 to the pile height), reported as "what would have to be true of the carpet for the walk to hold",
     with no verdict;
   - a *measurement*: a short bench sheet in the style of the delay sheet (pull a weighted pad across the carpet with a
     luggage scale for friction; a ruler or feeler gauge under a known weight for the sink), which turns the bracket
     into one point.
4. Reading the PLOS ONE paper and the two carpet-friction papers needs network access to their hosts; it would add
   the damping comparison and possibly a friction table, and would not replace the measurement.

Sources named by the lookup (not read): Forbo Tessera and Interface carpet-tile data sheets; Hirai and Gunji,
"Slipperiness and Coefficient of Friction on the Carpets", J. Textile Engineering 47(2); "The Frictional Advantages of
Wool Carpet in Slip Prevention"; Budinski, "On the Mechanism of Shoe Versus Walking Surface Friction" (2018); the PLOS
ONE 2025 paper above; ISO 3415, 3416, 2094, 1765, 1766, 8543.
