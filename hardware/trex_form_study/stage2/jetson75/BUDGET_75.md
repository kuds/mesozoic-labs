# The 75 cm build's prices, by the study's cost model

**Changed, 14 Nov 2026 (P31).** After the first version of this document (commit `4db227e`) was summarised for the
owner, the owner decided: "Please remove any budget concerns at this point and just document the prices. I think explore
the jetson nano hardware is a good exercise even with the extra cost." (decision-log row P31). So this is now a price
record. No price, kind label, source or sum changed. The places that changed:

- **Title.** It was "The 75 cm build's budget, rebuilt with the study's cost model".
- **Opening paragraphs.** The first called this "a planning estimate", said "nothing is adopted: the owner decides", and
  said that the 75 cm budget is decided after this rebuild (P30). It now says what the record is and, once, that P4's
  figure is not applied. "The owner decides" is removed. It now ends with when the choice of computer is made (P30). In
  the second, "Every budget number" is now "Every dollar figure", and a last sentence is added: the file names keep the
  word "budget".
- **"In brief, for the owner".** Rewritten as a short list of prices. The old brief's two opening sentences, its items 1
  to 4 and 8 and its cautions are kept under two new headings, "The prices, group by group" and "What to treat with
  care". Removed: item 5 (the totals set against P4), item 6's reading of P4, item 7 (the trims), the caution on how
  near the nominal totals are to $20,000, and the closing paragraph "The decision this prepares".
- **"The prices, group by group"** holds the old brief's numbered items, with these changes:
  - item 4: "the budget keeps" is now "the record keeps";
  - old item 6 is item 5: its self-assembled totals are kept, "(Jetson)" is now "(Jetson, microcontroller loop)", and in
    place of the reading of P4 it says that the frame and the coverings are still made outside and that the owner's
    time is not priced;
  - item 6 is the old brief's last caution, on the optional lines, moved here, with the head's and the audio's own
    figures added from Table 2;
  - old item 8 is item 7: its last sentence, "Quotes from drawings for those three lots would settle more than any other
    step (section 4)", is now "Section 4 ranks every line this way".
- **"What to treat with care"** holds the old brief's other cautions, with these changes:
  - the boom: "the boom may cost nothing" is now "the line is $0";
  - "The Jetson's extra cost is a range" is now "What the Jetson adds is a range", and "costs $160 to $235 more" and
    "costs $81 more" are now "adds $160 to $235" and "adds $81";
  - "This budget" is now "This record" in two cautions, and "the record's frame" is now "the round-two record's frame".
- **Section 1.** The heading was "How the budget was rebuilt". The steps are unchanged.
- **Table 4.** The column "Turnkey nominal against $20,000" and the sentence on P4 are removed.
- **Table 5, "The trim options",** is removed with its figures: the totals without the boom, and with and without the
  audio. The boom's and the audio's own prices are still in Table 2.
- **Table 6 is now Table 5,** "Where the prices are least certain"; it was "Quotes to ask for". The lines, the figures
  and their order are the same. Its introduction is reworded. Its last column, "What to ask for", is now "What a quote
  would cover": each cell describes the line and no longer says to ask, buy or decide. The `spares` cell said "at the
  quoted drive and pack prices"; no quote exists, so it now says "at the prices that quotes for the drives and the pack
  would give".
- **Table 7 is now Table 6,** "PR #668's price slips"; it was "budget slips". Its fourth row, which total PR #668 read
  against P4, is removed, and its third column, "The catalog or the decision log", is now "The catalog".
- **Sections 3.2 to 3.6.** "The budget", "this budget" and "This rebuild" are now "this record": twice in section 3.2
  and once in section 3.6. Section 3.3's tread item ended "If a fabricator's quote for the soles includes cast treads,
  drop the kit"; "quote" is now "price", and it ends "the kit is not a separate line". Section 3.5 was "Turnkey,
  self-assembled and P4": the reading of P4 is removed and the two totals are still explained.
- **Section 4.** It was "Quotes to ask for", with "the order to ask in". It is now "Where the prices are least certain":
  the same six entries and figures, described and not asked for. "No line has a quote" is added to its introduction. In
  item 2, "the budget" is now "the record". A closing paragraph is added; it holds figures of the old section 6.
- **Section 5.** It was "PR #668's budget slips". The reading of P4 is no longer listed among them, and "This rebuild"
  is now "This record".
- **Section 6, "What to do with this",** is removed. Its figures on where the gap between the low and high cases sits
  are under "The prices, group by group" and at the end of section 4.
- **Sections 7 and 8 are now 6 and 7,** and the references to them and to the renumbered tables follow. In the draft
  decision-log rows, the introduction gained two sentences: each draft records prices or the assumptions under them and
  none asks for a decision, and P31 is in the log already. Two drafts are removed: "P4's reading" and "The 75 cm budget"
  (accept, trim or raise). The other three are kept. "Budget finding" is now "Price finding", and its statement opens
  "The 75 cm build's prices, by" where it read "The 75 cm build's budget, rebuilt with". "Budget lines outside the cost
  model" is now "Lines outside the cost model".
- **`budget_75.json` and `budget_75.py`.** Removed: `p4`, each scenario's `p4` block and `turnkey_minus_ceiling`,
  `trims`, `trims_both`, the entries `nominal_over_ceiling`, `nominal_over_ceiling_share` and `high_over_ceiling` of
  `derived`, and the fourth slip. Renamed: `quotes` to `least_certain`, its rows' `ask` to `quote_scope`, and
  `derived`'s `quotes_subtotal_nominal` and `quotes_split_nominal` to `adopted_subtotal_nominal` and
  `adopted_split_nominal`. Added: `p31`. Reworded: the `study` string, as the title is, and the `quote_scope` texts, as
  Table 5's last column is. Every other value but `script_sha256` is the same. In the script, `ASK` is now
  `QUOTE_SCOPE` and `quotes()` is now `spreads()`; `P4_USD`, `TRIMS` and the code of the removed tables are removed; the
  docstring and the comments are reworded to match.
- **`tests/test_stage2_budget75.py`** follows these changes. One test is added: nothing outside this note sets a total
  against P4.

14 Nov 2026. A price record for one prototype of `75-M-H-10`, the 75 cm Hybrid with the targeted drives and bigger feet:
what the study's cost model and its parts catalog price that build at. Nothing here is built, quoted or measured, and
nothing is adopted. It carries out task 9 of `JETSON_PLEM_REVIEW.md` section 5. P4's $10-20k is not applied at this
point (decision-log row P31), so this document sets no total against it. The Raspberry Pi 5 + AI HAT+ stays adopted
(P14) and the Jetson Orin Nano Super is studied beside it (P30); which computer the build gets is decided after the
Jetson body's screens and the delay bench test.

Paths are relative to `hardware/trex_form_study/`. Every dollar figure comes from `stage2/jetson75/budget_75.json`,
written by `stage2/jetson75/budget_75.py`; the tables in section 2 are printed by `python stage2/jetson75/budget_75.py
md`. The only other numbers are PR #668's, marked "(unverified)", and the web search summaries of section 6. Tests:
`tests/test_stage2_budget75.py`. The file names keep the word "budget" from the first version.

**Kinds of number.** The study labels every number: (a) a prescribed-walk or static model result, (b) a closed-loop
simulation, (c) a sourced fact, (d) an estimate. **Every dollar figure below is kind (d), a cost-model estimate, unless
it is marked (c), a catalog price named by its catalog id.** A line marked c+d is a catalog price with an estimate on
top of it or in it: the cost model's allowance, or an assumed count or choice of part. A total is always (d), even where
catalog prices go into it. The few numbers that are not dollars are of two kinds: the hip height, the masses and the
cover volume read from the body's records are static model results (a), and the parts' ratings (amperes, volts, watts)
are catalog facts (c) under the ids named beside them. No figure is a closed-loop result or a hardware measurement.
Section 1 gives the details and the words used.

## In brief, for the owner

One prototype of the 75 cm build is priced at **$20,057** with the adopted computer, the Raspberry Pi 5 set (P14), and
at **$20,174** with the Jetson dev kit in its place. Both are US dollars, the cost model's nominal case, turnkey: made
and fitted together by outside shops, with bench equipment, spares and a contingency. Figures in threes are its low /
nominal / high cases.

- **With the Pi 5 set: $14,228 / $20,057 / $28,215.** That is the round-two record, $13,576 / $18,521 / $25,658, plus
  P21's four items, $652 / $1,536 / $2,557.
- **With the Jetson dev kit: $14,343 / $20,174 / $28,333.** The kit adds $115 / $117 / $118 ($399 against $290, c), or
  $157 / $192 / $194 if its 12 V regulator is bought separately.
- **With the Jetson driving the legs directly:** $435 / $853 / $895 to $477 / $928 / $1,025 more.
- **Self-assembled,** without the outside fit-up: $12,532 / $17,244 / $24,076 with the Pi 5 set, $12,647 / $17,360 /
  $24,194 with the Jetson.
- **Optional, in no total above:** the expressive head, $86 / $252 / $417; audio, $26 / $75 / $163.
- **Every figure is a cost-model estimate (d) or a catalog price (c). None is a quote.** The least certain lines are the
  frame, the coverings and the integration, scaled up from a 60 cm estimate.

## The prices, group by group

Figures in threes are the cost model's low / nominal / high cases. "Turnkey" is the model's total: outsourced
fabrication, a subcontractor fitting the robot together, bench equipment, spares and a contingency.

1. **The round-two record's own totals reproduce exactly: $13,576 low, $18,521 nominal, $25,658 high**, turnkey. The
   cost model, called on the record's parts, hip height and masses, returns all 16 lines, the three totals and the
   drivers equal to the record's. That record prices the adopted computer, the Raspberry Pi 5 with the AI HAT+
   accelerator board (P14; "the Pi 5 set" below). It does not price four of the materials and power items you adopted in
   P21.
2. **With P21's items the adopted build is $14,228 low, $20,057 nominal, $28,215 high**, turnkey. The four items add
   $652 / $1,536 / $2,557 with contingency (each rounded on its own below):
   - the overhead boom for the power cable, $316 / $965 / $1,626;
   - the external DC input, $211 / $322 / $434;
   - casting rubber and moulds for the polyurethane foot treads, and the grip test on your floors, $105 / $214 / $434;
   - coverings in the tougher nylon PA11 with thinner walls, in place of nylon PA12, $20 / $36 / $63.

   The 6061-T6 aluminium frame was already in the record.
3. **The Jetson Orin Nano Super developer kit ("the dev kit") in place of the Pi 5 set adds $115 / $117 / $118**, with
   the walking loop kept on the microcontroller: **$14,343 low, $20,174 nominal, $28,333 high**. The dev kit is $399 (c,
   catalog id `nvidia_orin_nano_super_devkit`) against $290 for the Pi 5 set (c, `rpi5_8gb_ai_hat_plus_13tops`).
   - Its carrier board needs a 12 V regulator fed from the six-cell (6S) pack: $39.95 or $69.95 (c, `pololu_d36v50f12`,
     `pololu_d42v110f12`). **If that regulator is bought on top, the Jetson adds $157 / $192 / $194: $14,385 low,
     $20,249 nominal, $28,409 high.**
   - It is probably not an extra. The power lot is carried over from the 60 cm robot, whose hardware list already has a
     12 V regulator for its computer. The Pi 5 set also needs a regulator of its own, a 5 V one, and that has no line
     either (section 3.2).
4. **If the Jetson also drives the legs directly, add $435 / $853 / $895 to $477 / $928 / $1,025 more.** That option
   needs USB adapters for CAN, the leg drives' data bus, at $368 each (c, `pcan_usb_fd`; one in the low case, two
   otherwise), and adapters for the servo buses (an estimate). It also needs a supervisor that stops the drives if the
   Jetson fails.
   - If the microcontroller stack that the record keeps serves as that supervisor, and the regulator is in the power
     lot: $14,778 low, $21,027 nominal, $29,228 high.
   - With a separate supervisor board (an estimate, $40 / $70 / $120) and a separately bought regulator: $14,862 low,
     $21,177 nominal, $29,434 high.
5. **Self-assembled, that is, without the subcontractor's fit-up, the same builds are $12,532 / $17,244 / $24,076** (Pi
   5 set) and $12,647 / $17,360 / $24,194 to $12,689 / $17,435 / $24,270 (Jetson, microcontroller loop). The frame and
   the coverings are still made outside, and your own time is not priced (section 3.5).
6. **The optional lines are outside every total above: $112 / $328 / $579 together**, turnkey, for the audio and for the
   expressive head's two extra servos (one turns the neck, one opens the jaw) with their brackets and the yaw bearing.
   The head alone is $86 / $252 / $417 and the audio $26 / $75 / $163, each rounded on its own. With the jaw deferred,
   as P2 and P29 have it, the head is one servo, one bracket and the bearing: $53 / $215 / $371. The servos and brackets
   are modelled; the bearing and the audio are not.
7. **Most of the uncertainty sits in three lots that were scaled up from an earlier 60 cm estimate**, not in the
   computer. The frame, the coverings and the integration differ by $8,054 between their low and high cases, out of
   $12,516 for all the lines of the adopted build together. The computer's own line differs by $140, and the choice of
   computer moves the nominal total by $117 to $192. Section 4 ranks every line this way.

## What to treat with care

- Every total is a cost-model estimate. About half of the nominal total is lots scaled from the 60 cm estimate
  (decision-log row A8), and none has a quote.
- **The boom's nominal value is the midpoint of a wide range**, $300 to $1,500, and the line is $0 if the bench frame
  can carry the cable.
- **What the Jetson adds is a range, not one figure.** This record prices it at $117 to $192 nominal, depending on the
  regulator. Two more things move it (section 3.2). The cost model counts storage twice for the Pi 5 set, whose $290
  already includes $40 for a drive (the catalog's note); with storage counted once for each computer the Jetson adds
  $160 to $235. At the street price the catalog notes for the Pi 5 set, about $323, the kit alone adds $81.
- **Three lines may be counted twice** against lots carried over from the 60 cm robot, whose contents the cost model
  does not list: the 12 V regulator and the DC input's regeneration clamp against the power lot, and the supervisor
  against the control lot. The Jetson totals are given both ways for the regulator and the supervisor. The clamp sits
  inside the DC input's $200 to $400 estimate and could not be separated.
- No Jetson body has been built in the model yet (task 6 of the same review). This record assumes it keeps the round-two
  record's frame, coverings, harness and battery pack.

## 1. How the prices were put together

**The kinds of number here.**

- **(c)** a catalog price, read from `designlab/data/parts.json` or `designlab/data/actuators.json` and named by its
  catalog id. The catalog records each price's source page and the date it was read (7 Nov 2026). Section 3.2 also
  quotes the two set prices of the study's compute survey (`COMPUTE.md`), with that source; one of them holds the
  survey's estimate for a drive.
- **(d)** everything else in dollars: every lot, allowance, scaling, contingency and total of the cost model
  (`designlab/cost.py`; decision-log row A8), and every line this document adds that the catalog does not hold. Where an
  added line is a guess with no source, the tables say so.
- **c+d** a catalog price with an estimate on top of it or in it: the cost model's allowance, or a count or choice of
  part that this document assumes.
- **(a)** the numbers that are not dollars and come from the body's records: the hip height, the masses, the cover
  volume.

**Words used.**

- Low, nominal and high are the cost model's three cases. Nominal is its target; low and high are its range, not bounds.
- Turnkey is the cost model's total. It includes outsourced fabrication, a subcontractor fitting the robot together (the
  model's `integration` line), the bench equipment, spares and a contingency.
- Self-assembled is the same total without the `integration` line and without the contingency on it. The frame and the
  coverings are still made outside. The owner's own time is not priced.
- A lot is a line priced as one sum, not part by part.
- H is the hip height, 0.502 m on this body (a). The cost model scales some lots by H / 0.39 m.
- 6S is a six-cell lithium pack, 21.0-25.2 V. CAN is the data bus of the 12 leg drives, and CAN FD its faster variant.
  TTL is the serial bus of the neck and tail servos. IMU is the inertial measurement unit. SSD is a solid-state drive;
  NVMe is the kind that plugs into the computer's board.
- MJF is Multi Jet Fusion, a nylon powder printing process. PA11 and PA12 are two nylons. TPU is thermoplastic
  polyurethane, a rubber that can be printed. Shore A is a hardness scale for rubber.
- sha256 is a file's fingerprint; it shows which file a number was made from. PR is a pull request.
- P, E, F and A rows are in `DECISION_LOG.md`. "The review" is `JETSON_PLEM_REVIEW.md`.

**The steps.**

1. **The baseline first.** The script reads the committed round-two record of `75-M-H-10`,
   `round2/refine/feet_75/candidates/75-M-H-10.json`. It hands `designlab/cost.py`'s `estimate` function the three
   things that function reads of a design: the selected parts, the hip height (0.502 m) and the mass by category. The
   result must equal the record's `cost` block exactly. If it does not, the script stops and prints each difference. It
   is equal: 16 lines, 3 totals and 6 drivers (Table 1).
2. **Each variant is a difference from that baseline, with its own lines** (Table 2). Where the cost model covers a
   line, the script calls it again with the changed input: the computer's catalog id, the covering material and volume,
   or the expressive head's own committed record. Lines the model does not cover are added one by one, each with its
   kind, its source and how its low, nominal and high values were set. Added lines take the model's own contingency
   (5.3 / 7.2 / 8.4 %).
3. **Where a study document gives only a range, the nominal value is its midpoint.** That rule sets the external DC
   input ($200-400, nominal $300) and the boom ($300-1,500, nominal $900), both from `round2/refine/MATERIALS.md`
   section 6.
4. **Each total is given two ways**, turnkey and self-assembled (Table 4).
5. **Before adding P21's items, the script checks what the record already holds** (Table 3), so that nothing is counted
   twice.
6. **A line that a kept lot may already hold is priced both ways.** The 12 V regulator (against the power lot) and the
   supervisor (against the kept control lot) are in their blocks, and every Jetson total is also given without them
   (Tables 2 and 4).
7. **Rounding.** Totals are rounded to the dollar after summing, as the cost model does. So a difference of two printed
   totals can be $1 off a block's printed figure in Table 2. The text quotes differences of printed totals, so its sums
   work to the dollar.

What the cost model leaves out, by its own header: engineering salaries, facilities and firm quotes. Freight and taxes
are inside its contingency.

## 2. The tables

Printed from `budget_75.json` by `python stage2/jetson75/budget_75.py md`; a test checks that this block is that output.

<!-- budget_75.py md: begin -->
### Table 1. The round-two record's cost block, recomputed

`round2/refine/feet_75/candidates/75-M-H-10.json` (sha256 `c24dcc4e928e`). The recorded and the recomputed values are
equal on every line, so one set of numbers is shown. Kind c+d: catalog prices (c) plus the model's allowance (d). Kind
d: the cost model.

| Line | Low | Nominal | High | Kind | Equal | What it is |
|---|---|---|---|---|---|---|
| `leg_actuators` | $3,546 | $3,546 | $3,723.30 | c+d | yes | 12 leg drives at catalog prices; the high case adds 5 % |
| `expression_actuators` | $504.89 | $504.89 | $530.13 | c+d | yes | neck and tail servos at catalog prices; the high case adds 5 % |
| `compute` | $350 | $410 | $490 | c+d | yes | the computer's catalog price + $60 / $120 / $200 for storage, fan duct, mounts and leads |
| `rgbd_camera` | $284 | $304 | $349 | c+d | yes | catalog price + $15 / $35 / $80 for the lead and mount |
| `tof` | $39.90 | $49.90 | $69.90 | c+d | yes | two sensors at catalog price + $0 / $10 / $30 |
| `battery` | $137.19 | $167.19 | $207.19 | c+d | yes | one pack at catalog price + $30 / $60 / $100 for the cassette |
| `control` | $125 | $180 | $270 | d | yes | PR #662's 60 cm lot, not scaled (Teensy 4.1, two CAN transceivers, IMU; the 60 cm hardware basis also names a TTL servo-bus interface) |
| `foot_sensing` | $113.44 | $204.19 | $340.31 | d | yes | PR #662's 60 cm lot x (H / 0.39 m)^0.5 |
| `power` | $275 | $420 | $650 | d | yes | PR #662's 60 cm lot for a 24 V bus, not scaled |
| `harness` | $193.02 | $308.83 | $450.38 | d | yes | PR #662's 60 cm lot x (H / 0.39 m) |
| `frame` | $2,947.58 | $4,503.25 | $6,550.19 | d | yes | PR #662's 60 cm lot x (structure mass / 1.79 kg)^0.6 |
| `covers` | $1,102.87 | $1,890.63 | $3,308.60 | d | yes | PR #662's 60 cm lot x (cover volume / 480 cm3)^0.85 |
| `bench` | $550 | $850 | $1,300 | d | yes | PR #662's 60 cm lot, not scaled (H is not above 0.6 m) |
| `spares` | $1,113.19 | $1,313.19 | $1,613.19 | c+d | yes | two of the dearest leg drive and one pack at catalog prices + $150 / $350 / $650 |
| `integration` | $1,610.62 | $2,624.71 | $3,817.76 | d | yes | PR #662's 60 cm lot x (H / 0.39 m)^0.7 |
| `contingency` | $683.31 | $1,243.93 | $1,988.28 | d | yes | 5.3 / 7.2 / 8.4 % of the other lines |
| **Total (turnkey)** | **$13,576** | **$18,521** | **$25,658** | d | yes | the sum of the lines above, as recorded |
| Self-assembled | $11,880 | $15,707 | $21,520 | d | | the same without `integration` and the contingency on it |

### Table 2. The variants, each as a difference from the record

Each row is one line's change. A block's row "Block, with contingency" is the change in the record's turnkey total when
that block alone is applied: its lines with the cost model's contingency (x 1.053 / 1.072 / 1.084). Totals are rounded
to the dollar after summing, so a block's figure can be $1 off its lines times those factors, and a difference of two
totals in Table 4 can be $1 off a block's figure. Kind c+d on an added line: the price is the catalog's (c), the count
or the choice of part is an assumption (d).

**`jetson`: Jetson Orin Nano Super dev kit in place of the Raspberry Pi 5 + AI HAT+, with the walking loop on the
microcontroller (the baseline of the owner's 14 Nov decision, P30).**

| Line | Low | Nominal | High | Kind | Source and derivation |
|---|---|---|---|---|---|
| `compute` (a line of the record, changed) | +$109 | +$109 | +$109 | c+d | Catalog: `nvidia_orin_nano_super_devkit` at $399 in place of `rpi5_8gb_ai_hat_plus_13tops` at $290; the model's allowance for storage, fan duct, mounts and leads is unchanged. Not like for like on storage: the Pi 5 set's price already holds a USB SSD and the dev kit's holds none; the compute survey's set prices, each with its storage, are $439 and $290 (COMPUTE.md; the dev kit's holds the survey's $40 estimate for a drive), a difference of $149 |
| 6S-to-12 V regulator for the dev kit's carrier (decision-log E7), if it is bought separately | +$39.95 | +$69.95 | +$69.95 | c+d | Catalog: `pololu_d36v50f12` at $39.95, `pololu_d42v110f12` at $69.95. The prices are the catalog's (c); the choice of part per case is an assumption (d). Low: the 4.5 A part; nominal and high: the 9 A part (no power mode is chosen, and the larger part covers both modes with room for USB loads). Both accept the 6S bus; E7 rules the D36V50F12 out only on 12S. Caution: probably already in the power lot, and not a cost of the Jetson alone (section 3.2). The lot is the 60 cm robot's, whose hardware basis lists a D36V50F12 for its computer (r0_posture_study_2026-11-07/study/hardware/component_basis.md); E7 calls that part the incumbent; the compute survey lists the dev kit's 12 V converter as existing and not counted, and the Pi 5 set's 5 V / 5 A regulator as not priced. The lot's own breakdown is not in this repository |
| `control` (a line of the record, unchanged at $125 / $180 / $270) | +$0 | +$0 | +$0 | d | The microcontroller stack stays: it is the record's control lot (Teensy 4.1, two CAN transceivers, IMU). The catalog has no Teensy entry |
| **Block, with contingency (turnkey)** | **+$157** | **+$192** | **+$194** | d | Every line above |
| **The same if the power lot already holds the 12 V regulator** | **+$115** | **+$117** | **+$118** | d | Without `regulator_12v` |

**`jetson_direct_parts`: Added only if the Jetson drives the leg drives directly over USB-CAN (the alternative the bench
test measures), on top of `jetson`.**

| Line | Low | Nominal | High | Kind | Source and derivation |
|---|---|---|---|---|---|
| USB-CAN FD adapter(s), PCAN-USB FD (one CAN channel each) | +$368 | +$736 | +$736 | c+d | Catalog: `pcan_usb_fd` at $368.00. The price is the catalog's (c); the count is an assumption (d): one adapter in the low case (PR #668's layout, unverified); two in the nominal and high cases, to keep the two CAN buses of the study's build (designlab/build.py, control stack: 'CAN x2'; JETSON_PLEM_REVIEW.md section 3). The bus plan (review task 7) decides |
| Supervisor, if it is a separate board: a small microcontroller board with a CAN transceiver and a stop output | +$40 | +$70 | +$120 | d | No catalog entry; this document's estimate from the price class of a Teensy 4.1 (about $34-36 at its maker's store, a web search summary) plus a transceiver, a relay driver and wiring (a guess, no source). Caution: overlaps the kept control lot, whose Teensy 4.1 and CAN transceivers are such a board: if that stack serves as the supervisor, this line is not needed |
| USB adapter(s) for the neck and tail servos' TTL buses | +$45 | +$60 | +$90 | d | No catalog entry; this document's estimate: a ROBOTIS U2D2 (about $44, a web search summary) plus a Feetech bus adapter (about $15, a guess, no source). Caution: the kept control lot's 60 cm basis also names a TTL servo-bus interface; if the kept stack keeps driving the servo buses, this line is not needed |
| `control` (a line of the record, unchanged at $125 / $180 / $270) | +$0 | +$0 | +$0 | d | Kept in full as a conservative choice, because the lot has no breakdown here. In the direct option its Teensy and two CAN transceivers overlap the supervisor line, and its servo-bus interface overlaps the USB servo-bus adapters; only the IMU is certainly still needed |
| **Block, with contingency (turnkey)** | **+$477** | **+$928** | **+$1,026** | d | Every line above |
| **The same if the kept Teensy stack is the supervisor** | **+$435** | **+$853** | **+$896** | d | Without `supervisor` |

**`p21`: P21's adopted materials and power items that the record does not hold.**

| Line | Low | Nominal | High | Kind | Source and derivation |
|---|---|---|---|---|---|
| `covers` (a line of the record, changed) | +$19.41 | +$33.29 | +$58.25 | d | Cost.py's material path: PA11 with walls x0.75 in place of PA12 at full walls (the study's own lever, re-priced on this body; an estimate, since the lever was never run on it) |
| Cast polyurethane pad treads: casting rubber, printed moulds and the grip test on the owner's floors | +$100 | +$200 | +$400 | d | Low: MATERIALS.md section 5's estimate for the floor test alone (about one day and $100), with a printed TPU trial tread. Nominal adds $100 and high adds $300 for casting rubber, release agent and printed moulds: this document's guess, with no source beyond the price class of a trial unit of 60 Shore A urethane rubber (about $25-45, a web search summary); the high case allows for several compounds and remakes. The soles themselves are in the frame lot |
| External DC input: the input, the 24 V priority power path, a regeneration clamp and the dummy puck | +$200 | +$300 | +$400 | d | MATERIALS.md section 6's estimate, $0.2-0.4k 'for the input, power path, clamp and puck'; the nominal value is its midpoint. That estimate does not name the 5 m cable or the stop tie-in; this document assumes they fit inside it. Caution: the clamp may already be in the power lot: the 60 cm robot's hardware basis names a regenerative clamp with its regulators (component_basis.md), and cost.py gives that lot no scope, so its contents could not be checked |
| Overhead boom for the power cable, with a balancer under 0.5 kg | +$300 | +$900 | +$1,500 | d | MATERIALS.md section 6's estimate, $0.3-1.5k 'if the bench lot's catch frame cannot carry it'; the nominal value is its midpoint; $0 if the catch frame can |
| `frame` (a line of the record, unchanged at $2,947.58 / $4,503.25 / $6,550.19) | +$0 | +$0 | +$0 | d | The 6061-T6 frame is already the record's frame line, and the 2.54 mm plates cost the same in the model (checks, `frame_6061`) |
| `battery` (a line of the record, unchanged at $137.19 / $167.19 / $207.19) | +$0 | +$0 | +$0 | c+d | The removable pack stays (P21: the battery plus an external DC input) |
| `bench` (a line of the record, unchanged at $550 / $850 / $1,300) | +$0 | +$0 | +$0 | d | The bench supply for the tether is already in the bench lot |
| **Block, with contingency (turnkey)** | **+$652** | **+$1,536** | **+$2,557** | d | |

**`head_x`: Optional: the expressive head's two extra servos (one for the neck yaw, one for the jaw), their brackets and
the yaw-stage bearing. The servos and brackets are modelled (the head's committed record); the bearing is not. P2's
deferral of a jaw for the build stands (P29), so the build's head as decided today has the neck-yaw servo only.**

| Line | Low | Nominal | High | Kind | Source and derivation |
|---|---|---|---|---|---|
| `expression_actuators` (a line of the record, changed) | +$44 | +$44 | +$46.20 | c+d | Two more STS3215 servos (`feetech_sts3215_c018`), one for the neck yaw and one for the jaw, at the catalog price, as the head's committed record holds them (round2/refine/head_x_75); the high case adds 5 % |
| `frame` (a line of the record, changed) | +$17.34 | +$26.49 | +$38.53 | d | The two servos' brackets: the head's record carries 0.040 kg more structure (a), which the frame lot's mass scaling prices |
| Optional, unmodelled: the neck-yaw stage's bearing (expressive head) | +$20 | +$165 | +$300 | d | No catalog entry and not in any record (stage2/x/X_COSTS.md section 5). This document's estimate; low: two thin-section ball bearings in a machined seat (a guess, no source); nominal: a 20 mm-bore polymer slewing ring at the cheapest listing found (about $164); high: the same ring at the dearest listing found (about $300); the listings are from a web search summary |
| **Block, with contingency (turnkey)** | **+$86** | **+$252** | **+$417** | d | |
| **The same with the jaw deferred (P2, P29): the neck-yaw servo, its bracket and the bearing** | **+$53** | **+$215** | **+$371** | d | An estimate: the head's record less the jaw servo ($22, c) and its bracket, priced by the cost model; no record of that body exists. Without the bearing: +$32 / +$38 / +$46 |

**`audio`: Optional: audio.**

| Line | Low | Nominal | High | Kind | Source and derivation |
|---|---|---|---|---|---|
| Optional, unmodelled: audio (amplifier, speaker, microphones) | +$25 | +$70 | +$150 | d | No catalog entry and not in any record. This document's estimate; low: a small amplifier board (about $6, a web search summary) with a small speaker and wiring (a guess, no source); high: with a far-field USB microphone array (about $60-78 in US listings, a web search summary) and a larger speaker (a guess); nominal: a guess between the two |
| **Block, with contingency (turnkey)** | **+$26** | **+$75** | **+$163** | d | |

### Table 3. What the record already holds of P21's items

| P21 item | In the record's totals? | How it was checked | Line added (low / nominal / high) |
|---|---|---|---|
| 6061-T6 frame (2.54 mm plates where a leg bench test allows) | yes | The record has no structure_material override, and its frame line equals cost.REF['frame'] x its frame_factor with no material factor; the committed 2.54 mm plate lever on 75-M-T-10 (fixed_drives.json, mat75_thinal) costs exactly what its reference does | none ($0) |
| MJF PA11 coverings, walls about a quarter thinner | no: PA12 at full walls | The record has no cover_material or cover_wall_scale override, and its cover volume driver equals its cover mass over the default material's density, so its covers line is PA12 at full walls. The re-pricing rule, run on the body the lever was recorded on, gives its recorded covers line within $0.17 and its totals $13,506 / $18,407 / $25,474 against $13,506 / $18,406 / $25,474 recorded | `covers` +$19.41 / +$33.29 / +$58.25 |
| Cast polyurethane pad treads, measured on the owner's floors | soles only | The frame lot of cost.py names soles in its scope, and the record's foot mass (with P20's bigger feet: toe_scale 1.15, pad_scale 1.25) is inside the structure mass that scales that lot; no line names a tread compound or a floor test | `pad_treads` +$100 / +$200 / +$400 |
| The battery, with an external DC input | the battery: yes; the input: no; the clamp: not known | The record has no power_mode override and its battery line is the pack plus the cassette allowance; the bench lot's scope names the tether supply and the catch frame, so the bench supply is not added again; no line names an onboard DC input, a power path or a boom. Not checked: cost.py gives the power lot no scope, so its contents could not be checked: the 60 cm robot's hardware basis names a regenerative clamp with its regulators, and the DC input line prices a clamp too | `dc_input` +$200 / +$300 / +$400 |
| The cable hung from a boom | no (the bench supply: yes) | cost.py's bench lot covers "tether supply, charger, cradle/catch frame, instrumentation"; MATERIALS.md section 6 prices a boom only "if the bench lot's catch frame cannot carry it" | `cable_boom` +$300 / +$900 / +$1,500 |

### Table 4. Totals: turnkey and self-assembled

Turnkey is the cost model's total. Self-assembled leaves out the subcontract integration line and the contingency on it.
Every total is kind d. A Jetson build has two rows: with every line of its blocks, and without the lines a kept lot may
already hold (Table 2).

| Build | Turnkey low | Turnkey nominal | Turnkey high | Self-assembled low | Self-assembled nominal | Self-assembled high |
|---|---|---|---|---|---|---|
| `record`: The round-two record: Raspberry Pi 5 + AI HAT+ (P14), P21's items not priced | $13,576 | $18,521 | $25,658 | $11,880 | $15,707 | $21,520 |
| `pi_p21`: Raspberry Pi 5 + AI HAT+ (P14, adopted) with P21's items | $14,228 | $20,057 | $28,215 | $12,532 | $17,244 | $24,076 |
| `jetson_mcu_p21`: Jetson dev kit, walking loop on the microcontroller, with P21's items | $14,385 | $20,249 | $28,409 | $12,689 | $17,435 | $24,270 |
| `jetson_mcu_p21`, if the power lot already holds the 12 V regulator | $14,343 | $20,174 | $28,333 | $12,647 | $17,360 | $24,194 |
| `jetson_direct_p21`: Jetson dev kit driving the legs directly, with P21's items | $14,862 | $21,177 | $29,434 | $13,166 | $18,364 | $25,296 |
| `jetson_direct_p21`, if the power lot already holds the 12 V regulator and the kept Teensy stack is the supervisor | $14,778 | $21,027 | $29,228 | $13,082 | $18,214 | $25,090 |
| `pi_p21_optional`: Raspberry Pi 5 + AI HAT+ with P21's items and every optional line | $14,340 | $20,385 | $28,794 | $12,644 | $17,571 | $24,656 |
| `jetson_mcu_p21_optional`: Jetson dev kit, microcontroller loop, P21's items and every optional line | $14,497 | $20,577 | $28,988 | $12,801 | $17,763 | $24,850 |
| `jetson_mcu_p21_optional`, if the power lot already holds the 12 V regulator | $14,455 | $20,502 | $28,912 | $12,759 | $17,688 | $24,774 |

### Table 5. Where the prices are least certain, the widest lines first

The lines of `pi_p21`, the adopted build (P14) with P21's items, ranked by high minus low: the wider that gap, the less
the estimate says about the line's price. The contingency is not ranked: it is 5.3 / 7.2 / 8.4 % of whatever the other
lines come to. A Jetson build has the same lines in the same order, with its `compute` line at $459 / $519 / $599 and
the lines of the second table. The last column says what a quote for the line would have to cover; no quote has been
asked for.

| Line | High minus low | Low / nominal / high | How it was priced | What a quote would cover |
|---|---|---|---|---|
| `frame` | $3,602.61 | $2,947.58 / $4,503.25 / $6,550.19 | scaled from the 60 cm estimate | Laser-cut and machined 6061-T6 plates and brackets, bearings, hubs, soles and fasteners, from drawings, at three shops; the plates at 2.54 mm and at 3 mm |
| `covers` | $2,244.57 | $1,122.28 / $1,923.92 / $3,366.85 | scaled from the 60 cm estimate | MJF PA11 shells from the exported cover shapes with graded 1.4 / 1.2 / 1.0 mm walls, inserts and finish, at three print services |
| `integration` | $2,207.14 | $1,610.62 / $2,624.71 / $3,817.76 | scaled from the 60 cm estimate | Subcontract fit-up of one prototype: hours and rate for mechanical assembly, harness routing and bench checkout |
| `cable_boom` | $1,200 | $300 / $900 / $1,500 | new line, estimated here | Whether the bench frame can carry the 5 m cable (the line is then $0); if not, a jib or gantry boom with a balancer rated under 0.5 kg |
| `bench` | $750 | $550 / $850 / $1,300 | 60 cm lot, carried over unscaled | 24 V bench supply (about 31 A, RSP-750-24 class), a 6S charger, a cradle or catch frame and instrumentation |
| `spares` | $500 | $1,113.19 / $1,313.19 / $1,613.19 | catalog price + allowance | Two spare hip-pitch drives and one spare pack, at the prices that quotes for the drives and the pack would give |
| `power` | $375 | $275 / $420 / $650 | 60 cm lot, carried over unscaled | The power bay as one assembled module: distribution board, fuses, the stop circuit, the regeneration clamp and its regulators, each regulator named (the computer's, the logic rail's, the servos') |
| `pad_treads` | $300 | $100 / $200 / $400 | new line, estimated here | A urethane casting kit (50-60 Shore A polyether), printed moulds and the grip test on the owner's floors |
| `harness` | $257.36 | $193.02 / $308.83 / $450.38 | scaled from the 60 cm estimate | The leg and trunk harness with locking connectors and the neck service loop, from a wiring list |
| `foot_sensing` | $226.87 | $113.44 / $204.19 / $340.31 | scaled from the 60 cm estimate | Foot contact or load sensors for two feet with their conditioning boards |
| `dc_input` | $200 | $200 / $300 / $400 | new line, estimated here | The input connector, the priority power-path board (LTC4417 class), the dummy puck and the cable; the clamp and the stop tie-in belong to the power bay's price |
| `leg_actuators` | $177.30 | $3,546 / $3,546 / $3,723.30 | catalog price + allowance | Twelve drives landed: unit price, shipping, duty and lead time (10 x RMD-X4-P36-36-E-L, 2 x RMD-X6-P20-60-E), and which X6 variant ships at the listed price |
| `control` | $145 | $125 / $180 / $270 | 60 cm lot, carried over unscaled | Teensy 4.1, two CAN transceivers, an IMU and the servo-bus interface as a board stack |
| `compute` | $140 | $350 / $410 / $490 | catalog price + allowance | The computer at today's price, its storage and its mounting tray |
| `battery` | $70 | $137.19 / $167.19 / $207.19 | catalog price + allowance | One pack and the tail-base cassette |
| `rgbd_camera` | $65 | $284 / $304 / $349 | catalog price + allowance | The camera and a locking lead, and the remaining stock of the discontinued camera chip |
| `tof` | $30 | $39.90 / $49.90 / $69.90 | catalog price + allowance | Stock parts at their list prices |
| `expression_actuators` | $25.24 | $504.89 / $504.89 / $530.13 | catalog price + allowance | The neck and tail servos at today's prices |

Lines only a Jetson build adds (`jetson_direct_p21`). `usb_can`, `supervisor`, `servo_bus_adapter` are added only if the
Jetson drives the legs directly.

| Line | High minus low | Low / nominal / high | How it was priced | What a quote would cover |
|---|---|---|---|---|
| `usb_can` | $368 | $368 / $736 / $736 | new line, catalog price; the count or the part is assumed | The adapter at today's price, and the two-channel version's price |
| `supervisor` | $80 | $40 / $70 / $120 | new line, estimated here | No firm price until the supervisor's design exists; not needed if the kept Teensy stack is the supervisor |
| `servo_bus_adapter` | $45 | $45 / $60 / $90 | new line, estimated here | Stock parts at their list prices |
| `regulator_12v` | $30 | $39.95 / $69.95 / $69.95 | new line, catalog price; the count or the part is assumed | A stock part at its list price; an extra only if the power bay holds no 12 V regulator for the computer |

### Table 6. PR #668's price slips, corrected

| Slip | PR #668 (unverified) | The catalog | Used here |
|---|---|---|---|
| The Jetson dev kit's price | $249 in its low and nominal cases | `nvidia_orin_nano_super_devkit` is $399 in `designlab/data/parts.json` (c) | $399, the price of `nvidia_orin_nano_super_devkit` |
| The tail servo's catalog id | `feetech_sts3215_12v` | No such id in `designlab/data`; the record names `feetech_sts3215_c018` | `feetech_sts3215_c018` ($22.00), the record's own id |
| The neck servo's catalog id | `robotis_xm540_w270` | No such id in `designlab/data`; the record names `robotis_xm540_w270_t` | `robotis_xm540_w270_t` ($482.89), the record's own id |
<!-- budget_75.py md: end -->

## 3. Reading the tables

### 3.1 The baseline (Table 1)

- The record's totals are $13,576 / $18,521 / $25,658. `LESSONS_LEARNED.md` section 7 quotes the nominal and high as
  $18.5k and $25.7k.
- By how they were priced, the nominal $17,277 before contingency splits into:
  - $9,532 (55 %) in five lots scaled from the 60 cm estimate: frame, coverings, integration, harness, foot sensing;
  - $1,450 (8 %) in three 60 cm lots carried over unscaled: bench, power, control;
  - $6,295 (36 %) in catalog prices with the model's allowances: the 12 leg drives, the two servos, the computer, the
    camera, the two distance sensors, the pack and the spares.
- The catalog prices inside those lines are (c): $269 for each of the ten `myactuator_rmd_x4_p36_36_e_l` drives, $428
  for each of the two `myactuator_rmd_x6_p20_60_e` hip-pitch drives, $482.89 for the `robotis_xm540_w270_t` neck servo,
  $22 for the `feetech_sts3215_c018` tail servo, $269 for the `luxonis_oak_d_lite` camera, $19.95 for each
  `st_vl53l5cx_carrier` distance sensor and $107.19 for the `lipo_6s_3300` pack.

### 3.2 The computer (Table 2, blocks `jetson` and `jetson_direct_parts`)

- **(i) The Pi 5 set, as adopted (P14).** It is the record's computer, so it adds nothing: its line is
  $350 / $410 / $490, the $290 catalog price (c) plus the model's $60 / $120 / $200 allowance for storage, fan duct,
  mounts and leads. Two things about that line matter for the comparison below.
  - The $290 already includes a USB SSD at $40 (the catalog's note: $175 + $5 + $70 + $40; the $40 is the compute
    survey's estimate). The allowance stands for storage as well: `designlab/build.py` names it "NVMe, fan duct, mounts,
    leads". So the record's line counts the Pi 5 set's storage twice.
  - The set needs a 5 V, 5 A regulator fed from the pack (the catalog's note; the compute survey lists it as "not
    priced"). The catalog has no such part and this record has no line for it. A 5 V, 5.5 A regulator from the maker of
    the two 12 V parts lists at about $40 (a web search summary, d).
- **(ii) The Jetson dev kit, at the catalog price.** The catalog id is `nvidia_orin_nano_super_devkit` and its
  `price_usd` is 399 (c). The catalog's own sources explain the stale figure: NVIDIA's datasheet still prints the launch
  price of $249, and the retailer listing read on 7 Nov 2026 shows $399. `COMPUTE.md` says the same. The swap raises the
  compute line by $109 in each case, to $459 / $519 / $599. No storage line is added, because the model's allowance
  already stands for it.
  - **That $109 is not like for like on storage.** The dev kit's $399 holds no drive and the Pi 5 set's $290 holds one.
    The compute survey behind `COMPUTE.md` prices the two sets each with its storage at $439 and $290, a difference of
    $149. The $439 is the $399 kit (c) plus the survey's estimate of about $40 (d) for an NVMe SSD.
  - The catalog also notes that the Pi 5 set costs about $323 at street prices, which would cut the difference by $33.
- **The regulator.** The dev kit's carrier has a 19 V input jack (c, the catalog's entry), below the 6S pack's 25.2 V at
  full charge, so it needs a step-down regulator; the study's choice is 12 V. Both catalog parts accept the 6S bus; E7
  rules the smaller one out only on a 12S bus. The low case takes `pololu_d36v50f12` (4.5 A, $39.95, c). The nominal and
  high cases take `pololu_d42v110f12` (9 A, $69.95, c), because no power mode is chosen and the larger part covers both
  modes.
  - **It is probably already in the power lot, and it is not a cost of the Jetson alone.** Four things in the repository
    say so:
    - the cost model's power lot is the 60 cm reference robot's, used unchanged, and that robot's hardware basis lists
      its three regulators with the D36V50F12 as the computer's
      (`r0_posture_study_2026-11-07/study/hardware/component_basis.md`);
    - E7 calls the D36V50F12 "the incumbent";
    - the compute survey lists the dev kit's 12 V converter as "existing" and "not counted"
      (`designlab/data/research/compute_2026-11-07.json`);
    - `designlab/build.py` puts the 48 V version's 48-to-12 V converter in the same bay.
  - The lot's own breakdown is not in this repository, so this is likely, not established. If the lot holds the 4.5 A
    part and the Jetson turns out to need the 9 A part, the upgrade is $30 (c, the difference of the two catalog
    prices).
  - So each Jetson total is given both ways. The regulator is $42 / $75 / $76 with contingency.
- **The difference between the two computers, read five ways** (turnkey, with contingency; all d):
  - $115 / $117 / $118: the kit alone, as the cost model prices it (the $109 catalog difference);
  - $157 / $192 / $194: with a 12 V regulator bought on top;
  - $157 / $160 / $162: the kit alone, with storage counted once for each computer (the $149 difference; that its low
    case equals the line above is a coincidence);
  - $199 / $235 / $237: both of those together;
  - $80 / $81 / $82: the kit alone, with the Pi 5 set at the street price the catalog notes.

  The first two are the ones this record carries into its totals.
- **The real-time options, listed separately.**
  - **Walking loop on the microcontroller** (the baseline; `COMPUTE.md`'s design, in which a Linux crash still ends in a
    controlled stop). Nothing is added. The microcontroller stack is the record's `control` lot, $125 / $180 / $270: a
    Teensy 4.1, two CAN transceivers and an IMU, and by the 60 cm hardware basis a TTL interface for the servos. The
    catalog has no Teensy entry.
  - **Jetson driving the legs directly** (the alternative the delay bench test measures). Added parts:
    - USB-CAN adapters: `pcan_usb_fd`, $368 each (c). Each carries one CAN channel. How many are needed is an
      assumption: one in the low case, as in PR #668 (unverified); two in the nominal and high cases, to keep the two
      CAN buses of the study's build (`designlab/build.py`, control stack: "CAN x2"; `JETSON_PLEM_REVIEW.md` section 3).
      The bus plan (review task 7) decides.
    - USB adapters for the neck and tail servo buses: no catalog entry; $45 / $60 / $90, this document's estimate.
    - A supervisor that stops the drives when the Jetson's heartbeat is lost: no catalog entry; $40 / $70 / $120 for a
      separate board, this document's estimate ($42 / $75 / $130 with contingency).
    - The `control` lot is kept in full. That is a conservative choice, made because the lot has no breakdown here. In
      the direct option its Teensy and two CAN transceivers overlap the supervisor line, and its servo interface
      overlaps the USB servo adapters; only the IMU is certainly still needed.
    - So the direct option adds $435 / $853 / $895 if the kept Teensy stack is the supervisor, and $477 / $928 / $1,025
      with a separate supervisor board. (Table 2's block figures are $1 higher in the high case, because of rounding.)

### 3.3 P21's items (Tables 2 and 3)

P21 adopted a 6061-T6 frame, MJF PA11 coverings with walls about a quarter thinner, cast polyurethane pad treads
measured on your floors, and the battery plus an external DC input with the cable hung from a boom. Table 3 shows what
the record already holds of each and how that was checked. In short:

- **Frame: in the record, $0 added.** The record's frame line is the reference lot times its mass factor, with no
  material factor. The 2.54 mm plates that P21 allows "where a leg bench test allows" cost the same as 3 mm plates in
  the model: the committed lever on the 75 cm Tapered body costs exactly what its reference does.
- **Coverings: not in the record.** The record prices PA12 at full wall thickness. Re-priced through the cost model's
  own material path as PA11 with walls ×0.75, the line rises by $19 / $33 / $58. This is an estimate for this body: the
  study ran that lever on the 75 cm Tapered body, not on this one. The same rule, run on the Tapered body, reproduces
  the lever's recorded coverings line within $0.17.
- **Treads: the soles are in the frame lot; the tread compound and the floor test are not named anywhere.** The added
  line is $100 / $200 / $400. Its low case is `MATERIALS.md` section 5's estimate for the grip test (about one day and
  $100). The nominal and high cases add $100 and $300 for casting rubber, release agent and printed moulds: this
  document's guess, with no source beyond the price class of a trial unit of urethane rubber. If a fabricator's price
  for the soles includes cast treads, the kit is not a separate line.
- **External DC input: not in the record.** $200 / $300 / $400, `MATERIALS.md` section 6's estimate "for the input,
  power path, clamp and puck". Three cautions:
  - that estimate does not name the 5 m cable or the tie-in to the stop circuit, and this document assumes they fit
    inside it;
  - the regeneration clamp may already be in the power lot, because the 60 cm robot's hardware basis names one with its
    regulators, and `designlab/cost.py` gives the power lot no scope, so its contents could not be checked;
  - the bench supply that feeds the input is already in the bench lot and is not added again.
- **Boom: not in the record.** $300 / $900 / $1,500, `MATERIALS.md` section 6's range, which applies only "if the bench
  lot's catch frame cannot carry it". If the catch frame can, the line is $0.

### 3.4 The optional lines (Table 2, blocks `head_x` and `audio`)

These are optional and outside every main total. The two servos and their brackets are modelled: the servo has a catalog
id, and the head has its own committed record. The yaw stage's bearing and the audio are not modelled: neither has a
catalog entry or a place in any record.

- **The expressive head's two extra servos and their brackets:** $65 / $75 / $92, the difference between the head's own
  committed record (`round2/refine/head_x_75/candidates/75-M-H-10.json`: $13,641 / $18,596 / $25,750) and the baseline.
  The cost model reproduces that record exactly too. The servos are two `feetech_sts3215_c018` at $22 each (c): one
  turns the neck (the neck-yaw servo) and one opens the jaw.
- **With the jaw deferred** (P2; P29 keeps the jaw in simulation only), the build's head adds one servo, $22 (c), and
  one bracket: $32 / $38 / $46 with contingency. This is an estimate made by taking the jaw servo and its 0.020 kg
  bracket (a) out of the head's record; no record of that body exists.
- **The yaw stage's bearing:** $20 / $165 / $300, this document's estimate. The low case is a guess with no source (two
  thin-section ball bearings in a machined seat). The nominal and high cases are the cheapest and the dearest listing
  found for a 20 mm-bore polymer slewing ring. `stage2/x/X_COSTS.md` section 5 records that the bearing's mass, cost and
  friction are not modelled.
- **Audio** (an amplifier, a speaker, microphones): $25 / $70 / $150, this document's estimate; Table 2 gives how each
  case was set.

With every optional line, the builds are $14,340 / $20,385 / $28,794 (Pi 5 set) and $14,455 / $20,502 / $28,912 to
$14,497 / $20,577 / $28,988 (Jetson, microcontroller loop). With the jaw deferred, the head with its bearing is
$53 / $215 / $371 instead of $86 / $252 / $417.

### 3.5 Turnkey and self-assembled (Table 4)

- The cost model has one total. Its `integration` line is a subcontractor's fit-up, $1,611 / $2,625 / $3,818 for this
  body. That makes the model's total the turnkey figure: it holds the outsourced fabrication, the fit-up, the bench
  equipment, the spares and the contingency.
- The self-assembled totals are about $1,696 / $2,814 / $4,138 lower: the same builds without that line and without the
  contingency on it. The frame and the coverings are still made outside, and the owner's own time is not priced.

### 3.6 Why these totals differ from the review's

`JETSON_PLEM_REVIEW.md` section 6 gave "about $19.6k nominal turnkey before the Jetson" and "about $20.0-20.2k with it".
Those figures took the added lines from PR #668 (unverified): $60 for PA11, $300 for the DC input, $600 for the boom and
$65 for audio. This record gives $20,057 before the Jetson and $20,174 to $20,249 with it. The difference before the
Jetson, about $440, is, with contingency:

- +$322: the boom at the midpoint of `MATERIALS.md`'s range, $900, where PR #668 (unverified) took $600;
- +$214: the tread kit and floor test, which PR #668 did not list;
- −$29: the PA11 coverings, re-priced through the cost model at +$33 where PR #668 (unverified) took +$60;
- −$70: the audio, optional here and inside PR #668's total.

## 4. Where the prices are least certain

Table 5 ranks every line of the adopted build by the gap between its low and high cases, and lists the lines only a
Jetson build adds. No line has a quote. The widest, in order:

1. **The frame** ($2,948 to $6,550): the widest line, and the largest line at nominal. It is the 60 cm lot multiplied by
   (structure mass / 1.79 kg) to the power 0.6, a factor of 1.64 here.
2. **The coverings** ($1,122 to $3,367 as PA11): the 60 cm lot multiplied by (cover volume / 480 cm³) to the power 0.85,
   a factor of 3.15 before the material and wall change. This is the largest scaling in the record.
3. **The integration** ($1,611 to $3,818): the 60 cm lot multiplied by (hip height / 0.39 m) to the power 0.7, a factor
   of 1.19.
4. **The boom** ($300 to $1,500): `MATERIALS.md`'s range. The line is $0 if the bench frame can carry the cable.
5. **The bench lot** ($550 to $1,300) and **the power bay** ($275 to $650): 60 cm lots carried over unscaled. The power
   lot's price names no regulator and does not say whether a regeneration clamp is inside. A price that named them would
   settle whether the computer's regulator and the DC input's clamp are extras.
6. **The 12 drives, landed** ($3,546 at catalog prices, c): the largest catalog-priced line, and the second largest line
   at nominal after the frame. Its range covers only a 5 % allowance. A landed price would name shipping, duty and lead
   time, and which X6 variant ships at the listed $428: the catalog notes that the US store's page describes a different
   variant from the datasheet.

The three widest lots hold $8,054 of the $12,516 between the low and high cases of all the lines. The computer's own
line holds $140, and the choice of computer moves the nominal total by $117 to $192.

**Which lines are scalings of the 60 cm estimate (A8).** A8 names the frame (by structure mass to the power 0.6), the
coverings (by cover volume to the power 0.85), the integration (by size to the power 0.7) and the contingency fractions.
The code also scales the harness (by size) and the foot sensing (by the square root of size). The bench, power and
control lots are the 60 cm values unchanged. Together the scaled lots are $9,565 of the $18,710 nominal subtotal of the
adopted build with P21's items, and the unscaled 60 cm lots another $1,450.

## 5. PR #668's price slips

Table 6 lists them: the $249 Jetson price and the two catalog ids that do not exist. This record copies none of PR
#668's totals. One more difference, not a slip: PR #668 put the audio inside its main total, and here it is optional.

One coincidence, so that it is not mistaken for a copy: the Pi 5 build's high total here, $28,215, happens to equal PR
#668's (unverified) turnkey high for its Jetson build. The lines behind the two differ, and nothing was taken from it.
Table 2's `p21` block shows the lines this total is made of.

## 6. What this does not show, and what could not be established

- **No quote, no measurement.** Every total is the cost model's.
- **The power lot's contents.** That it holds a 12 V regulator for the computer is likely, not established, and whether
  it holds a regeneration clamp is not known (sections 3.2 and 3.3). The lot's breakdown, in PR #662's procurement
  model, is not in this repository.
- **The Pi 5 set's own regulator.** It needs a 5 V, 5 A regulator from the pack. The catalog has no entry for one and no
  line here prices it.
- **The control lot's contents.** It has no breakdown here, so what the direct option could drop from it is not priced.
- **The Jetson body.** No refine case with the dev kit in the chest exists yet. If its packaging changes the frame, the
  coverings or the harness, those lots change. Cooling for a 15-25 W computer (c, the catalog's entry) in a closed chest
  has no line beyond the model's fan-duct allowance.
- **The pack.** Every Jetson build here keeps the record's 6S pack. Whether the Jetson's higher draw calls for a larger
  one is not examined here; a larger pack would change the battery and spares lines.
- **The leg bench test.** P21 allows the 2.54 mm plates "where a leg bench test allows". That test's parts and time have
  no line here.
- **How many CAN adapters the direct option needs**, and whether a two-channel adapter would replace two single-channel
  ones. The catalog has no two-channel entry.
- **Firmware and engineering time.** The cost model excludes them. `COMPUTE.md` notes that the microcontroller's
  firmware is unwritten for every option.
- **Spares for the computer, the camera and the servos.** The spares line holds two leg drives and a pack only.
- **Today's prices.** The catalog prices were read on 7 Nov 2026. For this document only the summaries of web searches
  were read, not the pages themselves. No (c) figure rests on them; they inform the (d) estimates listed below.

**Web search summaries read on 14 Nov 2026** (generic part-number queries; the pages themselves were not opened):

| Page | What the summary said | Used for |
|---|---|---|
| https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/ | The Jetson Orin Nano Super dev kit went from $249 to $399 in July 2026 | Agrees with the catalog's $399; not used as a source |
| https://pjrc.com/store/teensy41_pins.html | Teensy 4.1 at $33.60, or $35.50 with pins and the Ethernet chip, at its maker's store (other sellers $27 to $51) | The supervisor's price class |
| https://gridconnect.com/products/can-usb-fd-adapter-pcan-usb-fd | PCAN-USB FD at $347 (another distributor: $369) | Agrees with the catalog's $368 within 6 %; not used as a source |
| https://gridconnect.com/products/can-usb-pro-fd-adapter-2-channels-can-lin | The two-channel PCAN-USB Pro FD at $675-708 | Section 6's open question only |
| https://robotshop.com/products/robotis-u2d2-usb-to-serial-adapter-for-dynamixel | ROBOTIS U2D2 servo-bus adapter at $43.57 | The servo-bus adapters' estimate |
| https://www.pololu.com/product/4091 | Pololu D36V50F5, a 5 V, 5.5 A step-down regulator, at $39.95 | Section 3.2: the Pi 5 set's own regulator, which has no line |
| https://www.zoro.com/igus-slewing-ring-bearing-rpm-250-prt-02-20-al/i/G9002533/ | A 20 mm-bore polymer slewing ring at $163.99; three other sellers at $229.49, $251.83 and $300.64 | The yaw bearing's nominal and high cases |
| https://shop.smooth-on.com/reoflex-60 | A trial unit of a 60 Shore A urethane rubber at $24.83; a pint of another at about $43 from a marketplace seller | The tread kit's price class |
| https://www.adafruit.com/products/3006 | A 3 W amplifier board at $5.95 | The audio's low case |
| https://robotshop.com/products/seeedstudio-respeaker-mic-array-v20-far-field-4-pdm-microphones | A far-field USB microphone array at $66.81; other US listings from $59.93 to $78.06 | The audio's high case |

## 7. Draft decision-log rows

Drafts for the owner. Nothing here is adopted, and none of these rows is in `DECISION_LOG.md`. Each records prices or
the assumptions under them; none asks for a decision. P31 is in the log already and is not a draft.

| Draft | Kind | Statement | Evidence | Proposed status |
|---|---|---|---|---|
| Price finding | F | The 75 cm build's prices, by `designlab/cost.py`: the round-two record of `75-M-H-10` reproduces exactly ($13,576 / $18,521 / $25,658 turnkey). With P21's items not in that record (PA11 coverings, tread kit and floor test, external DC input, cable boom) it is $14,228 / $20,057 / $28,215 with the Pi 5 set. The Jetson dev kit at the catalog's $399, with the loop on the microcontroller, adds $115 / $117 / $118, or $157 / $192 / $194 if a 12 V regulator is bought on top of the power lot. Driving the legs directly adds $435 / $853 / $895 more if the kept Teensy stack is the supervisor, or $477 / $928 / $1,025 with a separate supervisor board. All are cost-model estimates. | `stage2/jetson75/BUDGET_75.md`, `budget_75.json` | open |
| Lines outside the cost model | A | Lines the cost model does not hold, as estimates: tread kit and floor test $100 / $200 / $400; external DC input $200 / $300 / $400 and cable boom $300 / $900 / $1,500 (`MATERIALS.md` ranges, midpoints as nominal); for a Jetson driving the legs directly, servo-bus adapters $45 / $60 / $90 and, if the kept Teensy stack is not the supervisor, a supervisor board $40 / $70 / $120; optional: yaw-stage bearing $20 / $165 / $300, audio $25 / $70 / $150. | `budget_75.py` (`EXTRA`) | open; replace with quotes |
| A8, made explicit | A (update) | Besides the lots A8 names, `cost.py` scales the harness by size and the foot sensing by the square root of size, and carries the bench, power and control lots over from 60 cm unscaled. The power lot has no stated scope: it probably holds a 12 V regulator for the computer and may hold a regeneration clamp. The compute line's allowance stands for storage, which the Pi 5 set's catalog price already includes ($40). | `designlab/cost.py`; `BUDGET_75.md` sections 3.2, 3.3 and 4 | open; replace with quotes from drawings |
