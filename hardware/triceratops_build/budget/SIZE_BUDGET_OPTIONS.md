# Size, speed and budget options for a $10k-$20k Triceratops

18 Nov 2026, for the owner's question: "30k is too much. What size could I build for 10k to 20k? What is the
performance difference, and is there a sweet spot between size and speed? Think about expressiveness, like moving its
head and mouth." Revised the same day after an independent check. **Rough planning estimates**: nothing is quoted,
bought or measured; masses, loads and speeds are scaled from the checked parts list
(`hardware/triceratops_kickoff/PARTS_PREMIUM.md`), not simulated. Numbers come from [size_budget.py](size_budget.py)
([output](size_budget_out.txt)), where every rule and price source is stated.

**Cost bases.** "Hardware" is the parts at list price. "T. rex way" adds 7.2 % for freight, duty and tax, the way the
T. rex's $17.2k (self-built) and $20.0k (turnkey) and the earlier $31.8k Triceratops figure were counted. "Landed" is the parts list's fuller rule:
freight 5 %, duty 25 % on the drives (an assumption), sales tax 7 %, then 7 % contingency.

## Decision (owner, 18 Nov 2026)

**Option C, 60 cm with the T. rex's drive set; the $20k ceiling counted landed, all-in; the wireless stop bought
before any public demo; the full expression package.** On that basis the build is about $24.2k-$25.0k landed with
contingency at the planning duty of 25 %.

**Update, same day: the handheld stop (D15).** The owner judged the certified Cattron wireless stop overkill. The
build now plans a fail-safe radio stop (about $250, an estimate) into the existing watchdog and Pilz relay: a press or
a lost signal makes the robot sit, then cut power. It is not certified; a certified wireless stop is bought only if a
host's risk assessment asks for one (+$2.45k of hardware net). **The chosen build is then $16.8k-$17.5k of
hardware; $20.1k-$20.9k landed from Fremont stock; about $17.9k-$18.8k landed with half the bench gear counted against
the T. rex and the spares bought later** (`size_budget_out.txt`, "Chosen build"). The figures below this paragraph
keep the certified stop, as the options were compared. The first demos are controlled, with no direct contact and a
safe distance kept from people (P8); barriers are assumed to be the host's (about $300 if the team brings its own,
an estimate; A6).

**Update, same day (the owner's duty information and the store pages):**
with all drives shipped from MyActuator's Fremont stock (no duty) it is $23.0k-$23.9k landed; dispatched from the
factory in China (about 41.5 % tariffs plus brokerage) $25.1k-$25.9k. With Fremont stock, the receiver included in
the stop kit, half the bench gear counted against the T. rex and the spares bought later, about $19.9k-$20.7k; a 20 %
educational / research discount on the drives, if the project qualifies, would take about $1.0k more off. Recorded in `../DECISION_LOG.md` (D1, D2, D12-D14).

## Answer

**The sweet spot is a 60 cm robot (to the top of the frill, 1.2 m long, about 12 kg) with the same drive types as the
T. rex (10 × X4 + 2 × X6 on a 48 V bus). Ready for public demos (moving head, jaw and tail, an industrial wired stop
plus a handler-held wireless stop, cooling, test gear) it is about $19.0k-$19.7k of hardware: $20.4k-$21.1k counted
the T. rex way, $23.9k-$24.7k landed with contingency; about $0.2k more for the full expression package.** It fits a $20k budget on the T. rex basis, or landed only if
some prices come in lower (see "Getting a demo-ready 60 cm robot to $20k" below).

- **$10k** does not buy a demo robot with industrial safety, painted shells and an expressive head; it buys the
  repository's $10k research-quadruped plan (owner-built frame, foam skin, passive tail; main's
  `docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md`, section 4).
- **No size is demo-ready under about $17.6k counted the T. rex way** (50 cm), because drives, computer, safety
  chain, power, cooling, bench gear and tools hardly shrink with size: about $9k-$10k of hardware with the wired stop,
  $12k-$12.5k demo-ready.
- **50 cm** saves about $1.0k-$1.4k over the budget 60 cm option and $2.2k-$2.6k over the recommended one, but walks slower
  (0.17-0.23 m/s) and is a smaller animal.
- **75 cm**, even lean and with mid-range choices, is about $22.9k of hardware demo-ready ($24.5k the T. rex way,
  $28.6k landed): over the budget. About $1.0k of its gap to 60 cm is the larger servos the parts list chose for the 75 cm head and tail.

## The options

| | A: 50 cm | B: 60 cm, budget | **C: 60 cm, recommended** | D: 75 cm, lean |
|---|---|---|---|---|
| Frill crown / length / hips | 0.50 / 1.00 / 0.27 m | 0.60 / 1.20 / 0.32 m | 0.60 / 1.20 / 0.32 m | 0.75 / 1.50 / 0.40 m |
| Mass (estimate) | 9.0-9.4 kg | 10.2-10.6 kg | 11.8-12.2 kg | 15 kg (parts list T15) |
| Leg drives | 12 × X4 | 12 × X4 | 10 × X4 + 2 × X6 (front elbows) | 8 × X4 + 4 × X6 at 1.3:1 |
| Battery bus | 6S (24 V) | 6S (24 V) | 12S (48 V) | 12S (48 V) |
| Walk | 0.17-0.23 m/s | 0.20-0.27 m/s | 0.25-0.27 m/s | 0.30 m/s |
| Short trot (Froude-similar to 0.60 m/s) | passes | elbow heat 0.91 | passes (knee heat 0.86) | roll heat 1.01: short bouts only |
| Run time per battery set | 28-39 min | 25-34 min | 43-60 min | 33-47 min |
| Tightest walking axis | pitch speed 1.08 at the Froude-similar walk | elbow heat 0.99 in a held "paw" pose; 1.02 if the frame is heavier | hind knee heat 0.93 in a held pose (0.96 if heavier) | roll peak 0.70, heat 0.87 |
| **Demo-ready, hardware** | $16.4k-$17.5k | $17.8k-$18.5k | **$19.0k-$19.7k** | $22.9k |
| Same, T. rex way | $17.6k-$18.7k | $19.1k-$19.8k | **$20.4k-$21.1k** | $24.5k |
| Same, landed with contingency | $20.6k-$21.9k | $22.3k-$23.1k | **$23.9k-$24.7k** | $28.6k |
| Without the wireless stop (wired stop only), landed | $17.4k-$18.6k | $19.0k-$19.9k | $20.7k-$21.5k | $25.4k |
| Full expression instead of core | +$0.2k | +$0.2k | +$0.2k | +$0.5k |

- Ranges: the low figure scales the outsourced frame's cost with its area; the high one assumes half of it is fixed
  (part count and set-ups do not shrink, and the drive mounts fit drives that do not shrink). The same pair scales the
  frame's mass with volume or area.
- Walk ranges: from PR #669's own swing profile with a 15 % drive-speed margin (low) to a smooth minimum-jerk swing at
  the parts list's limit (high); the Froude-similar pace is 0.24 / 0.27 / 0.27 / 0.30 m/s. PR #669's own target for
  its 60 cm companion was 0.218 m/s.
- "Tightest axis" uses the parts list's rules (≤ 1.00 passes): peak ≤ 0.70 of the maker's peak, continuous ≤ the
  thermal-equilibrium torque (open-air figures; inside a shell they are lower, hence the cooling in every option),
  speed ≤ 0.70 of no-load (or the X4's 83 rpm cap on 12S).

**What "demo-ready" holds:** the leg drives and a spare of each type; pushrod linkages; Jetson computer with a
real-time CAN board; IMU, depth camera and ranging sensors; an industrial wired stop chain (Pilz relay, two-channel
e-stops, contactor, watchdog) **plus** a PL d wireless stop with its receiver (+$2.7k: $1,900 found for the kit plus an
estimated $800 receiver; needs a quote); battery sets, tether, charger; drive-bay cooling; harness; outsourced CNC
frame; MJF shells painted by the team; bench supply, torque rig, CAN adapter, thermal camera and overhead catch gantry
(the gear the four-leg walking test needs); tools; the "core" expression package. Not included: foot load cells
(+$254), a premium IMU (+$392), a dim-hall camera (+$260), a professional painter (+$1,050), a road case (+$800),
labour, and PLEM.

## Getting a demo-ready 60 cm robot to $20k

From option C's $23.9k-$24.7k landed:
- **Duty:** items in stock at MyActuator's US store (Fremont, California) have already cleared customs, so they carry
  no duty (owner's information, 18 Nov 2026); all three drive pages showed "In Stock" that day. About -$1.1k against
  the 25 % planning assumption. Units dispatched from the factory in China instead pay about 41.5 % plus brokerage
  (+$0.9k against it). Confirm Fremont dispatch, or delivered-duty-paid terms, before ordering.
- **Wireless stop:** if the kit includes its receiver, about -$1.0k landed; other makers' quotes may be lower.
- **Bench gear:** the T. rex will reuse the $2.2k of bench gear (supply, torque rig, CAN adapter, camera, gantry); counting half of
  it against the T. rex is about -$1.3k landed. A team-built catch gantry saves more.
- **Spares:** buying the two spare drives after the first walk, about -$1.0k landed.

Taken together (about -$4.3k: the duty and spares savings overlap by $0.2k), C lands near $19.6k-$20.5k; with the
full expression package, about $0.3k more. Option B is $1.2k cheaper in hardware ($1.6k landed) but
has no heat margin at the elbows.

**Price risk.** All drive prices are MyActuator's own US store (read 7 Nov 2026; re-read 18 Nov 2026: X4-36-L $269,
X6 $428, X8-150 $725, each "In Stock", "Maximum 50 units per product"; the X6 page lists 0.78 kg and 50 N·m, the
X6-50-L's figures, which still pass option C's elbows at 0.53 of the peak rule; the pages also advertise 20 % off for
educational and research institutions). At US reseller prices found on
18 Nov 2026, option C's drives with spares rise from $4.2k to $5.9k (backordered listings) or $9.0k (RobotShop,
tariffs included; it showed 2 X6 left, and its X4 listing is the X4-P36-36-E). Ask the maker before budgeting.

## How size changes performance

- **Speed.** Animals and robots of different sizes look equally brisk at the same Froude number (speed² ÷ (g × leg
  length)): 0.30 m/s at 75 cm looks like 0.27 m/s at 60 cm and 0.24 m/s at 50 cm. Smaller robots need faster joints for
  it, and the X4 drive's speed limit bites first on a 24 V bus; option C's 48 V bus lets the 60 cm robot reach its
  Froude-similar pace.
- **Mass.** About 12 kg against 15-20 kg: less energy in a fall or a bump, and one person can lift it. Foot and pinch
  forces do not fall with size, because the same drives sit on shorter links (an X4 pitch drive gives about 160 N at a
  60 cm robot's knee against 128 N at 75 cm); they are set by torque limits in software.
- **Presence.** 60 cm to the frill crown and 1.2 m long is about a large dog's size; 50 cm is closer to a medium dog;
  75 cm is about a Great Dane's shoulder height.
- **Run time.** One 6S pack (A, B) gives 25-39 minutes; two in series (C, D) 33-60 minutes. Long sessions run on the
  tether in every option.

## Expressiveness: what moves, and what it costs

At 60 cm the head weighs about 0.6 kg. Holding it level takes about 0.17-0.57 N·m with no spring (the lower figure
from the 75 cm model's head scaled down, the higher one from the parts list's conservative lever) and about a third
of that with the 70 % counterbalance spring. That is within ROBOTIS's mid-range XC430-W240 (1.9 N·m stall at 12 V; about
0.63 N·m taken as its continuous limit, an estimate). Prices from web-search summaries, 18 Nov 2026: XC430-W240 about
$100-197 (typically $120-138), XL430-W250 about $45-70.

| Package | What moves | Servos | At 50-60 cm | At 75 cm |
|---|---|---|---|---|
| **Core** | Neck nod (spring-assisted), neck turn to look at onlookers, jaw/beak open and close, tail sway; eye LEDs; speaker, amplifier and microphone | 3 × XC430-W240 + 1 × XL430-W250 (75 cm: XM540 + 3 × XM430) | about $750 | about $1,800 |
| **Full** | Core plus blinking eyelids, tail lift, frill blush lighting | + 2 × XL330 + 1 × XL430 (75 cm: + XM430) | about $980 | about $2,300 |

- **Most character per dollar:** the neck turning to follow an onlooker (the camera and computer are already there), the
  jaw opening with a sound, and blinking.
- **Free expression:** the legs can act too: lowering the head, shifting weight, a paw scrape, sitting down. That costs
  software time, not parts.
- **One wiring caveat:** the XC430 and XL430 are TTL-bus parts; the parts list chose RS-485 servos for the long cable
  through the neck. Keep the bus adapter in the head or add an RS-485 converter (a few dollars to tens of dollars).

## What choosing 60 cm would change in the project

- **New model and simulation:** the current models are 75 cm. A 60 cm model needs PR #669's 60 cm geometry (its PDF
  and OpenSCAD file), the new parts' masses and positions, then the standing check and the crawl again: a few days of
  session work. Simulation then confirms or corrects this document's scaled loads.
- **Renders:** the simulation stills show the 75 cm robot; they would be redone at 60 cm before 8 Dec 2026.
- **The parts list:** a new checked parts list for the chosen option replaces the T20 / T15 lists.
- **The demo walk speed:** the decision log's 0.30 m/s (P1, P6) becomes about 0.25-0.27 m/s. At 0.30 m/s option
  C's pitch drive would use 0.90 of the X4's 83 rpm cap with a smooth swing and 1.04 with PR #669's swing profile.
- **The foot finding stays:** a sole rigid in roll tips onto its edges in a statically stable crawl, at any size.
- **The T. rex:** option C uses the same drive types and counts as the T. rex's set (10 × X4 + 2 × X6), so the foreleg
  bench, spare types, firmware and test gear serve both; the T. rex still needs its own set. Putting the T. rex on the
  same 48 V bus is the parts list's suggestion, pending the T. rex's re-check against the X4's 83 rpm cap.

## What this does not cover

No simulation at 50 or 60 cm yet; frame and shell costs are scaled estimates; no quotes; labour, turnkey integration
and PLEM are not included; the mid-range ROBOTIS prices and stall torques are web-search summaries, and their
continuous limits are estimates.
