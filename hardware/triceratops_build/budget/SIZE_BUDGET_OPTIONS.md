# Size, speed and budget options for a $10k-$20k Triceratops

18 Nov 2026, for the owner's question: "30k is too much. What size could I build for 10k to 20k? What is the
performance difference, and is there a sweet spot between size and speed? Think about expressiveness, like moving its
head and mouth." **Rough planning estimates**: nothing is quoted, bought or measured; masses, loads and speeds are
scaled from the checked parts list (`hardware/triceratops_kickoff/PARTS_PREMIUM.md`), not simulated. The numbers come
from [size_budget.py](size_budget.py) ([output](size_budget_out.txt)), with every rule and price source in the script.

## Answer

**The sweet spot is a 60 cm robot (to the top of the frill, 1.2 m long, about 10-12 kg), at about $15.2k-$15.4k of
hardware ($16.3k-$16.5k counted the T. rex way, $19.2k-$19.4k landed with contingency) with a moving head, jaw and
tail.** It walks at 0.23-0.27 m/s, which is
the same pace as the 75 cm design when measured in body lengths per second (about 0.2), so it looks just as lively.
It costs about $4k less in hardware than a lean 75 cm robot ($5k landed), weighs about 12 kg against 15-20 kg, and
can use exactly the T. rex's drive set.

- **$10k** does not buy a museum robot with industrial safety, painted shells and an expressive head. It buys the
  repository's existing $10k research-quadruped plan (owner-built frame, foam skin, passive tail; main's
  `docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md`, section 4).
- **50 cm** saves only about $1.4k over the cheapest 60 cm option, because about $9k of every build does not shrink with size (drives,
  computer, safety chain, power, bench and tools), and it walks slower (about 0.19 m/s).
- **75 cm**, even lean, is about $19-20k of hardware, $24k landed, and $27k with a wireless stop: over the budget.
- **The biggest single choice inside the budget is the wireless safety stop** (+$2.7k: the Cattron PL d kit found at
  $1,900 plus an estimated $800 receiver). A wired industrial stop chain is in every option below; a handler-held
  wireless stop is strongly advised before public demos and needs a quote.

## The options (nominal prices; "landed" adds freight 5 %, duty 25 % on the drives, tax 7 % and contingency 7 %)

| | A: 50 cm | B: 60 cm, budget | **C: 60 cm, recommended** | D: 75 cm, lean |
|---|---|---|---|---|
| Frill crown / length / hips | 0.50 / 1.00 / 0.27 m | 0.60 / 1.20 / 0.32 m | 0.60 / 1.20 / 0.32 m | 0.75 / 1.50 / 0.40 m |
| Mass (estimate) | about 9 kg | about 10 kg | about 12 kg | 15 kg (parts list T15) |
| Leg drives | 12 × X4 | 12 × X4 | 10 × X4 + 2 × X6 (the T. rex's set) | 8 × X4 + 4 × X6 |
| Battery bus | 6S (24 V) | 6S (24 V) | 12S (48 V) | 12S (48 V) |
| Walk, with a 15 % drive-speed margin | 0.19 m/s | 0.23 m/s | 0.27 m/s | 0.30 m/s |
| Same, in body lengths per second | 0.19 | 0.19 | 0.22 | 0.20 |
| Tightest axis | pitch speed (needs 6S margin) | elbow heat 0.99 in a held "paw" pose | hind knee heat 0.93 in a held pose | roll peak 0.70, heat 0.87 |
| Hardware, core expression, wired stop | $12.6k | $14.0k | **$15.2k** | $19.2k |
| Same, landed with contingency | $15.9k | $17.6k | **$19.2k** | $24.0k |
| + wireless stop (hardware / landed) | $15.3k / $19.1k | $16.7k / $20.8k | $17.9k / $22.4k | $21.9k / $27.2k |
| Full expression instead of core | +$0.2k | +$0.2k | +$0.2k | +$0.5k |

"Tightest axis" uses the parts list's rules (≤ 1.00 passes): peak ≤ 0.70 of the maker's peak, continuous ≤ the
thermal-equilibrium torque, speed ≤ 0.70 of no-load (or the X4's 83 rpm cap on 12S).

**Price risk.** These use MyActuator's own US store prices (read 7 Nov 2026). At the US reseller prices found on
18 Nov 2026, option C's drives with spares rise from $4.2k to $5.9k (backordered resellers) or $9.0k (RobotShop,
in stock, tariffs included). Ask MyActuator whether its store price is landed (drafted question M4).

## How size changes performance

- **Speed.** Animals and robots of different sizes look equally brisk at the same Froude number (speed² ÷ (g × leg
  length)). At 75 cm, 0.30 m/s; the same "look" is 0.27 m/s at 60 cm and 0.24 m/s at 50 cm. Smaller robots need
  faster joints for it, and the X4 drive's speed limit bites first: on 6S it allows the 60 cm robot 0.23 m/s and the
  50 cm robot 0.19 m/s; on 12S (option C) the 60 cm robot reaches 0.27 m/s.
- **Mass and safety.** 10-12 kg against 15-20 kg: less energy in a fall or a bump with a visitor, smaller pinch forces,
  and one person can lift it.
- **Presence.** 60 cm to the frill crown and 1.2 m long is about a large dog's size; 50 cm reads closer to a medium
  dog; 75 cm is about a Great Dane's shoulder height.
- **Run time** stays about 25-35 minutes at each size (the battery scales with the robot); long sessions run on the
  tether in every option.
- **Margins.** Option C keeps every axis under its limits with the drives the T. rex needs anyway; option B saves
  $1.2k but its elbows reach their heat limit if a paw is held up for long; option D's roll drives are the tight axis
  (the parts list's T15 finding).

## Expressiveness: what moves, and what it costs

Moving the head and mouth is cheap next to the legs, and at 60 cm the head is light enough for ROBOTIS's mid-range
servos (XC430 about $100-197, typically $120-138; XL430 about $45-70; prices from web-search summaries, 18 Nov 2026).

| Package | What moves | Servos | Cost at 50-60 cm | Cost at 75 cm |
|---|---|---|---|---|
| **Core** | Neck nod (spring-assisted), neck turn to look at visitors, jaw/beak open and close, tail sway; eye LEDs; speaker, amplifier and microphone | 3 × XC430-W240 + 1 × XL430-W250 (75 cm: XM540 + 3 × XM430) | about $750 | about $1,800 |
| **Full** | Core plus blinking eyelids, tail lift, frill blush lighting | + 2 × XL330 + 1 × XL430 (75 cm: + XM430) | about $980 | about $2,300 |

- **Most character per dollar:** neck turn with camera tracking ("it looks at you"), the jaw opening with a sound, and
  blinking. The camera and computer are already in every option.
- **Free expression:** the legs can also act: lowering the head to the ground, shifting weight, a paw scrape, sitting
  down. That costs software time, not parts.
- **What it costs besides money:** head mass (the neck servo and spring are sized for it), cable loops through the
  neck, and animation work.

## What choosing 60 cm would change in the project

- **New model and simulation.** The current models are 75 cm. A 60 cm model needs PR #669's 60 cm geometry (in its
  PDF and OpenSCAD file), the new parts' masses and positions, then the standing check and crawl again: a few days of
  session work.
- **Renders.** The December stills show the 75 cm robot; they would be redone at 60 cm before 8 Dec 2026.
- **The parts list.** A new checked parts list for the chosen option replaces the T20 / T15 lists.
- **The foot finding stays:** a sole rigid in roll tips onto its edges in a statically stable crawl, at any size.
- **The T. rex.** Option C buys the T. rex's own drive set (10 × X4 + 2 × X6), so the foreleg bench, spares and
  firmware serve both robots directly; the 12S choice would also apply to the T. rex.

## What this does not cover

No simulation at 50 or 60 cm yet; frame and shell costs scaled by area are estimates; no quotes; labour, turnkey
integration and PLEM are not included; the mid-range ROBOTIS prices and torques are web-search summaries, not the
maker's pages.
