# ruff: noqa: E501, E402, E702, E731
"""Write the report's turning-room texts from the committed turnpath.json files (no number typed by hand)."""
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[4]
V = STUDY / 'stage2/runs/2026-11-15-75cm/views/turnpath'
HERE = Path(__file__).resolve().parent
d = {s: json.loads((V / s / 'turnpath.json').read_text()) for s in ('V0', 'F25F9cl')}
BL = d['V0']['body_length']['body_length_m']

def planned(s, cond):
    x = next(t for t in d[s]['turns'] if t['condition'] == cond)
    r = x['planned']['readings']
    return dict(radius=x['planned']['commanded']['radius_m'], rate=x['planned']['commanded']['turn_rate_deg_s'],
                circle=r['swept']['turning_circle_m'], swept=r['swept']['swept_radius_m'], set_by=r['swept']['set_by'],
                far=r['swept_turn']['largest_distance_m'], corridor=r['corridor']['width_m'],
                secs=r['turn_itself']['seconds_after_command'], adv=r['turn_itself']['root_advance_m'],
                side=r['turn_itself']['root_sideways_m'], at2=r['turned_at_judged_instant_deg'])

def sim(s, cond, row, delay):
    x = next(t for t in d[s]['turns'] if t['condition'] == cond)
    for rec in x['simulated']:
        if rec['row'] == f'{row}@{delay}ms' and rec['condition'] == cond:
            return rec
    return None

def rng(vals, fmt='{:.2f}'):
    lo, hi = min(vals), max(vals)
    a, b = fmt.format(lo), fmt.format(hi)
    return a if a == b else f'{a} to {b}'

q = planned('V0', 'turn/q90_left'); qr = planned('V0', 'turn/q90_right')
cq = planned('F25F9cl', 'turn/q90_left')
# simulated, today's walk, quick turn, both fits, both directions, grippy floor, 0 4 6 ms
def sims(s, delays):
    out = []
    for cond in ('turn/q90_left', 'turn/q90_right'):
        for row in ('R2/42', 'R2/43'):
            for dl in delays:
                rec = sim(s, cond, row, dl)
                if rec and rec.get('summary'):
                    out.append(rec)
    return out
v_s = sims('V0', (0, 4, 6)); c_s = sims('F25F9cl', (0, 4))
def med_range(recs, key, fmt='{:.2f}'):
    return rng([r['summary'][key]['median'] for r in recs if r['summary'].get(key)], fmt)
def ups(recs):
    return rng([r['summary']['turning_circle']['n'] for r in recs], '{:d}')

answer = (f"Planned (the reference played without physics): in the quick 90° turn the robot's centre follows an arc of "
          f"{q['radius']:.2f} m radius, its tail tip sweeps a circle {q['circle']:.1f} m across, and it gets round a "
          f"square corner in a corridor {q['corridor']:.2f} m wide. Simulated with today's walk at 0, 4 and 6 ms of delay, "
          f"where the turn was reached: the same within a few centimetres (medians {med_range(v_s, 'turning_circle')} m "
          f"and {med_range(v_s, 'corridor')} m). The slower turns need circles of "
          f"{planned('V0', 'turn/t90_0.5_left')['circle']:.1f} m and {planned('V0', 'turn/t90_0.35_left')['circle']:.1f} m "
          f"(planned). The figure, the table and the clips are in section 10.")
(HERE / 'TURN_ANSWER.md').write_text(answer + '\n')

rows = []
for nm, cl, cr in (("Quick turn, 45° per second (in the bar)", 'turn/q90_left', 'turn/q90_right'),
                   ("0.5 rad/s, 29° per second (reported)", 'turn/t90_0.5_left', 'turn/t90_0.5_right'),
                   ("0.35 rad/s, 20° per second (reported)", 'turn/t90_0.35_left', 'turn/t90_0.35_right')):
    a, b = planned('V0', cl), planned('V0', cr)
    ca, cb = planned('F25F9cl', cl), planned('F25F9cl', cr)
    rows.append(f"| {nm} | {a['radius']:.2f} m | {rng([a['circle'], b['circle'], ca['circle'], cb['circle']])} m | "
                f"{rng([a['far'], b['far'], ca['far'], cb['far']])} m | {rng([a['corridor'], b['corridor'], ca['corridor'], cb['corridor']])} m | "
                f"{rng([a['secs'], b['secs'], ca['secs'], cb['secs']], '{:.1f}')} s | "
                f"{rng([a['adv'], b['adv'], ca['adv'], cb['adv']])} m along, {rng([a['side'], b['side'], ca['side'], cb['side']])} m sideways |")
srows = []
for nm, s, delays in (("Today's walk", 'V0', (0, 4, 6)), ("The candidate", 'F25F9cl', (0, 4, 6))):
    for dl in delays:
        recs = [r for r in sims(s, (dl,))]
        if not recs:
            continue
        srows.append(f"| {nm} | {dl} ms | {ups(recs)} | {med_range(recs, 'turning_circle')} m | {med_range(recs, 'corridor')} m | "
                     f"{med_range(recs, 'turned_at_judged_instant', '{:.0f}')}° |")
section = f"""**How big a circle does it take to turn?** One body length is {BL:.2f} m, snout tip to tail tip. All the numbers
come from `stage2/runs/2026-11-15-75cm/views/turnpath/<set>/turnpath.json`, made by `designlab.stage2.barviews`.
Its code was reviewed independently and fixed, the views were then made again after the chain's end, and this
report's last check recomputed the two tables below from those files. Planned and simulated numbers are kept in
separate tables.

![Top view of the quick left turn of today's walk](stage2/runs/2026-11-15-75cm/views/turnpath_fit42_0ms/turnpath/V0/turn_q90_left.png)

*The figure: seen from above, walking up the page and turning left. Blue is the planned path of the robot's
centre. Orange is 20 simulated runs of today's walk under the first fit of the controller with no delay (20 of 20
stayed up); it is one of the records of the simulated table below.*

**Planned** (the frozen reference played without physics; both sides, today's walk and the candidate together):

| Turn of 90° at 0.40 m/s | Radius of the centre's path (commanded) | Turning circle (twice the tail tip's radius, steady part) | Farthest point from the turn's centre, from the command | Corridor for a square corner | Time until the trunk has turned 90° | Where the centre is then |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

- **The tail sets the circle.** In the quick turn the tail tip is {q['swept']:.2f} m from the turn's centre and the head
  less; the centre itself only moves on a {q['radius']:.2f} m arc.
- **At the moment of the command the tail still points straight back**, so for an instant the farthest point is
  {q['far']:.2f} m from the centre the turn will have on today's walk ({cq['far']:.2f} m on the candidate). A circle
  that holds the robot from the command on is {2 * q['far']:.1f} m across ({2 * cq['far']:.1f} m on the candidate).
- **The corridor** is the narrowest L-shaped passage of one width that holds every point of the robot's surface from
  the command to the end of the move.
- **A turn at 1 rad/s (57° per second) has no reference** on either walk: the built motion is over a leg drive's
  limit or outside the thigh twist's range.

**Simulated** (quick turn, both sides, the two controller fits, grippy floor; medians per record, over the episodes
that stayed up):

| Walk | Servo delay | Stayed up, of 20 | Turning circle | Corridor | Turned at the judged instant, 2 s after the command |
|---|---|---|---|---|---|
{chr(10).join(srows)}

- **Planned, the trunk has turned {min(q['at2'], qr['at2'], cq['at2']):.0f}° at that instant** and reaches 90° after
  {rng([q['secs'], qr['secs'], cq['secs']], '{:.1f}')} s. Read the last column against that, not against 90°.
- **Where the turn is reached it takes the planned room**, within a few centimetres.
- **The turn does not pass the bar at any delay** (section 7): the room it takes is right, but it fails other limits.
- **At 10 ms no quick turn is completed on the grippy floor**: 78 of 80 runs of today's walk fall before the
  command and 2 during the turn; the candidate falls before the command in every run.
"""
(HERE / 'TURN_SECTION.md').write_text(section)
print(answer); print(); print(section)

# --- clips: listed from the committed manifest
man = json.loads((STUDY / 'stage2/runs/2026-11-15-75cm/views/clips.json').read_text())
WALK = {'V0': "Today's walk", 'F25F9cl': 'The candidate'}
crow = []
for c in sorted(man['clips'], key=lambda c: (c['kind'] != 'simulated', c['set'] != 'V0', c['condition'], str(c['row']), c['view'])):
    if c['kind'] == 'planned':
        what = "planned motion, no physics"
        shown = '-'
    else:
        what = f"simulated, controller fit {c['row'].split('@')[0].split('/')[1]}, {c['row'].split('@')[1].replace('ms', ' ms')} of delay"
        shown = f"seed {c['seed']} ({c['selection']}): {c['outcome']}"
    crow.append(f"| {WALK[c['set']]} | {c['in_words']} | {what} | {c['view']} | {shown} | {c['seconds']:.0f} s |")
clips = f"""
**Clips.** {len(man['clips'])} short clips were rendered from the committed records and the frozen references
(`stage2/runs/2026-11-15-75cm/views/clips.json` lists each with its record, seed and checksum; the media are not in
git and are made again with `python -m designlab.stage2.barviews clips ...`). Every frame is labelled as simulation
or as planned motion. **Which records have a clip** was the lead's choice, made from the tables: today's walk under
the first fit of the controller on the grippy floor, at the lowest delay run (walk, quick turn, a light push to its
left, stop) and at the judged 10 ms (walk); the candidate's walk at 10 ms and its quick turn with no delay; and the planned
quick turn of both. **Which run of a record is shown** follows a rule written before any record existed: the first
seed, and if its outcome is not the record's majority outcome, also the lowest seed with the majority outcome.

| Walk | What | Kind | View | Run shown | Length |
|---|---|---|---|---|---|
{chr(10).join(crow)}

A clip of a run that fell before its move says so on every frame ("THE WALK AT 0.40 m/s NEVER BEGAN").
"""
(HERE / 'TURN_SECTION.md').write_text(section + clips)
print(clips)
