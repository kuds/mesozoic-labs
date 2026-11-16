**How big a circle does it take to turn?** One body length is 1.76 m, snout tip to tail tip. All the numbers
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
| Quick turn, 45° per second (in the bar) | 0.51 m | 2.22 m | 1.12 to 1.19 m | 0.80 to 0.81 m | 2.1 to 2.2 s | 0.61 m along, 0.54 m sideways |
| 0.5 rad/s, 29° per second (reported) | 0.80 m | 2.61 m | 1.31 to 1.36 m | 0.80 m | 3.2 s | 0.84 to 0.86 m along, 0.82 to 0.85 m sideways |
| 0.35 rad/s, 20° per second (reported) | 1.14 m | 3.13 m | 1.57 to 1.61 m | 0.80 to 0.81 m | 4.5 s | 1.17 to 1.22 m along, 1.16 m sideways |

- **The tail sets the circle.** In the quick turn the tail tip is 1.11 m from the turn's centre and the head
  less; the centre itself only moves on a 0.51 m arc.
- **At the moment of the command the tail still points straight back**, so for an instant the farthest point is
  1.19 m from the centre the turn will have on today's walk (1.12 m on the candidate). A circle
  that holds the robot from the command on is 2.4 m across (2.2 m on the candidate).
- **The corridor** is the narrowest L-shaped passage of one width that holds every point of the robot's surface from
  the command to the end of the move.
- **A turn at 1 rad/s (57° per second) has no reference** on either walk: the built motion is over a leg drive's
  limit or outside the thigh twist's range.

**Simulated** (quick turn, both sides, the two controller fits, grippy floor; medians per record, over the episodes
that stayed up):

| Walk | Servo delay | Stayed up, of 20 | Turning circle | Corridor | Turned at the judged instant, 2 s after the command |
|---|---|---|---|---|---|
| Today's walk | 0 ms | 20 | 2.23 to 2.24 m | 0.81 to 0.82 m | 82 to 83° |
| Today's walk | 4 ms | 20 | 2.23 m | 0.81 to 0.82 m | 84 to 85° |
| Today's walk | 6 ms | 20 | 2.23 m | 0.81 to 0.82 m | 83 to 85° |
| The candidate | 0 ms | 20 | 2.23 m | 0.81 m | 84 to 85° |
| The candidate | 4 ms | 19 to 20 | 2.23 m | 0.81 m | 83 to 84° |
| The candidate | 6 ms | 1 | 2.23 to 2.25 m | 0.81 to 0.83 m | 77 to 91° |

- **Planned, the trunk has turned 85° at that instant** and reaches 90° after
  2.1 to 2.2 s. Read the last column against that, not against 90°.
- **Where the turn is reached it takes the planned room**, within a few centimetres.
- **The turn does not pass the bar at any delay** (section 7): the room it takes is right, but it fails other limits.
- **At 10 ms no quick turn is completed on the grippy floor**: 78 of 80 runs of today's walk fall before the
  command and 2 during the turn; the candidate falls before the command in every run.

**Clips.** 18 short clips were rendered from the committed records and the frozen references
(`stage2/runs/2026-11-15-75cm/views/clips.json` lists each with its record, seed and checksum; the media are not in
git and are made again with `python -m designlab.stage2.barviews clips ...`). Every frame is labelled as simulation
or as planned motion. **Which records have a clip** was the lead's choice, made from the tables: today's walk under
the first fit of the controller on the grippy floor, at the lowest delay run (walk, quick turn, a light push to its
left, stop) and at the judged 10 ms (walk); the candidate's walk at 10 ms and its quick turn with no delay; and the planned
quick turn of both. **Which run of a record is shown** follows a rule written before any record existed: the first
seed, and if its outcome is not the record's majority outcome, also the lowest seed with the majority outcome.

| Walk | What | Kind | View | Run shown | Length |
|---|---|---|---|---|---|
| Today's walk | nudge left (0.25 of the push bound) while walking at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | top | seed 3042 (first seed): stayed up | 5 s |
| Today's walk | nudge left (0.25 of the push bound) while walking at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | visitor | seed 3042 (first seed): stayed up | 5 s |
| Today's walk | stop from 0.40 m/s within two steps | simulated, controller fit 42, 0 ms of delay | side | seed 3042 (first seed): stayed up | 5 s |
| Today's walk | stop from 0.40 m/s within two steps | simulated, controller fit 42, 0 ms of delay | visitor | seed 3042 (first seed): stayed up | 5 s |
| Today's walk | quick 90° turn to the left at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | top | seed 3042 (first seed): stayed up | 6 s |
| Today's walk | quick 90° turn to the left at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | visitor | seed 3042 (first seed): stayed up | 6 s |
| Today's walk | walk at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | side | seed 3042 (first seed): stayed up | 10 s |
| Today's walk | walk at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | visitor | seed 3042 (first seed): stayed up | 10 s |
| Today's walk | walk at 0.40 m/s | simulated, controller fit 42, 10 ms of delay | side | seed 3042 (first seed): fell | 9 s |
| Today's walk | walk at 0.40 m/s | simulated, controller fit 42, 10 ms of delay | visitor | seed 3042 (first seed): fell | 9 s |
| The candidate | quick 90° turn to the left at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | top | seed 3042 (first seed): stayed up | 7 s |
| The candidate | quick 90° turn to the left at 0.40 m/s | simulated, controller fit 42, 0 ms of delay | visitor | seed 3042 (first seed): stayed up | 7 s |
| The candidate | walk at 0.40 m/s | simulated, controller fit 42, 10 ms of delay | side | seed 3042 (first seed): fell | 6 s |
| The candidate | walk at 0.40 m/s | simulated, controller fit 42, 10 ms of delay | visitor | seed 3042 (first seed): fell | 6 s |
| Today's walk | quick 90° turn to the left at 0.40 m/s | planned motion, no physics | top | - | 6 s |
| Today's walk | quick 90° turn to the left at 0.40 m/s | planned motion, no physics | visitor | - | 6 s |
| The candidate | quick 90° turn to the left at 0.40 m/s | planned motion, no physics | top | - | 7 s |
| The candidate | quick 90° turn to the left at 0.40 m/s | planned motion, no physics | visitor | - | 7 s |

A clip of a run that fell before its move says so on every frame ("THE WALK AT 0.40 m/s NEVER BEGAN").
