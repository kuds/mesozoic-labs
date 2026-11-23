# Reusing the walking courses

The shared helpers build MuJoCo terrain, issue timed curvature commands and measure foot contact and body motion. Each species supplies its MJCF, foot collision geoms, controller, and acceptance criteria. The working examples below run from a source checkout with MuJoCo and NumPy installed.

## Run a species on uneven ground

This example runs the Stegosaurus controller over a 3.5 cm mound and counts complete swings of the left forefoot. `surface_loads[1]` measures whether a foot actually loaded the mound. The entries in `loads` and `clearance` follow the insertion order of the foot mapping. [Stegosaurus integration test](../../environments/shared/tests/test_locomotion_courses.py)

```python
import mujoco
import numpy as np

from examples.shared.footfalls import swing_events
from examples.shared.ground_contacts import GroundContacts
from examples.shared.locomotion_courses import Course, Mound
from examples.stegosaurus_stenops.build import MODEL_PATH
from examples.stegosaurus_stenops.walk import WalkController

course = Course(mounds=(Mound(x=1.42, y=0.25, x_radius=0.35, y_radius=0.25, height=0.035),))
model = course.model(MODEL_PATH)
data = mujoco.MjData(model)
course.reset(model, data)
controller = WalkController(model)
pads = {}
for leg in ("lf", "rf", "lh", "rh"):
    foot = model.body(f"{leg}_foot").id
    ids = np.flatnonzero((model.geom_bodyid == foot) & (model.geom_contype != 0))
    pads[leg] = [model.geom(i).name for i in ids]
probe = GroundContacts(model, pads, course.surface_names())

body_weight = mujoco.mj_getTotalmass(model) * abs(model.opt.gravity[2])
loaded, gaps, mound_load = [], [], []
for step in range(round(7.0 / model.opt.timestep)):
    controller.step(data)
    mujoco.mj_step(model, data)
    if step % 10 == 0:
        mujoco.mj_forward(model, data)
        sample = probe.measure(model, data)
        loaded.append(sample.loads[0] > 0.01 * body_weight)
        gaps.append(sample.clearance[0])
        mound_load.append(sample.surface_loads[1])
landings, peak_gaps, shallow_reloads = swing_events(loaded, gaps, minimum_clearance=0.005)
print(len(landings), "complete left forefoot swings. Peak mound load (N)", max(mound_load))
```

For a slope, use `Course(pitch_deg=5)` to rise towards +X or `Course(bank_deg=3)` to rise towards +Y. Negative values reverse the gradient. Mounds are elliptical caps on a level base. Their positions, radii and peak heights are in metres. `course.height(x, y)` gives the exposed surface height for a controller that uses terrain information. The MJCF needs a fixed plane centered on the world origin and a free root joint. Models with different names can pass `floor_name` to both `course.model` and `course.surface_names`, and `root_joint` to `course.reset`. Reset aligns the neutral body with the initial plane once. [Terrain implementation](locomotion_courses.py)

For another species, map each foot to **all of its ground-contacting geom names**. The probe sums normal forces per foot and per terrain geom in newtons. It also returns the contact positions projected onto horizontal XY, force on other body parts, signed local clearance, and load-weighted slip along the contact surface in m/s. Clearance queries use MuJoCo's geometry-distance support and saturate at `distance_limit` (default 0.5 m). `swing_events` counts an unloading and reloading cycle only when its peak gap exceeds the chosen clearance. Partial swings at either edge of the sample window are excluded. Shallow reloads have their own count. [Ground measurements](ground_contacts.py) · [Footfall measurements](footfalls.py)

## Record a turn

`TurnCourse(radius_m, entry_s, turn_s, exit_s)` defines straight entry, a constant-curvature arc and straight exit. Radius is positive for a left turn and negative for a right turn. At each control update pass `course.curvature_at(t)` to the species controller, where `t` is elapsed simulation time. Record actual trunk XY and yaw (radians counterclockwise from +X) before stepping, including the onset and end of the arc. A controller must translate the curvature command into its own leg motions.

The adult Triceratops runner records time, trunk position, yaw, sole centers, foot names and normal loads. To reproduce the 5 m left turn and independently evaluate the recorded path, run the following from the repository root. [Turn runner](../triceratops_horridus/check_turns.py)

```sh
python -m examples.triceratops_horridus.check_turns --case left --trace-dir /tmp/turns
```

```python
import numpy as np

from examples.shared.turn_metrics import circular_path_metrics
from examples.triceratops_horridus.courses import TURN_COURSES

course = TURN_COURSES["left"]
with np.load("/tmp/turns/turn_left.npz") as trace:
    time = trace["time"]
    first = np.argmin(abs(time - course.entry_s))
    last = np.argmin(abs(time - (course.entry_s + course.turn_s)))
    arc = slice(first, last + 1)
    result = circular_path_metrics(trace["trunk"][arc, :2], trace["yaw"][arc], course.radius_m)
print(result["heading_change_deg"], result["rms_radial_error_m"])
```

The first recorded arc position, its heading and the signed radius define the reference circle. Body heading change and angular travel around that circle are separate outputs. Check both to distinguish a pivot in place from forward progress through the bend. The exit segment tests whether the body keeps walking along its new heading. [Turn measurements](turn_metrics.py)

## Choose species-specific limits

Set course radius, entry time, slope and obstacle size for the animal's stride and body scale. Specify which contacts count as support, the minimum clearance for a step, expected footfall order and acceptable pad slip and path error. For example, the Triceratops tests look for sustained three-foot support and a lateral-sequence walk. A biped needs its own support and phase criteria. Keep thresholds, reported results and any controller adjustments with the species so the same physical measurements can serve different gaits. [Triceratops course definitions](../triceratops_horridus/courses.py) · [Triceratops terrain checks](../triceratops_horridus/check_terrain.py)
