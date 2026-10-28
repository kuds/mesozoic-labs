# 60 cm skeleton engineering example

See the [design, parts and evidence](../../hardware/tyrannosaurus_rex_60cm/m01_skeleton/README.md)
and [hardware/training milestones](../../hardware/tyrannosaurus_rex_60cm/README.md).

`build.py` generates the standalone `model.xml` and explicit `components.json`
from the hardware manifest. `validate.py` screens packaging, shaft alignment,
mass accounting, fixed-PD stance/crouch and a separate combined-range collision
probe. `render.py` produces views of the actual model and recorded poses.

This example is not registered with the shared SB3 notebook. The existing mini
training plant remains unchanged. There is no head, jaw, arm or tail actuator.
