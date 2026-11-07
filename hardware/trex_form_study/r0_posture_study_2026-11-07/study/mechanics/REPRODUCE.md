# Mechanical screen reproduction

Run these commands from the directory containing `study/` and `source/`. The calculations used Python 3.14.6, MuJoCo 3.10.0, NumPy 2.3.5 and SciPy 1.18.1. Matplotlib 3.11.2 generated the chart.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install "mujoco==3.10.0" "numpy==2.3.5" "scipy==1.18.1" "matplotlib==3.11.2"
python study/mechanics/screen.py source/pr662/model.xml study/model/generated/*.xml
python study/mechanics/motion.py source/pr662/model.xml study/model/generated/*.xml --drive study/mechanics/upstream/intermediate_drive.py
python study/mechanics/clearance.py study/model/generated/*.xml
python study/mechanics/pelvis_probe.py source/pr662/model.xml study/model/generated/*10.xml
python study/mechanics/tabulate.py
python study/mechanics/load_figure.py study/model/generated/C10.xml
```

The supplied drive is the exact `environments/shared/intermediate_drive.py` from `kuds/mesozoic-labs` at `e103f7ff2609fa1d7eb08507fb50ed9bb410b632`. Its Git blob ID is `c9973312a540571413bf02f15bf7ea2aec9f540d` and SHA-256 is `8d9a426dc2e9afb8210a37d9502e0ef624eb161bc52a38b9ab23f77cf4b4f734`. `tabulate.py` verifies that the main calculations refer to identical candidate MJCF bytes. No checkpoint or normalization archive is required for the stand or transfer test.

The alternate pelvis probe varies signed absolute pelvis roll from −10° to +10° in 5° increments, support leg counter-roll from 0.12 to 0.36 rad in 0.02 rad increments, and swing-leg adduction by −0.08, 0 or +0.08 rad. Stance hip and ankle rotations sum to the inverse pelvis rotation to keep the sole flat. All joints must stay within limits, the four loaded corners must agree in height within 0.5 mm, the support margin must reach 3 mm and the swing foot must clear 8 mm. Trials with a feasible support wrench are ranked by actual stance hip-roll demand. A pose passing these gravity conditions still needs a full interference screen and physical temperature qualification. The samples do not establish a global optimum.
