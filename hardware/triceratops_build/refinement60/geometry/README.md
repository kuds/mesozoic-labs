# Foreleg placement and shell clearance

Both 60 cm MuJoCo models open directly from the checkout. The reference directory holds the original open-rib study's input meshes, textures, assembled exhibit and physical model. The refined exhibit references common assets in that directory. Its local meshes carry the changed shoulder, forearm and wrist surfaces.

From the `refinement60` directory, the following command rebuilds the selected model from those reference assets. It first moves the simulated X6 masses and their visible housings, both crank planes, neutral pushrods and motor supports. It then fits the backing plates, opens the forward rib cage and reshapes the two shoulder panels around the original X4 drives.

```sh
python geometry/build.py
```

The foreleg retains a 128 mm upper link, a 184 mm mechanical lower link and the baseline passive ankle. Each 80 mm X6 shifts 20 mm aft, 6 mm up and 5 mm outboard. The 114 mm visible forearm ends in a fixed tapered cuff. The sculpt uses the shell mass allowance carried by the physical model. The short links, plate, cuff and open ribs still need weighing and tolerance checks on the guarded assembly.

## Reproduce the measurements

With the dependencies in `requirements.txt`, run these examples from `refinement60`. Each command writes its last positional argument unless an option follows it.

```sh
python geometry/check_models.py models/refined equivalence.json
python geometry/validate_fit.py models/refined/physics.xml hardware-gaps.json
python geometry/rod_fit.py models/refined/physics.xml linkage-gaps.json
python geometry/skin_fit.py models/refined/triceratops.xml panels.json
python geometry/skin_fit.py models/refined/triceratops.xml pitch-panels.json --drive x4_pitch_stator
python geometry/skin_fit.py models/refined/triceratops.xml right-panels.json --side right
python geometry/check_trim.py models/refined/triceratops.xml shoulder-panels.json
```

The signed hardware gaps use 17 samples along each roll, pitch and elbow range, giving 4,913 poses per side. The X6 barrel to upper strut improves from 9.20 mm of intersection to 8.70 mm of separation. The backing plate clears the rail by 4.68 mm. The separate ideal four-bar check rotates both 28 mm crank pins and moves the rod at each pose. It finds 3.50 mm between the nominal rod and X6 case. In the dynamic model the elbow actuator applies torque directly and the displayed rod stays at its neutral orientation on the upper link.

The exterior check uses signed cylinder distances for the panel vertices, face centres and edge midpoints in 1,125 poses. Pass `--trajectory path/to/recording.npz` to sample every third recorded pose. The shoulder check independently examines 441 combinations of pitch and roll. The sampled minima are nominal model gaps. Fastener heads, the moving cable exit, rod-end width, elastic deflection, tool access, airflow and guard openings need independent verification against the manufactured article.

[Selected dimensions and signed sweep results](../evidence/geometry/)
