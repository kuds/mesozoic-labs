# The 160 mm foreassembly

Both variants measure 600 mm at the frill crown. The steering and walking comparisons use the same 280 mm shoulder, level sole position, 184 mm lower link, passive ankle, X4/X4/X6 drives and torque limits. The X6 barrel and crank shaft keep their fitted position on the upper link. Changing the upper pivot span to 160 mm moves the neutral elbow 26.0 mm rearward and 18.9 mm down. Its floor height changes from 193.8 to 174.9 mm. The 80 mm barrel spans 50% of the longer link, compared with 62.5% of the shorter one.

The 160 mm assembly has longer upper struts and covers, a 103.83 mm neutral rod pin spacing (formerly 81.50 mm) and a backing plate aligned to the longer strut. Both crank radii remain 28 mm, and both versions have a 43° minimum ideal transmission angle. Mass scales with stock length for the upper links, upper covers, rods and backing plates, bringing the whole model from 12.844 kg to 12.925 kg. The model preserves the mass allowance for sculpted skin. The humeral sculpt is lengthened along the centreline with its transverse dimensions preserved, then locally relieved around the X6. The same lower sculpt and wrist cuff are reoriented to follow the shank.

| Minimum sampled gap, both sides | 128 mm | 160 mm |
|---|---:|---:|
| X6 case to upper strut | 8.70 mm | 7.07 mm |
| X6 case to upper sculpt | 6.42 mm | 6.92 mm |
| X6 case to closest sampled outer panel (anterior rib) | 4.97 mm | 4.97 mm |
| Backing plate to frame rail | 4.68 mm | 4.68 mm |
| Moving rod to X6 case | 3.50 mm | 3.50 mm |
| Moving rod to output-crank guard | 3.30 mm | 3.30 mm |
| Elbow output crank to torso panels | 38.71 mm | 23.47 mm |

The physical sweep samples 17 values of each shoulder-roll, shoulder-pitch and elbow angle per side (4,913 combinations). The motor-to-sculpt sweep uses 1,125 poses per side. The additional rod and elbow-crank screen uses 2,025 poses per side and samples panel vertices, face centres and edge midpoints. It translates the rod with both moving crank pins through the ideal parallelogram. The measured surfaces include torso ribs, saddle, scapula, coracoid, limb sculpt, cuff and partial crank guards. The renderer displays the rod at its locally fixed neutral position. The dynamics apply elbow torque directly, and shell/guard meshes are cosmetic. Tolerances, real rod ends, screw heads, cable sweep, deflection and guard effectiveness need checks on the instrumented leg. [Bilateral measurements and model-equivalence results](geometry.json)

From the refinement directory:

```sh
python mid160/build.py
python mid160/check_geometry.py
```

Keep the [physics model](../models/mid160/physics.xml) and [dressed model](../models/mid160/triceratops.xml) alongside the existing `models/refined` and `models/reference` folders for the shared assets.
