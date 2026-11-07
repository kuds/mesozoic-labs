# Building and inspecting the nine alternatives

The first letter selects A (tapered flank), B (deep ribcage) or C (open rib frame). The two digits give the chest inclination in degrees. Every supplied model and component ledger has revision 2. The earlier clearance experiment is preserved in `prior_geometry_v1.zip`.

Use the top-level reproduction script with `--only-models` to compile all nine from the pinned #662 MJCF and its component ledger. Each candidate XML embeds its meshes. The companion JSON records the transformation of each moved assembly, replacement inertials, per-piece wall thickness and mass, and centroid and inertia tensor. All component tensors are about their listed centroid, in the axes of the named body. Coordinates and angles in the station table are millimetres and degrees. In the compiled MJCF, lengths are metres. Model +x points forward, +y left and +z up.

## Shape controls

Edit `shapes.py` to change the ribcage stations, the windows between panels, cervical rise/run, skull depth and breadth, thigh extent or the tail root and curvature. Each torso station specifies longitudinal position, half breadth, bottom and top. Intermediate stations use shape-preserving interpolation. The two rail markers at x = 65 and 165 mm define chest pitch. It is the angle above the horizontal of their connecting line toward the head. The pelvis, ankle/foot configuration and tail mount keep their own reference frames. The neutral skull pitch is 0° in A, −3° in B and +2° in C. The neutral leg and sole contact positions come from the pinned model. The mechanics scripts search additional moving-leg and pelvis configurations.

## Panels and frame

The mid-surface triangles carry uniform unfilled PA12 at a design density of 1,010 kg/m³. The generator offsets the two faces and closes the exposed edges for visual geometry. Panel mass is summed from the area remaining after reliefs and vents, multiplied by local wall thickness and density. The plate inertia comes from the integrated lamina, translated into its rigid body. Default body-wall values are 1.45, 1.60 and 1.50 mm for A, B and C. Different parts use local wall values. Captive M2.5 screw, insert and mounting allowances weigh 2.5 g per counted attachment. The nasal cap shares its skull carrier. Detailed fastener bosses, pullout strength, flange stiffness and wiring are physical-validation items. The frame ledger accounts for the revised crossmembers, spar and tray allocations. The visual meshes and local convex collision patches serve different purposes. The clearance study tests selected exact surface pairs independently of the solver contact masks.

`generated/` contains the frozen revisions and part data. See `../hardware/README.md` for the extraction sequence and connector reservations, `../mechanics/SCREENING.md` for the tested joint configurations and limitations, and `../report/manufacturing.md` for physical qualification.
