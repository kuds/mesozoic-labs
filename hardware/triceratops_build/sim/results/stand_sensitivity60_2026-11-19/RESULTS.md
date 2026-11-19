# Standing balance and design sensitivity, 60 cm (simulation)

Written by `stand_sensitivity60.py`. Standing at the neutral pose, 10 s, means over the last second. Reported, not judged.

## Battery position (closed variant)

| Battery centre x | Front share | Left share | Front elbow | Hind knee | Pitch max | Roll max | Battery's tightest gap |
|---|---|---|---|---|---|---|---|
| -200 mm | 54.7 % | 49.8 % | 3.95 N·m | 3.25 N·m | 0.71 N·m | 0.81 N·m | -0.6 mm (pelvic_crossmember) INTERSECTS |
| -180 mm | 55.0 % | 49.8 % | 3.98 N·m | 3.22 N·m | 0.71 N·m | 0.82 N·m | -0.0 mm (removable_service_tray) |
| -160 mm | 55.3 % | 49.8 % | 4.00 N·m | 3.20 N·m | 0.71 N·m | 0.83 N·m | -0.0 mm (removable_service_tray) |
| -140 mm | 55.6 % | 49.8 % | 4.02 N·m | 3.18 N·m | 0.71 N·m | 0.84 N·m | -9.0 mm (jetson_orin_nano_super_kit) INTERSECTS |
| -120 mm | 55.9 % | 49.8 % | 4.05 N·m | 3.15 N·m | 0.71 N·m | 0.85 N·m | -29.0 mm (jetson_orin_nano_super_kit) INTERSECTS |
| -100 mm | 56.2 % | 49.8 % | 4.07 N·m | 3.13 N·m | 0.71 N·m | 0.85 N·m | -34.5 mm (jetson_orin_nano_super_kit) INTERSECTS |
| -80 mm | 56.5 % | 49.8 % | 4.09 N·m | 3.11 N·m | 0.71 N·m | 0.86 N·m | -34.5 mm (jetson_orin_nano_super_kit) INTERSECTS |

## Mass

| Case | Mass | Front share | Front elbow | Hind knee | ESTIMATE: pipe1 gait's heat x mass ratio (left hind knee / fore pitch) |
|---|---|---|---|---|---|
| listed (no reserve) | 11.540 kg | 55.1 % | 3.62 N·m | 2.92 N·m | 1.00 / 0.93 |
| closed | 12.694 kg | 55.3 % | 4.00 N·m | 3.20 N·m | 1.10 / 1.02 |
| closed scaled to the budget's 12.2 kg | 12.200 kg | 55.3 % | 3.84 N·m | 3.07 N·m | 1.06 / 0.98 |
| closed + 0.5 kg payload at the chassis | 13.194 kg | 55.1 % | 4.15 N·m | 3.35 N·m | 1.14 / 1.06 |

The last column scales the first learned gait's measured heat (mean of episodes) by mass; it assumes walking torque scales with weight and is not a simulation of a gait at that mass.

Chart: `battery_position.png`.
