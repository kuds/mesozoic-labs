# Reading the visual plates

Start with `plates/pr663_side_comparison.png` for the published R2 and three candidate side profiles. The 10° candidates appear beside the verified nominal plant in `plates/family_exteriors.png`. Open a candidate’s four-view sheet for its crown, hip height, length, width, model mass, center of mass and XML digest. Labels describe the installed mass represented by the model.

| Plate | Comparison |
| --- | --- |
| published_baselines | R2 from the PR #663 redesign plate and R3 from the same PR’s partner brief. Each retains its original camera and is independently enlarged. |
| pr663_side_comparison | Published R2 side view and three generated candidate profiles. R2 is calibrated using the published 599.8 mm crown. |
| family_exteriors | Front three-quarter views from one camera direction at one physical scale. |
| silhouettes | Black side silhouettes at one physical scale, then uniformly rescaled to one crown height. Actual crown and magnification are shown. |
| posture_matrix | All nine body and posture combinations, at the same scale and camera origin. |
| hardware_cutaways | Each assembled candidate beside the same view with the covers and shell-mounted teeth removed. |
| *_four_views | Side, front three-quarter, top and front at one physical scale. Crosshairs mark projected whole-robot center of mass in front and side views. |

Every generated view uses compiled physical surfaces in the model’s geometric home pose. The camera looks from −y for the side, +x for the front and +z for the top. The three-quarter camera looks from direction (0.61, −1, 0.41). The top and side images place the snout to the right. Lighting is fixed in world coordinates. The side and front floor line marks z = 0. The subtle shadow in three-quarter views is an illustration of foot placement.

The renderer includes opaque groups 1, 2 and 4, and omits invisible outlines, clearance reserves and duplicate collision patches. Width includes exposed leg-drive faces. The head measurement covers the visible head assembly; hip height uses the hip-pitch axis. Each candidate sheet records its XML SHA-256. The normalized silhouettes use the maximum crown of the compiled comparison set as the target, so physical pixel size changes only by the printed scale factor. Keep-outs and service clearances are covered by the engineering screens.

From the package root, with Python, MuJoCo 3.10, NumPy, Pillow and a C compiler installed:

```sh
python study/visuals/plates.py --pr662 source/pr662/model.xml \
    --models study/model/generated \
    --pr-image source/pr663/visual/redesign-at-60cm.png
```

For a verified R3-based run, replace `--pr662` and its argument with `--r3 /path/to/the/verified/R3/model.xml` and supply the corresponding regenerated candidate files. `--r2` accepts a verified R2 MJCF as an additional comparison. The original raster source is a 1328 × 1270 PNG, and the R3 partner view is the single embedded 904 × 584 RGB image in the accompanying PDF. The PNG must be present beside the original R2 plate as `published-r3-partner-view.png`. The snapshot of the original #663 partner brief has Git blob `eff8efa2374dd425f353df06710fd873a66e2dc4`. Its image can be extracted with `pdfimages -png source/pr663/Partner-brief.pdf /tmp/partner`; the output is `/tmp/partner-000.png`.

Fonts fall back to DejaVu where Arial and Georgia are unavailable. Font rasterization may change the image digest across systems; neither camera coordinates nor the geometry depend on the fonts.
