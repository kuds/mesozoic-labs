# Mesozoic Labs · Intermediate robotic T. rex
## Body shape and posture design study · 7 November 2026

Open **T_rex_body_study.pdf** for the visual comparison, mechanical results and downselection. For an offline, rotatable model, open **study/visuals/Inspect-3D.html** in a browser. Full-size review plates are under **study/visuals/plates/**. The original R2 and R3 views are separately identified in **source/pr663/**.

| Question | Where to look |
|---|---|
| How do the alternatives look beside R2? | `study/visuals/plates/pr663_side_comparison.png`, `family_exteriors.png` and `silhouettes.png` |
| What changes between 0°, 10° and 20°? | `study/visuals/plates/posture_matrix.png` plus the individually dimensioned four-view plates |
| What do the cross-sections and real hardware look like? | `study/sections/transverse_sections.png`, `study/visuals/plates/hardware_cutaways.png` and `study/hardware/service_paths.png` |
| What data defines each candidate? | `study/model/shapes.py`, `study/model/build.py` and `study/model/generated/`. Each JSON lists component moves, cover geometry, allocation and inertia. |
| Did the components fit? What can be removed? | `study/hardware/README.md`, `component_basis.md`, `installed_layout.csv` and `screen.json` |
| Where did the mechanisms interfere? How much torque was required? | `study/mechanics/SCREENING.md`, `metrics.json`, `motion.json`, `clearance.json` and `pelvis_probe.json` |
| What guided the form? | `study/reference/references.md` and `visual_review.md` reference SUE, Stan and the adult skull comparisons. The reference plate is an original qualitative diagram. |
| What is still required to build it? | `study/report/manufacturing.md` and the hardware and mechanics appendices |
| Can I inspect the superseded design? | `study/model/prior_geometry_v1.zip` preserves the earlier models; the early clearance witnesses document the reasons for the revisions. |

### Source and limits

The generated files use the nominal executable MJCF and component ledger at PR #662 head `e103f7ff2609fa1d7eb08507fb50ed9bb410b632`. The nominal XML’s SHA-256 is `06019e3c43c48376d7d048f0e9c4b540df0094459247125b9d7c35f7d9ce6f66`. It has 224 mm hip and sole-center spacing. The original R2 in the #663 comparison plate is 10.282 kg. The R3 walking model is 10.370 kg, uses 214 mm hip and 175 mm sole spacing and remounted fixed arms. Its published model digest is `87e36ef3cffc87b83dc33abffd5bed4acaee62a768de11b273143febbf64d3fe`. The original executable handoff was identified by name, size and release commit; byte-level archive and R3 replay verification were unavailable for this study. The 9 N·m R3 value comes from the source report. Candidate operating loads were calculated independently.

The work creates nine separate free-base, 14-actuator design models. The existing working models, training archives and policies were preserved. No new learned gait, autonomy trial or hardware qualification is claimed.

[PR #662](https://github.com/kuds/mesozoic-labs/pull/662) · [PR #663](https://github.com/kuds/mesozoic-labs/pull/663) · [Original executable handoff](https://chatgpt.com/library?file_id=libfile_fde3_f-ab3addefb9c48191998ce0b68aa8e7c1_5a238845a59881919f692be20e4c3c2e&account_id=personal)

### Reproduce

From an extracted copy of the archive on macOS or Linux, with Python, a C compiler and the listed packages installed:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install 'mujoco==3.10.0' numpy scipy pillow trimesh matplotlib reportlab
python reproduce.py --only-models
python -m mujoco.viewer --mjcf=study/model/generated/B10.xml
python reproduce.py
```

The first script command regenerates the nine self-contained XMLs and ledgers from the included inputs. The final command also recomputes the screens and figures and rebuilds the PDF. On macOS, use the MuJoCo installation’s `mjpython` if the interactive viewer needs the main thread. The full run updates generated files in the extracted study directory. It does not edit a repository checkout. The reference notes and reviewed decision text are supplied editorial evidence. See the model and mechanics documentation for numerical tolerances and environment versions. New dependency versions or fonts may alter the last digits or render pixels.

Verify the distributed release before regenerating files:

```sh
shasum -a 256 -c SHA256SUMS
```
