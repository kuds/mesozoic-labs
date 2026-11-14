# Source audit

| Record | Identity | Check performed here |
|---|---|---|
| PR #662, draft body/locomotion platform | Base `edd76a2828222174016712eaf6d38da8ea78b852`, head `e103f7ff2609fa1d7eb08507fb50ed9bb410b632` | Read PR metadata and fetched the nominal MJCF and mass ledger directly at the head. Both Git blob IDs below match locally recomputed blob objects. Compiled the XML and ran the comparison methods on it. |
| PR #663, findings/open | Base is the #662 head; head `b5c18bce1e576542a51532ffb442c238d83b0c9a` | Read findings, original R2 redesign plate, original R3 partner brief and clean-checkout record. The ZIP contains their relevant project snapshots. |
| R2 compact comparison source | Published XML prefix `857d189b8a0d` | The redesign plate labels its redesigned specimen as R2. Used its own original pixels with stated scale for the side comparison. It shows approximately 60 cm height, 599.8 mm crown, 1,104 mm length, 10.282 kg. The published prefix is printed in the partner brief. |
| R3 walking platform | Embodiment `intermediate_r3_175`, handoff `f7d9857791e0b75d388118fd38fac2d2ce2c0794`, path `examples/intermediate_rex_r2/successor/model.xml` | Release record binds physical XML SHA-256 `87e36ef3cffc87b83dc33abffd5bed4acaee62a768de11b273143febbf64d3fe`. Read partner brief’s R3 hero and gait/report evidence. Hips 214 mm, sole centers 175 mm, 10.370 kg. R2 and R3 differ at least in stance, mass and arm brackets. The exact remaining torso mesh equality is undetermined without running the R3 and R2 input models side by side. |
| Executable handoff | `Intermediate-T-Rex-Simulation-Handoff.zip`, 269,274,463 bytes, Library file ID `libfile_fde3_f-ab3addefb9c48191998ce0b68aa8e7c1_5a238845a59881919f692be20e4c3c2e` | Resolved its metadata and the handoff revision referenced by #663. The local data transfer failed; the browser had no authenticated session for the file. No independent archive SHA, checkout, policy load or R3 replay was possible here. The earlier 100,124,954-byte `Intermediate-Rex-Handoff.zip` is from an earlier exercise at 85ae1705 and is distinct. |

The #663 release-check record reports 983 passing tests at checkout `eb0e984a6eebdcefc324268e03f9e6ba94c721f3`, and 450 identical replay frames. These are upstream claims and use an earlier verification checkout in the history of the final handoff. The present study independently verified and executed the #662 model. The study retains the reported R3 9 N·m hold demand for context; its numerical candidate comparisons use the #662 leg geometry and actuator schedule.

### Pinned inputs

* `examples/tyrannosaurus_rex_60cm/platform/nominal/model.xml`, Git blob `43152d1ca4220b90f86a220f65cda3126dee19ff`, SHA-256 `06019e3c43c48376d7d048f0e9c4b540df0094459247125b9d7c35f7d9ce6f66`.
* `examples/tyrannosaurus_rex_60cm/platform/nominal/components.json`, Git blob `5da76a30adde5070604b3bd49b15862db6735e2d`, SHA-256 `6beb0ea7b6d1961b87d4163dcd9f96a2dcfcfb80acaffd3891444b44c08fa077`.
* The original drive file is under `study/mechanics/upstream/`. All nine compiled XMLs and report inputs are in the package digest. The #663 images include the project’s original materials. The anatomical diagram is an original interpretive schematic; no external museum photographs or published paleontological figures have been redistributed.

[PR #662](https://github.com/kuds/mesozoic-labs/pull/662) · [PR #663](https://github.com/kuds/mesozoic-labs/pull/663) · [Pinned findings](https://github.com/kuds/mesozoic-labs/blob/b5c18bce1e576542a51532ffb442c238d83b0c9a/hardware/intermediate_rex_r2/FINDINGS.md) · [Original R3 handoff](https://chatgpt.com/library?file_id=libfile_fde3_f-ab3addefb9c48191998ce0b68aa8e7c1_5a238845a59881919f692be20e4c3c2e&account_id=personal)
