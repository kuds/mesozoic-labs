# Reference attribution and modifications

Inherited leg transforms, inertias and the 14 STL files in
`../assets/meshes/` derive from **Open Duck Mini v2**, by **apirrone and
contributors**, commit `b23317a485b3cec7d8417f352478778b3475173c`:

https://github.com/apirrone/Open_Duck_Mini/tree/b23317a485b3cec7d8417f352478778b3475173c/mini_bdx/robots/open_duck_mini_v2

The upstream Apache 2.0 license is retained in `LICENSE_Open_Duck_Mini`.
The project MIT license does not remove these third-party obligations.
No upstream endorsement is implied.

`compso_rev_b.xml` is the preserved generated reference from **Mesozoic Labs
Leg Mechanism Rev B, September 6, 2026**. That adaptation fixed the head
and tail, allocated core mass, added 19.5 g per hip-roll servo assembly and
replaced the reference feet with proposed ankle-roll adapters and broader
soles. Its file hash and original home-pose source are recorded in
`../data/model_parameters.json`.

This September 7 adaptation adds floor/inter-leg/head-tail collisions,
limited position servos, explicit joint dynamics, foot sensor volumes,
an IMU, encoders, one camera, a mocap target, home keyframe and materials.
Mechanical transforms and inertias are unchanged. Decorative geometry adds
no mass outside the previous allowances.

The anatomical model is new primitive geometry following Mesozoic Labs
conventions. Scientific references and limitations are in `../README.md`.
No external skeletal illustration or animal mesh is redistributed.
