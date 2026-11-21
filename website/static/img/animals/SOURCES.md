# Animal model plates

The catalog model plates show each animal’s `home` keyframe with studio lighting. The framing scales each model independently.

The Allosaurus plate shows its standing keyframe in MuJoCo 3.10.0. The walking video uses the same anatomical model with live floor contacts. [Model source](../../../../environments/allosaurus_fragilis/anatomical/assets/allosaurus_fragilis_anatomical.xml)

The anatomical Compsognathus plate shows physics and visual revision 3, rendered in MuJoCo 3.10.0. [Source model](https://github.com/kuds/mesozoic-labs/blob/50208916d4e3266b58b005c972d2915ae8c89f08/environments/compsognathus_longipes/anatomical/assets/compsognathus_longipes_anatomical.xml)

The anatomical Brachiosaurus plate shows physics revision 6, rendered in MuJoCo 3.10.0 from the standing keyframe. [Source model](https://github.com/kuds/mesozoic-labs/blob/fb42a60c1949a2d4a925b0e45096fa4280d07de8/environments/brachiosaurus_altithorax/anatomical/assets/brachiosaurus_altithorax_anatomical.xml)

The anatomical Velociraptor plate shows the home pose of physics revision 5 and visual revision 6, rendered in MuJoCo 3.10.0. [Source model](https://github.com/kuds/mesozoic-labs/blob/185b2f5964b20335d6515a694df3b412eb6361ce/environments/velociraptor_mongoliensis/anatomical/assets/velociraptor_mongoliensis_anatomical.xml)

The Stegosaurus plate shows the standing pose of the specimen-scale reference model, rendered in MuJoCo 3.10.0. The walking clip follows the same model for ten simulated seconds, with the bars beneath it showing measured foot loads above 1% of body weight. [Source model and recording code](https://github.com/kuds/mesozoic-labs/tree/93fe00ad606068056ff08607cce2cf3888010ded/examples/stegosaurus_stenops)

The anatomical Dibothrosuchus and Tyrannosaurus plates and the Compsognathus robot plate were rendered in MuJoCo 3.17.0. For each `<species>-<embodiment>.webp`, the corresponding MJCF is `environments/<species>/<embodiment>/assets/<species>_<embodiment>.xml` at the linked revision. The robot includes Open Duck Mini hardware meshes. [Source revision](https://github.com/kuds/mesozoic-labs/tree/4b86f5c6fec8d1b427ba86c1c9a7cc11855ec875/environments) · [Upstream license](https://github.com/kuds/mesozoic-labs/blob/4b86f5c6fec8d1b427ba86c1c9a7cc11855ec875/environments/compsognathus_longipes/robot/references/LICENSE_Open_Duck_Mini)

The two Triceratops plates crop the anatomical and robot views from the lab’s design study. The robot view shows the 75 cm frill-height concept. [Original study figure](https://github.com/kuds/mesozoic-labs/blob/84a13e1e7729ff4e92f1376db6464975836b3bcc/hardware/triceratops_design_study/models/preview.png)
