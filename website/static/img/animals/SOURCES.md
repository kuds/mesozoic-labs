# Animal model plates

The original catalog plates were rendered from each model’s `home` keyframe in MuJoCo 3.17.0. The plates use a common camera angle and studio lighting. Model geometry and materials come from the lab’s MJCF files. The framing scales each model independently. [Source revision](https://github.com/kuds/mesozoic-labs/tree/4b86f5c6fec8d1b427ba86c1c9a7cc11855ec875/environments)

For a file named `<species>-<embodiment>.webp`, the source model is `environments/<species>/<embodiment>/assets/<species>_<embodiment>.xml` in that revision. The Compsognathus robot render includes Open Duck Mini hardware meshes referenced by its model. [Upstream license](https://github.com/kuds/mesozoic-labs/blob/4b86f5c6fec8d1b427ba86c1c9a7cc11855ec875/environments/compsognathus_longipes/robot/references/LICENSE_Open_Duck_Mini)

The current Brachiosaurus plate shows physics revision 6, rendered in MuJoCo 3.10.0 from the standing keyframe. [Revised model](../../../../environments/brachiosaurus_altithorax/anatomical/assets/brachiosaurus_altithorax_anatomical.xml)

The two Triceratops plates crop the anatomical and robot views from the lab’s design study. The robot view shows the 75 cm frill-height concept. [Original study figure](https://github.com/kuds/mesozoic-labs/blob/84a13e1e7729ff4e92f1376db6464975836b3bcc/hardware/triceratops_design_study/models/preview.png)
