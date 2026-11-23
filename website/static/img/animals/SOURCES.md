# Animal model plates

The catalog model plates show each animal’s `home` keyframe with studio lighting. The framing scales each model independently.

The Allosaurus plate shows its standing keyframe in MuJoCo 3.10.0. The walking video uses the same anatomical model with live floor contacts. [Model source](../../../../environments/allosaurus_fragilis/anatomical/assets/allosaurus_fragilis_anatomical.xml)

The anatomical Compsognathus plate shows physics and visual revision 3, rendered in MuJoCo 3.10.0. [Source model](https://github.com/kuds/mesozoic-labs/blob/50208916d4e3266b58b005c972d2915ae8c89f08/environments/compsognathus_longipes/anatomical/assets/compsognathus_longipes_anatomical.xml)

The anatomical Brachiosaurus plate shows physics revision 6, rendered in MuJoCo 3.10.0 from the standing keyframe. [Source model](https://github.com/kuds/mesozoic-labs/blob/fb42a60c1949a2d4a925b0e45096fa4280d07de8/environments/brachiosaurus_altithorax/anatomical/assets/brachiosaurus_altithorax_anatomical.xml)

The anatomical Velociraptor plate shows the home pose of physics revision 5 and visual revision 6, rendered in MuJoCo 3.10.0. [Source model](https://github.com/kuds/mesozoic-labs/blob/185b2f5964b20335d6515a694df3b412eb6361ce/environments/velociraptor_mongoliensis/anatomical/assets/velociraptor_mongoliensis_anatomical.xml)

The Stegosaurus plate shows the standing pose of the specimen-scale reference model, rendered in MuJoCo 3.10.0. The walking clip follows the same model for ten simulated seconds, with the bars beneath it showing measured foot loads above 1% of body weight. [Source model and recording code](https://github.com/kuds/mesozoic-labs/tree/93fe00ad606068056ff08607cce2cf3888010ded/examples/stegosaurus_stenops)

The anatomical Tyrannosaurus plate shows the home pose of physics revision 11 and visual revision 5, rendered in MuJoCo 3.10.0. [Source model](../../../../environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml)

The anatomical Dibothrosuchus plate and the Compsognathus robot plate were rendered in MuJoCo 3.17.0. For each `<species>-<embodiment>.webp`, the corresponding MJCF is `environments/<species>/<embodiment>/assets/<species>_<embodiment>.xml` at the linked revision. The robot includes Open Duck Mini hardware meshes. [Source revision](https://github.com/kuds/mesozoic-labs/tree/4b86f5c6fec8d1b427ba86c1c9a7cc11855ec875/environments) · [Upstream license](https://github.com/kuds/mesozoic-labs/blob/4b86f5c6fec8d1b427ba86c1c9a7cc11855ec875/environments/compsognathus_longipes/robot/references/LICENSE_Open_Duck_Mini)

The adult Triceratops plate shows the neutral standing pose. The real-time walking clip follows the same model under its reference controller. [Adult model](../../../../examples/triceratops_horridus/anatomical.xml) · [Walking study and measurements](../../../../examples/triceratops_horridus/README.md)

The 75 cm robot plate is a crop from the original design study. [Original study figure](https://github.com/kuds/mesozoic-labs/blob/84a13e1e7729ff4e92f1376db6464975836b3bcc/hardware/triceratops_design_study/models/preview.png)

## Rotating views

The Animals collection and specimen pages use 30-second camera orbits rendered from each model’s standing pose in MuJoCo 3.10.0. Each orbit has a light and dark H.264 version and a transparent WebP poster made from the opening frame. A keyframe every half second keeps the video responsive when visitors drag to change the view. The anatomical and robot Compsognathus views use their respective models, including the robot hardware meshes credited above.

To regenerate all rotating views, install the repository’s rendering dependencies and an FFmpeg build with libx264, then run this command from the repository root.

```sh
python website/scripts/render_turntables.py
```

## Triceratops scale comparison

The two transparent profiles show neutral standing poses in a level, orthographic side view. Each image is cropped from the tip of the tail to the tip of the brow horn and from the lowest foot surface to the crown of the frill. The adult model spans 7.19 m in this view. The selected 60 cm robot spans 1.20 m and its frill reaches 0.60 m above the soles. The rubber pads have a lighter studio finish so they remain visible on the dark background. [Adult reference model](../../../../examples/triceratops_horridus/anatomical.xml) · [60 cm option C model with roll ankles](https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/sim/generated/trike60_C_closed_ankleroll.xml) · [60 cm dimensions and build report](https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/sim/REPORT_60CM.md)
