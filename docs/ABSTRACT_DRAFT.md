# CoRL 2026 Abstract Draft

**Target venue:** Conference on Robot Learning (CoRL) 2026
**Paper angle:** Multi-morphology platform with domain randomization for sim-to-real

---

## Title

**Mesozoic Labs: A Unified Curriculum Learning Framework for Multi-Morphology Legged Locomotion**

## Abstract

Training legged robots with diverse body plans remains a fundamental challenge
in robot learning, as each morphology typically demands bespoke reward
functions, training pipelines, and hyperparameter configurations. We present
Mesozoic Labs, a unified reinforcement learning framework that trains
locomotion policies across radically different morphologies — bipedal and
quadrupedal — using a shared curriculum learning architecture. Our platform
models three dinosaur-inspired body plans (Velociraptor, Tyrannosaurus Rex,
and Brachiosaurus) in MuJoCo, spanning 67–83 observation dimensions and
21–22 actuators, with a three-stage curriculum that progresses from postural
balance through coordinated locomotion to species-specific behaviors such as
predatory strikes and food reaching. Through systematic evaluation across 80+
training runs, we identify critical cross-morphology patterns including
catastrophic forgetting at curriculum transitions and high sensitivity to
reward component weighting. We further incorporate domain randomization over
physical parameters, sensor noise, and terrain variation to evaluate policy
robustness for sim-to-real transfer. Our results demonstrate that a single
configurable training infrastructure — with externalized TOML-based reward
specifications and automatic stage advancement — can produce stable locomotion
across morphologies that differ by over 10x in mass and limb count, while
reducing per-species engineering effort. We release the full platform,
including MJCF models, training configurations, and evaluation tools, as an
open-source benchmark for multi-morphology legged locomotion research.

---

## Notes

- **Word count:** ~200 words (within typical CoRL abstract limits)
- **Key contributions claimed:**
  1. Unified framework for multi-morphology locomotion (bipedal + quadrupedal)
  2. Three-stage curriculum learning with automatic advancement
  3. Empirical findings from 80+ training runs on cross-morphology patterns
  4. Domain randomization analysis for sim-to-real robustness
  5. Open-source benchmark release
- **Before submission:**
  - [ ] Complete domain randomization experiments
  - [ ] Finalize quantitative results for all 3 species
  - [ ] Record supplementary video (recommended, max 3 min / 250 MB)
  - [ ] Verify double-blind compliance (no identifying info)
  - [ ] Add mandatory Limitations section to paper body
  - [ ] Author list finalized before abstract deadline (cannot change after)
