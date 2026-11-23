---
name: New Species Proposal
about: Propose a new dinosaur species for the project
title: "[SPECIES] "
labels: new-species
assignees: ''
---

**Species name**
Scientific name (the binomial, e.g. "Deinonychus antirrhopus") and any common name.

**Species ID and embodiment**
Names follow [docs/SPECIES_NAMING.md](https://github.com/kuds/mesozoic-labs/blob/main/docs/SPECIES_NAMING.md).
- Species ID (the binomial in snake_case, e.g. `deinonychus_antirrhopus`):
- Embodiment:
  - [ ] `anatomical`: the animal-proportioned research model; a new species starts with this one
  - [ ] `robot`: a buildable robot of an existing species ([Adding an embodiment][embodiment])

**Locomotion type**
- [ ] Bipedal
- [ ] Quadrupedal

**Body plan**
Brief description of body proportions, limb structure, and notable features.

**Unique behavior**
What behavior would this species demonstrate that differs from existing species?
(e.g., pack hunting, burrowing, flying, swimming)

**Estimated complexity**
- Approximate number of joints/actuators:
- Similar to existing species: [e.g. "similar to Velociraptor but smaller"]

**References**
Links to paleontological literature, biomechanics papers, or anatomical references
that could inform the MJCF model design.

[embodiment]: https://github.com/kuds/mesozoic-labs/blob/main/CONTRIBUTING.md#adding-an-embodiment-a-robot
