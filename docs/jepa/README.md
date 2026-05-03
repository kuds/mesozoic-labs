# `docs/jepa/` — exploratory, paused

This directory holds the design notes for an exploratory JEPA + RL
pipeline that was scaffolded on branch
`claude/jepa-dinosaur-locomotion-B6Qkr` and **paused before completion**.

After scaffolding, the premise was re-examined and judged a poor fit
for the current state-based velociraptor task. The analysis lives in
[`CHOICE.md`](./CHOICE.md) — see the "Is JEPA the right tool?" section
for the verdict and recommended alternatives (TD-MPC2 on proprio,
DreamerV3, SPR-style auxiliary heads).

The reference code is preserved on-branch under
`environments/shared/jepa/` and `environments/shared/wrappers/` so the
ViT-Tiny encoder, predictor, masking, EMA loss, and pixel-observation
wrapper can be reused if a future task (e.g. predator vision, sim-to-
real with a real camera) actually needs them.

**No further work is planned on this branch.**
