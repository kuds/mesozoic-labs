# Triceratops simulation clips (data branch, never merged)

Video clips from the Triceratops build session, kept here because they exceed the repository's 500 KB file limit
(owner's decision D11, 18 Nov 2026, in `hardware/triceratops_build/DECISION_LOG.md` on the build branch). **Every clip
is a simulation render, labelled "SIMULATION" on every frame.** Never merge this branch.

| File | What | Made by |
|---|---|---|
| `crawl_T20_ankleroll_2x.mp4` | 20 s at 2× speed, 1280 × 720, H.264, 5.6 MB: the 20 kg robot's scripted crawl (0.02 m/s), design variant with passive roll ankles, seed 1. The variant's pass is thin (fails under a stiffer friction model); see `hardware/triceratops_build/sim/CRAWL_REPORT.md` | `hardware/triceratops_build/renders/render_crawl_clip.py --model ../sim/generated/trike_T20_closed_ankleroll.xml --speed 2 --seconds 20` at build-branch commit 6247dc78 |
