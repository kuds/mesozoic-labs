# Triceratops simulation clips (data branch, never merged)

Video clips from the Triceratops build session, kept here because they exceed the repository's 500 KB file limit
(owner's decision D11, 18 Nov 2026, in `hardware/triceratops_build/DECISION_LOG.md` on the build branch). **Every clip
is a simulation render, labelled "SIMULATION" on every frame.** Never merge this branch.

| File | What | Made by |
|---|---|---|
| `crawl_T20_ankleroll_2x.mp4` | 20 s at 2× speed, 1280 × 720, H.264, 5.7 MB (5,721,193 bytes): the 20 kg robot's scripted crawl (0.02 m/s), design variant with passive roll ankles, seed 1. The variant's pass is thin (fails under a stiffer friction model); see `hardware/triceratops_build/sim/CRAWL_REPORT.md` | `hardware/triceratops_build/renders/render_crawl_clip.py --model ../sim/generated/trike_T20_closed_ankleroll.xml --speed 2 --seconds 20` at build-branch commit 6247dc78 |
| `walk60_trot_0.25_1x.mp4` | 20 s shown at actual speed (1x), 1280 × 720, H.264, 5.7 MB (5,687,229 bytes): the 60 cm robot (12.7 kg) trotting at its planned walking speed, 0.25 m/s, with the model-based controller that passed its pre-registered rules on flat floors (`hardware/triceratops_build/walk/REPORT_WALK60.md`); design variant with passive roll ankles (not adopted; the foot is open, D5); 10 ms servo delay and sensor noise simulated, no randomisation, episode seed 101. Straight ahead on a flat floor only: no turns; its weak side is ramps (same report). The second render of 22 Nov 2026: second badge, no floor reflection, each frame at the first 1 ms physics step at or after its 1/30 s time, new title | `hardware/triceratops_build/renders/render_walk_clip.py --out walk60_trot_0.25_1x.mp4 --seconds 20` at build-branch commit 57699759 |
