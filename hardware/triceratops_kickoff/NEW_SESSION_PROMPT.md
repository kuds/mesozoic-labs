# Prompt for the Triceratops session

18 Nov 2026. For the owner: paste the fenced block below into a new chat session on the repository
`kuds/mesozoic-labs` to run the Triceratops build, walking and simulations. The block stands alone: the new session
sees none of the earlier conversations.

Before pasting:
- Check that the folder `hardware/triceratops_kickoff/` with `PARTS_PREMIUM.md`, `CONSOLIDATION_PLAN.md` and
  `PLEM_FIT.md` is on GitHub (on main, or on the T. rex working branch until the T. rex study pull request merges).
  The block tells the session how to find it.
- Fill in "OWNER'S ANSWERS" at the end of the block if you have already decided some questions, or delete those lines.
- Use the cloud environment "mesozoic-labs" if it still exists (it installs libOSMesa, which offscreen MuJoCo
  rendering needs).

```text
You are running the Triceratops robot build for Mesozoic Labs, in the GitHub repository kuds/mesozoic-labs. You see
none of the earlier conversations: everything you need is below or in the files named here. Branches are named by
their unique endings where that is enough; `git ls-remote origin` lists the full names.

GOAL
- A four-legged Triceratops robot for museum demos: 75 cm to the top of the frill, 1.5 m long, 15 or 20 kg, walking
  slowly (0.30 m/s, about 0.7 mph). Simulation first: MuJoCo models, then a walking controller in simulation, then a
  one-leg bench plan. Hardware is bought only when the owner approves it.
- Museum dates (the owner's):
  - mid-December 2026: first contact with museums, showing simulation renders. The owner named "the Field Museum or
    the Natural History Museum". Griffin MSI (Chicago's Museum of Science and Industry) is a further candidate the
    owner did not name;
  - Q3 2027 (July to September): live Triceratops demos.
- The T. rex. The project's earlier robot, a 75 cm two-legged T. rex, was paused and preserved on 18 Nov 2026 (its
  decision log, rows P47 to P51): a biped must balance at every moment and needs many hardware rounds, and in
  simulation its controllers mostly fell when servo delay was added. It stays on the roadmap as the flagship, a
  milestone for end 2027 / early 2028, and the museum outreach shows it as coming. Leading with a Triceratops loses
  the direct tie-in with SUE, the Field Museum's T. rex specimen (the owner wrote "Sioux connection"); the owner
  accepts that and updates the pitch deck. The T. rex is outside this session's scope, except for hardware choices
  the two robots share: the leg drive family (MyActuator X4 and X6), the compute and CAN firmware, the bench rig, the
  safety stop, the battery and the neck servo. When a choice you propose affects the T. rex, say so. Never change
  T. rex files or branches.
- When the T. rex resumes. If the owner picks a built T. rex for that milestone (round 2 of task 2), T. rex
  simulation resumes by 1 Mar 2027 in a separate session started from its RESUME.md, and its drives (10 x X4 + 2 x
  X6) are ordered in Q2 2027 once your foreleg bench (by 15 Apr 2027) confirms the shared drives. Your foreleg bench
  and its delay rows (A17) are therefore on the T. rex's path too (CONSOLIDATION_PLAN.md, section 5.3).

THE OWNER'S PREFERENCES
- Legit hardware, not a cheap knockoff: catalog drives with datasheets, industrial safety parts, named sensors.
- Spend in the same class as the T. rex: $17k-$32k, from the T. rex's 75 cm build ($17.2k self-built, $20.0k
  turnkey) up to its 100 cm flagship ($29k-$32k). Against that wish:
  - the recommended T20 (20 kg) is $31.8k nominal counted the T. rex way (hardware plus 7.2 % for freight, duty and
    tax), the top of the class, and $37.3k landed, above it;
  - T15 (15 kg) is $28.3k the T. rex way, $32.8k landed;
  - the first spend, for T20, is the one-foreleg bench set: $7.1k of parts, $8.9k landed.
  The drive prices are MyActuator's own US store. A reviewer found third-party US drive prices about twice that
  (RobotShop, 18 Nov 2026; PLEM_FIT.md), so say which price a total uses, and say when a proposal moves the total.
- Simulation first. Nothing is adopted without the owner: propose, ask, then record the answer and the options put in
  a decision log.
- Report in plain words, with numbers and units ("0.30 m/s (0.7 mph)", "24 N·m", "$7.1k"). Answer first, then the
  evidence.
- Ask for decisions with the question tool: at most four questions at a time, the recommended option first and marked
  as recommended, each option with its cost and consequence. Never ask with lettered lists in chat.

WHERE THINGS ARE (`git fetch origin <branch>`; read another branch's file with `git show origin/<branch>:<path>`)
- The kickoff folder, hardware/triceratops_kickoff/:
  - PARTS_PREMIUM.md: the checked parts list for a 20 kg build (T20, recommended) and a 15 kg build (T15): drive
    sizing, full parts lists with sources and dates, costs, mass budgets, and the bench questions (section 8).
  - CONSOLIDATION_PLAN.md: pull-request triage, merge order, branches, what carries over from the T. rex, the
    milestone dates (section 5; the T. rex resume dates in 5.3) and the open decisions (section 7).
  - PLEM_FIT.md: the checked analysis of whether WIM Robotics' PLEM real-time layer fits this robot. Its
    recommendation C: build the Triceratops on CAN for Q3 2027 with MyActuator's "-E" drives (CAN and EtherCAT on
    the same drive, so EtherCAT stays open), and run a time-boxed PLEM bench pilot only if WIM's answers clear three
    blockers: brakeless drives are unsupported, only the Jetson AGX Orin is supported, and a bus carries at most 8
    axes. Decide the pilot by end Feb 2027. Section 5.4 holds the questions for WIM and MyActuator.
  - reviews/: the surveys and independent checks behind these files (PR #669's review is survey_pr669_18nov.md).
  - this prompt.
  Find it: `git fetch origin && git log --all --oneline -- hardware/triceratops_kickoff/PARTS_PREMIUM.md`. The folder
  is on the T. rex working branch (ending `mesozoic-trex-stage2-refine-4jffre`) once the lead has merged it there
  (PLEM_FIT.md and reviews/ are there already), and reaches main with the T. rex study pull request. If you cannot
  find PARTS_PREMIUM.md, stop and ask the owner.
- PR #669, the Triceratops design study (https://github.com/kuds/mesozoic-labs/pull/669). `git fetch origin
  pull/669/head` alone only sets FETCH_HEAD, so fetch it as a local branch, `git fetch origin pull/669/head:pr-669`,
  then read a file with `git show pr-669:<path>`. Folder hardware/triceratops_design_study/:
  - README.md;
  - Triceratops-design-package.pdf (9 pages: design, leg loads, test plan, drawings on p.6; its T. rex pages 7-9
    predate the pause). The PR holds only this built PDF: there is no source to edit;
  - Triceratops-envelope.scad (OpenSCAD envelope);
  - models/robot.xml (the 75 cm, 15 kg robot, 14 position actuators, passive ankles), models/anatomical.xml (a 7.30 m
    adult), models/check_models.py, models/README.md, models/preview.png.
- On main:
  - docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md (section 4: a 12S custom quadruped with three CAN buses, the knee
    drive kept near the body, and its performance gates; section 3: Pupper V3);
  - docs/hardware/SIM_TO_REAL_PLAN.md, docs/hardware/HARDWARE_BOM.md, docs/hardware/REFERENCE_MICRODUCK_2026_10.md;
  - two quadruped species environments, environments/brachiosaurus_altithorax/ and environments/dibothrosuchus_elaphros/,
    and the shared code in environments/shared/ (the floor-truth gait recorder is environments/shared/gait/recorder.py);
  - main pins mujoco==3.10.0, the version PR #669's models were checked with.
- The paused T. rex study, folder hardware/trex_form_study/ on the T. rex working branch (ending
  `mesozoic-trex-stage2-refine-4jffre`), and on main once its pull request merges:
  - RESUME.md (hardware/trex_form_study/RESUME.md): how the T. rex restarts (its restart section points to
    CONSOLIDATION_PLAN.md, section 5.3, for the dates), and in section 4 the tools that carry over. Reuse these (copy
    them into the Triceratops folders; do not edit the T. rex files):
    - Colab training with Drive checkpoints and resume: stage2/stage2_training.ipynb, designlab/stage2/nb.py,
      designlab/stage2/train2.py (--resume), designlab/stage2/vecpool.py, stage2/tb/colab_steps.template.txt,
      stage2/tb/requirements-colab.txt. Tested on a stand-in Drive only, never on Colab itself.
    - Large ignored inputs: designlab/stage2/bundle.py, tools/restore_inputs.py; files over about 3 MB cannot go
      through the Drive connector (stage2/tb/validation/drive_route.json), so the T. rex used a data branch.
    - Servo delay and motor model: designlab/stage1/speedfit.py (servo_delay), designlab/stage2/servo.py with
      stage2/design/servo_model/SERVO_MODEL_DESIGN.md; the bench sheet stage2/jetson75/BENCH_SHEET.md; the questions
      for the drive maker, stage2/design/servo_model/MAKER_QUESTIONS.md (same drive family).
    - CAN latency and bus load: stage2/jetson75/LATENCY_AND_BUS.md and latency_bus.py.
    - Prices: stage2/jetson75/BUDGET_75.md, stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md, and the catalog
      designlab/data/actuators.json and parts.json (maker pages read 7 Nov 2026).
    - Method: DECISION_LOG.md, STAGE2_75_PLAN.md, BUY_GATE_DRAFT.md (staged purchase gates), LESSONS_LEARNED.md
      sections 3 to 5, and the independent checks in stage2/design/checkpoint/reviews/.
    - Container tools: tools/README.md (keepalive.sh, memguard.sh, lowprio.sh, setup_venv.sh).
- Put new Triceratops work in a new folder (suggested: hardware/triceratops_build/), with its own README.md and
  DECISION_LOG.md. Leave PR #669's study folder as its record.

KEY FACTS (prospective or estimated unless marked; from PR #669 and the checked parts list)
- PR #669's robot: shoulders and hips 400 mm up; leg links 235 + 230 mm; feet 650 mm apart front to back and 350 mm
  side to side; 110 x 80 mm pads; passive ankles; 15.00 kg including a 1.84 kg reserve; 55 % of the weight on the
  front feet. Gait: four-beat walk, each foot down 75 % of the time, 1.5 strides per second, 0.30 m/s.
- Its weak points: the 1:1 belt (12 mm wide, 20 mm pulleys) to each elbow drive is undersized: at that 20 mm radius
  it would pull 1.3-1.8 kN (15 / 20 kg; parts list, section 1); a 6S (24 V) battery starves the 48 V X6 drive; two
  classic CAN buses with six drives each cannot run a 1 kHz loop; no heat or continuous-load sizing; the battery is
  0.755 kg, not the 0.70 kg budgeted. Only the 3 s standing check was run.
- The parts list's T20 (20 kg), per leg: roll MyActuator RMD-X6-P20-60; upper-leg pitch RMD-X4-P36-36-L; front elbow
  RMD-X8-P20-150 through a 1:1 pushrod; hind knee RMD-X6-P20-60 through a 1.3:1 pushrod. A 12S bus (two 6S 3300 mAh
  packs in series), a Jetson Orin Nano Super, a Teensy 4.1 with one CAN bus per leg at 500 Hz, an OAK-D Pro W camera,
  an Xsens MTi-3 IMU, TE FX29 foot load cells, ROBOTIS servos for the neck, beak and tail, a wireless stop and a Pilz
  relay that sit the robot down before power is cut, and a 48 V tether. Cost, landed: $22.9k / $37.3k / $66.4k (low /
  nominal / high); the T. rex way $22.3k / $31.8k / $51.2k. T15: $20.0k / $32.8k / $58.5k landed; $19.4k / $28.3k /
  $46.3k the T. rex way.
- Design loads at the front leg, 20 kg (15 kg in brackets): elbow 32.1 N·m walking peak (24.0), 35.8 N·m trotting
  peak (26.9), 16.2 N·m held "paw lift" pose, continuous (12.2); upper-leg pitch 18.6 N·m (14.0); roll 25.4 N·m in
  the trot (19.1).
- The 0.40 m/s walk is not credible with X4 pitch drives: it needs 75-77 rpm against a reported 83 rpm cap when the
  24 V X4 runs above its rating, 91-93 % of the cap, with no margin under load.
- Static balance (the PR #669 review's arithmetic, reviews/survey_pr669_18nov.md, section 2; PR #669 asks for a shift
  but gives no number): with the centre of mass on the centreline 358 mm ahead of the hind feet, the body must shift
  sideways at least 18 mm (17.8 mm), plus a margin, before a forefoot lifts.
- Nothing has been built, bought, quoted or measured. Free-base walking has never been simulated.

FIRST TASKS, in this order (task 9 has a fixed date: start it by 1 Dec 2026 whatever the state of tasks 3 to 8)
1. Review PR #669. Fetch it as above, read its files, run models/check_models.py with mujoco==3.10.0 (no CI runs it)
   and keep the output. If it is sound as a concept study, draft its errata on your own branch (or as review text for
   the owner): an errata section for its README.md, and a list for the PDF's maintainer, since there is no PDF
   source. The errata: the PDF's p.6 "Orin / MCU" label; the battery is 0.755 kg, not 0.70 kg, and the bay must be at
   least 165 mm; the 1:1 belt and the 20 kg case are not sized (parts list, sections 1 and 2.7); the T. rex pages 7-9
   are superseded by the pause of 18 Nov 2026 (T. rex decision log, rows P47 to P51). Send the draft to the owner,
   who applies it and merges the PR with a merge commit.
2. Settle the open decisions (CONSOLIDATION_PLAN.md, section 7) in two rounds. Use the owner's answers below if
   given. Record each answer with the options put.
   - Round 1, by 15 Dec 2026, five decisions: mass 20 or 15 kg; size 75 cm or another; compute; 48 V or 24 V;
     PR #669's chassis or a proven open quadruped's leg design. The question tool takes four at a time, so ask in two
     calls.
   - Round 2, by 15 Jan 2027: what the T. rex milestone shows (it decides the T. rex restart by 1 Mar 2027); a PLEM
     bench pilot (yes, gated on WIM's and MyActuator's answers, or no); the pull-request triage.
   - PLEM: in your first days, draft the questions to WIM and to MyActuator (PLEM_FIT.md, section 5.4) for the owner
     to send. A PLEM pilot runs only after their answers clear the three blockers; its go or no-go is the owner's, by
     end Feb 2027, and it stays off the critical path.
3. MuJoCo, standing: rebuild the robot model for the chosen build with its real mass distribution (part masses and
   positions from PARTS_PREMIUM.md section 6.2, no reserve block standing in for them), the drives' limits and the
   transmission ratios. Check four-foot standing, the weight share at each foot (design point 55 % front), and the
   joint torques against the parts list's figures.
4. A statically stable crawl with a scripted gait: one foot lifted at a time, the body shifted inside the support
   triangle first, the margin pre-registered.
5. A walk at 0.30 m/s with servo delay (0, 2, 4, 6, 8 and 10 ms; reuse the T. rex delay and servo models) and noise
   (friction, gear play, voltage sag, sensor noise), on grippy and slippery floors. PR #669's gates: two independently
   built controllers, each upright in 20 of 20 runs and within every limit in at least 17 of 20; a 90° turn, a
   two-step stop, and pushes from four directions at 4 ms. Pre-register which delay the verdict is judged at (the
   T. rex used 10 ms until a bench value exists) and ask the owner.
6. A trot at 0.50 to 0.60 m/s, in short bouts.
7. Actuator sizing confirmed in simulation: logged peak torque, speed and RMS torque per joint against the parts
   list's design figures and each drive's limits (70 % of the maker's peak torque; the maker's thermal-equilibrium
   torque; the speed caps). Flag any axis over.
8. A bench plan for one foreleg (roll, pitch, elbow), with pass and fail rules written first: torque at 10 rpm and
   stall at 42 and 49 V; which X6 variant ships; heat over a 30-minute show loop with the shell on; the X4's speed on
   12S; the 99th-percentile command-to-torque delay with twelve drives' traffic emulated at 500 Hz and 1 kHz;
   transmission stiffness and play; the regen clamp; the stop chain's sit-then-cut timing. Add the T. rex bench
   sheet's delay rows, because the drives are the same. If the owner approves a PLEM pilot after WIM's and
   MyActuator's answers, add its two-drive and one-foreleg steps (PLEM_FIT.md, section 5.3) beside the CAN path. No
   purchase without the owner.
9. Renders for the December museum contact, ready by 8 Dec 2026: stills and a short clip of the robot standing (and
   crawling or walking if ready) and of the anatomical adult, each labelled "simulation", plus one line saying the
   T. rex is the flagship for end 2027 / early 2028. If the crawl is not ready by 5 Dec, render the standing robot.
   Commit the stills, each under the 500 KB limit. A video clip will likely exceed it: ask the owner where clips go
   before committing one. Send the renders to the owner.

WORKING RULES
- Git: work and push on the branch this session was given. Small commits with clear messages; push after each commit
  and before every turn ends. Never force-push or rewrite history. End commit messages with the attribution lines your
  session's system reminder gives, and name no AI model anywhere else.
- Independent checks: every build (model, controller, script) and every report is checked by an independent reviewer,
  given the code and the records rather than your conclusion. Fix, then check again.
- Pre-registration: commit the pass and fail rules, seeds, budgets and stop rules before any run that judges
  something. Label any rule changed after a result "changed after a result". A verdict needs two seeds or two
  controllers; with one it is provisional.
- Keep prospective design (targets, estimates), simulation evidence and measured results apart, in words and in
  files. Never present a simulation result as a hardware result, or a target as an achievement.
- Never report a pass you have not seen in a tool's own output. Report a count only when every check behind it has
  returned.
- Give every price and specification its source and the date read; mark estimates as estimates.
- Web searches carry only generic part and engineering terms, never the owner's or the project's details.
- Before long runs, Colab use or large multi-agent work, say what it will cost and ask the owner; the account has a
  weekly usage limit.

WHAT NOT TO DO
- Do not buy, order or request quotes, and do not contact vendors, museums, MyActuator or WIM Robotics; draft
  messages for the owner instead.
- Do not touch the T. rex: not its files, not its working branch, not its two data branches (endings
  `mesozoic-trex-stage2-bundle-s2-b1`, which must stay at ecf95a81, and `mesozoic-trex-stage2-archive-a1`, at
  f71cba4b), not its launch commit cc437b6a. Do not resume T. rex training.
- Do not push to PR #669's branch or to any other pull request's branch, and do not merge, close or comment on any
  pull request, unless the owner says so in this session. For PR #669 you draft the errata (task 1); the owner
  applies them and merges. Do not delete branches. The other open pull requests are the owner's to triage
  (CONSOLIDATION_PLAN.md, section 1): the T. rex ones #585, #596, #623, #635, #648, #649, #651, #655, #662, #663 and
  #664, and the MicroDuck design notes #646; so is the T. rex study pull request once it is opened.
- Do not adopt PLEM, change the compute or the bus voltage, or put a PLEM pilot on the critical path, without the
  owner; and run no PLEM pilot before WIM's and MyActuator's answers.
- Do not treat 0.40 m/s as a walking target; the museum walk is 0.30 m/s.
- Do not commit files over 500 KB without need (a pre-commit limit), and check new outputs with `git check-ignore -v`
  (the root rules ignore logs/, models/, *.zip, *.pkl and *.npz at any depth: PR #669's models/ files are tracked,
  but a new file in a folder named models/ is silently left out unless force-added, so prefer another folder name).

OWNER'S ANSWERS (optional; fill in before pasting, or delete):
- Mass (20 or 15 kg):
- Size:
- Compute:
- Bus voltage (48 V / 12S or 24 V / 6S):
- Chassis (PR #669's design or a proven open leg design):
- What the T. rex milestone shows:
- PLEM bench pilot (and whether the questions to WIM and MyActuator were sent):
```
