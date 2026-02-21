# Website Improvement Plan

## Priority 1: Landing Page — Implement Already-Designed Sections

The CSS in `index.module.css` already defines styles for features, simulation preview, roadmap, and CTA sections — but `index.tsx` only renders the hero. These sections need to be built out in the React component.

### 1a. Features Section
- Add a `FeaturesSection` component to `index.tsx` showcasing 4 key features:
  - **MuJoCo Physics** — Accurate dinosaur biomechanics simulation
  - **Reinforcement Learning** — PPO & SAC curriculum training
  - **3 Species** — T-Rex, Velociraptor, Brachiosaurus
  - **Open Source** — MIT licensed, extensible framework
- CSS classes already exist: `.featuresSection`, `.featuresGrid`, `.featureCard`, `.featureCardBorder`, `.featureCorner`, etc.

### 1b. Simulation Preview Section
- Add a `SimulationSection` component with a styled terminal window showing a real code snippet (e.g., creating an env, running a step, seeing output)
- Use actual project API: `gym.make("MesozoicLabs/Raptor-v0")`, `env.step()`, etc.
- CSS classes already exist: `.simulationSection`, `.terminalWindow`, `.terminalHeader`, `.terminalBody`, `.codeLine`, `.codeKeyword`, etc.

### 1c. Roadmap Section
- Add a `RoadmapSection` component showing the 6 phases as milestone cards
- Phase 0 (v0.2.0): Complete
- Phase 1 (v0.3.0): Active/In Progress
- Phases 2-6: Upcoming
- CSS classes already exist: `.roadmapSection`, `.milestoneCard`, `.milestoneCard.complete`, `.milestoneCard.active`, `.milestoneCard.upcoming`, etc.

### 1d. CTA Section
- Add a `CTASection` with a "View on GitHub" button linking to `https://github.com/kuds/mesozoic-labs`
- CSS classes already exist: `.ctaSection`, `.ctaButton`, `.ctaDinoTrack`, etc.

### 1e. Remove "COMING SOON" Badge
- The project has working code, 3 environments, docs, and training infrastructure — it is not "coming soon"
- Remove the `.comingSoonBadge` from the hero and update the page `<title>` from "Coming Soon" to something like "Robotic Dinosaur Locomotion"

### 1f. Use the Training GIFs
- `static/img/ppo_apex.gif` and `static/img/sac_apex.gif` exist but are unused
- Embed at least one in the hero or simulation section to give visitors an immediate visual of what the project does

---

## Priority 2: Documentation Accuracy Fixes

### 2a. Fix CHANGELOG Version Status
- `CHANGELOG.md` line 8: `[0.2.0] - Unreleased` → update to a release date (or remove "Unreleased" since docs already describe v0.2.0 as current)

### 2b. Fix `editUrl` in `docusaurus.config.ts`
- Line 42: points to `mesozoic-labs/mesozoic-labs.github.io` but actual code is at `kuds/mesozoic-labs`
- Line 50: same issue for blog editUrl
- Update both to `https://github.com/kuds/mesozoic-labs/tree/main/website/`

### 2c. Fix GitHub Discussions Link
- `docusaurus.config.ts` line 104: points to `mesozoic-labs/mesozoic-labs.github.io/discussions`
- Update to `https://github.com/kuds/mesozoic-labs/discussions`

### 2d. Update Privacy Policy Date
- `docs/privacy.md` line 8: "Last updated: December 2024" → update to "February 2026"

### 2e. Clarify "Basic Dinosaur" in Results Tables
- `docs/intro.md`, `docs/training/ppo.md`, `docs/training/sac.md` all reference "Basic Dinosaur" training results without context
- Either clarify what "Basic Dinosaur" refers to (e.g., "Velociraptor Stage 2") or add the per-species results from README alongside

---

## Priority 3: Fill or Remove Incomplete Doc Placeholders

### 3a. Custom Models Page (`docs/models/custom-models.md`)
- Currently just a skeleton XML snippet and "Coming Soon"
- Either flesh out with real guidance (using `BaseDinoEnv`, required observation/action space structure, MJCF tips from the existing 3 species) or remove from sidebar until ready

### 3b. PPO Training Page (`docs/training/ppo.md`)
- Add actual hyperparameter guidance from the TOML configs (learning rate schedules, n_steps, batch_size)
- Document the 3-stage curriculum approach with PPO
- Add per-species results (Velociraptor PPO: 118.37 avg reward, 6M steps)

### 3c. SAC Training Page (`docs/training/sac.md`)
- Same treatment: add TOML config details, curriculum integration, per-species results

### 3d. Hyperparameters Page (`docs/training/hyperparameters.md`)
- Currently 4 rows and 4 bullet tips
- Expand with the actual TOML config parameters, per-stage tuning guidance, and links to the config files

---

## Priority 4: Minor Design & Performance Improvements

### 4a. Optimize Logo SVG
- `static/img/logo.svg` is 32K+ tokens — this is extremely large for an SVG
- Run through SVGO or manually clean up (remove editor metadata, simplify paths)
- This affects page load performance

### 4b. Footer Cleanup
- The "Finding Theta" link in the footer has no context for visitors — either add a description or remove it
- Add more useful links (e.g., link to API Reference, link to Roadmap, link to Training docs)

### 4c. Intro Page Results Table
- `docs/intro.md` shows only "Basic Dinosaur" results — add the Velociraptor PPO result (118.37 avg reward) and note which species/stages are represented

---

## Implementation Order

| Step | Task | Files Changed |
|------|------|---------------|
| 1 | Build FeaturesSection, SimulationSection, RoadmapSection, CTASection in `index.tsx` | `website/src/pages/index.tsx` |
| 2 | Remove "COMING SOON" badge, update page title | `website/src/pages/index.tsx` |
| 3 | Embed training GIF in landing page | `website/src/pages/index.tsx` |
| 4 | Fix `editUrl` and GitHub Discussions link | `website/docusaurus.config.ts` |
| 5 | Update CHANGELOG version status | `CHANGELOG.md` |
| 6 | Update privacy policy date | `website/docs/privacy.md` |
| 7 | Clarify/update results tables across docs | `website/docs/intro.md`, `docs/training/ppo.md`, `docs/training/sac.md` |
| 8 | Flesh out training docs (PPO, SAC, hyperparameters) | `website/docs/training/*.md` |
| 9 | Flesh out or remove custom models placeholder | `website/docs/models/custom-models.md` |
| 10 | Optimize logo SVG | `website/static/img/logo.svg` |
| 11 | Clean up footer links | `website/docusaurus.config.ts` |
