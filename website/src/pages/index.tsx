import React, { useEffect, useRef, useState } from 'react';
import Layout from '@theme/Layout';
import {
  ALL_MODELS,
  PROJECT_CAPABILITIES,
  backendLabel,
  headlineFor,
  posterFor,
  requireModel,
  TOTAL_ACTUATORS,
  TOTAL_TRAINED_STEPS,
} from '@site/src/data/species';
import styles from './index.module.css';

/* =============================================================================
   SCROLL ANIMATION HOOK
   ============================================================================= */

function useScrollReveal() {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    // Sections are VISIBLE by default; the hidden "pending" state is added
    // here (after first paint) and only for sections below the fold — so
    // no-JS visitors, printing, and full-page rendering always see content.
    if (el.getBoundingClientRect().top >= window.innerHeight) {
      el.classList.add(styles.scrollRevealPending);
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.remove(styles.scrollRevealPending);
          el.classList.add(styles.revealed);
          observer.unobserve(el);
        }
      },
      // Pre-reveal 150px before the section enters so fast scrolling
      // doesn't flash blank sections.
      { threshold: 0.05, rootMargin: '0px 0px 150px 0px' },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return ref;
}

function formatStatus(status: string): string {
  return status.replaceAll('_', ' ');
}

function capabilitySummary(id: string): string {
  const capability = PROJECT_CAPABILITIES[id];
  return `${capability.label} — ${formatStatus(capability.status)}`;
}

const repo = 'https://github.com/kuds/mesozoic-labs';
const conceptStudy = `${repo}/pull/669`;
const buildStudy = `${repo}/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build`;
const hardwareRoadmap = `${repo}/blob/4b86f5c6fec8d1b427ba86c1c9a7cc11855ec875/hardware/trex_form_study/DECISION_LOG.md`;

const studyViews = [
  {
    label: '60 cm gait trial', src: '/img/triceratops-gait.jpg', width: 960, height: 596,
    alt: 'A simulated four-legged robot with a green frill and three horns crossing a checkered floor',
    caption: 'Learned controller, six seconds into a simulated gait trial.',
    href: `${buildStudy}/rl/analysis/REPORT_GAIT_PIPE1.md`, link: 'Gait analysis',
  },
  {
    label: '75 cm concept', src: '/img/triceratops-study.webp', width: 1650, height: 930,
    alt: 'Four views comparing the adult anatomy, earlier robot concept, skeletal landmarks and foreleg assembly',
    caption: 'Earlier design study. The anatomy and robot views use independent scales.',
    href: conceptStudy, link: 'Design study',
  },
];

function HeroSection() {
  const [viewIndex, setViewIndex] = useState(0);
  const view = studyViews[viewIndex];
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroSpread}>
        <div className={styles.heroCopy}>
          <p className={styles.kicker}>Mesozoic Labs / Locomotion research</p>
          <h1>How does a<br /><em>Triceratops</em><br />walk?</h1>
          <p className={styles.heroIntro}>
            We study dinosaur-inspired locomotion in MuJoCo. Follow our 60 cm robot from its first simulated steps toward a planned museum demo.
          </p>
          <div className={styles.heroActions}>
            <a className={styles.primaryLink} href="#research">Explore the research</a>
            <a className={styles.textLink} href="/docs">Get started</a>
          </div>
        </div>
        <figure className={styles.studyPlate}>
          <div className={styles.plateHeader} role="group" aria-label="Triceratops model views">
            {studyViews.map((option, index) => (
              <button type="button" key={option.label} aria-pressed={viewIndex === index}
                onClick={() => setViewIndex(index)}>{option.label}</button>
            ))}
          </div>
          <div className={styles.robotView}>
            <img src={view.src} width={view.width} height={view.height} alt={view.alt} />
          </div>
          <figcaption className={styles.plateCaption} aria-live="polite">
            <span>{view.caption}</span>
            <a href={view.href}>{view.link}</a>
          </figcaption>
        </figure>
      </div>
    </header>
  );
}

const researchNotes = [
  {
    label: 'Design study',
    title: 'Four feet on the ground',
    body: 'An earlier 75 cm concept stands on all four feet for three seconds in MuJoCo. The study pairs the robot with an adult anatomical model to show how the proportions change.',
    href: conceptStudy,
    link: 'Inspect the models',
  },
  {
    label: 'Flat-floor trials',
    title: 'Forty trots without a fall',
    body: 'A model-based controller carried the 60 cm robot across four simulated floor conditions at 0.247–0.254 m/s. A separate 5° ramp test cut its speed by 27%.',
    href: `${buildStudy}/walk/REPORT_WALK60.md`,
    link: 'Read the trial results',
  },
  {
    label: 'Hardware roadmap',
    title: 'Triceratops leads the way',
    body: 'The lab is targeting a live Triceratops demo in Q3 2027. The T. rex form study is paused, with a flagship milestone targeted for late 2027 or early 2028.',
    href: hardwareRoadmap,
    link: 'Read decision P51',
  },
];

function ResearchSection() {
  return (
    <section className={styles.research} id="research" aria-labelledby="research-heading">
      <div className={styles.researchHeader}>
        <h2 id="research-heading">Research notes</h2>
        <time dateTime="2026-11">November 2026</time>
      </div>
      <div className={styles.notes}>
        {researchNotes.map((note) => (
          <article key={note.title}>
            <p className={styles.kicker}>{note.label}</p>
            <h3>{note.title}</h3>
            <p>{note.body}</p>
            <a className={styles.textLink} href={note.href}>{note.link}</a>
          </article>
        ))}
      </div>
    </section>
  );
}

/* =============================================================================
   HOW IT WORKS (Curriculum Pipeline)
   ============================================================================= */

// Stage colors reference the palette variables from index.module.css so
// they follow the light/dark theme (hex literals here would not).
const curriculumSteps = [
  {
    stage: 1,
    title: 'Balance',
    subtitle: 'Stand & Stabilize',
    description: 'The agent learns to stand upright and maintain balance using proprioceptive feedback and joint torques.',
    color: 'var(--color-primary)',
    glow: 'var(--color-primary-dim)',
  },
  {
    stage: 2,
    title: 'Locomotion',
    subtitle: 'Walk & Run',
    description: 'Building on balance skills, the agent develops forward locomotion with coordinated movement patterns.',
    color: 'var(--color-accent)',
    glow: 'var(--color-accent-dim)',
  },
  {
    stage: 3,
    title: 'Behavior',
    subtitle: 'Species-Specific Task',
    description:
      'Species-specific simulator tasks add sickle-claw contact, a fixed head-contact proxy, or head-tip target proximity.',
    color: 'var(--color-secondary)',
    glow: 'rgba(var(--color-secondary-rgb), 0.25)',
  },
];

function HowItWorksSection() {
  const ref = useScrollReveal();
  return (
    <section
      className={`${styles.howItWorksSection} ${styles.scrollReveal}`}
      aria-labelledby="howitworks-heading"
      ref={ref}
    >
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ CURRICULUM ]'}</div>
        <h2 className={styles.sectionTitle} id="howitworks-heading">How It Works</h2>
        <p className={styles.sectionSubtitle}>
          The lab trains stance, locomotion and species-specific behaviors in MuJoCo with reinforcement learning. Each behavior has its own success criteria.
        </p>
      </div>
      <div className={styles.pipelineContainer}>
        <div className={styles.pipelineLine} aria-hidden="true"></div>
        {curriculumSteps.map((step, idx) => (
          <div className={styles.pipelineStep} key={idx}>
            <div
              className={styles.pipelineNode}
              style={{ borderColor: step.color, boxShadow: `0 0 20px ${step.glow}` }}
              aria-hidden="true"
            >
              <span style={{ color: step.color }}>{step.stage}</span>
            </div>
            <div className={styles.pipelineContent}>
              <div className={styles.pipelineStage} style={{ color: step.color }}>
                STAGE {step.stage}
              </div>
              <h3 className={styles.pipelineTitle}>{step.title}</h3>
              <div className={styles.pipelineSubtitle}>{step.subtitle}</div>
              <p className={styles.pipelineDescription}>{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/* =============================================================================
   SIMULATION PREVIEW SECTION
   ============================================================================= */

/** The model the code sample creates; its observation width is printed from the catalog. */
const VELOCIRAPTOR_MONGOLIENSIS_ANATOMICAL = requireModel('velociraptor_mongoliensis', 'anatomical');

function SimulationSection() {
  const ref = useScrollReveal();
  return (
    <section
      className={`${styles.simulationSection} ${styles.scrollReveal}`}
      aria-labelledby="simulation-heading"
      ref={ref}
    >
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ SIMULATION ]'}</div>
        <h2 className={styles.sectionTitle} id="simulation-heading">See It In Action</h2>
      </div>
      <div className={styles.simulationContainer}>
        <div className={styles.simulationColumns}>
          <div className={styles.terminalWindow} role="img" aria-label="Python code example showing how to create a Velociraptor environment with Gymnasium and run a training step">
            <div className={styles.terminalHeader} aria-hidden="true">
              <span className={styles.terminalDot} style={{ background: '#ff5f56' }}></span>
              <span className={styles.terminalDot} style={{ background: '#ffbd2e' }}></span>
              <span className={styles.terminalDot} style={{ background: '#27c93f' }}></span>
              <span className={styles.terminalTitle}>mesozoic-labs</span>
            </div>
            <div className={styles.terminalBody} aria-hidden="true">
              <div className={styles.codeLine}>
                <span className={styles.codeComment}>{'# Create a Velociraptor environment'}</span>
              </div>
              <div className={styles.codeLine}>
                <span className={styles.codeKeyword}>import </span>
                <span>gymnasium </span>
                <span className={styles.codeKeyword}>as </span>
                <span>gym</span>
              </div>
              <div className={styles.codeLine}>
                <span className={styles.codeKeyword}>import </span>
                <span>environments</span>
              </div>
              <div className={styles.codeLine}>&nbsp;</div>
              <div className={styles.codeLine}>
                <span>env = gym.make(</span>
              </div>
              <div className={styles.codeLine}>
                <span>{'    '}</span>
                <span className={styles.codeString}>"MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0"</span>
              </div>
              <div className={styles.codeLine}>
                <span>)</span>
              </div>
              <div className={styles.codeLine}>
                <span>obs, info = env.reset()</span>
              </div>
              <div className={styles.codeLine}>&nbsp;</div>
              <div className={styles.codeLine}>
                <span className={styles.codeComment}>{'# Run a training step'}</span>
              </div>
              <div className={styles.codeLine}>
                <span>action = env.action_space.sample()</span>
              </div>
              <div className={styles.codeLine}>
                <span>obs, reward, done, trunc, info = env.step(action)</span>
              </div>
              <div className={styles.codeLine}>&nbsp;</div>
              <div className={styles.codeLine}>
                <span className={styles.codeKeyword}>print</span>
                <span>(</span>
                <span className={styles.codeString}>"Obs shape:"</span>
                <span>, obs.shape)</span>
              </div>
              <div className={styles.codeLine}>
                <span className={styles.codeOutput}>{`>>> Obs shape: (${VELOCIRAPTOR_MONGOLIENSIS_ANATOMICAL.observationDim},)`}</span>
              </div>
              <div className={styles.codeLine}>
                <span className={styles.cursor}>_</span>
              </div>
            </div>
          </div>
          <figure className={styles.previewPane}>
            <div className={styles.previewLabel}>PPO Stage 1 — Balance</div>
            <img
              src="/img/velociraptor_mongoliensis_anatomical_stance_ppo.gif"
              alt="Velociraptor learning to balance using PPO reinforcement learning"
              className={styles.previewGif}
            />
            <figcaption className={styles.previewCaption}>
              Historical, unverified PPO / Stable-Baselines3 artifact; backend version not recorded. Not evidence for
              the current model or configuration.
            </figcaption>
          </figure>
        </div>
      </div>
    </section>
  );
}

/* =============================================================================
   SPECIES SHOWCASE (Tabbed)
   ============================================================================= */

// Model/stage/video data lives in src/data/species.ts (shared with the
// model docs pages so the two can't drift apart).

/** A tab and panel id for a model, built from its species ID and embodiment (decision D-D33). */
function modelKey(model: {species: string; embodiment: string}): string {
  return `${model.species}-${model.embodiment}`;
}

function SpeciesShowcase() {
  const [activeSpecies, setActiveSpecies] = useState(0);
  const ref = useScrollReveal();
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const species = ALL_MODELS[activeSpecies];
  const result = species.featuredResult;

  // WAI-ARIA tabs keyboard support: arrows move + activate, Home/End jump.
  const onTabKeyDown = (e: React.KeyboardEvent, idx: number) => {
    const last = ALL_MODELS.length - 1;
    let next: number | null = null;
    if (e.key === 'ArrowRight') next = idx === last ? 0 : idx + 1;
    else if (e.key === 'ArrowLeft') next = idx === 0 ? last : idx - 1;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = last;
    if (next !== null) {
      e.preventDefault();
      setActiveSpecies(next);
      tabRefs.current[next]?.focus();
    }
  };

  return (
    <section
      className={`${styles.speciesSection} ${styles.scrollReveal}`}
      aria-labelledby="species-heading"
      ref={ref}
    >
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ SPECIES ]'}</div>
        <h2 className={styles.sectionTitle} id="species-heading">Published Run Summaries</h2>
        <p className={styles.sectionSubtitle}>
          Explore {ALL_MODELS.length} models with {TOTAL_ACTUATORS} actuators in total and {Math.round(TOTAL_TRAINED_STEPS / 1e6)} million published training steps. Select a model to inspect its recorded runs and videos.
        </p>
      </div>

      <div className={styles.speciesTabs} role="tablist" aria-label="Select model">
        {ALL_MODELS.map((sp, idx) => (
          <button
            key={modelKey(sp)}
            ref={(el) => {
              tabRefs.current[idx] = el;
            }}
            className={`${styles.speciesTab} ${idx === activeSpecies ? styles.speciesTabActive : ''}`}
            onClick={() => setActiveSpecies(idx)}
            onKeyDown={(e) => onTabKeyDown(e, idx)}
            role="tab"
            tabIndex={idx === activeSpecies ? 0 : -1}
            aria-selected={idx === activeSpecies}
            aria-controls={`species-panel-${modelKey(sp)}`}
          >
            <span className={styles.speciesTabName}>{sp.name}</span>
            <span className={styles.speciesTabMeta}>{sp.actuators} actuators &middot; {sp.gait}</span>
          </button>
        ))}
      </div>

      <div
        className={styles.speciesPanel}
        role="tabpanel"
        id={`species-panel-${modelKey(species)}`}
        aria-label={species.name}
      >
        <div className={styles.speciesInfo}>
          <span className={styles.speciesBadge}>{species.specialty}</span>
          {result ? (
            <>
              <dl
                className={styles.speciesStats}
                aria-label={`${species.name} historical run summary (${result.algorithm}, ${result.date})`}
              >
                <div className={styles.speciesStat}>
                  {/* A ladder summary headlines "Task success" exactly as before;
                      a schema-4 result headlines its certified primary
                      deliverable's gate-kind metric (decisions D-A9, D-A10). */}
                  <dt>{headlineFor(result).label}</dt>
                  <dd>{headlineFor(result).value}</dd>
                </div>
                <div className={styles.speciesStat}>
                  <dt>Max reported avg. velocity</dt>
                  <dd>
                    {result.maxAverageForwardVelocity === null
                      ? '—'
                      : `${result.maxAverageForwardVelocity.toFixed(2)} m/s`}
                  </dd>
                </div>
                <div className={styles.speciesStat}>
                  <dt>Timesteps</dt>
                  <dd>{Math.round(result.totalTimesteps / 1e6)}M</dd>
                </div>
                <div className={styles.speciesStat}>
                  <dt>Run</dt>
                  <dd>{result.algorithm} &middot; {backendLabel(result.backend)} &middot; {result.date}</dd>
                </div>
              </dl>
              <p className={styles.speciesProvenance}>
                {formatStatus(result.provenance.modelRevisionStatus)} model &middot;{' '}
                {formatStatus(result.provenance.verificationStatus)} &middot;{' '}
                {result.provenance.evaluationEpisodes === null
                  ? 'evaluation episode count not recorded'
                  : `${result.provenance.evaluationEpisodes} evaluation episodes`} &middot;{' '}
                {result.backendVersion === null
                  ? `${backendLabel(result.backend)} version not recorded`
                  : `${backendLabel(result.backend)} ${result.backendVersion}`}. Metrics are retained experiment
                records, not claims about the current model revision.
              </p>
            </>
          ) : (
            <p className={styles.speciesProvenance}>No run summary has been published for this model.</p>
          )}
        </div>

        <div className={styles.speciesVideos}>
          {/* Cards are keyed and labelled by the manifest stage id/label, never
              the legacy number, which is null for a semantic-only stage such as
              recovery and collided as a React key (decision D-A13). */}
          {species.stages.map((stage) => (
            <div className={styles.speciesVideoCard} key={stage.id}>
              <div className={styles.speciesVideoHeader}>
                <span className={styles.speciesStageLabel}>STAGE {stage.label.toUpperCase()}</span>
                <span className={styles.speciesAlgoBadge}>
                  {stage.video
                    ? `${stage.video.algorithm} · ${backendLabel(stage.video.backend)} · ${stage.video.modelRevisionStatus}`
                    : 'NO VIDEO'}
                </span>
              </div>
              <h3 className={styles.speciesStageTitle}>{stage.title}</h3>
              <p className={styles.speciesStageDesc}>{stage.description}</p>
              {stage.video ? (
                <>
                  <p className={styles.speciesVideoProvenance}>
                    {stage.video.verificationStatus} artifact;{' '}
                    {stage.video.backendVersion === null
                      ? 'backend version not recorded'
                      : `${backendLabel(stage.video.backend)} ${stage.video.backendVersion}`}; not evidence for the
                    current model or config.
                  </p>
                  <div className={styles.speciesVideoWrap}>
                    <video
                      src={stage.video.path}
                      controls
                      preload="none"
                      poster={posterFor(stage.video.path)}
                      className={styles.speciesVideoPlayer}
                      aria-label={`${species.name} stage ${stage.label}: ${stage.title}`}
                    />
                  </div>
                </>
              ) : (
                <div className={styles.speciesVideoUnavailable} role="status">
                  No published video is available for this stage.
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* =============================================================================
   ROADMAP SECTION (Timeline)
   ============================================================================= */

const milestones = [
  {
    phase: 'PHASE 0 — v0.2.0',
    title: 'Clean Slate',
    status: 'complete' as const,
    statusLabel: 'COMPLETE',
    items: ['TOML configs', 'Gymnasium registration', 'Developer tooling', 'Testing improvements'],
  },
  {
    phase: 'PHASE 1 — v0.3.0',
    title: 'First Steps',
    status: 'active' as const,
    statusLabel: 'IN PROGRESS',
    items: ['Curriculum manager', 'W&B tracking', 'Locomotion metrics', 'Species training runs'],
  },
  {
    phase: 'PHASE 2 — v0.4.0',
    title: 'Into the Wild',
    status: 'active' as const,
    statusLabel: 'IN PROGRESS',
    items: [capabilitySummary('domain_randomization'), 'Terrain diversity — piloted', 'Turning & steering — piloted'],
  },
  {
    phase: 'PHASE 3 — v0.5.0',
    title: 'Evolution',
    status: 'active' as const,
    statusLabel: 'IN PROGRESS',
    items: ['Custom policy networks', 'New species — 2 of 4 delivered', 'Benchmark suite'],
  },
  {
    phase: 'PHASE 4 — v0.6.0',
    title: 'The Pack',
    status: 'upcoming' as const,
    statusLabel: 'PLANNED',
    items: ['Multi-agent envs', 'Cooperative hunting', 'Predator-prey'],
  },
  {
    phase: 'PHASE 6',
    title: 'Sim-to-Real',
    status: 'upcoming' as const,
    statusLabel: 'PLANNED',
    items: [
      capabilitySummary('hardware_prototype'),
      capabilitySummary('ros2_bridge'),
      capabilitySummary('sim_to_real_validation'),
    ],
  },
];

function RoadmapSection() {
  const ref = useScrollReveal();
  return (
    <section
      className={`${styles.roadmapSection} ${styles.scrollReveal}`}
      aria-labelledby="roadmap-heading"
      ref={ref}
    >
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ ROADMAP ]'}</div>
        <h2 className={styles.sectionTitle} id="roadmap-heading">Simulation platform roadmap</h2>
      </div>
      <div className={styles.timelineContainer} role="list">
        <div className={styles.timelineLine} aria-hidden="true"></div>
        {milestones.map((milestone, idx) => (
          <article
            className={`${styles.timelineItem} ${styles[milestone.status]}`}
            key={idx}
            role="listitem"
          >
            <div
              className={styles.timelineDot}
              aria-hidden="true"
            ></div>
            <div className={styles.timelineCard}>
              <div className={styles.milestonePhase}>{milestone.phase}</div>
              <h3 className={styles.milestoneTitle}>{milestone.title}</h3>
              <ul className={styles.milestoneItems} aria-label={`${milestone.title} items`}>
                {milestone.items.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
              <span className={styles.milestoneStatus} aria-label={`Status: ${milestone.statusLabel}`}>
                {milestone.statusLabel}
              </span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

/* =============================================================================
   CTA SECTION
   ============================================================================= */

function CTASection() {
  return (
    <section className={styles.ctaSection} aria-labelledby="cta-heading">
      <div className={styles.ctaDinoTrack} aria-hidden="true"></div>
      <div className={styles.ctaContent}>
        <h2 className={styles.ctaTitle} id="cta-heading">Ready to Train a Dinosaur-Inspired Agent?</h2>
        <p className={styles.ctaDescription}>
          Explore the docs, train your first model, or contribute to the project.
        </p>
        <div className={styles.ctaButtons}>
          <a href="/docs" className={styles.ctaButton}>
            GET STARTED
          </a>
          <a
            href="https://github.com/kuds/mesozoic-labs"
            className={styles.ctaButtonSecondary}
            target="_blank"
            rel="noopener noreferrer"
          >
            VIEW ON GITHUB
            <span className="sr-only"> (opens in new tab)</span>
          </a>
        </div>
      </div>
    </section>
  );
}

/* =============================================================================
   MAIN PAGE
   ============================================================================= */

export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="Dinosaur-Inspired Locomotion Research"
      description="Mesozoic Labs - Open-source dinosaur-inspired locomotion research with MuJoCo simulation and reinforcement learning"
    >
      <a className="skip-nav" href="#main-content">
        Skip to main content
      </a>
      <main className={styles.main} id="main-content">
        <HeroSection />
        <div className={styles.sectionDivider} aria-hidden="true"></div>
        <ResearchSection />
        <div className={styles.sectionDivider} aria-hidden="true"></div>
        <HowItWorksSection />
        <div className={styles.sectionDivider} aria-hidden="true"></div>
        <SimulationSection />
        <div className={styles.sectionDivider} aria-hidden="true"></div>
        <SpeciesShowcase />
        <div className={styles.sectionDivider} aria-hidden="true"></div>
        <RoadmapSection />
        <div className={styles.sectionDivider} aria-hidden="true"></div>
        <CTASection />
      </main>
    </Layout>
  );
}
