import React, { useEffect, useRef, useState } from 'react';
import Layout from '@theme/Layout';
import styles from './index.module.css';

/* =============================================================================
   SCROLL ANIMATION HOOK
   ============================================================================= */

function useScrollReveal() {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add(styles.revealed);
          observer.unobserve(el);
        }
      },
      { threshold: 0.15 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return ref;
}

/* =============================================================================
   HERO SECTION
   ============================================================================= */

function HeroSection() {
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroOverlay} aria-hidden="true"></div>
      <div className={styles.gridLines} aria-hidden="true"></div>
      <div className={styles.scanLines} aria-hidden="true"></div>
      <div className={styles.heroParticles} aria-hidden="true">
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={i}
            className={styles.particle}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 6}s`,
              animationDuration: `${4 + Math.random() * 4}s`,
            }}
          />
        ))}
      </div>

      <div className={styles.heroContent}>
        <div className={styles.logoContainer}>
          <div className={styles.logoGlow} aria-hidden="true"></div>
          <div className={styles.logo} role="img" aria-label="Mesozoic Labs">
            <span className={styles.logoDino}>MESOZOIC</span>
            <span className={styles.logoLabs}>LABS</span>
          </div>
        </div>

        <div className={styles.taglineContainer}>
          <div className={styles.circuitLine} aria-hidden="true"></div>
          <h2 className={styles.tagline}>
            <span className={styles.taglineWord}>PREHISTORIC</span>
            <span className={styles.taglineDivider} aria-hidden="true">//</span>
            <span className={styles.taglineWord}>ROBOTIC</span>
            <span className={styles.taglineDivider} aria-hidden="true">//</span>
            <span className={styles.taglineWord}>INTELLIGENCE</span>
          </h2>
          <div className={styles.circuitLine} aria-hidden="true"></div>
        </div>

        <p className={styles.description}>
          Open-source platform for building robotic dinosaurs
          <br />
          powered by physics simulation and reinforcement learning
        </p>

        <div className={styles.heroButtons}>
          <a href="/docs/" className={styles.heroPrimaryBtn}>
            GET STARTED
            <span className={styles.btnArrow} aria-hidden="true">&rarr;</span>
          </a>
          <a
            href="https://github.com/kuds/mesozoic-labs"
            className={styles.heroSecondaryBtn}
            target="_blank"
            rel="noopener noreferrer"
          >
            VIEW ON GITHUB
            <span className="sr-only"> (opens in new tab)</span>
          </a>
        </div>

        <dl className={styles.statsContainer}>
          <div className={styles.statItem}>
            <dt className={styles.statLabel}>SPECIES</dt>
            <dd className={styles.statNumber}>3</dd>
          </div>
          <div className={styles.statDivider} aria-hidden="true"></div>
          <div className={styles.statItem}>
            <dt className={styles.statLabel}>TRAINING STAGES</dt>
            <dd className={styles.statNumber}>3</dd>
          </div>
        </dl>
      </div>

      <div className={styles.heroScrollIndicator} aria-hidden="true">
        <span className={styles.scrollChevron}></span>
      </div>
    </header>
  );
}

/* =============================================================================
   FEATURES SECTION
   ============================================================================= */

const FeatureIcons = {
  physics: (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#00ff88" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
      <circle cx="12" cy="12" r="8" strokeDasharray="4 2" opacity="0.4" />
    </svg>
  ),
  brain: (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#00ff88" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a5 5 0 0 1 4.5 2.8A4 4 0 0 1 20 9a4 4 0 0 1-1.5 3.1A4.5 4.5 0 0 1 16 17H8a4.5 4.5 0 0 1-2.5-4.9A4 4 0 0 1 4 9a4 4 0 0 1 3.5-4.2A5 5 0 0 1 12 2z" />
      <path d="M12 2v15" opacity="0.4" />
      <path d="M8 9h8" opacity="0.4" />
      <path d="M9 13h6" opacity="0.4" />
      <circle cx="12" cy="20" r="2" />
      <line x1="12" y1="17" x2="12" y2="18" />
    </svg>
  ),
  species: (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#00ff88" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 20l2-4 3 1 2-5 2 3 3-2 4-8" />
      <circle cx="4" cy="20" r="1.5" fill="#00ff8830" />
      <circle cx="13" cy="12" r="1.5" fill="#00ff8830" />
      <circle cx="20" cy="5" r="1.5" fill="#00ff8830" />
      <path d="M2 22h20" opacity="0.3" />
      <path d="M2 22V2" opacity="0.3" />
    </svg>
  ),
  openSource: (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#00ff88" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-3" />
      <path d="M15 3h6v6" />
      <path d="M10 14L21 3" />
      <path d="M7 8h4" opacity="0.4" />
      <path d="M7 12h3" opacity="0.4" />
      <path d="M7 16h6" opacity="0.4" />
    </svg>
  ),
};

const features = [
  {
    icon: FeatureIcons.physics,
    title: 'MuJoCo Physics',
    description:
      'Accurate dinosaur biomechanics simulation with articulated joints, contact dynamics, and actuator models.',
  },
  {
    icon: FeatureIcons.brain,
    title: 'Reinforcement Learning',
    description:
      'PPO and SAC algorithms via Stable-Baselines3 with automated 3-stage curriculum learning.',
  },
  {
    icon: FeatureIcons.species,
    title: '3 Species',
    description:
      'T-Rex (18 actuators), Velociraptor (17 actuators), and Brachiosaurus (26 actuators).',
  },
  {
    icon: FeatureIcons.openSource,
    title: 'Open Source',
    description:
      'MIT licensed with TOML configs, Gymnasium registration, W&B tracking, and Docker support.',
  },
];

function FeaturesSection() {
  const ref = useScrollReveal();
  return (
    <section
      className={`${styles.featuresSection} ${styles.scrollReveal}`}
      aria-labelledby="features-heading"
      ref={ref}
    >
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ CAPABILITIES ]'}</div>
        <h2 className={styles.sectionTitle} id="features-heading">Core Features</h2>
      </div>
      <div className={styles.featuresGrid} role="list">
        {features.map((feature, idx) => (
          <div
            className={styles.featureCard}
            key={idx}
            role="listitem"
            style={{ transitionDelay: `${idx * 0.1}s` }}
          >
            <div className={styles.featureCardBorder} aria-hidden="true"></div>
            <div className={styles.featureIcon} aria-hidden="true">{feature.icon}</div>
            <h3 className={styles.featureTitle}>{feature.title}</h3>
            <p className={styles.featureDescription}>{feature.description}</p>
            <div className={styles.featureCorner} aria-hidden="true"></div>
          </div>
        ))}
      </div>
    </section>
  );
}

/* =============================================================================
   HOW IT WORKS (Curriculum Pipeline)
   ============================================================================= */

const curriculumSteps = [
  {
    stage: 1,
    title: 'Balance',
    subtitle: 'Stand & Stabilize',
    description: 'The agent learns to stand upright and maintain balance using proprioceptive feedback and joint torques.',
    color: '#00ff88',
  },
  {
    stage: 2,
    title: 'Locomotion',
    subtitle: 'Walk & Run',
    description: 'Building on balance skills, the agent develops forward locomotion with natural gait patterns.',
    color: '#00d4ff',
  },
  {
    stage: 3,
    title: 'Behavior',
    subtitle: 'Strike & Hunt',
    description: 'Advanced species-specific behaviors emerge: sickle claw strikes, head attacks, and tail defense.',
    color: '#ff6b35',
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
          Each species learns through a 3-stage curriculum, progressively mastering more complex behaviors
        </p>
      </div>
      <div className={styles.pipelineContainer}>
        <div className={styles.pipelineLine} aria-hidden="true"></div>
        {curriculumSteps.map((step, idx) => (
          <div className={styles.pipelineStep} key={idx}>
            <div
              className={styles.pipelineNode}
              style={{ borderColor: step.color, boxShadow: `0 0 20px ${step.color}40` }}
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
                <span className={styles.codeString}>"MesozoicLabs/Raptor-v0"</span>
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
                <span className={styles.codeOutput}>{'>>> Obs shape: (73,)'}</span>
              </div>
              <div className={styles.codeLine}>
                <span className={styles.cursor}>_</span>
              </div>
            </div>
          </div>
          <figure className={styles.previewPane}>
            <div className={styles.previewLabel}>PPO Stage 1 — Balance</div>
            <img
              src="/img/raptor_balance_ppo.gif"
              alt="Velociraptor learning to balance using PPO reinforcement learning"
              className={styles.previewGif}
            />
            <figcaption className={styles.previewCaption}>
              Velociraptor learning to balance via PPO curriculum training
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

const speciesData = [
  {
    id: 'velociraptor',
    name: 'Velociraptor',
    tagline: 'Swift Bipedal Predator',
    actuators: 17,
    gait: 'Bipedal',
    specialty: 'Sickle claw strikes',
    stages: [
      { number: 1, title: 'Balance', desc: 'Learning to stand upright', video: '/videos/velociraptor_ppo_stage1_best.mp4' },
      { number: 2, title: 'Locomotion', desc: 'Walking and running forward', video: '/videos/velociraptor_ppo_stage2_best.mp4' },
      { number: 3, title: 'Strike', desc: 'Sprinting and attacking with claws', video: '/videos/velociraptor_ppo_stage3_best.mp4' },
    ],
  },
  {
    id: 'trex',
    name: 'T-Rex',
    tagline: 'Apex Predator',
    actuators: 18,
    gait: 'Bipedal',
    specialty: 'Jaw strike attacks',
    stages: [
      { number: 1, title: 'Balance', desc: 'Stabilizing massive frame', video: '/videos/trex_ppo_stage1_best.mp4' },
      { number: 2, title: 'Locomotion', desc: 'Heavy bipedal gait', video: '/videos/trex_ppo_stage2_best.mp4' },
      { number: 3, title: 'Strike', desc: 'Head-strike attack patterns', video: '/videos/trex_ppo_stage3_best.mp4' },
    ],
  },
];

function SpeciesShowcase() {
  const [activeSpecies, setActiveSpecies] = useState(0);
  const ref = useScrollReveal();
  const species = speciesData[activeSpecies];

  return (
    <section
      className={`${styles.speciesSection} ${styles.scrollReveal}`}
      aria-labelledby="species-heading"
      ref={ref}
    >
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ SPECIES ]'}</div>
        <h2 className={styles.sectionTitle} id="species-heading">Training Results</h2>
        <p className={styles.sectionSubtitle}>
          Watch each species progress through the curriculum stages
        </p>
      </div>

      <div className={styles.speciesTabs} role="tablist" aria-label="Select species">
        {speciesData.map((sp, idx) => (
          <button
            key={sp.id}
            className={`${styles.speciesTab} ${idx === activeSpecies ? styles.speciesTabActive : ''}`}
            onClick={() => setActiveSpecies(idx)}
            role="tab"
            aria-selected={idx === activeSpecies}
            aria-controls={`species-panel-${sp.id}`}
          >
            <span className={styles.speciesTabName}>{sp.name}</span>
            <span className={styles.speciesTabMeta}>{sp.actuators} actuators &middot; {sp.gait}</span>
          </button>
        ))}
      </div>

      <div
        className={styles.speciesPanel}
        role="tabpanel"
        id={`species-panel-${species.id}`}
        aria-label={species.name}
      >
        <div className={styles.speciesInfo}>
          <span className={styles.speciesBadge}>{species.specialty}</span>
        </div>

        <div className={styles.speciesVideos}>
          {species.stages.map((stage) => (
            <div className={styles.speciesVideoCard} key={stage.number}>
              <div className={styles.speciesVideoHeader}>
                <span className={styles.speciesStageLabel}>STAGE {stage.number}</span>
                <span className={styles.speciesAlgoBadge}>PPO</span>
              </div>
              <h3 className={styles.speciesStageTitle}>{stage.title}</h3>
              <p className={styles.speciesStageDesc}>{stage.desc}</p>
              <div className={styles.speciesVideoWrap}>
                <video
                  src={stage.video}
                  controls
                  className={styles.speciesVideoPlayer}
                  aria-label={`${species.name} stage ${stage.number}: ${stage.title}`}
                />
              </div>
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
    status: 'upcoming' as const,
    statusLabel: 'PLANNED',
    items: ['Domain randomization', 'Terrain diversity', 'Turning & steering'],
  },
  {
    phase: 'PHASE 3 — v0.5.0',
    title: 'Evolution',
    status: 'upcoming' as const,
    statusLabel: 'PLANNED',
    items: ['Custom policy networks', 'New species', 'Benchmark suite'],
  },
  {
    phase: 'PHASE 4 — v0.6.0',
    title: 'The Pack',
    status: 'upcoming' as const,
    statusLabel: 'PLANNED',
    items: ['Multi-agent envs', 'Cooperative hunting', 'Predator-prey'],
  },
  {
    phase: 'PHASES 5-6',
    title: 'Hyperdrive & Sim-to-Real',
    status: 'upcoming' as const,
    statusLabel: 'PLANNED',
    items: ['JAX/MJX backend', 'Hardware prototype', 'ROS 2 bridge'],
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
        <h2 className={styles.sectionTitle} id="roadmap-heading">Project Roadmap</h2>
      </div>
      <div className={styles.timelineContainer}>
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
        <h2 className={styles.ctaTitle} id="cta-heading">Ready to Build Robotic Dinosaurs?</h2>
        <p className={styles.ctaDescription}>
          Explore the docs, train your first model, or contribute to the project.
        </p>
        <div className={styles.ctaButtons}>
          <a href="/docs/" className={styles.ctaButton}>
            GET STARTED
            <span className={styles.btnArrow} aria-hidden="true">&rarr;</span>
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
      title="Robotic Dinosaur Locomotion"
      description="Mesozoic Labs - Open-source platform for building robotic dinosaurs through simulation and reinforcement learning"
    >
      <a className="skip-nav" href="#main-content">
        Skip to main content
      </a>
      <main className={styles.main} id="main-content">
        <HeroSection />
        <div className={styles.sectionDivider} aria-hidden="true"></div>
        <FeaturesSection />
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
