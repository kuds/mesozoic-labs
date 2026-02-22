import React from 'react';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HeroSection() {
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroOverlay} aria-hidden="true"></div>
      <div className={styles.gridLines} aria-hidden="true"></div>
      <div className={styles.scanLines} aria-hidden="true"></div>

      <div className={styles.heroContent}>
        <div className={styles.logoContainer}>
          <div className={styles.logoGlow} aria-hidden="true"></div>
          <div className={styles.logo} role="img" aria-label="Mesozoic Labs">
            <span className={styles.logoDino} aria-hidden="true">MESOZOIC</span>
            <span className={styles.logoLabs} aria-hidden="true">LABS</span>
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

        <dl className={styles.statsContainer}>
          <div className={styles.statItem}>
            <dt className={styles.statLabel}>SPECIES</dt>
            <dd className={styles.statNumber}>3</dd>
          </div>
          <div className={styles.statDivider} aria-hidden="true"></div>
          <div className={styles.statItem}>
            <dt className={styles.statLabel}>TESTS PASSING</dt>
            <dd className={styles.statNumber}>155</dd>
          </div>
          <div className={styles.statDivider} aria-hidden="true"></div>
          <div className={styles.statItem}>
            <dt className={styles.statLabel}>TRAINING STAGES</dt>
            <dd className={styles.statNumber}>3</dd>
          </div>
        </dl>
      </div>

      <div className={styles.dinoSilhouette} aria-hidden="true"></div>
    </header>
  );
}

const features = [
  {
    icon: '\u2699\uFE0F',
    title: 'MuJoCo Physics',
    description:
      'Accurate dinosaur biomechanics simulation with articulated joints, contact dynamics, and actuator models using MuJoCo.',
  },
  {
    icon: '\uD83E\uDDE0',
    title: 'Reinforcement Learning',
    description:
      'PPO and SAC algorithms via Stable-Baselines3 with automated 3-stage curriculum learning: balance, locomotion, behavior.',
  },
  {
    icon: '\uD83E\uDD96',
    title: '3 Species',
    description:
      'T-Rex (14 actuators), Velociraptor (12 actuators), and Brachiosaurus (22 actuators) — bipedal and quadrupedal gaits.',
  },
  {
    icon: '\uD83D\uDCE6',
    title: 'Open Source',
    description:
      'MIT licensed. TOML configs, Gymnasium registration, W&B tracking, Docker support, and Vertex AI cloud training.',
  },
];

function FeaturesSection() {
  return (
    <section className={styles.featuresSection} aria-labelledby="features-heading">
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ CAPABILITIES ]'}</div>
        <h2 className={styles.sectionTitle} id="features-heading">Core Features</h2>
      </div>
      <div className={styles.featuresGrid} role="list">
        {features.map((feature, idx) => (
          <div className={styles.featureCard} key={idx} role="listitem">
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

function SimulationSection() {
  return (
    <section className={styles.simulationSection} aria-labelledby="simulation-heading">
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
                <span className={styles.codeOutput}>{'>>> Obs shape: (69,)'}</span>
              </div>
              <div className={styles.codeLine}>
                <span className={styles.codeKeyword}>print</span>
                <span>(</span>
                <span className={styles.codeString}>"Actions:"</span>
                <span>, env.action_space.shape)</span>
              </div>
              <div className={styles.codeLine}>
                <span className={styles.codeOutput}>{'>>> Actions: (12,)'}</span>
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
    items: [
      'Curriculum manager',
      'W&B tracking',
      'Locomotion metrics',
      'Species training runs',
    ],
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
  return (
    <section className={styles.roadmapSection} aria-labelledby="roadmap-heading">
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">{'[ ROADMAP ]'}</div>
        <h2 className={styles.sectionTitle} id="roadmap-heading">Project Roadmap</h2>
      </div>
      <div className={styles.roadmapContainer} role="list">
        {milestones.map((milestone, idx) => (
          <article
            className={`${styles.milestoneCard} ${styles[milestone.status]}`}
            key={idx}
            role="listitem"
          >
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
          </article>
        ))}
      </div>
    </section>
  );
}

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
            <span className={styles.ctaButtonIcon} aria-hidden="true">📖</span>
            GET STARTED
          </a>
          <a
            href="https://github.com/kuds/mesozoic-labs"
            className={styles.ctaButtonSecondary}
            target="_blank"
            rel="noopener noreferrer"
          >
            <span className={styles.ctaButtonIcon} aria-hidden="true">⭐</span>
            VIEW ON GITHUB
            <span className="sr-only"> (opens in new tab)</span>
          </a>
        </div>
      </div>
    </section>
  );
}

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
        <FeaturesSection />
        <SimulationSection />
        <RoadmapSection />
        <CTASection />
      </main>
    </Layout>
  );
}
