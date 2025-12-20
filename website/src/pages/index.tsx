import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HeroSection() {
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroOverlay}></div>
      <div className={styles.gridLines}></div>
      <div className={styles.scanLines}></div>

      <div className={styles.heroContent}>
        <div className={styles.logoContainer}>
          <div className={styles.logoGlow}></div>
          <div className={styles.logo}>
            <span className={styles.logoDino}>MESOZOIC</span>
            <span className={styles.logoLabs}>LABS</span>
          </div>
        </div>

        <div className={styles.taglineContainer}>
          <div className={styles.circuitLine}></div>
          <h2 className={styles.tagline}>
            <span className={styles.taglineWord}>PREHISTORIC</span>
            <span className={styles.taglineDivider}>//</span>
            <span className={styles.taglineWord}>ROBOTIC</span>
            <span className={styles.taglineDivider}>//</span>
            <span className={styles.taglineWord}>INTELLIGENCE</span>
          </h2>
          <div className={styles.circuitLine}></div>
        </div>

        <p className={styles.description}>
          Open-source platform for building robotic dinosaurs
          <br />
          powered by physics simulation and reinforcement learning
        </p>

        <div className={styles.comingSoonBadge}>
          <span className={styles.badgePulse}></span>
          <span className={styles.badgeText}>COMING SOON</span>
        </div>

        <div className={styles.statsContainer}>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>3+</span>
            <span className={styles.statLabel}>Dinosaur Models</span>
          </div>
          <div className={styles.statDivider}></div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>10M+</span>
            <span className={styles.statLabel}>Training Steps</span>
          </div>
          <div className={styles.statDivider}></div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>2</span>
            <span className={styles.statLabel}>RL Algorithms</span>
          </div>
        </div>
      </div>

      <div className={styles.dinoSilhouette}></div>
    </header>
  );
}

function FeatureSection() {
  const features = [
    {
      icon: '🦖',
      title: 'Dinosaur Physics Models',
      description: 'Accurate MuJoCo physics simulations of T-Rex, Velociraptor, and more prehistoric creatures with realistic joint dynamics.',
    },
    {
      icon: '🤖',
      title: 'Reinforcement Learning',
      description: 'Train dinosaurs to walk, run, and move naturally using state-of-the-art PPO and SAC algorithms.',
    },
    {
      icon: '⚡',
      title: 'Real-World Robotics',
      description: 'Sim-to-real transfer learning for building actual robotic dinosaurs that move like the real thing.',
    },
    {
      icon: '🔧',
      title: 'Open Source',
      description: 'Fully open-source codebase. Contribute models, algorithms, and improvements to the project.',
    },
  ];

  return (
    <section className={styles.featuresSection}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon}>[ SYSTEMS ]</div>
        <h2 className={styles.sectionTitle}>Core Technology</h2>
      </div>

      <div className={styles.featuresGrid}>
        {features.map((feature, idx) => (
          <div key={idx} className={styles.featureCard}>
            <div className={styles.featureCardBorder}></div>
            <div className={styles.featureIcon}>{feature.icon}</div>
            <h3 className={styles.featureTitle}>{feature.title}</h3>
            <p className={styles.featureDescription}>{feature.description}</p>
            <div className={styles.featureCorner}></div>
          </div>
        ))}
      </div>
    </section>
  );
}

function SimulationPreview() {
  return (
    <section className={styles.simulationSection}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon}>[ PREVIEW ]</div>
        <h2 className={styles.sectionTitle}>Simulation in Action</h2>
      </div>

      <div className={styles.simulationContainer}>
        <div className={styles.terminalWindow}>
          <div className={styles.terminalHeader}>
            <div className={styles.terminalDot} style={{background: '#ff5f56'}}></div>
            <div className={styles.terminalDot} style={{background: '#ffbd2e'}}></div>
            <div className={styles.terminalDot} style={{background: '#27ca40'}}></div>
            <span className={styles.terminalTitle}>mesozoic-training.py</span>
          </div>
          <div className={styles.terminalBody}>
            <div className={styles.codeLine}>
              <span className={styles.codeComment}>{'# Initialize dinosaur model'}</span>
            </div>
            <div className={styles.codeLine}>
              <span className={styles.codeKeyword}>from</span> mesozoic <span className={styles.codeKeyword}>import</span> DinoEnv, SACAgent
            </div>
            <div className={styles.codeLine}>&nbsp;</div>
            <div className={styles.codeLine}>
              env = DinoEnv(<span className={styles.codeString}>"trex"</span>)
            </div>
            <div className={styles.codeLine}>
              agent = SACAgent(env.observation_space, env.action_space)
            </div>
            <div className={styles.codeLine}>&nbsp;</div>
            <div className={styles.codeLine}>
              <span className={styles.codeComment}>{'# Train for 3.6M steps'}</span>
            </div>
            <div className={styles.codeLine}>
              agent.train(steps=<span className={styles.codeNumber}>3_600_000</span>)
            </div>
            <div className={styles.codeLine}>&nbsp;</div>
            <div className={styles.codeLine}>
              <span className={styles.codeOutput}>{'>'} Avg Reward: 3091.31</span>
            </div>
            <div className={styles.codeLine}>
              <span className={styles.codeOutput}>{'>'} Training complete! Dinosaur learned to walk.</span>
            </div>
            <div className={clsx(styles.codeLine, styles.cursor)}>_</div>
          </div>
        </div>
      </div>
    </section>
  );
}

function RoadmapSection() {
  const milestones = [
    { phase: '01', title: 'Foundation', status: 'complete', items: ['MuJoCo physics models', 'Basic dinosaur locomotion', 'PPO & SAC training'] },
    { phase: '02', title: 'Expansion', status: 'active', items: ['Additional species', 'Improved gaits', 'Public documentation'] },
    { phase: '03', title: 'Hardware', status: 'upcoming', items: ['Sim-to-real pipeline', 'Robot prototypes', 'Motion controllers'] },
    { phase: '04', title: 'Ecosystem', status: 'upcoming', items: ['Community models', 'Plugin system', 'Educational tools'] },
  ];

  return (
    <section className={styles.roadmapSection}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon}>[ ROADMAP ]</div>
        <h2 className={styles.sectionTitle}>Development Timeline</h2>
      </div>

      <div className={styles.roadmapContainer}>
        {milestones.map((milestone, idx) => (
          <div key={idx} className={clsx(styles.milestoneCard, styles[milestone.status])}>
            <div className={styles.milestonePhase}>PHASE {milestone.phase}</div>
            <h3 className={styles.milestoneTitle}>{milestone.title}</h3>
            <ul className={styles.milestoneItems}>
              {milestone.items.map((item, itemIdx) => (
                <li key={itemIdx}>{item}</li>
              ))}
            </ul>
            <div className={styles.milestoneStatus}>
              {milestone.status === 'complete' && '✓ COMPLETE'}
              {milestone.status === 'active' && '◉ IN PROGRESS'}
              {milestone.status === 'upcoming' && '○ UPCOMING'}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className={styles.ctaSection}>
      <div className={styles.ctaContent}>
        <h2 className={styles.ctaTitle}>Join the Mesozoic Era of Robotics</h2>
        <p className={styles.ctaDescription}>
          Star the project on GitHub to follow our progress and be notified when we launch.
        </p>
        <a
          href="https://github.com/mesozoic-labs"
          className={styles.ctaButton}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span className={styles.ctaButtonIcon}>★</span>
          <span>Star on GitHub</span>
        </a>
      </div>
      <div className={styles.ctaDinoTrack}></div>
    </section>
  );
}

export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="Coming Soon"
      description="Mesozoic Labs - Open-source platform for building robotic dinosaurs through simulation and reinforcement learning"
    >
      <main className={styles.main}>
        <HeroSection />
        <FeatureSection />
        <SimulationPreview />
        <RoadmapSection />
        <CTASection />
      </main>
    </Layout>
  );
}
