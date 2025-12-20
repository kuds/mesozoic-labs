import React from 'react';
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
      </div>

      <div className={styles.dinoSilhouette}></div>
    </header>
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
      </main>
    </Layout>
  );
}
