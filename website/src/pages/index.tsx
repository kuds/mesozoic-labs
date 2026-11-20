import React from 'react';
import Link from '@docusaurus/Link';
import useBrokenLinks from '@docusaurus/useBrokenLinks';
import Layout from '@theme/Layout';
import styles from './index.module.css';

const study = 'https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build';
const walkingReport = study + '/walk/REPORT_WALK60.md';

const featured = [
  {
    href: '/animals/triceratops',
    name: 'Triceratops',
    description: 'From a horned quadruped to the first planned build.',
    link: 'Explore the project',
  },
  {
    href: '/animals/compsognathus_longipes?version=robot',
    name: 'Compsognathus',
    description: 'A small biped and its twelve-servo robot model.',
    link: 'Explore the robot model',
  },
  {
    href: '/animals/brachiosaurus_altithorax',
    name: 'Brachiosaurus',
    description: 'How does a long-necked quadruped reach for food?',
    link: 'Explore the anatomy',
  },
];

const roadmap = [
  ['Now', '60 cm Triceratops in simulation'],
  ['Next', 'Single-foreleg bench tests'],
  ['Q3 2027 target', 'Demonstrate the 60 cm Triceratops walking on hardware'],
  ['2028 target', 'Build and test the 75 cm T. rex'],
];

function SectionHeading({id, children}: {id: string; children: string}) {
  useBrokenLinks().collectAnchor(id);
  return <h2 id={id}>{children}</h2>;
}

function TriceratopsStudy() {
  return (
    <figure>
      <div className={styles.plates}>
        <figure>
          <img
            src="/img/animals/triceratops-anatomical.webp"
            width={800}
            height={400}
            alt="Anatomical Triceratops with a broad frill, three horns and four weight-bearing legs"
          />
          <figcaption>Anatomical model</figcaption>
        </figure>
        <figure>
          <img
            src="/img/animals/triceratops-concept.webp"
            width={800}
            height={400}
            alt="Earlier open-frame Triceratops robot design with jointed legs, a frill and horns"
          />
          <figcaption>Robot model · 75 cm concept</figcaption>
        </figure>
      </div>
      <figcaption className={styles.caption}>
        <strong>Triceratops</strong>
        <span>MuJoCo design study · Views scaled independently</span>
      </figcaption>
    </figure>
  );
}

export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="Building robots inspired by Mesozoic animals"
      description="Explore Mesozoic animal models, follow a Triceratops robot from anatomy into simulation, and read the lab’s build journal."
    >
      <main className={styles.page} id="main-content">
        <header className={styles.intro}>
          <h1>Building robots inspired by Mesozoic animals.</h1>
          <div>
            <p>We start with an animal’s anatomy. When there’s a path to a physical robot, we design one and test how it moves. Today, every trial takes place in simulation.</p>
            <Link className={styles.primary} to="/animals/triceratops">Follow the Triceratops build</Link>
          </div>
        </header>

        <TriceratopsStudy />

        <section className={styles.featured} aria-labelledby="explore">
          <div className={styles.sectionHeading}>
            <SectionHeading id="explore">Explore the animals</SectionHeading>
            <Link className={styles.textLink} to="/animals">All animals</Link>
          </div>
          <div className={styles.featureGrid}>
            {featured.map(animal => (
              <Link key={animal.href} to={animal.href} className={styles.feature}>
                <h3>{animal.name}</h3>
                <p>{animal.description}</p>
                <span>{animal.link}</span>
              </Link>
            ))}
          </div>
        </section>

        <div className={styles.research}>
          <section aria-labelledby="journal">
            <SectionHeading id="journal">Build journal</SectionHeading>
            <div className={styles.metadata}><time dateTime="2026-11-19">19 November 2026</time> · Simulation</div>
            <h3>Forty flat-floor trials</h3>
            <p>The 60 cm Triceratops trotted through four simulated floor conditions without falling. On a 5° ramp, its speed dropped by 27%.</p>
            <a href={walkingReport} className={styles.textLink}>See the trial results</a>
          </section>
          <section aria-labelledby="roadmap">
            <SectionHeading id="roadmap">Build roadmap</SectionHeading>
            <dl className={styles.roadmap}>
              {roadmap.map(([when, milestone]) => (
                <div key={when}><dt>{when}</dt><dd>{milestone}</dd></div>
              ))}
            </dl>
          </section>
        </div>
      </main>
    </Layout>
  );
}
