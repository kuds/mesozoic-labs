import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useBrokenLinks from '@docusaurus/useBrokenLinks';
import Layout from '@theme/Layout';
import SharePreview from '@site/src/components/SharePreview';
import styles from './index.module.css';

const study = 'https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build';
const walkingReport = study + '/walk/results/walk60_2026-11-19/RESULTS.md';
const rampResults = study + '/walk/results/sweep_walk60/summary_walk60.md';
const targetSpeed = 0.25;
const slopeResults = [
  {terrain: 'Level ground', speed: 0.246},
  {terrain: '5° uphill', speed: 0.179},
  {terrain: '8° uphill', speed: 0.129},
];

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
    href: '/animals/brachiosaurus_altithorax#fossil-evidence',
    name: 'Brachiosaurus',
    description: 'What can the surviving bones tell us about this long-necked animal?',
    link: 'Explore the fossil evidence',
  },
];

function SectionHeading({id, children}: {id: string; children: string}) {
  useBrokenLinks().collectAnchor(id);
  return <h2 id={id}>{children}</h2>;
}

function TriceratopsStudy() {
  const robot = useBaseUrl('/img/animals/triceratops-60cm-profile.webp');
  const adult = useBaseUrl('/img/animals/triceratops-adult-profile.webp');
  return (
    <figure>
      <div className={styles.plates}>
        <figure>
          <img
            src={robot}
            width={800}
            height={400}
            alt="Side view of the robot with a horned head, open chassis and jointed legs"
          />
          <figcaption>Current robot design · 60 cm at the frill</figcaption>
        </figure>
        <figure>
          <img
            src={adult}
            width={1600}
            height={706}
            alt="Side view of the adult anatomical model with a broad frill, three horns and four weight-bearing legs"
          />
          <figcaption>Adult anatomical model · 7.3 m long</figcaption>
        </figure>
      </div>
      <figcaption className={styles.caption}>
        <strong>Triceratops</strong>
        <span>MuJoCo models · Views scaled independently</span>
        <Link className={styles.textLink} to="/animals/triceratops#scale-comparison">Compare the adult and the 60 cm robot to scale</Link>
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
      <SharePreview page="home" />
      <main className={styles.page} id="main-content">
        <header className={styles.intro}>
          <h1>Building robots inspired by Mesozoic animals.</h1>
          <div>
            <p>We start with an animal’s anatomy. Our first planned build is a Triceratops robot that stands 60 cm at the frill. Today, every walking trial takes place in simulation.</p>
            <div className={styles.introLinks}>
              <Link className={styles.primary} to="/animals/triceratops">Follow the Triceratops build</Link>
              <Link className={styles.textLink} to="/collaborate">Collaborate with us</Link>
            </div>
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
            <h3>From flat floors to ramps</h3>
            <p>The 60 cm robot stayed upright in all 40 judged trials. Using a model-based controller, it trotted across four simulated floor conditions, ten trials per condition.</p>
            <p>In a separate exploratory test, the same controller targeted {targetSpeed} m/s in eight simulations per slope. At 5°, it moved 27% slower than on level ground.</p>
            <figure className={styles.slopeStudy}>
              <figcaption>Mean forward speed by slope <span>(m/s)</span></figcaption>
              <dl>
                {slopeResults.map(({terrain, speed}) => (
                  <div key={terrain} className={styles.slopeRow}>
                    <dt>{terrain}</dt>
                    <dd>
                      <span className={styles.speedTrack} aria-hidden="true">
                        <span style={{width: `${speed / targetSpeed * 100}%`}} />
                      </span>
                      <span>{speed.toFixed(3)}</span>
                    </dd>
                  </div>
                ))}
              </dl>
            </figure>
            <span>
              <a href={walkingReport} className={styles.textLink}>Read the flat-floor trial results</a>
              {' · '}
              <a href={rampResults} className={styles.textLink}>Read the ramp results</a>
            </span>
          </section>
          <section aria-labelledby="roadmap">
            <SectionHeading id="roadmap">Build roadmap</SectionHeading>
            <p>The next hardware test is a single Triceratops foreleg on the bench. The target for a walking demonstration of the 60 cm robot is Q3 2027.</p>
            <Link className={styles.textLink} to="/timeline">Explore the project timeline</Link>
          </section>
        </div>
      </main>
    </Layout>
  );
}
