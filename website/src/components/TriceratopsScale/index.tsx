import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Heading from '@theme/Heading';
import {anatomicalStudy, triceratopsVersions} from '@site/src/data/triceratops';
import styles from './styles.module.css';

export default function TriceratopsScale() {
  const [matched, setMatched] = useState(false);
  const adult = useBaseUrl('/img/animals/triceratops-adult-profile.webp');
  const robot = useBaseUrl('/img/animals/triceratops-60cm-profile.webp');
  const robotReport = triceratopsVersions.find(version => version.slug === '60-cm')?.source;

  return (
    <section className={styles.comparison} aria-labelledby="scale-comparison">
      <div className={styles.heading}>
        <Heading as="h2" id="scale-comparison">Triceratops at scale</Heading>
        <fieldset className={styles.selector}>
          <legend className="sr-only">Comparison scale</legend>
          <label>
            <input type="radio" name="triceratops-scale" checked={!matched} onChange={() => setMatched(false)} />
            <span>Actual scale</span>
          </label>
          <label>
            <input type="radio" name="triceratops-scale" checked={matched} onChange={() => setMatched(true)} />
            <span>Match body lengths</span>
          </label>
        </fieldset>
      </div>
      <figure className={styles.figure}>
        <div className={styles.plate}>
          <div className={matched ? styles.matched : styles.actual}>
            <img src={adult} width={1600} height={706} className={styles.adult} alt="Adult Triceratops in side profile" />
            <img src={robot} width={800} height={400} className={styles.robot} alt="Triceratops robot in side profile" />
            <div className={styles.ground} aria-hidden="true" />
            {!matched && (
              <span className={styles.ruler} role="img" aria-label="1 metre"><span>1 m</span></span>
            )}
          </div>
        </div>
        <figcaption className={styles.caption}>
          <div className={styles.labels}>
            <div>
              <strong>Adult anatomical model</strong>
              <span>7.3 m overall length</span>
              <a href={anatomicalStudy}>Anatomical model notes</a>
            </div>
            <div>
              <strong>Current robot design</strong>
              <span>1.2 m overall length · 60 cm at the frill</span>
              {robotReport && <a href={robotReport.href}>60 cm model report</a>}
            </div>
          </div>
          <p className={styles.modeNote} aria-live="polite">
            {matched
              ? 'The profiles are the same length on screen so you can compare their frills, legs and tails.'
              : 'At the same scale, about six of the robot models would fit along the length of the adult.'}
          </p>
        </figcaption>
      </figure>
      <Link className={styles.milestone} to="/timeline#foreleg-bench-test">Foreleg bench-test milestone</Link>
    </section>
  );
}
