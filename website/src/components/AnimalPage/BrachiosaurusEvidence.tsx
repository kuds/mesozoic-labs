import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Heading from '@theme/Heading';
import styles from './brachiosaurusEvidence.module.css';

export default function BrachiosaurusEvidence({image}: {image: string}) {
  return (
    <section className={styles.section} aria-labelledby="fossil-evidence">
      <Heading as="h2" id="fossil-evidence">Which details come from fossils?</Heading>
      <figure className={styles.figure}>
        <div className={styles.plate}>
          <img src={useBaseUrl(image)} width={1600} height={800} loading="lazy"
            alt="Side view of the model. Label 1 marks the upper arm and thigh, 2 the lower forelimb, and 3 the head." />
          <svg className={styles.lines} viewBox="0 0 800 400" aria-hidden="true" focusable="false">
            <path d="M309 270 L337 291 H404 L432 270 M431 341 H511 M655 32 L695 66 H724" />
          </svg>
          <span className={styles.upperLimbs} aria-hidden="true">1</span>
          <span className={styles.forelimb} aria-hidden="true">2</span>
          <span className={styles.head} aria-hidden="true">3</span>
        </div>
        <figcaption>
          <ol className={styles.key}>
            <li>
              <strong>Upper arm and thigh</strong>
              <p>The Field Museum holds the specimen used to describe this species. Its upper arm
                and thigh bones are nearly equal in length, a proportion carried into the model. <a href="https://www.miketaylor.org.uk/dino/pubs/taylor2009/">Taylor (2009)</a></p>
            </li>
            <li>
              <strong>The tall forelimb</strong>
              <p>The model’s forearm and hand proportions come from a second skeleton found at
                Potter Creek. Its assignment to <i>B. altithorax</i> is tentative. <a href="https://scholarlyworks.adelphi.edu/esploro/outputs/journalArticle/Redescription-of-Brachiosaurid-Sauropod-Dinosaur-Material/991004306399906266">D’Emic and Carrano (2020)</a></p>
            </li>
            <li>
              <strong>Head shape and browsing posture</strong>
              <p>The model’s skull outline and browsing angle draw on comparisons with related
                long-necked dinosaurs. Its feeding pose is a reconstruction. <a href="https://github.com/kuds/mesozoic-labs/blob/main/docs/reviews/BRACHIOSAURUS_ANATOMICAL_REFERENCE.md#anatomy-and-design">Model measurements and assumptions</a></p>
            </li>
          </ol>
        </figcaption>
      </figure>
      <details className={styles.prediction}>
        <summary>Would a food target 0.6 m from the model’s head tip count as a reach?</summary>
        <p>Yes. In the simulation, a reach counts when the head tip comes within 0.8 m of the target.
          A different neck posture would change which targets fall within that distance.{' '}
          <Link to="/docs/models/brachiosaurus_altithorax/anatomical">Read the model notes</Link></p>
      </details>
      <p className={styles.feedback}>
        Which label would you change to make the evidence clearer to visitors?{' '}
        <Link to="/collaborate">Share feedback with the lab</Link>
      </p>
    </section>
  );
}
