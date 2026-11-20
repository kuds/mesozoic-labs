import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {ANIMALS} from '@site/src/data/animals';
import styles from './index.module.css';

export default function Animals(): React.JSX.Element {
  return (
    <Layout
      title="Animals"
      description="Explore Mesozoic Labs’ anatomical models, robot designs and simulated behaviors by animal."
    >
      <main className={styles.page}>
        <header className={styles.intro}>
          <h1>Animals</h1>
          <p>Explore the anatomy of each animal, see how its models move, and follow the robot designs it inspired.</p>
        </header>
        <ul className={styles.catalog}>
          {ANIMALS.map(animal => (
            <li key={animal.slug}>
              <article>
                <Link to={`/animals/${animal.slug}`} className={styles.entry}>
                  <img src={animal.image} alt="" loading="lazy" />
                  <div className={styles.label}>
                    <h2>{animal.name}</h2>
                    <i>{animal.scientificName}</i>
                  </div>
                </Link>
                <p>{animal.summary}</p>
              </article>
            </li>
          ))}
        </ul>
      </main>
    </Layout>
  );
}
