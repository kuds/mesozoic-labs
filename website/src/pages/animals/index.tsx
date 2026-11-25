import React from 'react';
import Layout from '@theme/Layout';
import {RotationButton, Turntable, useTurntableMotion} from '@site/src/components/AnimalPage/Turntable';
import {ANIMALS} from '@site/src/data/animals';
import styles from './index.module.css';

export default function Animals(): React.JSX.Element {
  const motion = useTurntableMotion();
  return (
    <Layout
      title="Animals"
      description="Explore Mesozoic Labs’ anatomical models, robot designs and simulated behaviors by animal."
    >
      <main className={styles.page}>
        <header className={styles.intro}>
          <h1>Animals</h1>
          <div className={styles.introTools}>
            <p>Drag, swipe or use the left and right arrow keys to rotate an animal. Select it to explore its anatomy, movement and robot designs.</p>
            <RotationButton motion={motion} plural />
          </div>
        </header>
        <ul className={styles.catalog}>
          {ANIMALS.map(animal => (
            <li key={animal.slug}>
              <article>
                <Turntable image={animal.image} alt="" className={styles.specimen} motion={motion}
                  catalogLink={{to: `/animals/${animal.slug}`, className: styles.entry}}>
                  <div className={styles.label}>
                    <h2>{animal.name}</h2>
                    <i>{animal.scientificName}</i>
                  </div>
                </Turntable>
                <p>{animal.summary}</p>
              </article>
            </li>
          ))}
        </ul>
      </main>
    </Layout>
  );
}
