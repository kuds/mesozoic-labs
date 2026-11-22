import React from 'react';
import Layout from '@theme/Layout';
import {AnimalHeader} from '@site/src/components/AnimalPage/AnimalHeader';
import {StudyRecording} from '@site/src/components/AnimalPage/MovementStudy';
import styles from '@site/src/components/AnimalPage/styles.module.css';
import {STEGOSAURUS} from '@site/src/data/animals';
import {posterFor} from '@site/src/data/species';
import pageStyles from './stegosaurus.module.css';

const repository = 'https://github.com/kuds/mesozoic-labs/blob/main';
const study = repository + '/examples/stegosaurus_stenops/README.md';
const walk = '/videos/stegosaurus_stenops_anatomical_reference_walk.mp4';

export default function StegosaurusPage() {
  return (
    <Layout title={STEGOSAURUS.name} description={STEGOSAURUS.summary}>
      <main className={styles.page}>
        <AnimalHeader
          name={STEGOSAURUS.name}
          scientificName={STEGOSAURUS.scientificName}
          classification="Late Jurassic · Stegosaur"
          collection="Anatomical studies"
          modelLabel="Anatomical reference"
          title="A low head and high hips"
          description="This 5.79 m reconstruction follows the proportions of Sophie, a young adult Stegosaurus. The short forelimbs end in broad pads. Each hindfoot has a long heel cushion and three blunt hooves."
          image={STEGOSAURUS.image}
          alt="Stegosaurus with hips above the shoulders, staggered back plates and spikes near the tail tip"
          caption="MuJoCo · Standing pose"
          source={study}
          sourceLabel="Model and anatomical references"
        />
        <section aria-labelledby="walk">
          <div className={styles.sectionHead}>
            <h2 id="walk">The four-beat walk</h2>
            <a href={study + '#measured-balance-and-walking'}>Gait measurements</a>
          </div>
          <div className={pageStyles.walk}>
            <StudyRecording recording={{
              src: walk,
              poster: posterFor(walk),
              label: 'Stegosaurus walking on level ground with a timeline of foot contact',
              caption: 'Each bar shows when that foot carried weight over the previous three seconds.',
              source: repository + '/website/static' + walk,
            }} />
            <div>
              <p>Watch the right hindfoot touch down, followed by the right forefoot. The left pair repeats the sequence. In the steady part of the simulation, three or four feet carried weight at every sample.</p>
              <dl className={pageStyles.measurements}>
                <div><dt>Walking speed</dt><dd>0.25 m/s</dd></div>
                <div><dt>Stride length</dt><dd>0.46 m</dd></div>
                <div><dt>Weight on hindfeet at rest</dt><dd>80.5%</dd></div>
              </dl>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
