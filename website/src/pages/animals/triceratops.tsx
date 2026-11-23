import React from 'react';
import Layout from '@theme/Layout';
import {AnimalHeader} from '@site/src/components/AnimalPage/AnimalHeader';
import RobotVersions from '@site/src/components/RobotVersions';
import TriceratopsScale from '@site/src/components/TriceratopsScale';
import {anatomicalStudy, triceratopsVersions} from '@site/src/data/triceratops';
import styles from '@site/src/components/AnimalPage/styles.module.css';

export default function TriceratopsPage() {
  return (
    <Layout title="Triceratops" description="Compare the adult Triceratops anatomical model, the 60 cm robot design and the earlier 75 cm concept through their simulation studies.">
      <main className={styles.page}>
        <AnimalHeader
          compact
          name="Triceratops"
          scientificName="Triceratops horridus"
          classification="Late Cretaceous · Ceratopsian"
          collection="Anatomy & robotics"
          modelLabel="Anatomical reference"
          title="Start with the animal"
          description="The adult model carries its head over flexed forelimbs. Rotate it to compare the smaller, outward-facing forefeet with the broader four-toed hindfeet."
          image="/img/animals/triceratops-anatomical.webp"
          alt="Anatomical Triceratops with a broad frill, three horns and four weight-bearing limbs"
          caption="MuJoCo · Adult standing pose"
          source={anatomicalStudy}
          sourceLabel="Anatomy and walking tests"
        />
        <TriceratopsScale />
        <RobotVersions speciesPath="/animals/triceratops" versions={triceratopsVersions} />
      </main>
    </Layout>
  );
}
