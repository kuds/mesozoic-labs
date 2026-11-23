import React from 'react';
import Layout from '@theme/Layout';
import SharePreview from '@site/src/components/SharePreview';
import {AnimalHeader} from '@site/src/components/AnimalPage/AnimalHeader';
import RobotVersions from '@site/src/components/RobotVersions';
import TriceratopsScale from '@site/src/components/TriceratopsScale';
import {anatomicalStudy, triceratopsVersions} from '@site/src/data/triceratops';
import styles from '@site/src/components/AnimalPage/styles.module.css';

export default function TriceratopsPage() {
  return (
    <Layout title="Triceratops" description="Compare the adult Triceratops anatomical model, the 60 cm robot design and the earlier 75 cm concept through their simulation studies.">
      <SharePreview page="triceratops" />
      <main className={styles.page}>
        <AnimalHeader
          compact
          name="Triceratops"
          scientificName="Triceratops horridus"
          classification="Late Cretaceous · Ceratopsian"
          collection="Anatomy & robotics"
          modelLabel="Anatomical reference"
          title="Start with the animal"
          description="The 7.3 m adult reference sets the proportions. The robot designs carry forward its four-legged stance, broad frill and three horns."
          image="/img/animals/triceratops-anatomical.webp"
          alt="Anatomical Triceratops with a broad frill, three horns and four weight-bearing limbs"
          caption="MuJoCo · Adult reference model"
          source={anatomicalStudy}
          sourceLabel="Anatomy & model notes"
        />
        <TriceratopsScale />
        <RobotVersions speciesPath="/animals/triceratops" versions={triceratopsVersions} />
      </main>
    </Layout>
  );
}
