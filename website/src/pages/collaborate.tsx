import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import SharePreview from '@site/src/components/SharePreview';
import styles from './collaborate.module.css';

export default function Collaborate() {
  return (
    <Layout title="Collaborate with Mesozoic Labs" description="Contact the lab about fossil interpretation, museum education and simulated locomotion.">
      <SharePreview page="collaborate" />
      <main className={styles.page}>
        <h1>Collaborate with Mesozoic Labs</h1>
        <p>We welcome feedback from paleontologists, educators and robotics researchers on how to communicate what our models can teach.</p>
        <section>
          <h2>Which parts of Brachiosaurus can we reconstruct from fossils?</h2>
          <p>Our annotated model traces the limb proportions to particular specimens and identifies choices informed by related animals. We’d like to know which labels help visitors see that distinction and which need a clearer explanation.</p>
          <Link to="/animals/brachiosaurus_altithorax#fossil-evidence">Review the Brachiosaurus diagram</Link>
        </section>
        <section>
          <h2>What can we test in simulation?</h2>
          <p>The 60 cm Triceratops is our first planned robot build. We’ve tested it on level ground and ramps in simulation. Next, we plan to measure the response of one physical foreleg under load.</p>
          <Link to="/#journal">See the walking trial results</Link>
        </section>
        <section>
          <h2>Get in touch</h2>
          <p>For project inquiries, contact Michael Kudlaty, Mesozoic Labs project author, through Finding Theta at <a href="mailto:hello@findingtheta.com?subject=Mesozoic%20Labs%20collaboration">hello@findingtheta.com</a>.</p>
        </section>
      </main>
    </Layout>
  );
}
