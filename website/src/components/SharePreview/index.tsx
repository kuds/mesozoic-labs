import React from 'react';
import {PageMetadata} from '@docusaurus/theme-common';

const descriptions = {
  home: 'November 22 simulation of the current Triceratops robot standing on rounded capsule feet. Its frill is 60 cm high.',
  triceratops: 'The 60 cm Triceratops robot with rounded capsule feet, standing in the November 22 simulation still.',
  timeline: 'The earlier roll-ankle Triceratops in a simulated walking trial. A physical foreleg bench test is planned.',
  collaborate: 'An anatomical Brachiosaurus model for a proposed review of its fossil evidence and reconstruction.',
  brachiosaurus: 'Brachiosaurus in its reconstructed standing pose, with four supporting legs and an elevated neck.',
};

export default function SharePreview({page}: {page: keyof typeof descriptions}) {
  return (
    <PageMetadata image={`/img/share/${page}.jpg`}>
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="630" />
      <meta property="og:image:alt" content={descriptions[page]} />
      <meta name="twitter:image:alt" content={descriptions[page]} />
      <meta name="twitter:card" content="summary_large_image" />
    </PageMetadata>
  );
}
