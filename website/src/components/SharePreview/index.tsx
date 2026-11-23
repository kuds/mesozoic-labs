import React from 'react';
import {PageMetadata} from '@docusaurus/theme-common';

const descriptions = {
  home: 'The lab’s current Triceratops robot design in simulation. Its frill stands 60 cm high.',
  triceratops: 'Side view of the 60 cm Triceratops robot model, showing the horned head and four mechanical legs.',
  timeline: 'The 60 cm Triceratops in a simulated walking trial. A physical foreleg bench test is planned.',
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
