import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/installation', 'getting-started/quick-start'],
    },
    {
      type: 'category',
      label: 'Models',
      items: ['models/trex', 'models/velociraptor', 'models/brachiosaurus', 'models/custom-models'],
    },
    {
      type: 'category',
      label: 'Training',
      items: ['training/ppo', 'training/sac', 'training/hyperparameters', 'training/sweeps', 'training/vertex-ai'],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: ['api/overview'],
    },
  ],
};

export default sidebars;
