import {ALL_MODELS, type Species} from './species';

export interface Animal {
  slug: string;
  name: string;
  scientificName: string;
  summary: string;
  image: string;
  models: Species[];
}

const summaries: Record<string, string> = {
  allosaurus_fragilis: 'A long tail balances a crested skull as three forward toes spread the load on each foot.',
  velociraptor_mongoliensis: 'Balance on two feet, run, and aim a raised sickle claw at simulated prey.',
  tyrannosaurus_rex: 'A heavy tail balances the skull as the model stands, recovers from pushes, and advances on prey.',
  brachiosaurus_altithorax: 'Four supporting legs and a long, articulated neck work together to reach elevated food.',
  dibothrosuchus_elaphros: 'A small crocodylomorph walks on upright limbs and guides its narrow snout toward prey.',
  compsognathus_longipes: 'An anatomical model and a robot take on the same challenge of walking to a target and stopping there.',
};

export function animalModelImage(model: Species): string {
  return '/img/animals/' + model.species + '-' + model.embodiment + '.webp';
}

const bySpecies = new Map<string, Animal>();
for (const model of ALL_MODELS) {
  const existing = bySpecies.get(model.species);
  if (existing) {
    existing.models.push(model);
  } else {
    const [name, ...epithets] = model.speciesName.split(' ');
    bySpecies.set(model.species, {
      slug: model.species,
      name,
      scientificName: [name, ...epithets.map((epithet) => epithet.toLowerCase())].join(' '),
      summary: summaries[model.species] ?? model.specialty,
      image: animalModelImage(model),
      models: [model],
    });
  }
}

export const STEGOSAURUS: Animal = {
  slug: 'stegosaurus',
  name: 'Stegosaurus',
  scientificName: 'Stegosaurus stenops',
  summary: 'Watch a slow, four-beat walk and see how the short forefeet and long, cushioned hindfeet share the load.',
  image: '/img/animals/stegosaurus_stenops-anatomical.webp',
  models: [],
};

export const ANIMALS: Animal[] = [
  {
    slug: 'triceratops',
    name: 'Triceratops',
    scientificName: 'Triceratops horridus',
    summary: 'An adult anatomical reference and two robot designs share a four-legged stance beneath the frill.',
    image: '/img/animals/triceratops-anatomical.webp',
    models: [],
  },
  STEGOSAURUS,
  ...bySpecies.values(),
];
