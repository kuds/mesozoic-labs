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
  velociraptor_mongoliensis: 'Balance on two feet and run, with the sickle claw on each foot held clear of the ground.',
  tyrannosaurus_rex: 'A heavy tail balances the skull as the model stands, recovers from pushes, and walks.',
  brachiosaurus_altithorax: 'Four supporting legs carry the body while a long, articulated neck holds the head high.',
  dibothrosuchus_elaphros: 'A small crocodylomorph stands and walks on upright limbs, its narrow snout held ahead.',
  compsognathus_longipes: 'An anatomical model and a robot take on the same challenges of standing, recovering from pushes and walking.',
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
