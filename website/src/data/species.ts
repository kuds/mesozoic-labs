/**
 * Single source of truth for the species/stage/video data shown on the
 * homepage species showcase and on each model's docs page — previously
 * duplicated in both places, which let the copies drift.
 */

export interface SpeciesStage {
  number: number;
  title: string;
  description: string;
  video: string;
}

export interface Species {
  id: string;
  name: string;
  tagline: string;
  actuators: number;
  gait: 'Bipedal' | 'Quadrupedal';
  specialty: string;
  stages: SpeciesStage[];
}

/** Poster image generated from the video's first frame (static/img/posters). */
export function posterFor(video: string): string {
  return video.replace('/videos/', '/img/posters/').replace('.mp4', '.jpg');
}

export const VELOCIRAPTOR: Species = {
  id: 'velociraptor',
  name: 'Velociraptor',
  tagline: 'Swift Bipedal Predator',
  actuators: 22,
  gait: 'Bipedal',
  specialty: 'Sickle claw strikes',
  stages: [
    { number: 1, title: 'Balance', description: 'Learning to stand upright', video: '/videos/velociraptor_ppo_stage1_best.mp4' },
    { number: 2, title: 'Locomotion', description: 'Walking and running forward', video: '/videos/velociraptor_ppo_stage2_best.mp4' },
    { number: 3, title: 'Strike', description: 'Sprinting and attacking with sickle claws', video: '/videos/velociraptor_ppo_stage3_best.mp4' },
  ],
};

export const TREX: Species = {
  id: 'trex',
  name: 'T-Rex',
  tagline: 'Apex Predator',
  actuators: 21,
  gait: 'Bipedal',
  specialty: 'Jaw strike attacks',
  stages: [
    { number: 1, title: 'Balance', description: 'Stabilizing a massive frame', video: '/videos/trex_ppo_stage1_best.mp4' },
    { number: 2, title: 'Locomotion', description: 'Heavy bipedal gait', video: '/videos/trex_ppo_stage2_best.mp4' },
    { number: 3, title: 'Strike', description: 'Head-strike attack patterns', video: '/videos/trex_ppo_stage3_best.mp4' },
  ],
};

export const BRACHIOSAURUS: Species = {
  id: 'brachiosaurus',
  name: 'Brachiosaurus',
  tagline: 'Gentle Giant Herbivore',
  actuators: 30,
  gait: 'Quadrupedal',
  specialty: 'Neck food reaching',
  stages: [
    { number: 1, title: 'Balance', description: 'Stable quadrupedal stance', video: '/videos/brachiosaurus_ppo_stage1_best.mp4' },
    { number: 2, title: 'Locomotion', description: 'Coordinated four-legged walking', video: '/videos/brachiosaurus_ppo_stage2_best.mp4' },
    { number: 3, title: 'Food Reach', description: 'Walking to food and reaching with neck', video: '/videos/brachiosaurus_ppo_stage3_best.mp4' },
  ],
};

export const ALL_SPECIES: Species[] = [VELOCIRAPTOR, TREX, BRACHIOSAURUS];
