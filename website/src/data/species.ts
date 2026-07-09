/**
 * Single source of truth for the species/stage/video data shown on the
 * homepage species showcase and on each model's docs page — previously
 * duplicated in both places, which let the copies drift.
 *
 * Headline training numbers are imported AT BUILD TIME from the shipped
 * result summaries in ../../results/, so the site can't drift from the
 * published runs — retrain, update summary.json, rebuild.
 */

import raptorPpo from '../../../results/velociraptor/ppo/summary.json';
import trexPpo from '../../../results/trex/ppo/summary.json';
import brachioPpo from '../../../results/brachiosaurus/ppo/summary.json';
import raptorSac from '../../../results/velociraptor/sac/summary.json';

export interface SpeciesStage {
  number: number;
  title: string;
  description: string;
  video: string;
}

export interface SpeciesResults {
  algorithm: string;
  date: string;
  /** Stage-3 task success rate over the 30-episode final eval, 0..1. */
  successRate: number;
  /** Best per-stage average forward velocity (m/s) — the locomotion stage. */
  topSpeed: number;
  totalTimesteps: number;
}

export interface Species {
  id: string;
  name: string;
  tagline: string;
  actuators: number;
  gait: 'Bipedal' | 'Quadrupedal';
  specialty: string;
  stages: SpeciesStage[];
  results: SpeciesResults;
}

interface RunSummary {
  algorithm: string;
  date: string;
  stages: Record<string, {timesteps?: number; avg_forward_vel?: number; mean_success_rate?: number}>;
}

function summarizeRun(summary: RunSummary): SpeciesResults {
  const stages = Object.values(summary.stages);
  return {
    algorithm: summary.algorithm,
    date: summary.date,
    successRate: summary.stages['3']?.mean_success_rate ?? 0,
    topSpeed: Math.max(...stages.map((s) => s.avg_forward_vel ?? 0)),
    totalTimesteps: stages.reduce((t, s) => t + (s.timesteps ?? 0), 0),
  };
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
  results: summarizeRun(raptorPpo as RunSummary),
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
  results: summarizeRun(trexPpo as RunSummary),
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
  results: summarizeRun(brachioPpo as RunSummary),
  stages: [
    { number: 1, title: 'Balance', description: 'Stable quadrupedal stance', video: '/videos/brachiosaurus_ppo_stage1_best.mp4' },
    { number: 2, title: 'Locomotion', description: 'Coordinated four-legged walking', video: '/videos/brachiosaurus_ppo_stage2_best.mp4' },
    { number: 3, title: 'Food Reach', description: 'Walking to food and reaching with neck', video: '/videos/brachiosaurus_ppo_stage3_best.mp4' },
  ],
};

export const ALL_SPECIES: Species[] = [VELOCIRAPTOR, TREX, BRACHIOSAURUS];

/** Sum of actuators across the three species (hero stat). */
export const TOTAL_ACTUATORS = ALL_SPECIES.reduce((t, sp) => t + sp.actuators, 0);

/** Total environment steps across all published runs, incl. the SAC run. */
export const TOTAL_TRAINED_STEPS = [raptorPpo, trexPpo, brachioPpo, raptorSac]
  .map((run) => summarizeRun(run as RunSummary).totalTimesteps)
  .reduce((a, b) => a + b, 0);
