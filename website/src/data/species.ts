/** Typed website adapter for the generated public species catalog. */

import generatedCatalog from './species.generated.json';

export type TrainingBackend = 'stable-baselines3' | 'jax-mjx';

export interface SpeciesVideo {
  path: string;
  algorithm: string;
  backend: TrainingBackend;
  backendVersion: string | null;
  modelRevisionStatus: 'current' | 'historical';
  verificationStatus: 'verified' | 'unverified';
  repositoryCommit: string | null;
  modelHash: string | null;
  configHash: string | null;
}

export interface AdvancementGate {
  minAverageReward: number | null;
  minAverageEpisodeLength: number | null;
  minAverageForwardVelocity: number | null;
  minSuccessRate: number | null;
  minEvaluationEpisodes: number;
  requiredConsecutive: number;
}

export interface PlantLayerContract {
  schema: string;
  revision: number;
  sha256: string;
}

export interface PlantContract {
  bundleSha256: string;
  sourceClosureSha256: string;
  policyInterface: PlantLayerContract & {observationSchema: string};
  physics: PlantLayerContract;
  visual: PlantLayerContract;
}

export interface SpeciesStage {
  number: number;
  name: string;
  title: string;
  description: string;
  configPath: string;
  timesteps: number;
  advancementGate: AdvancementGate;
  video: SpeciesVideo | null;
}

export interface ResultProvenance {
  modelRevisionStatus: 'current' | 'historical';
  verificationStatus: 'verified' | 'unverified';
  evaluationEpisodes: number | null;
  repositoryCommit: string | null;
  modelHash: string | null;
  configHash: string | null;
}

export interface ResultStage {
  number: number;
  name: string;
  description: string;
  timesteps: number;
  bestEvalReward: number | null;
  finalEvalReward: number | null;
  averageForwardVelocity: number | null;
  averageEpisodeLength: number | null;
  successRate: number | null;
  trainingTime: string | null;
  passed: boolean | null;
}

export interface PublishedResult {
  summaryPath: string;
  algorithm: string;
  backend: TrainingBackend;
  backendVersion: string | null;
  date: string;
  hardware: string | null;
  seed: number | null;
  parallelEnvironments: number | null;
  totalTimesteps: number;
  totalTrainingTime: string | null;
  finalAverageReward: number | null;
  maxAverageForwardVelocity: number | null;
  stage3SuccessRate: number | null;
  provenance: ResultProvenance;
  stages: ResultStage[];
}

export interface Species {
  id: string;
  name: string;
  tagline: string;
  observationDim: number;
  actionDim: number;
  actuators: number;
  gait: 'Bipedal' | 'Quadrupedal';
  specialty: string;
  aliases: string[];
  environmentEntrypoint: string;
  model: {
    path: string;
    nq: number;
    nv: number;
    nu: number;
    dynamicMassKg: number;
    plantContract: PlantContract;
  };
  successMetrics: Array<{
    backends: TrainingBackend[];
    key: string;
    label: string;
    definition: string;
  }>;
  stages: SpeciesStage[];
  historicalResults: PublishedResult[];
  featuredResult: PublishedResult | null;
}

interface RawStage {
  number: number;
  name: string;
  title: string;
  description: string;
  config_path: string;
  timesteps: number;
  advancement_gate: {
    min_avg_reward: number | null;
    min_avg_episode_length: number | null;
    min_avg_forward_velocity: number | null;
    min_success_rate: number | null;
    min_eval_episodes: number;
    required_consecutive: number;
  };
  video: {
    path: string;
    algorithm: string;
    backend: TrainingBackend;
    backend_version: string | null;
    model_revision_status: 'current' | 'historical';
    verification_status: 'verified' | 'unverified';
    repository_commit: string | null;
    model_hash: string | null;
    config_hash: string | null;
  } | null;
}

interface RawResultStage {
  number: number;
  name: string;
  description: string;
  timesteps: number;
  best_eval_reward: number | null;
  final_eval_reward: number | null;
  avg_forward_vel: number | null;
  avg_episode_length: number | null;
  mean_success_rate: number | null;
  training_time: string | null;
  stage_passed: boolean | null;
}

interface RawResult {
  summary_path: string;
  algorithm: string;
  backend: TrainingBackend;
  backend_version: string | null;
  date: string;
  hardware: string | null;
  seed: number | null;
  parallel_envs: number | null;
  total_timesteps: number;
  total_training_time: string | null;
  final_avg_reward: number | null;
  max_average_forward_velocity: number | null;
  stage3_success_rate: number | null;
  provenance: {
    model_revision_status: 'current' | 'historical';
    verification_status: 'verified' | 'unverified';
    evaluation_episodes: number | null;
    repository_commit: string | null;
    model_hash: string | null;
    config_hash: string | null;
  };
  stages: RawResultStage[];
}

interface RawPlantLayerContract {
  schema: string;
  revision: number;
  sha256: string;
}

interface RawSpecies {
  id: string;
  display_name: string;
  tagline: string;
  gait: 'Bipedal' | 'Quadrupedal';
  specialty: string;
  aliases: string[];
  environment: {
    entrypoint: string;
    observation_dim: number;
    action_dim: number;
  };
  model: {
    path: string;
    nq: number;
    nv: number;
    nu: number;
    dynamic_mass_kg: number;
    plant_contract: {
      bundle_sha256: string;
      source_closure_sha256: string;
      policy_interface: RawPlantLayerContract & {observation_schema: string};
      physics: RawPlantLayerContract;
      visual: RawPlantLayerContract;
    };
  };
  success_metrics: Array<{
    backends: TrainingBackend[];
    key: string;
    label: string;
    definition: string;
  }>;
  stages: RawStage[];
  historical_results: RawResult[];
}

interface RawCatalog {
  schema_version: number;
  plant_manifest: {
    path: string;
    schema: string;
    fingerprint_tool_version: number;
    generated_with: {mujoco: string; float_significant_digits: number};
  };
  project_capabilities: Record<string, {label: string; status: string}>;
  notebooks: Array<{id: string; label: string; description: string; path: string}>;
  species: RawSpecies[];
}

const catalog = generatedCatalog as RawCatalog;

function adaptResult(result: RawResult): PublishedResult {
  return {
    summaryPath: result.summary_path,
    algorithm: result.algorithm,
    backend: result.backend,
    backendVersion: result.backend_version,
    date: result.date,
    hardware: result.hardware,
    seed: result.seed,
    parallelEnvironments: result.parallel_envs,
    totalTimesteps: result.total_timesteps,
    totalTrainingTime: result.total_training_time,
    finalAverageReward: result.final_avg_reward,
    maxAverageForwardVelocity: result.max_average_forward_velocity,
    stage3SuccessRate: result.stage3_success_rate,
    provenance: {
      modelRevisionStatus: result.provenance.model_revision_status,
      verificationStatus: result.provenance.verification_status,
      evaluationEpisodes: result.provenance.evaluation_episodes,
      repositoryCommit: result.provenance.repository_commit,
      modelHash: result.provenance.model_hash,
      configHash: result.provenance.config_hash,
    },
    stages: result.stages.map((stage) => ({
      number: stage.number,
      name: stage.name,
      description: stage.description,
      timesteps: stage.timesteps,
      bestEvalReward: stage.best_eval_reward,
      finalEvalReward: stage.final_eval_reward,
      averageForwardVelocity: stage.avg_forward_vel,
      averageEpisodeLength: stage.avg_episode_length,
      successRate: stage.mean_success_rate,
      trainingTime: stage.training_time,
      passed: stage.stage_passed,
    })),
  };
}

function adaptSpecies(raw: RawSpecies): Species {
  const historicalResults = raw.historical_results.map(adaptResult);
  return {
    id: raw.id,
    name: raw.display_name,
    tagline: raw.tagline,
    observationDim: raw.environment.observation_dim,
    actionDim: raw.environment.action_dim,
    actuators: raw.environment.action_dim,
    gait: raw.gait,
    specialty: raw.specialty,
    aliases: raw.aliases,
    environmentEntrypoint: raw.environment.entrypoint,
    model: {
      path: raw.model.path,
      nq: raw.model.nq,
      nv: raw.model.nv,
      nu: raw.model.nu,
      dynamicMassKg: raw.model.dynamic_mass_kg,
      plantContract: {
        bundleSha256: raw.model.plant_contract.bundle_sha256,
        sourceClosureSha256: raw.model.plant_contract.source_closure_sha256,
        policyInterface: {
          schema: raw.model.plant_contract.policy_interface.schema,
          revision: raw.model.plant_contract.policy_interface.revision,
          sha256: raw.model.plant_contract.policy_interface.sha256,
          observationSchema: raw.model.plant_contract.policy_interface.observation_schema,
        },
        physics: raw.model.plant_contract.physics,
        visual: raw.model.plant_contract.visual,
      },
    },
    successMetrics: raw.success_metrics,
    stages: raw.stages.map((stage) => ({
      number: stage.number,
      name: stage.name,
      title: stage.title,
      description: stage.description,
      configPath: stage.config_path,
      timesteps: stage.timesteps,
      advancementGate: {
        minAverageReward: stage.advancement_gate.min_avg_reward,
        minAverageEpisodeLength: stage.advancement_gate.min_avg_episode_length,
        minAverageForwardVelocity: stage.advancement_gate.min_avg_forward_velocity,
        minSuccessRate: stage.advancement_gate.min_success_rate,
        minEvaluationEpisodes: stage.advancement_gate.min_eval_episodes,
        requiredConsecutive: stage.advancement_gate.required_consecutive,
      },
      video: stage.video === null ? null : {
        path: stage.video.path,
        algorithm: stage.video.algorithm,
        backend: stage.video.backend,
        backendVersion: stage.video.backend_version,
        modelRevisionStatus: stage.video.model_revision_status,
        verificationStatus: stage.video.verification_status,
        repositoryCommit: stage.video.repository_commit,
        modelHash: stage.video.model_hash,
        configHash: stage.video.config_hash,
      },
    })),
    historicalResults,
    featuredResult: historicalResults[0] ?? null,
  };
}

/** Poster image generated from the video's first frame (static/img/posters). */
export function posterFor(video: string): string {
  return video.replace('/videos/', '/img/posters/').replace('.mp4', '.jpg');
}

export function backendLabel(backend: TrainingBackend): string {
  return backend === 'stable-baselines3' ? 'Stable-Baselines3' : 'JAX/MJX';
}

export function successMetricForBackend(species: Species, backend: TrainingBackend) {
  const metric = species.successMetrics.find((candidate) => candidate.backends.includes(backend));
  if (!metric) throw new Error(`Generated catalog is missing ${backend} success semantics for ${species.id}`);
  return metric;
}

export const PROJECT_CAPABILITIES = catalog.project_capabilities;
export const PUBLIC_NOTEBOOKS = catalog.notebooks;
export const PLANT_MANIFEST = catalog.plant_manifest;
export const ALL_SPECIES: Species[] = catalog.species.map(adaptSpecies);

function requireSpecies(speciesId: string): Species {
  const species = ALL_SPECIES.find((entry) => entry.id === speciesId);
  if (!species) throw new Error(`Generated catalog is missing species: ${speciesId}`);
  return species;
}

export const VELOCIRAPTOR = requireSpecies('velociraptor');
export const TREX = requireSpecies('trex');
export const BRACHIOSAURUS = requireSpecies('brachiosaurus');
export const DIBOTHROSUCHUS = requireSpecies('dibothrosuchus');

/** Sum of current actuator/action dimensions across implemented species. */
export const TOTAL_ACTUATORS = ALL_SPECIES.reduce((total, species) => total + species.actionDim, 0);

/** Total environment steps across every published historical run. */
export const TOTAL_TRAINED_STEPS = ALL_SPECIES.flatMap((species) => species.historicalResults)
  .reduce((total, result) => total + result.totalTimesteps, 0);
