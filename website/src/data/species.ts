/** Typed website adapter for the generated public species catalog. */

import generatedCatalog from './species.generated.json';

export type TrainingBackend = 'stable-baselines3' | 'jax-mjx';

/** Units a deliverable headline metric can carry (species_catalog.HEADLINE_UNITS). */
export type HeadlineUnit = 'percent' | 'm/s' | 'ratio';

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
  /** Declared curriculum gate kind; none/v1 marks a non-advancing pilot. */
  gateKind: string | null;
  /** Measured gate kind a none/v1 pilot is waiting on (recovery_quality/v1). */
  pendingGateKind: string | null;
  minAverageReward: number | null;
  minAverageEpisodeLength: number | null;
  minAverageForwardVelocity: number | null;
  minSuccessRate: number | null;
  /** stance_quality/v1 criteria; null on every other gate kind. */
  minFullHorizonFraction: number | null;
  maxUnsupportedDuty: number | null;
  maxUnsupportedDutyUcb: number | null;
  /** recovery_quality/v1 criteria; null on every other gate kind. */
  minRecoverySuccessLcb: number | null;
  minPairedSuccessDeltaLcb: number | null;
  recoveryTRecoverSteps: number | null;
  recoveryDwellSteps: number | null;
  /** task_success/v1 bar (the binomial LCB95 on task success); null on every other gate kind. */
  minSuccessLcb: number | null;
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
  /** Semantic stage id from the stage manifest (stance/recovery/...). */
  id: string;
  /** 1-based curriculum position; display only, never identity. */
  position: number;
  /** Historical stage number, or null for semantic-only stages (recovery). */
  number: number | null;
  /** Canonical display reference: the legacy number, or the id. */
  label: string;
  name: string;
  title: string;
  description: string;
  configPath: string;
  timesteps: number;
  /** True when the stage's checkpoint is a published behavior deliverable. */
  deliverable: boolean;
  /** Stage id this node warm-starts from, or null for a root node. */
  warmStartFrom: string | null;
  /** Behavior recipe label (stand/walk/hunt) the node belongs to, or null. */
  recipe: string | null;
  /** Distinct-seed runs a deliverable needs before it stops being provisional (plan §4.5, D-B9; default 1). */
  certificationSeeds: number;
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
  id: string;
  position: number;
  number: number | null;
  label: string;
  /** Recipe label and deliverable flag from the CURRENT manifest entry. */
  recipe: string | null;
  deliverable: boolean;
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
  /** Gate kind the summary recorded the verdict under; null when unrecorded. */
  gateKind: string | null;
  /** Gate kind the species declares for this stage today. */
  currentGateKind: string | null;
  /** True when the verdict was earned under an unrecorded or no-longer-declared gate. */
  gateRetired: boolean;
}

/** One statistic a deliverable's certifying gate measured (decision D-A9). */
export interface HeadlineMetric {
  key: string;
  label: string;
  /** Null when the summary does not record the statistic (stance/recovery until a later phase — decision D-B15). */
  value: number | null;
  unit: HeadlineUnit;
}

/** One published behavior checkpoint of a schema-4 result, in manifest order. */
export interface ResultDeliverable {
  stageId: string;
  /** The summary's stage key ("2" for a numbered stage, the id otherwise). */
  stageKey: string;
  label: string;
  recipe: string | null;
  gateKind: string | null;
  certified: boolean;
  modelHash: string;
  /** Runs certifying this node in the published bundle: this run plus its replicates (plan §4.5, D-B16). */
  replicationCount: number;
  /** The CURRENT config's certification_seeds; provisional is replicationCount < certificationSeeds (D-B10/D-B11). */
  certificationSeeds: number;
  provisional: boolean;
  headline: HeadlineMetric[];
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
  /** Historical ladder headline; read only through headlineFor. */
  stage3SuccessRate: number | null;
  provenance: ResultProvenance;
  /** Empty for every schema-2/3 ladder summary; never synthesized. */
  deliverables: ResultDeliverable[];
  primaryDeliverable: string | null;
  targetDeliverable: string | null;
  stages: ResultStage[];
}

export interface SuccessMetric {
  backends: TrainingBackend[];
  key: string;
  label: string;
  definition: string;
}

export interface DeliverableMetric extends SuccessMetric {
  /** The manifest spelling: a stage id or a recipe label. */
  deliverable: string;
  /** The stage the deliverable resolved to. */
  stageId: string;
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
  trainingNotebooks: string[];
  model: {
    path: string;
    nq: number;
    nv: number;
    nu: number;
    dynamicMassKg: number;
    plantContract: PlantContract;
  };
  successMetrics: SuccessMetric[];
  deliverableMetrics: DeliverableMetric[];
  stages: SpeciesStage[];
  historicalResults: PublishedResult[];
  featuredResult: PublishedResult | null;
}

// The Raw* types below declare the generated JSON's shape key for key; the
// Python test test_website_adapter_declares_every_exported_key pins each of
// them two-sided against the rows build_catalog() exports.

interface RawStage {
  id: string;
  position: number;
  number: number | null;
  label: string;
  name: string;
  title: string;
  description: string;
  config_path: string;
  timesteps: number;
  deliverable: boolean;
  warm_start_from: string | null;
  recipe: string | null;
  certification_seeds: number;
  advancement_gate: {
    gate_kind: string | null;
    pending_gate_kind: string | null;
    min_avg_reward: number | null;
    min_avg_episode_length: number | null;
    min_avg_forward_velocity: number | null;
    min_success_rate: number | null;
    min_full_horizon_fraction: number | null;
    max_unsupported_duty: number | null;
    max_unsupported_duty_ucb: number | null;
    min_recovery_success_lcb: number | null;
    min_paired_success_delta_lcb: number | null;
    recovery_t_recover_steps: number | null;
    recovery_dwell_steps: number | null;
    min_success_lcb: number | null;
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
  id: string;
  position: number;
  number: number | null;
  label: string;
  recipe: string | null;
  deliverable: boolean;
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
  gate_kind: string | null;
  current_gate_kind: string | null;
  gate_retired: boolean;
}

interface RawHeadlineMetric {
  key: string;
  label: string;
  value: number | null;
  unit: HeadlineUnit;
}

interface RawResultDeliverable {
  id: string;
  stage_key: string;
  label: string;
  recipe: string | null;
  gate_kind: string | null;
  certified: boolean;
  model_hash: string;
  replication_count: number;
  certification_seeds: number;
  provisional: boolean;
  headline: RawHeadlineMetric[];
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
  deliverables: RawResultDeliverable[];
  primary_deliverable: string | null;
  target_deliverable: string | null;
  stages: RawResultStage[];
}

interface RawPlantLayerContract {
  schema: string;
  revision: number;
  sha256: string;
}

interface RawSuccessMetric {
  backends: TrainingBackend[];
  key: string;
  label: string;
  definition: string;
}

interface RawDeliverableMetric {
  deliverable: string;
  stage_id: string;
  backends: TrainingBackend[];
  key: string;
  label: string;
  definition: string;
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
  training_notebooks: string[];
  success_metrics: RawSuccessMetric[];
  deliverable_metrics: RawDeliverableMetric[];
  stages: RawStage[];
  historical_results: RawResult[];
}

interface RawCatalog {
  schema_version: number;
  manifest_path: string;
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

// The adapter maps the catalog key for key, so a regenerated JSON of another
// schema would be read through stale field names; refuse it at module load
// (build time) rather than render undefined fields (decision D-A8).
if (catalog.schema_version !== 4) throw new Error(`Generated catalog schema_version must be 4, got ${catalog.schema_version}`);

function adaptSuccessMetric(metric: RawSuccessMetric): SuccessMetric {
  return {
    backends: metric.backends,
    key: metric.key,
    label: metric.label,
    definition: metric.definition,
  };
}

function adaptDeliverableMetric(metric: RawDeliverableMetric): DeliverableMetric {
  return {
    deliverable: metric.deliverable,
    stageId: metric.stage_id,
    backends: metric.backends,
    key: metric.key,
    label: metric.label,
    definition: metric.definition,
  };
}

function adaptHeadlineMetric(metric: RawHeadlineMetric): HeadlineMetric {
  return {
    key: metric.key,
    label: metric.label,
    value: metric.value,
    unit: metric.unit,
  };
}

function adaptDeliverable(deliverable: RawResultDeliverable): ResultDeliverable {
  return {
    stageId: deliverable.id,
    stageKey: deliverable.stage_key,
    label: deliverable.label,
    recipe: deliverable.recipe,
    gateKind: deliverable.gate_kind,
    certified: deliverable.certified,
    modelHash: deliverable.model_hash,
    replicationCount: deliverable.replication_count,
    certificationSeeds: deliverable.certification_seeds,
    provisional: deliverable.provisional,
    headline: deliverable.headline.map(adaptHeadlineMetric),
  };
}

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
    deliverables: result.deliverables.map(adaptDeliverable),
    primaryDeliverable: result.primary_deliverable,
    targetDeliverable: result.target_deliverable,
    stages: result.stages.map((stage) => ({
      id: stage.id,
      position: stage.position,
      number: stage.number,
      label: stage.label,
      recipe: stage.recipe,
      deliverable: stage.deliverable,
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
      gateKind: stage.gate_kind,
      currentGateKind: stage.current_gate_kind,
      gateRetired: stage.gate_retired,
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
    trainingNotebooks: raw.training_notebooks,
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
    successMetrics: raw.success_metrics.map(adaptSuccessMetric),
    deliverableMetrics: raw.deliverable_metrics.map(adaptDeliverableMetric),
    stages: raw.stages.map((stage) => ({
      id: stage.id,
      position: stage.position,
      number: stage.number,
      label: stage.label,
      name: stage.name,
      title: stage.title,
      description: stage.description,
      configPath: stage.config_path,
      timesteps: stage.timesteps,
      deliverable: stage.deliverable,
      warmStartFrom: stage.warm_start_from,
      recipe: stage.recipe,
      certificationSeeds: stage.certification_seeds,
      advancementGate: {
        gateKind: stage.advancement_gate.gate_kind,
        pendingGateKind: stage.advancement_gate.pending_gate_kind,
        minAverageReward: stage.advancement_gate.min_avg_reward,
        minAverageEpisodeLength: stage.advancement_gate.min_avg_episode_length,
        minAverageForwardVelocity: stage.advancement_gate.min_avg_forward_velocity,
        minSuccessRate: stage.advancement_gate.min_success_rate,
        minFullHorizonFraction: stage.advancement_gate.min_full_horizon_fraction,
        maxUnsupportedDuty: stage.advancement_gate.max_unsupported_duty,
        maxUnsupportedDutyUcb: stage.advancement_gate.max_unsupported_duty_ucb,
        minRecoverySuccessLcb: stage.advancement_gate.min_recovery_success_lcb,
        minPairedSuccessDeltaLcb: stage.advancement_gate.min_paired_success_delta_lcb,
        recoveryTRecoverSteps: stage.advancement_gate.recovery_t_recover_steps,
        recoveryDwellSteps: stage.advancement_gate.recovery_dwell_steps,
        minSuccessLcb: stage.advancement_gate.min_success_lcb,
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

/** The certified primary deliverable of a schema-4 result, or null for a ladder summary. */
export function primaryDeliverableOf(result: PublishedResult): ResultDeliverable | null {
  if (result.deliverables.length === 0) return null;
  const primary = result.deliverables.find((deliverable) => deliverable.stageKey === result.primaryDeliverable);
  if (!primary) {
    throw new Error(`Published result ${result.summaryPath} names no primary among its deliverables`);
  }
  return primary;
}

/** A headline metric's value in its unit: 97% / 3.47 m/s / 0.02, or a dash when unrecorded. */
export function formatHeadlineValue(metric: HeadlineMetric): string {
  if (metric.value === null) return '—';
  if (metric.unit === 'percent') return `${Math.round(metric.value * 100)}%`;
  if (metric.unit === 'm/s') return `${metric.value.toFixed(2)} m/s`;
  return metric.value.toFixed(2);
}

/**
 * The one statistic that headlines a published result on the landing page.
 *
 * A schema-4 result headlines its primary deliverable's first gate-kind
 * metric (a walk its velocity, a hunt its task success; stance and recovery
 * name the metric with no value until a later phase — decisions D-A9 and
 * D-B15), and a provisional primary — fewer certifying runs than the
 * current config's certification_seeds — says so with its count (plan
 * §4.5, decision D-B11), so a one-seed pass never headlines as settled.  A
 * ladder summary publishes no deliverable and keeps the historical "Task
 * success" headline from stage3_success_rate, rendered exactly as before
 * (D-A10).
 */
export function headlineFor(result: PublishedResult): {label: string; value: string} {
  const primary = primaryDeliverableOf(result);
  if (primary === null) {
    return {
      label: 'Task success',
      value: result.stage3SuccessRate === null ? '—' : `${Math.round(result.stage3SuccessRate * 100)}%`,
    };
  }
  const provisional = primary.provisional
    ? ` (provisional, ${primary.replicationCount} of ${primary.certificationSeeds} seeds)`
    : '';
  const metric = primary.headline[0];
  if (metric === undefined) {
    return {label: `${primary.recipe ?? primary.stageId} headline`, value: `—${provisional}`};
  }
  const label = metric.label.charAt(0).toUpperCase() + metric.label.slice(1);
  return {label, value: `${formatHeadlineValue(metric)}${provisional}`};
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
export const COMPSOGNATHUS = requireSpecies('compsognathus');
export const COMPSOGNATHUS_ROBOT = requireSpecies('compsognathus_robot');

/** Sum of current actuator/action dimensions across implemented species. */
export const TOTAL_ACTUATORS = ALL_SPECIES.reduce((total, species) => total + species.actionDim, 0);

/** Total environment steps across every published historical run. */
export const TOTAL_TRAINED_STEPS = ALL_SPECIES.flatMap((species) => species.historicalResults)
  .reduce((total, result) => total + result.totalTimesteps, 0);
