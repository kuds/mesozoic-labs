import React from 'react';
import {
  backendLabel,
  successMetricForBackend,
  type AdvancementGate,
  type HeadlineMetric,
  type PublishedResult,
  type ResultDeliverable,
  type ResultStage,
  type Species,
  type SpeciesStage,
} from '@site/src/data/species';

function formatMillions(value: number): string {
  return `${(value / 1_000_000).toLocaleString(undefined, {maximumFractionDigits: 1})}M`;
}

function formatNumber(value: number | null, digits = 2): string {
  return value === null ? '—' : value.toFixed(digits);
}

function formatPercent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function provenanceLabel(result: PublishedResult): string {
  const model = result.provenance.modelRevisionStatus === 'historical' ? 'Historical model' : 'Current model';
  const verification = result.provenance.verificationStatus === 'verified' ? 'verified' : 'unverified';
  const episodes = result.provenance.evaluationEpisodes === null
    ? 'evaluation episode count not recorded'
    : `${result.provenance.evaluationEpisodes} evaluation episodes`;
  const backend = result.backendVersion === null
    ? `${backendLabel(result.backend)} (version not recorded)`
    : `${backendLabel(result.backend)} ${result.backendVersion}`;
  return `${model} · ${verification} · ${episodes} · ${backend}`;
}

function formatVerdict(stage: ResultStage): string {
  // A verdict earned under a gate the species no longer declares -- or under
  // no recorded gate, which every pre-provenance summary is -- reads as a
  // pass/fail of that RETIRED gate, never as a bare "Yes" beneath the current
  // gate's description (same rule as _format_verdict in species_catalog.py).
  if (stage.passed === null) return '—';
  if (!stage.gateRetired) return stage.passed ? 'Yes' : 'No';
  const gate = stage.gateKind ?? 'reward gate';
  return `${stage.passed ? 'passed' : 'failed'} retired gate (${gate})`;
}

function requireCriterion(value: number | null, key: string): number {
  // The Python renderer indexes a frozen recovery threshold directly and
  // fails on a missing one; a recovery_quality/v1 gate without its
  // thresholds is a catalog bug, not a row to render with blanks.
  if (value === null) throw new Error(`recovery_quality/v1 gate is missing ${key}`);
  return value;
}

function formatGate(gate: AdvancementGate): string {
  // Mirrors _format_advancement_gate in species_catalog.py branch for
  // branch; test_website_gate_formatter_mirrors_python pins the phrases and
  // the branch order on both sides.
  //
  // A none/v1 stage is a recorded non-advancing pilot; listing the episode
  // defaults would dress the placeholder up as a permissive gate.
  if (gate.gateKind === 'none/v1') {
    const pending = gate.pendingGateKind === null ? '' : `; gate ${gate.pendingGateKind} pending calibration (P5)`;
    return `non-advancing pilot (gate_kind none/v1); never advances${pending}`;
  }
  // A recovery_quality/v1 verdict is produced once, post-stage, from the
  // frozen gate_resolution.json -- the in-training scheduler refuses the
  // kind outright -- so the generic consecutive-passes tail below would
  // publish hysteresis that never applies to it.
  if (gate.gateKind === 'recovery_quality/v1') {
    const recoveryCriteria = [
      `recovery success LCB95 ≥ ${requireCriterion(gate.minRecoverySuccessLcb, 'min_recovery_success_lcb').toLocaleString()}`,
    ];
    if (gate.minPairedSuccessDeltaLcb !== null) {
      recoveryCriteria.push(
        `paired Δ vs each required frozen null LCB95 ≥ ${gate.minPairedSuccessDeltaLcb.toLocaleString()}`,
      );
    }
    const reentry = requireCriterion(gate.recoveryTRecoverSteps, 'recovery_t_recover_steps').toLocaleString();
    const dwell = requireCriterion(gate.recoveryDwellSteps, 'recovery_dwell_steps').toLocaleString();
    recoveryCriteria.push(
      `re-entry ≤ ${reentry} steps + ${dwell}-step dwell`,
      `≥ ${gate.minEvaluationEpisodes} episodes/evaluation`,
      'verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale)',
    );
    return recoveryCriteria.join('; ');
  }
  const criteria: string[] = [];
  if (gate.minAverageReward !== null) criteria.push(`reward ≥ ${gate.minAverageReward.toLocaleString()}`);
  if (gate.minAverageEpisodeLength !== null) {
    criteria.push(`episode length ≥ ${gate.minAverageEpisodeLength.toLocaleString()}`);
  }
  if (gate.minAverageForwardVelocity !== null) {
    criteria.push(`avg. velocity ≥ ${gate.minAverageForwardVelocity.toLocaleString()} m/s`);
  }
  if (gate.minSuccessRate !== null) criteria.push(`task success ≥ ${formatPercent(gate.minSuccessRate)}`);
  if (gate.minFullHorizonFraction !== null) {
    criteria.push(`full-horizon episodes ≥ ${formatPercent(gate.minFullHorizonFraction)}`);
  }
  if (gate.maxUnsupportedDuty !== null) criteria.push(`unsupported duty ≤ ${gate.maxUnsupportedDuty.toLocaleString()}`);
  if (gate.maxUnsupportedDutyUcb !== null) {
    criteria.push(`unsupported duty 95% upper bound ≤ ${gate.maxUnsupportedDutyUcb.toLocaleString()}`);
  }
  criteria.push(`≥ ${gate.minEvaluationEpisodes} episodes/evaluation`);
  criteria.push(`${gate.requiredConsecutive} consecutive passes`);
  return criteria.join('; ');
}

/** `1 — Balance` / `recovery — Recovery`: the label the tables address a stage by (_stage_heading). */
function stageHeading(stage: SpeciesStage): string {
  return `${stage.label} — ${stage.title}`;
}

/** The behavior label a stage belongs to, tagged when its checkpoint is published (_format_recipe). */
function formatRecipe(stage: SpeciesStage): string {
  if (stage.recipe === null) return '—';
  return stage.deliverable ? `${stage.recipe} (deliverable)` : stage.recipe;
}

/** The parent row's heading, or a dash for a root node (_format_warm_start). */
function formatWarmStart(stage: SpeciesStage, headings: Map<string, string>): string {
  if (stage.warmStartFrom === null) return '—';
  const heading = headings.get(stage.warmStartFrom);
  if (heading === undefined) {
    // The stage manifest only accepts edges to declared earlier entries, so
    // this is a catalog bug, not a manifest error.
    throw new Error(`stage ${stage.id} warm-starts from ${stage.warmStartFrom}, which the species does not list`);
  }
  return heading;
}

/** `task success 96.7%` / `unsupported duty 95% UCB not recorded` (_format_headline_metric, D-A9). */
function formatHeadlineMetric(metric: HeadlineMetric): string {
  if (metric.value === null) return `${metric.label} not recorded`;
  if (metric.unit === 'percent') return `${metric.label} ${formatPercent(metric.value)}`;
  if (metric.unit === 'm/s') return `${metric.label} ${metric.value.toFixed(2)} m/s`;
  return `${metric.label} ${formatNumber(metric.value)}`;
}

function formatCertification(deliverable: ResultDeliverable, primary: boolean): string {
  const status = deliverable.certified ? 'certified' : 'not certified';
  return primary ? `${status}, primary` : status;
}

export function SpeciesSpecifications({species}: {species: Species}): React.JSX.Element {
  const plant = species.model.plantContract;
  return (
    <>
      <h2>Generated specifications</h2>
      <p>
        These values are generated from the environment and compiled MJCF model. “Dynamic mass” is a simulator
        property, not an anatomical weight estimate.
      </p>
      <table>
        <tbody>
          <tr><th>Gait</th><td>{species.gait}</td></tr>
          <tr><th>Observation dimension</th><td>{species.observationDim}</td></tr>
          <tr><th>Action dimension / actuators</th><td>{species.actionDim}</td></tr>
          <tr><th>Generalized coordinates (nq)</th><td>{species.model.nq}</td></tr>
          <tr><th>Generalized velocities (nv)</th><td>{species.model.nv}</td></tr>
          <tr><th>Compiled dynamic model mass</th><td>{species.model.dynamicMassKg.toFixed(1)} kg</td></tr>
          <tr>
            <th>Plant contract revisions</th>
            <td>
              Policy r{plant.policyInterface.revision} · physics r{plant.physics.revision} · visual r{plant.visual.revision}
            </td>
          </tr>
          <tr><th>Model</th><td><code>{species.model.path}</code></td></tr>
        </tbody>
      </table>
    </>
  );
}

export function SpeciesStages({species}: {species: Species}): React.JSX.Element {
  const headings = new Map<string, string>(species.stages.map((stage) => [stage.id, stageHeading(stage)]));
  return (
    <>
      <h2>Current curriculum configuration</h2>
      <p>
        The Stable-Baselines3 budgets and early-advancement gates below come from the current TOML files and can differ
        from older published runs. A stage also ends when its configured budget is exhausted. JAX/MJX uses the same task
        sequence, but its CLI and notebook gate behavior is not yet equivalent. The recipe column names the behavior a
        stage belongs to and whether its checkpoint is a published deliverable; the warm-start column names the stage
        it starts from.
      </p>
      <table>
        <thead>
          <tr>
            <th>Stage</th><th>Recipe</th><th>Warm-start from</th><th>Objective</th>
            <th>SB3 configured budget</th><th>SB3 early-advancement gate</th>
          </tr>
        </thead>
        <tbody>
          {species.stages.map((stage) => (
            <tr key={stage.id}>
              <td>{stageHeading(stage)}</td>
              <td>{formatRecipe(stage)}</td>
              <td>{formatWarmStart(stage, headings)}</td>
              <td>{stage.description}</td>
              <td>{formatMillions(stage.timesteps)}</td>
              <td>{formatGate(stage.advancementGate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>Backend-specific task-success semantics</h3>
      <ul>
        {species.successMetrics.map((metric) => (
          <li key={metric.backends.join('-')}>
            <strong>{metric.backends.map(backendLabel).join(' / ')} — {metric.label}:</strong>{' '}
            {metric.definition}
          </li>
        ))}
      </ul>
      <h3>Per-deliverable success semantics</h3>
      <ul>
        {species.deliverableMetrics.map((metric) => (
          <li key={`${metric.stageId}-${metric.backends.join('-')}`}>
            <strong>
              {metric.deliverable} ({headings.get(metric.stageId) ?? metric.stageId}) ·{' '}
              {metric.backends.map(backendLabel).join(' / ')} — {metric.label}:
            </strong>{' '}
            {metric.definition}
          </li>
        ))}
      </ul>
    </>
  );
}

export function PublishedResults({species}: {species: Species}): React.JSX.Element {
  return (
    <>
      <h2>Published run summaries</h2>
      <p>
        These are historical experiment records. A result is not evidence for the current model revision unless its
        provenance is explicitly marked current and verified.
      </p>
      {species.historicalResults.length === 0 && <p>No run summary has been published for this species.</p>}
      {species.historicalResults.map((result) => (
        <section key={result.summaryPath}>
          <h3>{result.algorithm} · {backendLabel(result.backend)} · {result.date}</h3>
          <p><strong>Provenance:</strong> {provenanceLabel(result)}</p>
          <table>
            <thead>
              <tr>
                <th>Stage</th><th>Trained steps</th><th>Best eval reward</th>
                <th>Avg. forward velocity</th><th>Task success ({backendLabel(result.backend)})</th><th>Passed</th>
              </tr>
            </thead>
            <tbody>
              {result.stages.map((stage) => (
                <tr key={stage.id}>
                  <td>{stage.label} — {stage.name.replace('_', ' ')}</td>
                  <td>{formatMillions(stage.timesteps)}</td>
                  <td>{formatNumber(stage.bestEvalReward)}</td>
                  <td>{stage.averageForwardVelocity === null ? '—' : `${stage.averageForwardVelocity.toFixed(2)} m/s`}</td>
                  <td>{formatPercent(stage.successRate)}</td>
                  <td>{formatVerdict(stage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {/* Only a schema-4 result publishes deliverables; the committed ladder
              summaries publish none, so their rendering is byte-identical to
              the pre-Phase-A page (decision D-A10). */}
          {result.deliverables.length > 0 && (
            <>
              <h4>Deliverables</h4>
              <table>
                <thead>
                  <tr>
                    <th>Deliverable</th><th>Behavior</th><th>Status</th><th>Gate</th><th>Runs</th><th>Headline</th>
                  </tr>
                </thead>
                <tbody>
                  {result.deliverables.map((deliverable) => (
                    <tr key={deliverable.stageId}>
                      <td>{deliverable.label} — {deliverable.stageId}</td>
                      <td>{deliverable.recipe ?? deliverable.stageId}</td>
                      <td>{formatCertification(deliverable, deliverable.stageKey === result.primaryDeliverable)}</td>
                      <td>{deliverable.gateKind ?? 'not recorded'}</td>
                      <td>{deliverable.replicationCount}</td>
                      <td>
                        {deliverable.headline.length === 0 ? '—' : deliverable.headline.map(formatHeadlineMetric).join('; ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          <p>
            <strong>Current {backendLabel(result.backend)} definition for this task label:</strong>{' '}
            {successMetricForBackend(species, result.backend).definition}
          </p>
        </section>
      ))}
    </>
  );
}

export default function SpeciesCatalog({species}: {species: Species}): React.JSX.Element {
  return (
    <>
      <SpeciesSpecifications species={species} />
      <SpeciesStages species={species} />
      <PublishedResults species={species} />
    </>
  );
}
