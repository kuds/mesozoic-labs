import React from 'react';
import {
  backendLabel,
  successMetricForBackend,
  type AdvancementGate,
  type PublishedResult,
  type Species,
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

function formatGate(gate: AdvancementGate): string {
  // A none/v1 stage is a recorded non-advancing pilot; listing the episode
  // defaults would dress the placeholder up as a permissive gate (same rule
  // as the Python renderer in species_catalog.py).
  if (gate.gateKind === 'none/v1') {
    const pending = gate.pendingGateKind === null ? '' : `; gate ${gate.pendingGateKind} pending calibration (P5)`;
    return `non-advancing pilot (gate_kind none/v1); never advances${pending}`;
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
  criteria.push(`≥ ${gate.minEvaluationEpisodes} episodes/evaluation`);
  criteria.push(`${gate.requiredConsecutive} consecutive passes`);
  return criteria.join('; ');
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
  return (
    <>
      <h2>Current curriculum configuration</h2>
      <p>
        The Stable-Baselines3 budgets and early-advancement gates below come from the current TOML files and can differ
        from older published runs. A stage also ends when its configured budget is exhausted. JAX/MJX uses the same task
        sequence, but its CLI and notebook gate behavior is not yet equivalent.
      </p>
      <table>
        <thead>
          <tr><th>Stage</th><th>Objective</th><th>SB3 configured budget</th><th>SB3 early-advancement gate</th></tr>
        </thead>
        <tbody>
          {species.stages.map((stage) => (
            <tr key={stage.id}>
              <td>{stage.label} — {stage.title}</td>
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
                  <td>{stage.passed === null ? '—' : stage.passed ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
