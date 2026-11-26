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

function requireCriterion(gate: AdvancementGate, value: number | null, key: string): number {
  // The Python renderer indexes a kind's own threshold directly and fails
  // on a missing one; a recovery_quality/v1 or floor-truth stance gate without
  // its thresholds is a catalog bug, not a row to render with blanks.
  if (value === null) throw new Error(`${gate.gateKind} gate is missing ${key}`);
  return value;
}

function requireCommandBar(gate: AdvancementGate, value: number | null, key: string): number {
  // As requireCriterion: the Python renderer indexes the profile's own bars directly and fails on a missing one.
  if (value === null) throw new Error(`${gate.gateKind} gate is missing ${key}`);
  return value;
}

function formatCommandTrackingGate(gate: AdvancementGate): string {
  // Mirrors _format_command_tracking_gate in species_catalog.py: the profile's families, the gait clause, the
  // declared tolerance overrides and the verdict's source; test_website_gate_formatter_mirrors_python pins the
  // phrases on both sides.
  const command = gate.commandTracking;
  if (command === null) throw new Error(`${gate.gateKind} gate is missing its command-tracking bars`);
  const bar = (value: number | null, key: string) => requireCommandBar(gate, value, key);
  let criteria: string[];
  if (command.profile === 'follow') {
    criteria = [
      `command-following episodes success LCB95 ≥ ${bar(command.minFollowSuccessLcb, 'min_follow_success_lcb').toLocaleString()} and survival LCB95 ≥ ${bar(command.minFollowSurvivalLcb, 'min_follow_survival_lcb').toLocaleString()} over ≥ ${bar(command.minFollowEpisodes, 'min_follow_episodes').toLocaleString()} episodes of ≥ ${bar(command.minFollowHorizonS, 'min_follow_horizon_s').toLocaleString()} s`,
      `an episode succeeds when it survives, stays in tolerance on ≥ ${formatPercent(bar(command.minFollowTrackingFraction, 'min_follow_tracking_fraction'))} of its steps, settles ≥ ${formatPercent(bar(command.minFollowEventSettleFraction, 'min_follow_event_settle_fraction'))} of its commands, ≥ ${formatPercent(bar(command.minFollowStopSettleFraction, 'min_follow_stop_settle_fraction'))} of its stops and ≥ ${formatPercent(bar(command.minFollowTurnSettleFraction, 'min_follow_turn_settle_fraction'))} of its turns (each where the schedule draws them) and passes the gait clause`,
      `≥ ${bar(command.minFollowCommandEpisodes, 'min_follow_command_episodes').toLocaleString()} episodes exposed to each of stops and turns`,
    ];
    if (command.minFollowCourseFraction !== null) {
      criteria.push(
        `course ≥ ${formatPercent(command.minFollowCourseFraction)} of ${bar(command.followCourseDistanceM, 'follow_course_distance_m').toLocaleString()} m`,
      );
    }
    if (command.minFollowReferenceDeltaLcb !== null) {
      criteria.push(`paired Δ success vs the parent policy LCB95 ≥ ${command.minFollowReferenceDeltaLcb.toLocaleString()}`);
    }
  } else {
    criteria = [
      `cruise episodes (a non-zero reset command) success LCB95 ≥ ${bar(command.minCruiseSuccessLcb, 'min_cruise_success_lcb').toLocaleString()} and survival LCB95 ≥ ${bar(command.minCruiseSurvivalLcb, 'min_cruise_survival_lcb').toLocaleString()} over ≥ ${bar(command.minCruiseEpisodes, 'min_cruise_episodes').toLocaleString()} episodes`,
      `a cruise episode succeeds when it reaches the horizon, settles its initial heading within 1 s + |error| / yaw_rate_max + the stage's tracking window ([commands] tracking_window_s, decision D-D64) and holds it 1 s, stays in tolerance on ≥ ${formatPercent(bar(command.minCruiseTrackingFraction, 'min_cruise_tracking_fraction'))} of its steps after the settle (its velocity and yaw-rate errors read on that window's means), makes ≥ ${bar(command.minCruiseSpeedRatio, 'min_cruise_speed_ratio').toLocaleString()} × the commanded speed along the heading after 2 s and passes the gait clause`,
      `stand episodes (an exactly zero command) over ≥ ${bar(command.minStandEpisodes, 'min_stand_episodes').toLocaleString()}: ≥ ${formatPercent(bar(command.minStandSuccessFraction, 'min_stand_success_fraction'))} hold the stop (it settles, and holds on ≥ ${formatPercent(bar(command.minStandHoldFraction, 'min_stand_hold_fraction'))} of the steps after 1 s) and ≤ ${bar(command.maxStandFalls, 'max_stand_falls').toLocaleString()} fall`,
    ];
  }
  const bars = command.gaitCriteria.map(
    ({key, bar: value}) => `${key} ${key.startsWith('min_') ? '≥' : '≤'} ${value.toLocaleString()}`,
  );
  criteria.push(`gait clause (gait_limbs ${command.gaitLimbs}): ${bars.join(', ')}`);
  if (command.toleranceOverrides.length > 0) {
    const overrides = command.toleranceOverrides.map(({key, value}) => `${key} ${value.toLocaleString()}`);
    criteria.push(`tolerances overridden for the gate: ${overrides.join(', ')}`);
  }
  criteria.push(
    'handoff chosen by the gated statistic on a separate selection panel; verdict from command_tracking_report.json on that pair (post-stage; fail-closed when absent)',
  );
  return criteria.join('; ');
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
      `recovery success LCB95 ≥ ${requireCriterion(gate, gate.minRecoverySuccessLcb, 'min_recovery_success_lcb').toLocaleString()}`,
    ];
    if (gate.minPairedSuccessDeltaLcb !== null) {
      recoveryCriteria.push(
        `paired Δ vs each required frozen null LCB95 ≥ ${gate.minPairedSuccessDeltaLcb.toLocaleString()}`,
      );
    }
    const reentry = requireCriterion(gate, gate.recoveryTRecoverSteps, 'recovery_t_recover_steps').toLocaleString();
    const dwell = requireCriterion(gate, gate.recoveryDwellSteps, 'recovery_dwell_steps').toLocaleString();
    recoveryCriteria.push(
      `re-entry ≤ ${reentry} steps + ${dwell}-step dwell`,
      `≥ ${gate.minEvaluationEpisodes} episodes/evaluation`,
      'verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale)',
    );
    return recoveryCriteria.join('; ');
  }
  // A stance_quality/v2 verdict is produced once, post-stage, from the
  // floor-truth stance_gate_report.json rolled on the handoff pair at the
  // certification panel seeds; the in-training manager refuses the kind, so
  // the consecutive-passes tail is not rendered. Every declared per-episode
  // criterion is listed under its TOML key, in the gate's order; both rails
  // read "reward rail" so the generic path's "reward ≥ " stays unique to it.
  if (gate.gateKind === 'stance_quality/v2') {
    const lcb = requireCriterion(gate, gate.minCleanStanceLcb, 'min_clean_stance_lcb').toLocaleString();
    const settle = requireCriterion(gate, gate.settleSteps, 'settle_steps').toLocaleString();
    const stanceCriteria = [
      `clean stance episodes LCB95 ≥ ${lcb} over ≥ ${gate.minEvaluationEpisodes} episodes (settle ${settle} steps)`,
    ];
    const perEpisode = gate.stanceEpisodeCriteria.map(
      ({key, bar}) => `${key} ${key.startsWith('min_') ? '≥' : '≤'} ${bar.toLocaleString()}`,
    );
    stanceCriteria.push(`an episode is clean when it reaches the horizon with ${perEpisode.join(', ')}`);
    if (gate.minFullHorizonFraction !== null) {
      stanceCriteria.push(`full-horizon episodes ≥ ${formatPercent(gate.minFullHorizonFraction)}`);
    }
    if (gate.maxHopOrFallEpisodes !== null) {
      stanceCriteria.push(`hop-or-fall episodes ≤ ${gate.maxHopOrFallEpisodes.toLocaleString()}`);
    }
    if (gate.minAverageReward !== null) stanceCriteria.push(`reward rail ≥ ${gate.minAverageReward.toLocaleString()}`);
    if (gate.minAverageRewardStatueRatio !== null) {
      stanceCriteria.push(`reward rail ≥ ${gate.minAverageRewardStatueRatio.toLocaleString()} × the statue's`);
    }
    stanceCriteria.push(
      'verdict from the floor-truth stance_gate_report.json on the handoff pair (post-stage; fail-closed when absent)',
    );
    return stanceCriteria.join('; ');
  }
  // A stance_quality_heading/v1 verdict (the six-heading certificate) is
  // produced once, post-stage, from stance_heading_report.json: the v2
  // criteria at every heading of the fixed whole-scene grid, each heading's
  // full v2 decision, and the bound on seeds clean at every heading, so
  // min_eval_episodes and min_clean_stance_lcb read as both the per-heading
  // and the grouped panel. No consecutive-passes tail, as for v2.
  if (gate.gateKind === 'stance_quality_heading/v1') {
    const lcb = requireCriterion(gate, gate.minCleanStanceLcb, 'min_clean_stance_lcb').toLocaleString();
    const settle = requireCriterion(gate, gate.settleSteps, 'settle_steps').toLocaleString();
    const headingCriteria = [
      `seeds clean at all six headings (whole scene turned -90, -45, 0, 45, 90, 180°) LCB95 ≥ ${lcb} over ≥ ${gate.minEvaluationEpisodes} seeds (settle ${settle} steps)`,
      `at every heading, clean stance episodes LCB95 ≥ ${lcb} over the same seeds`,
    ];
    const perEpisode = gate.stanceEpisodeCriteria.map(
      ({key, bar}) => `${key} ${key.startsWith('min_') ? '≥' : '≤'} ${bar.toLocaleString()}`,
    );
    headingCriteria.push(`an episode is clean when it reaches the horizon with ${perEpisode.join(', ')}`);
    if (gate.minFullHorizonFraction !== null) {
      headingCriteria.push(`full-horizon episodes ≥ ${formatPercent(gate.minFullHorizonFraction)} at every heading`);
    }
    if (gate.maxHopOrFallEpisodes !== null) {
      headingCriteria.push(`hop-or-fall episodes ≤ ${gate.maxHopOrFallEpisodes.toLocaleString()} at every heading`);
    }
    if (gate.minAverageReward !== null) {
      headingCriteria.push(`reward rail ≥ ${gate.minAverageReward.toLocaleString()} at every heading`);
    }
    if (gate.minAverageRewardStatueRatio !== null) {
      headingCriteria.push(
        `reward rail ≥ ${gate.minAverageRewardStatueRatio.toLocaleString()} × the statue's at the same heading`,
      );
    }
    headingCriteria.push(
      'verdict from the six-heading stance_heading_report.json on the handoff pair (post-stage; fail-closed when absent, never the yaw-zero panel)',
    );
    return headingCriteria.join('; ');
  }
  // A command_tracking/v1 verdict (decision D-D61) is produced once, post-stage, from
  // command_tracking_report.json: the stage's command panel rolled on the handoff its gated statistic chose; the
  // in-training manager refuses the kind, so no consecutive-passes tail, as for the stance kinds.
  if (gate.gateKind === 'command_tracking/v1') return formatCommandTrackingGate(gate);
  const criteria: string[] = [];
  if (gate.minAverageReward !== null) criteria.push(`reward ≥ ${gate.minAverageReward.toLocaleString()}`);
  if (gate.minAverageEpisodeLength !== null) {
    criteria.push(`episode length ≥ ${gate.minAverageEpisodeLength.toLocaleString()}`);
  }
  if (gate.minAverageForwardVelocity !== null) {
    criteria.push(`avg. velocity ≥ ${gate.minAverageForwardVelocity.toLocaleString()} m/s`);
  }
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

/** `clean stance episodes 96.7%` / `unsupported duty 95% UCB not recorded` (_format_headline_metric, D-A9). */
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

function formatReplication(d: ResultDeliverable): string {
  // Mirrors _format_replication in species_catalog.py: the count AND the
  // current config's certification_seeds are always rendered, and a count
  // below the bar is labelled provisional (plan §4.5, decision D-B11);
  // test_website_replication_formatter_mirrors_python pins the phrases.
  const runs = `${d.replicationCount} run${d.replicationCount === 1 ? '' : 's'}`;
  const seeds = `${d.certificationSeeds} seed${d.certificationSeeds === 1 ? '' : 's'}`;
  return `${runs} of ${seeds}${d.provisional ? '; provisional' : ''}`;
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
        from older published runs. A stage also ends when its configured budget is exhausted. The recipe column names the
        behavior a stage belongs to and whether its checkpoint is a published deliverable; the warm-start column names
        the stage it starts from.
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
      {species.historicalResults.length === 0 && <p>No run summary has been published for this model.</p>}
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
                  <td>{stage.label} — {stage.id === 'stance' ? 'Stance' : stage.name.replace('_', ' ')}</td>
                  <td>{formatMillions(stage.timesteps)}</td>
                  <td>{formatNumber(stage.bestEvalReward)}</td>
                  <td>{stage.averageForwardVelocity === null ? '—' : `${stage.averageForwardVelocity.toFixed(2)} m/s`}</td>
                  <td>{formatPercent(stage.successRate)}</td>
                  <td>{formatVerdict(stage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {/* Every published result carries deliverables (decision D-D63 retired
              the schema-2 ladder summaries, which published none and rendered as
              the pre-Phase-A page did: decision D-A10). */}
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
                      <td>{formatReplication(deliverable)}</td>
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
