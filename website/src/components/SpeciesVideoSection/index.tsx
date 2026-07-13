import React from 'react';
import { backendLabel, posterFor, type SpeciesStage } from '@site/src/data/species';
import styles from './styles.module.css';

interface SpeciesVideoSectionProps {
  species: string;
  stages: SpeciesStage[];
}

function VideoCard({ species, stage }: { species: string; stage: SpeciesStage }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardBorder} aria-hidden="true" />
      <div className={styles.stageLabel}>STAGE {stage.number}</div>
      <h3 className={styles.stageTitle}>{stage.title}</h3>
      <p className={styles.stageDescription}>{stage.description}</p>
      {stage.video && (
        <p className={styles.videoProvenance}>
          {stage.video.algorithm} · {backendLabel(stage.video.backend)} artifact ·{' '}
          {stage.video.modelRevisionStatus} model · {stage.video.verificationStatus} ·{' '}
          {stage.video.backendVersion === null
            ? 'backend version not recorded'
            : `version ${stage.video.backendVersion}`}. It is not evidence for the current model or stage config.
        </p>
      )}
      <div className={styles.videoContainer}>
        {stage.video ? (
          <video
            src={stage.video.path}
            controls
            preload="none"
            poster={posterFor(stage.video.path)}
            className={styles.videoPlayer}
            aria-label={`${species} stage ${stage.number}: ${stage.title}`}
          />
        ) : (
          <p className={styles.videoUnavailable}>No published video is available for this stage.</p>
        )}
      </div>
    </div>
  );
}

export default function SpeciesVideoSection({
  species,
  stages,
}: SpeciesVideoSectionProps): React.JSX.Element {
  return (
    <div className={styles.section}>
      <p className={styles.subtitle}>
        Published videos for {species}, where available. Artifact provenance is shown on each stage.
      </p>
      <div className={styles.grid}>
        {stages.map((stage) => (
          <VideoCard key={stage.number} species={species} stage={stage} />
        ))}
      </div>
    </div>
  );
}
