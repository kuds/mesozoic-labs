import React from 'react';
import styles from './styles.module.css';

interface Stage {
  number: number;
  title: string;
  description: string;
  video: string;
}

interface SpeciesVideoSectionProps {
  species: string;
  stages: Stage[];
}

function VideoCard({ species, stage }: { species: string; stage: Stage }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardBorder} aria-hidden="true" />
      <div className={styles.stageLabel}>STAGE {stage.number}</div>
      <h3 className={styles.stageTitle}>{stage.title}</h3>
      <p className={styles.stageDescription}>{stage.description}</p>
      <div className={styles.videoContainer}>
        <video
          src={stage.video}
          controls
          className={styles.videoPlayer}
          aria-label={`${species} stage ${stage.number}: ${stage.title}`}
        />
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
        {species} PPO training progression across each curriculum stage.
      </p>
      <div className={styles.grid}>
        {stages.map((stage) => (
          <VideoCard key={stage.number} species={species} stage={stage} />
        ))}
      </div>
    </div>
  );
}
