import React from 'react';
import styles from './styles.module.css';

const stages = [
  {
    number: 1,
    title: 'Balance',
    description: 'Raptor learning to stand without falling',
    video: '/videos/velociraptor_ppo_stage1_best.mp4',
  },
  {
    number: 2,
    title: 'Locomotion',
    description: 'Raptor learning to walk and run forward',
    video: '/videos/velociraptor_ppo_stage2_best.mp4',
  },
  {
    number: 3,
    title: 'Strike',
    description: 'Raptor sprinting and attacking with sickle claws',
    video: '/videos/velociraptor_ppo_stage3_best.mp4',
  },
];

function VideoCard({ stage }: { stage: (typeof stages)[number] }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardBorder} aria-hidden="true" />
      <div className={styles.stageLabel}>STAGE {stage.number}</div>
      <div className={styles.algorithmBadge}>PPO</div>
      <h3 className={styles.stageTitle}>{stage.title}</h3>
      <p className={styles.stageDescription}>{stage.description}</p>
      <div className={styles.videoContainer}>
        <video
          src={stage.video}
          controls
          className={styles.videoPlayer}
          aria-label={`Raptor stage ${stage.number}: ${stage.title}`}
        />
      </div>
      <div className={styles.cardCorner} aria-hidden="true" />
    </div>
  );
}

export default function RaptorVideoUpload(): React.JSX.Element {
  return (
    <section className={styles.section} aria-labelledby="video-upload-heading">
      <div className={styles.sectionHeader}>
        <div className={styles.sectionIcon} aria-hidden="true">
          {'[ RAPTOR VIDEOS ]'}
        </div>
        <h2 className={styles.sectionTitle} id="video-upload-heading">
          Training Stage Videos
        </h2>
        <p className={styles.sectionSubtitle}>
          Velociraptor PPO training progression across each curriculum stage
        </p>
      </div>
      <div className={styles.grid}>
        {stages.map((stage) => (
          <VideoCard key={stage.number} stage={stage} />
        ))}
      </div>
    </section>
  );
}
