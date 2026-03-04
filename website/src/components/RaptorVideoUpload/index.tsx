import React, { useCallback, useRef, useState } from 'react';
import styles from './styles.module.css';

interface StageVideo {
  file: File;
  url: string;
}

const stages = [
  {
    number: 1,
    title: 'Balance',
    description: 'Raptor learning to stand without falling',
  },
  {
    number: 2,
    title: 'Locomotion',
    description: 'Raptor learning to walk and run forward',
  },
  {
    number: 3,
    title: 'Strike',
    description: 'Raptor sprinting and attacking with sickle claws',
  },
];

function UploadCard({
  stage,
  video,
  onSelect,
  onRemove,
}: {
  stage: (typeof stages)[number];
  video: StageVideo | null;
  onSelect: (file: File) => void;
  onRemove: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith('video/')) {
        onSelect(file);
      }
    },
    [onSelect],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div className={styles.card}>
      <div className={styles.cardBorder} aria-hidden="true" />
      <div className={styles.stageLabel}>STAGE {stage.number}</div>
      <h3 className={styles.stageTitle}>{stage.title}</h3>
      <p className={styles.stageDescription}>{stage.description}</p>

      {video ? (
        <div className={styles.videoContainer}>
          <video
            src={video.url}
            controls
            className={styles.videoPlayer}
            aria-label={`Raptor stage ${stage.number} video`}
          />
          <div className={styles.videoInfo}>
            <span className={styles.fileName}>{video.file.name}</span>
            <button
              type="button"
              className={styles.removeButton}
              onClick={onRemove}
              aria-label={`Remove stage ${stage.number} video`}
            >
              REMOVE
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`${styles.dropZone} ${dragOver ? styles.dropZoneActive : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          aria-label={`Upload video for stage ${stage.number}: ${stage.title}`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            onChange={handleChange}
            className={styles.fileInput}
            tabIndex={-1}
            aria-hidden="true"
          />
          <div className={styles.uploadIcon} aria-hidden="true">
            &#9654;
          </div>
          <span className={styles.uploadText}>Drop video here or click to upload</span>
          <span className={styles.uploadHint}>MP4, WebM, or MOV</span>
        </div>
      )}
      <div className={styles.cardCorner} aria-hidden="true" />
    </div>
  );
}

export default function RaptorVideoUpload(): React.JSX.Element {
  const [videos, setVideos] = useState<Record<number, StageVideo | null>>({
    1: null,
    2: null,
    3: null,
  });

  const handleSelect = useCallback((stageNumber: number, file: File) => {
    const url = URL.createObjectURL(file);
    setVideos((prev) => {
      const old = prev[stageNumber];
      if (old) URL.revokeObjectURL(old.url);
      return { ...prev, [stageNumber]: { file, url } };
    });
  }, []);

  const handleRemove = useCallback((stageNumber: number) => {
    setVideos((prev) => {
      const old = prev[stageNumber];
      if (old) URL.revokeObjectURL(old.url);
      return { ...prev, [stageNumber]: null };
    });
  }, []);

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
          Upload videos of the Velociraptor at each training stage
        </p>
      </div>
      <div className={styles.grid}>
        {stages.map((stage) => (
          <UploadCard
            key={stage.number}
            stage={stage}
            video={videos[stage.number]}
            onSelect={(file) => handleSelect(stage.number, file)}
            onRemove={() => handleRemove(stage.number)}
          />
        ))}
      </div>
    </section>
  );
}
