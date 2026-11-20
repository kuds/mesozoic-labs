import React, {useId, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './movement.module.css';

type Recording = {
  src: string;
  poster: string;
  label: string;
  caption: string;
  source: string;
};

export type MovementStudyProps = {
  label: string;
  title: string;
  description: string;
  href: string;
  linkLabel: string;
  video?: Recording;
};

export function StudyRecording({recording}: {recording: Recording}) {
  const [failed, setFailed] = useState(false);
  const captionId = useId();
  const src = useBaseUrl(recording.src);
  const poster = useBaseUrl(recording.poster);

  return (
    <figure className={styles.recording}>
      {!failed && (
        <video
          controls
          playsInline
          preload="none"
          src={src}
          poster={poster}
          aria-label={recording.label}
          aria-describedby={captionId}
          onPlay={(event) => {
            const active = event.currentTarget;
            active.closest('main')?.querySelectorAll('video').forEach((player) => {
              if (player !== active && !player.paused) player.pause();
            });
          }}
          onError={() => setFailed(true)}
        />
      )}
      <figcaption id={captionId}>{recording.caption}</figcaption>
      <a href={recording.source}>{failed ? 'Open recording at source' : 'Original recording'}</a>
    </figure>
  );
}

export default function MovementStudy({label, title, description, href, linkLabel, video}: MovementStudyProps) {
  return (
    <article className={styles.study}>
      <p className={styles.label}>{label}</p>
      <h3>{title}</h3>
      <p>{description}</p>
      {video && <StudyRecording key={video.src} recording={video} />}
      <a href={href}>{linkLabel}</a>
    </article>
  );
}
