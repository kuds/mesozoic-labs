import React, {useEffect, useRef, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {useColorMode} from '@docusaurus/theme-common';
import styles from './turntable.module.css';

export function useTurntableMotion() {
  const [enabled, setEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const applyPreference = () => setEnabled(!preference.matches);
    applyPreference();
    preference.addEventListener('change', applyPreference);
    return () => preference.removeEventListener('change', applyPreference);
  }, []);

  return {enabled, setEnabled};
}

type Motion = ReturnType<typeof useTurntableMotion>;

export function RotationButton({motion, plural = false}: {motion: Motion; plural?: boolean}) {
  if (motion.enabled === null) return null;
  return (
    <button type="button" className={styles.control} onClick={() => motion.setEnabled(!motion.enabled)}>
      {motion.enabled ? 'Pause' : 'Play'} {plural ? 'rotations' : 'rotation'}
    </button>
  );
}

interface TurntableProps {
  image: string;
  alt: string;
  motion: Motion;
  className: string;
}

export function Turntable({image, alt, motion: {enabled, setEnabled}, className}: TurntableProps) {
  const {colorMode} = useColorMode();
  const original = useBaseUrl(image);
  const asset = useBaseUrl(image.replace('/img/animals/', '/videos/turntables/').replace(/\.webp$/, ''));
  const poster = asset + '.webp';
  const src = asset + (colorMode === 'dark' ? '-dark.mp4' : '.mp4');
  const videoRef = useRef<HTMLVideoElement>(null);
  const [failedSrc, setFailedSrc] = useState<string>();
  const [hasPlayed, setHasPlayed] = useState(false);
  const failed = failedSrc === src;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !enabled || failed) return;

    let inView = false;
    let active = true;
    const updatePlayback = () => {
      if (!inView || document.hidden) {
        video.pause();
        return;
      }
      video.play().catch((error: DOMException) => {
        if (active && error.name === 'NotAllowedError') setEnabled(false);
      });
    };
    const observer = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      updatePlayback();
    });
    observer.observe(video);
    document.addEventListener('visibilitychange', updatePlayback);
    return () => {
      active = false;
      observer.disconnect();
      document.removeEventListener('visibilitychange', updatePlayback);
      video.pause();
    };
  }, [enabled, failed, setEnabled, src]);

  if (failed || (!enabled && !hasPlayed)) {
    return (
      <img className={className} src={poster} alt={alt} width={800} height={450}
        onError={(event) => {
          const fallback = event.currentTarget;
          if (fallback.getAttribute('src') !== original) fallback.src = original;
        }}
      />
    );
  }
  return (
    <video ref={videoRef} className={className} src={src} poster={poster}
      muted loop playsInline preload="none" width={800} height={450}
      role={alt ? 'img' : undefined} aria-label={alt || undefined} aria-hidden={alt ? undefined : true}
      onPlaying={() => setHasPlayed(true)}
      onError={() => setFailedSrc(src)}
    />
  );
}
