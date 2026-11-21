import React, {useCallback, useEffect, useRef, useState} from 'react';
import Link from '@docusaurus/Link';
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
  catalogLink?: {to: string; className: string};
  children?: React.ReactNode;
}

type Gesture = {
  pointerId: number;
  x: number;
  y: number;
  width: number;
  startAngle: number;
  direction: 'pending' | 'horizontal' | 'vertical';
};

export function Turntable({image, alt, motion: {enabled, setEnabled}, className, catalogLink, children}: TurntableProps) {
  const {colorMode} = useColorMode();
  const original = useBaseUrl(image);
  const asset = useBaseUrl(image.replace('/img/animals/', '/videos/turntables/').replace(/\.webp$/, ''));
  const poster = asset + '.webp';
  const src = asset + (colorMode === 'dark' ? '-dark.mp4' : '.mp4');
  const surfaceRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const visible = useRef(false);
  const gesture = useRef<Gesture | null>(null);
  const suppressClick = useRef(false);
  // Fractions of one revolution, independent of the clip's duration and theme.
  const angle = useRef(0);
  const pendingAngle = useRef<number | null>(null);
  const [activated, setActivated] = useState(false);
  const [failedSrc, setFailedSrc] = useState<string>();
  const failed = failedSrc === src;

  const syncPlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const canPlay = enabled && visible.current && !document.hidden &&
      !gesture.current && pendingAngle.current === null && !video.seeking;
    if (canPlay) {
      video.play().catch((error: DOMException) => {
        if (error.name === 'NotAllowedError' && video === videoRef.current) setEnabled(false);
      });
    } else {
      video.pause();
    }
  }, [enabled, setEnabled]);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      visible.current = entry.isIntersecting;
      syncPlayback();
    });
    observer.observe(surfaceRef.current!);
    document.addEventListener('visibilitychange', syncPlayback);
    return () => {
      observer.disconnect();
      document.removeEventListener('visibilitychange', syncPlayback);
    };
  }, [syncPlayback, src]);

  // Finish one decode before asking for the latest pointer position. Replacing a
  // seek on every pointermove can keep the browser from displaying any new frame.
  function flushSeek() {
    const video = videoRef.current;
    if (!video || video.seeking || !Number.isFinite(video.duration) || pendingAngle.current === null) return;
    const target = pendingAngle.current * video.duration;
    pendingAngle.current = null;
    if (Math.abs(video.currentTime - target) > 0.001) video.currentTime = target;
  }

  function turnTo(fraction: number) {
    angle.current = ((fraction % 1) + 1) % 1;
    pendingAngle.current = angle.current;
    setActivated(true);
    videoRef.current?.pause();
    flushSeek();
  }

  function currentAngle() {
    const video = videoRef.current;
    return pendingAngle.current ?? (video && video.duration > 0 ? video.currentTime / video.duration : angle.current);
  }

  function startPointer(event: React.PointerEvent<HTMLDivElement>) {
    if (!event.isPrimary || event.button !== 0 || failed) return;
    suppressClick.current = false;
    gesture.current = {
      pointerId: event.pointerId, x: event.clientX, y: event.clientY,
      width: event.currentTarget.clientWidth, startAngle: currentAngle(), direction: 'pending',
    };
    videoRef.current?.pause();
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function movePointer(event: React.PointerEvent<HTMLDivElement>) {
    const drag = gesture.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (drag.direction === 'pending') {
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 6) return;
      drag.direction = Math.abs(dx) >= Math.abs(dy) ? 'horizontal' : 'vertical';
      suppressClick.current = true;
    }
    if (drag.direction === 'horizontal') turnTo(drag.startAngle - dx / drag.width);
  }

  function endPointer(event: React.PointerEvent<HTMLDivElement>) {
    if (gesture.current?.pointerId !== event.pointerId) return;
    gesture.current = null;
    syncPlayback();
  }

  function pressKey(event: React.KeyboardEvent) {
    if (failed || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    turnTo(currentAngle() + (event.key === 'ArrowLeft' ? 1 : -1) / 24);
  }

  const surface = (
    <div ref={surfaceRef} className={failed ? undefined : styles.surface}
      tabIndex={catalogLink || failed ? undefined : 0}
      role={catalogLink || failed ? undefined : 'group'}
      aria-label={catalogLink || failed ? undefined : 'Rotate the model with the left and right arrow keys'}
      aria-keyshortcuts={catalogLink || failed ? undefined : 'ArrowLeft ArrowRight'}
      onKeyDown={catalogLink ? undefined : pressKey}
      onPointerDown={startPointer} onPointerMove={movePointer} onPointerUp={endPointer}
      onPointerCancel={endPointer} onLostPointerCapture={endPointer}
      onClickCapture={event => {
        if (suppressClick.current && event.detail !== 0) {
          event.preventDefault();
          event.stopPropagation();
        }
        suppressClick.current = false;
      }}>
      {failed || (!enabled && !activated) ? (
        <img className={className} src={poster} alt={alt} width={800} height={450} draggable={false}
          onError={event => {
            const fallback = event.currentTarget;
            if (fallback.getAttribute('src') !== original) fallback.src = original;
          }}
        />
      ) : (
        <video ref={videoRef} className={className} src={src} poster={poster}
          muted loop playsInline preload={activated ? 'auto' : 'none'} width={800} height={450} draggable={false}
          role={alt ? 'img' : undefined} aria-label={alt || undefined} aria-hidden={alt ? undefined : true}
          onPlaying={() => setActivated(true)}
          onTimeUpdate={event => {
            const v = event.currentTarget;
            if (!v.paused && !v.seeking && v.duration > 0) angle.current = v.currentTime / v.duration;
          }}
          onLoadedMetadata={() => {
            pendingAngle.current ??= angle.current;
            flushSeek();
            syncPlayback();
          }}
          onSeeked={() => { flushSeek(); syncPlayback(); }}
          onError={() => setFailedSrc(src)}
        />
      )}
    </div>
  );

  return catalogLink ? (
    <Link to={catalogLink.to} className={catalogLink.className} draggable={false}
      aria-keyshortcuts="ArrowLeft ArrowRight" onKeyDown={pressKey}>
      {surface}
      {children}
    </Link>
  ) : surface;
}
