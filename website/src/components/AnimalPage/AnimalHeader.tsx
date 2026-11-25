import React from 'react';
import Link from '@docusaurus/Link';
import {RotationButton, Turntable, useTurntableMotion} from './Turntable';
import styles from './styles.module.css';

interface AnimalHeaderProps {
  name: string;
  scientificName: string;
  classification: string;
  collection: string;
  modelLabel: string;
  title: string;
  description: string;
  image: string;
  alt: string;
  caption: string;
  source: string;
  sourceLabel?: string;
  compact?: boolean;
}

export function AnimalHeader({
  name, scientificName, classification, collection, modelLabel, title, description,
  image, alt, caption, source, sourceLabel = 'Model source', compact = false,
}: AnimalHeaderProps) {
  const motion = useTurntableMotion();
  return (
    <>
      <div className={styles.breadcrumb}>
        <Link to="/animals">Animal collection</Link>
        <span aria-hidden="true"> / </span>{collection}
      </div>
      <div className={styles.titleRow}>
        <div>
          <h1>{name}</h1>
          <p className={styles.scientificName}><i>{scientificName}</i></p>
        </div>
        <p className={styles.classification}>{classification}</p>
      </div>
      <figure className={compact ? styles.compactSpecimen : styles.specimen}>
        <div className={styles.plate}>
          <Turntable key={image} className={styles.specimenImage} image={image} alt={alt} motion={motion} />
          <div className={styles.plateText}>
            <p className={styles.meta}>{modelLabel}</p>
            <h2>{title}</h2>
            <p>{description}</p>
          </div>
        </div>
        <figcaption className={styles.caption}>
          <span className={styles.captionControls}>
            {caption} · Rotate by dragging, swiping or using the left and right arrow keys
            <RotationButton motion={motion} />
          </span>
          <a href={source}>{sourceLabel}</a>
        </figcaption>
      </figure>
    </>
  );
}
