import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
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
          <img className={styles.specimenImage} src={useBaseUrl(image)} alt={alt} width={1500} height={750} />
          <div className={styles.plateText}>
            <p className={styles.meta}>{modelLabel}</p>
            <h2>{title}</h2>
            <p>{description}</p>
          </div>
        </div>
        <figcaption className={styles.caption}>
          <span>{caption}</span>
          <a href={source}>{sourceLabel}</a>
        </figcaption>
      </figure>
    </>
  );
}
