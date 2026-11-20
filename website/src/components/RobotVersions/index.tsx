import React from 'react';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useIsBrowser from '@docusaurus/useIsBrowser';
import MovementStudy, {type MovementStudyProps} from '@site/src/components/AnimalPage/MovementStudy';
import styles from './styles.module.css';

type RobotVersion = {
  slug: string;
  label: string;
  name: string;
  studyLabel: string;
  title: string;
  description: string;
  image: string;
  alt: string;
  caption: string;
  darkPlate?: boolean;
  facts: {label: string; value: string}[];
  source: {label: string; href: string};
  studies: MovementStudyProps[];
};

export type RobotVersionList = readonly [RobotVersion, ...RobotVersion[]];

type RobotVersionsProps = {
  speciesPath: string;
  versions: RobotVersionList;
};

export default function RobotVersions({speciesPath, versions}: RobotVersionsProps) {
  const location = useLocation();
  const isBrowser = useIsBrowser();
  const slug = isBrowser ? new URLSearchParams(location.search).get('version') : null;
  const version = versions.find((option) => option.slug === slug) ?? versions[0];
  const imageUrl = useBaseUrl(version.image);

  return (
    <section className={styles.robots} aria-labelledby="robot-heading">
      <div className={styles.heading}>
        <h2 id="robot-heading">Robot versions</h2>
        <p>These designs are studied in simulation.</p>
      </div>
      <nav aria-label="Choose a robot version" className={styles.versions}>
        {versions.map((option) => (
          <Link
            key={option.slug}
            to={`${speciesPath}?version=${encodeURIComponent(option.slug)}`}
            aria-current={option.slug === version.slug ? 'true' : undefined}>
            <strong>{option.label}</strong>
            <span>{option.name}</span>
          </Link>
        ))}
      </nav>
      <div aria-live="polite" aria-atomic="true">
        <article className={styles.record}>
          <figure className={version.darkPlate ? styles.frame : undefined}>
            <img src={imageUrl} alt={version.alt} />
            <figcaption>{version.caption}</figcaption>
          </figure>
          <div className={styles.details}>
            <p className={styles.label}>{version.studyLabel}</p>
            <h3>{version.title}</h3>
            <p>{version.description}</p>
            <dl>
              {version.facts.map(({label, value}) => (
                <React.Fragment key={label}>
                  <dt>{label}</dt>
                  <dd>{value}</dd>
                </React.Fragment>
              ))}
            </dl>
            <div className={styles.sources}>
              <a href={version.source.href}>{version.source.label}</a>
            </div>
          </div>
        </article>
        <section className={styles.studies} aria-labelledby="movement-heading">
          <div className={styles.heading}>
            <h2 id="movement-heading">Movement studies · {version.label}</h2>
          </div>
          <div className={styles.studyGrid}>
            {version.studies.map((study) => <MovementStudy key={study.href} {...study} />)}
          </div>
        </section>
      </div>
    </section>
  );
}
