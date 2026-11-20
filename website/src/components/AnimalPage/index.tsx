import React from 'react';
import {useLocation} from '@docusaurus/router';
import useIsBrowser from '@docusaurus/useIsBrowser';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {ANIMALS, animalModelImage, type Animal} from '@site/src/data/animals';
import {headlineFor, posterFor, type Species} from '@site/src/data/species';
import {AnimalHeader} from './AnimalHeader';
import MovementStudy from './MovementStudy';
import {INTERPRETATIONS} from './interpretation';
import styles from './styles.module.css';

const REPO = 'https://github.com/kuds/mesozoic-labs';

function MovementStudies({animal, model}: {animal: Animal; model: Species}) {
  const study = INTERPRETATIONS[animal.slug];
  const result = model.featuredResult;
  const headline = result && headlineFor(result);
  const docsPath = '/docs/models/' + animal.slug + '/' + model.embodiment;
  const details = model.embodiment === 'robot' ? study?.robot : study;

  return (
    <>
      <section aria-labelledby="movements">
        <div className={styles.sectionHead}>
          <h2 id="movements">Movement studies</h2>
          <Link to={docsPath}>Open model notes</Link>
        </div>
        <div className={styles.studyGrid}>
          {model.stages.map((stage) => {
            const label = study?.studies[stage.id];
            const video = stage.video;
            return (
              <MovementStudy
                key={stage.id}
                label={stage.title}
                title={label?.title ?? stage.title}
                description={label?.description ?? stage.description}
                href={REPO + '/blob/main/' + stage.configPath}
                linkLabel="Study settings"
                video={video ? {
                  src: video.path,
                  poster: posterFor(video.path),
                  label: `${model.name} simulation, stage ${stage.label}: ${stage.title}`,
                  caption: `${video.algorithm} simulation · ${video.modelRevisionStatus} model · ${video.verificationStatus} recording`,
                  source: REPO + '/blob/main/website/static' + video.path,
                } : undefined}
              />
            );
          })}
        </div>
      </section>
      <dl className={styles.specs}>
        <div><dt>Support</dt><dd>{model.gait === 'Quadrupedal' ? 'Four limbs' : 'Two hind limbs'}</dd></div>
        <div>
          <dt>{details?.detailLabel ?? 'Observations'}</dt>
          <dd>{details?.detail ?? model.observationDim}</dd>
        </div>
        <div><dt>Whole model</dt><dd>{model.actuators} actuators</dd></div>
      </dl>
      {result && headline && (
        <section className={styles.result} aria-labelledby="run">
          <div>
            <h2 id="run">Published run</h2>
            <p>{result.algorithm} · {result.date} · {result.provenance.modelRevisionStatus} model · {result.provenance.verificationStatus}</p>
          </div>
          <p><strong>{headline.value}</strong> {headline.label}</p>
          <a href={REPO + '/blob/main/' + result.summaryPath}>Run record</a>
        </section>
      )}
    </>
  );
}

function SpeciesSpread({animal}: {animal: Animal}) {
  const location = useLocation();
  const isBrowser = useIsBrowser();
  const version = isBrowser ? new URLSearchParams(location.search).get('version') : null;
  const model = animal.models.find((entry) => entry.embodiment === version) ?? animal.models[0];
  const study = INTERPRETATIONS[animal.slug];
  const isRobot = model.embodiment === 'robot';
  const details = isRobot ? study?.robot : study;
  const hasVersions = animal.models.length > 1;

  return (
    <Layout title={animal.name} description={animal.summary}>
      <main className={styles.page}>
        <AnimalHeader
          name={animal.name}
          scientificName={animal.scientificName}
          classification={study?.classification ?? model.gait}
          collection={hasVersions ? 'Anatomy & robotics' : 'Anatomical studies'}
          modelLabel={isRobot ? 'Robot model' : 'Anatomical model'}
          title={details?.title ?? model.specialty}
          description={details?.anatomy ?? animal.summary}
          image={animalModelImage(model)}
          alt={details?.alt ?? model.name}
          caption={`MuJoCo · ${isRobot ? 'Robot' : 'Anatomical model'} in its reference pose`}
          source={REPO + '/blob/main/' + model.model.path}
        />
        {hasVersions && (
          <nav className={styles.versions} aria-label="Choose a model">
            {animal.models.map((option) => (
              <Link key={option.embodiment}
                to={'/animals/' + animal.slug + '?version=' + option.embodiment}
                aria-current={option.embodiment === model.embodiment ? 'true' : undefined}>
                {option.embodiment === 'robot' ? 'Robot' : 'Anatomical'} model
              </Link>
            ))}
          </nav>
        )}
        <MovementStudies key={model.embodiment} animal={animal} model={model} />
      </main>
    </Layout>
  );
}

export default function AnimalPage({slug}: {slug: string}) {
  const animal = ANIMALS.find((candidate) => candidate.slug === slug);
  if (!animal?.models.length) throw new Error('Unknown catalog animal: ' + slug);
  return <SpeciesSpread key={slug} animal={animal} />;
}
