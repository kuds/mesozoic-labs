import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useBrokenLinks from '@docusaurus/useBrokenLinks';
import Layout from '@theme/Layout';
import styles from './timeline.module.css';

const repository = 'https://github.com/kuds/mesozoic-labs/blob/';
const hardwareStudy = repository + 'c1feca1227ae2b16fc590c5e39e8e83360650031/hardware/';
const walkingReport = repository + '7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/walk/REPORT_WALK60.md';

interface Milestone {
  id: string;
  when: string;
  dateTime?: string;
  category: string;
  title: string;
  description: string;
  href: string;
  linkText: string;
}

const recorded: Milestone[] = [
  {
    id: 'lab-launch', when: '21 Dec 2025', dateTime: '2025-12-21', category: 'Project launch',
    title: 'The lab opens with simulated animals',
    description: 'Mesozoic Labs introduced an open-source project to explore how dinosaur-inspired bodies move. Articulated models in MuJoCo gave the lab a place to train and test controllers.',
    href: '/blog/welcome', linkText: 'Read the launch announcement',
  },
  {
    id: 'standing-study', when: 'Oct 2026', dateTime: '2026-10', category: 'Simulation study',
    title: 'What counts as standing?',
    description: 'A trained T. rex spread its feet at the start and leaned on one leg. Comparing its foot placement with a motionless reference showed how it had passed the earlier balance test.',
    href: '/blog/teaching-dinosaurs-to-stand', linkText: 'See the balance study',
  },
  {
    id: 'triceratops-build-decision', when: '18 Nov 2026', dateTime: '2026-11-18', category: 'Design decision',
    title: 'Triceratops becomes the first planned build',
    description: 'The lab chose a four-legged robot for its first hardware demonstration. A Triceratops can keep three feet down during a slow walk. The two-legged T. rex study was paused for a later build.',
    href: hardwareStudy + 'trex_form_study/RESUME.md', linkText: 'Read the build decision',
  },
  {
    id: 'triceratops-walking-trials', when: '19 Nov 2026', dateTime: '2026-11-19', category: 'Simulation result',
    title: 'Forty flat-floor trials',
    description: 'The 60 cm Triceratops completed ten trials in each of four simulated floor conditions. It stayed upright in all forty. A separate test on a 5° ramp reduced its speed by 27%.',
    href: walkingReport, linkText: 'See the walking trial results',
  },
];

const planned: Milestone[] = [
  {
    id: 'foreleg-bench-test', when: 'Next', category: 'Planned bench test',
    title: 'Measure one Triceratops foreleg',
    description: 'A single-leg test rig would let the lab measure turning force, speed, heat and the delay between a command and the motor’s response. Those measurements can be checked against the simulation before assembling four legs.',
    href: hardwareStudy + 'triceratops_kickoff/CONSOLIDATION_PLAN.md#52-triceratops-milestones-a-proposal-the-dates-are-planning-estimates',
    linkText: 'Read the proposed bench tests',
  },
  {
    id: 'triceratops-hardware-walk', when: 'Q3 2027', category: 'Hardware target',
    title: 'Walk the 60 cm Triceratops',
    description: 'The goal for July through September 2027 is to assemble the four-legged robot and demonstrate a physical walk. The current design measures 1.2 m overall.',
    href: '/animals/triceratops#scale-comparison', linkText: 'See the 60 cm design to scale',
  },
  {
    id: 'tyrannosaurus-hardware-build', when: '2028', dateTime: '2028', category: 'Hardware target',
    title: 'Build and test the 75 cm T. rex',
    description: 'The biped adds a balance challenge at every step. Its retained simulation studies include tests of how motor response time affects its ability to stay upright.',
    href: hardwareStudy + 'trex_form_study/RESUME.md', linkText: 'Explore the T. rex study and restart plan',
  },
];

function TimelineEntry({milestone}: {milestone: Milestone}) {
  const {id, when, dateTime, category, title, description, href, linkText} = milestone;
  useBrokenLinks().collectAnchor(id);
  return (
    <li className={styles.entry} id={id}>
      <div className={styles.when}>
        {dateTime ? <time dateTime={dateTime}>{when}</time> : when}
      </div>
      <div className={styles.story}>
        <p className={styles.category}>{category}</p>
        <h3>{title}</h3>
        <p>{description}</p>
        <Link to={href}>{linkText}</Link>
      </div>
    </li>
  );
}

export default function Timeline() {
  const {collectAnchor} = useBrokenLinks();
  collectAnchor('recorded');
  collectAnchor('planned');
  return (
    <Layout title="Project timeline" description="Follow Mesozoic Labs from its first simulations to the planned Triceratops bench tests and walking hardware.">
      <main className={styles.page}>
        <header className={styles.intro}>
          <div>
            <h1>Project timeline</h1>
            <p>The lab has tested its 60 cm Triceratops in simulation. Before building all four legs, the next step is to measure how one physical foreleg responds under load.</p>
            <nav className={styles.jump} aria-label="Timeline sections">
              <a href="#recorded">Recorded milestones</a>
              <a href="#planned">Hardware plans</a>
            </nav>
          </div>
          <figure>
            <img src={useBaseUrl('/img/triceratops-gait.jpg')} width={960} height={596}
              alt="The open-frame Triceratops robot taking a step on a checkerboard floor in simulation" />
            <figcaption>The 60 cm design during a simulated gait trial.</figcaption>
          </figure>
        </header>
        <section aria-labelledby="recorded">
          <h2 id="recorded">Recorded milestones</h2>
          <ol className={styles.entries}>
            {recorded.map(milestone => <TimelineEntry key={milestone.id} milestone={milestone} />)}
          </ol>
        </section>
        <section className={styles.plans} aria-labelledby="planned">
          <h2 id="planned">Hardware plans</h2>
          <ol className={styles.entries}>
            {planned.map(milestone => <TimelineEntry key={milestone.id} milestone={milestone} />)}
          </ol>
        </section>
      </main>
    </Layout>
  );
}
