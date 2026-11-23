import type {RobotVersionList} from '@site/src/components/RobotVersions';

const repo = 'https://github.com/kuds/mesozoic-labs';
const build = `${repo}/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build`;
const concept = `${repo}/blob/84a13e1e7729ff4e92f1376db6464975836b3bcc/hardware/triceratops_design_study`;
export const anatomicalStudy = `${repo}/blob/main/examples/triceratops_horridus/README.md`;

export const triceratopsVersions: RobotVersionList = [
  {
    slug: '60-cm',
    label: '60 cm',
    name: 'Current design',
    studyLabel: 'Current design · Walking trials',
    title: 'The 60 cm Triceratops design',
    description: 'The selected design stands 60 cm at the frill and measures 1.2 m from horn tip to tail tip. The Q3 2027 target is to assemble the four-legged robot and demonstrate a physical walk.',
    image: '/img/triceratops-gait.jpg',
    alt: 'A green-frilled Triceratops robot balancing on its legs over a checkerboard floor during a simulated gait trial',
    caption: 'Simulation still · Learned controller, 6 seconds into a gait trial',
    darkPlate: true,
    facts: [
      {label: 'Target mass', value: 'About 12 kg.'},
      {label: 'Next physical test', value: 'One foreleg on the bench.'},
    ],
    source: {label: 'Read the 60 cm model report', href: `${build}/sim/REPORT_60CM.md`},
    studies: [
      {
        label: 'Locomotion · Learned controller',
        title: 'Learning to trot',
        description: 'With the speed set to 0.25 m/s, the learned controller reached 0.238 m/s. The feet cycled in diagonal pairs about 2.4 times a second.',
        href: `${build}/rl/analysis/REPORT_GAIT_PIPE1.md`,
        linkLabel: 'Read the gait analysis',
      },
      {
        label: 'Locomotion · Model-based controller',
        title: 'Testing the trot',
        description: 'Across 40 flat-floor trials, the robot stayed upright and traveled at 0.247–0.254 m/s. A separate 5° ramp test reduced its speed by 27%.',
        href: `${build}/walk/REPORT_WALK60.md`,
        linkLabel: 'Read the walking report',
      },
    ],
  },
  {
    slug: '75-cm',
    label: '75 cm',
    name: 'Earlier concept',
    studyLabel: 'Earlier design · Stance study',
    title: 'The 75 cm concept',
    description: 'The first study paired the adult reference with a 1.5 m long robot. Twelve leg drives support the chassis, with separate servos for the neck and tail.',
    image: '/img/animals/triceratops-concept.webp',
    alt: 'The earlier Triceratops robot with an open chassis, a mechanical neck and four load-bearing legs',
    caption: 'MuJoCo still · 15 kg mechanical concept in its neutral pose',
    facts: [
      {label: 'Model height', value: '75 cm at the frill.'},
      {label: 'Foot spacing', value: '65 cm between the front and rear feet.'},
    ],
    source: {label: 'Open the design study', href: `${repo}/pull/669`},
    studies: [
      {
        label: 'Stance',
        title: 'Stand on four feet',
        description: 'With its joints set to their neutral positions, the 75 cm robot stood on all four feet for three seconds in MuJoCo.',
        href: `${concept}/models/README.md`,
        linkLabel: 'Read the stance study',
      },
    ],
  },
];
