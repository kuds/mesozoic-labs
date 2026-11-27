/** Museum labels for the models' distinguishing anatomy and movement experiments. */
export interface StudyLabel {
  title: string;
  description: string;
  referenceRecording?: {path: string; caption: string; controllerPath: string};
}

export interface Interpretation {
  classification: string;
  title: string;
  anatomy: string;
  alt: string;
  detailLabel: string;
  detail: string;
  /**
   * The robot page's own text. Its `studies`, keyed by stage id like the species' own, replace the species'
   * label of any stage whose anatomical text does not fit the robot. A species label shown on a robot page never
   * brings its `referenceRecording`, which records the anatomical model.
   */
  robot?: {
    title: string;
    anatomy: string;
    alt: string;
    detailLabel: string;
    detail: string;
    studies?: Record<string, StudyLabel>;
  };
  studies: Record<string, StudyLabel>;
}

/**
 * The label of one stage on a model's page: on a robot page the robot's own label first, then the species' label
 * without its reference recording (a recording of the anatomical model).
 */
export function studyLabel(
  interpretation: Interpretation | undefined,
  embodiment: string,
  stageId: string,
): StudyLabel | undefined {
  const label = interpretation?.studies[stageId];
  if (embodiment !== 'robot') return label;
  const robotLabel = interpretation?.robot?.studies?.[stageId];
  if (robotLabel) return robotLabel;
  return label && {title: label.title, description: label.description};
}

export const INTERPRETATIONS: Record<string, Interpretation> = {
  allosaurus_fragilis: {
    classification: 'Late Jurassic · Theropod',
    title: 'Walking on three toes',
    anatomy: 'The center toe reaches farthest. Two shorter toes spread to either side, and a small inner toe sits above the ground. Behind the hips, a long, tapering tail balances the trunk and crested skull.',
    alt: 'Allosaurus with a crest ahead of each eye, bent hind limbs and a tapered tail held above the floor',
    detailLabel: 'Weight-bearing toes', detail: 'Three on each foot',
    studies: {
      stance: {title: 'Share the load', description: 'The standing trial measures how much weight each foot carries and how far the center of mass lies inside the ground contacts.'},
      locomotion: {
        title: 'Follow a footfall',
        description: 'Watch one foot swing while the other supports the body. The force readings come from contact with the simulated floor.',
        referenceRecording: {
          path: '/videos/allosaurus_fragilis_anatomical_reference_walk.mp4',
          caption: 'Reference controller · Level ground · Real-time playback',
          controllerPath: 'environments/allosaurus_fragilis/scripts/reference_walk.py',
        },
      },
    },
  },
  brachiosaurus_altithorax: {
    classification: 'Late Jurassic · Sauropod',
    title: 'A long neck over four legs',
    anatomy: 'The raised shoulders sit above compact forefeet. The hindfeet have padded heels and three claws. A four-beat walk transfers weight between them while the long neck holds the head above the chest.',
    alt: 'Brachiosaurus with a deep chest, raised shoulders, tapered tail and elevated neck',
    detailLabel: 'Head and neck control', detail: '6 actuators',
    studies: {
      stance: {title: 'Hold steady', description: 'Can the body keep its balance on four feet with the neck held high?'},
      locomotion: {
        title: 'Coordinate a stride',
        description: 'Watch the hindfoot land, followed by the forefoot on the same side. The bars show when each foot carries weight.',
        referenceRecording: {
          path: '/videos/brachiosaurus_altithorax_anatomical_reference_walk.mp4',
          caption: 'Scripted reference walk · Physics revision 6 · Measured speed 0.34 m/s',
          controllerPath: 'environments/brachiosaurus_altithorax/scripts/reference_walk.py',
        },
      },
    },
  },
  velociraptor_mongoliensis: {
    classification: 'Late Cretaceous · Theropod',
    title: 'A claw held clear of the ground',
    anatomy: 'Each foot supports the body on a rounded pad and two toes. The sickle claw rides above them. The model can move that claw separately.',
    alt: 'Feathered Velociraptor with a long, low snout, folded arms and a raised sickle claw on each foot',
    detailLabel: 'Tail control', detail: '4 actuators',
    studies: {
      stance: {title: 'Balance on two feet', description: 'The stance test checks how each foot bears weight when the same animal faces six different directions.'},
      locomotion: {
        title: 'Place each foot',
        description: 'Watch the two long toes land as the sickle claw stays raised. The bars show when each foot carries weight.',
        referenceRecording: {
          path: '/videos/velociraptor_mongoliensis_anatomical_reference_walk.mp4',
          caption: 'Scripted reference walk · Level ground · Measured speed 0.258 m/s',
          controllerPath: 'environments/velociraptor_mongoliensis/scripts/reference_walk.py',
        },
      },
    },
  },
  tyrannosaurus_rex: {
    classification: 'Late Cretaceous · Theropod',
    title: 'Balancing a heavy head',
    anatomy: 'A five-segment tail extends behind the hips, counterbalancing a forward-leaning body and a large skull. The two hind limbs carry the model while the small forelimbs stay folded.',
    alt: 'Tyrannosaurus model with a large skull, small arms, two weight-bearing legs and a segmented tail',
    detailLabel: 'Tail structure', detail: '5 segments',
    studies: {
      stance: {title: 'Stand over the feet', description: 'The model balances its forward-leaning trunk and skull above two planted hind feet.'},
      recovery: {title: 'Recover from a push', description: 'Scheduled pushes disturb a standing model. The controller tries to recover its upright stance.'},
      locomotion: {
        title: 'Follow the weight from foot to foot',
        description: 'The two views show the same walk. The force readings under the feet track the transfer of weight through 23 alternating touchdowns in the final 12 seconds.',
        referenceRecording: {
          path: '/videos/tyrannosaurus_rex_anatomical_reference_walk.mp4',
          caption: 'Scripted reference walk · Level ground · Mean speed 0.638 m/s in the final 12 seconds',
          controllerPath: 'environments/tyrannosaurus_rex/scripts/reference_walk.py',
        },
      },
    },
  },
  dibothrosuchus_elaphros: {
    classification: 'Early Jurassic · Crocodylomorph',
    title: 'A high walk on four limbs',
    anatomy: 'Long hind limbs and a slender neck give this small crocodile relative its distinctive posture. Its four feet support the body from underneath while the tail stays above the ground.',
    alt: 'Slender, four-legged Dibothrosuchus model with an elongated snout and segmented tail',
    detailLabel: 'Hind limbs', detail: 'Longer than forelimbs',
    studies: {
      stance: {title: 'Stand tall', description: 'The foot pads hold the body above the ground as the animal maintains its erect stance.'},
      locomotion: {
        title: 'Trace a four-beat walk',
        description: 'The left hindfoot lands first, followed by the left forefoot, right hindfoot and right forefoot. The replay shows when each foot presses on the floor. The separate training task asks for coordination in diagonal pairs.',
        referenceRecording: {
          path: '/videos/dibothrosuchus_elaphros_anatomical_reference_walk.mp4',
          caption: 'Scripted reference walk · Level ground · Measured foot contacts',
          controllerPath: 'environments/dibothrosuchus_elaphros/scripts/reference_walk.py',
        },
      },
    },
  },
  compsognathus_longipes: {
    classification: 'Late Jurassic · Theropod',
    title: 'Two ways to balance a small biped',
    anatomy: 'The anatomical model carries a long, articulated tail. A second model fixes the head and tail to a robot chassis. Both use their legs to balance and walk.',
    alt: 'Small bipedal Compsognathus model with a narrow body, long hind legs and an extended tail',
    detailLabel: 'Anatomical tail', detail: 'Articulated',
    robot: {
      title: 'Balancing with a fixed head and tail',
      anatomy: 'Twelve leg servos move the robot. The head and tail are fixed to its chassis, which carries the head camera and the inertial sensor.',
      alt: 'Compsognathus robot model with a fixed head and tail and a pair of articulated legs',
      detailLabel: 'Head and tail', detail: 'Fixed to the chassis',
      studies: {
        stance: {title: 'Stand on two flat soles', description: 'Each foot rests on one flat sole. The toes are surface detail and carry no weight. The current check asks the robot to last almost every trial without falling, while a touch sensor under at least one foot reports contact nearly all the time. It is an older check, due to be replaced by a stricter one before the robot next trains to stand.'},
        locomotion: {title: 'Walk upright', description: 'Each trial gives the robot one command for its two feet: walk forward at the commanded speed, holding a heading close to the one it starts on, or, in about one trial in seven, stand still. A trial ends as soon as any other part of the robot touches the floor.'},
      },
    },
    studies: {
      stance: {title: 'Settle onto both feet', description: 'The stance test checks that the foot pads and toes share the load without hopping or tipping onto an edge.'},
      recovery: {title: 'Regain balance', description: 'In this pilot task, calibrated pushes from a random horizontal direction, about every two seconds, test how the standing model responds.'},
      locomotion: {title: 'Walk upright', description: 'The model advances on its hind feet while keeping the rest of its body clear of the floor.'},
    },
  },
};
