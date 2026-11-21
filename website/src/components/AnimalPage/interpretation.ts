/** Museum labels for the models' distinguishing anatomy and movement experiments. */
interface StudyLabel {
  title: string;
  description: string;
  referenceRecording?: {path: string; caption: string};
}

interface Interpretation {
  classification: string;
  title: string;
  anatomy: string;
  alt: string;
  detailLabel: string;
  detail: string;
  robot?: {title: string; anatomy: string; alt: string; detailLabel: string; detail: string};
  studies: Record<string, StudyLabel>;
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
        },
      },
    },
  },
  brachiosaurus_altithorax: {
    classification: 'Late Jurassic · Sauropod',
    title: 'Reaching from a four-legged stance',
    anatomy: 'The raised shoulders sit above compact forefeet. The hindfeet have padded heels and three claws. A four-beat walk transfers weight between them while the long neck holds the head above the chest.',
    alt: 'Brachiosaurus in profile with a deep chest, raised shoulders, tapered tail and elevated neck',
    detailLabel: 'Head and neck control', detail: '6 actuators',
    studies: {
      stance: {title: 'Hold steady', description: 'Can the body keep its balance on four feet with the neck held high?'},
      locomotion: {
        title: 'Coordinate a stride',
        description: 'Watch the hindfoot land, followed by the forefoot on the same side. The bars show when each foot carries weight.',
        referenceRecording: {
          path: '/videos/brachiosaurus_altithorax_anatomical_reference_walk.mp4',
          caption: 'Scripted reference walk · Physics revision 6 · Measured speed 0.34 m/s',
        },
      },
      behavior: {title: 'Guide the head', description: 'Food targets appear 2.0–3.5 m above the ground. A reach counts when the head tip comes within 0.8 m of the target.'},
    },
  },
  velociraptor_mongoliensis: {
    classification: 'Late Cretaceous · Theropod',
    title: 'A claw held clear of the ground',
    anatomy: 'Each foot supports the body on a rounded pad and two toes. The sickle claw rides above them. The model can move that claw separately as it approaches prey.',
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
        },
      },
      behavior: {title: 'Aim the sickle claw', description: 'Either raised claw can score a strike by touching the simulated prey.'},
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
      locomotion: {title: 'Carry the skull forward', description: 'The hind legs alternate support as the body advances and the tail balances the trunk.'},
      behavior: {title: 'Make head contact', description: 'A contact surface on the head represents a bite. Touching simulated prey with that surface counts as success.'},
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
      locomotion: {title: 'Work in diagonal pairs', description: 'Each foreleg coordinates with the opposite hind leg as the model walks.'},
      behavior: {title: 'Place the snout', description: 'The MuJoCo task scores a touch of the snout on the prey.'},
    },
  },
  compsognathus_longipes: {
    classification: 'Late Jurassic · Theropod',
    title: 'Two ways to balance a small biped',
    anatomy: 'The anatomical model carries a long, articulated tail. A second model fixes the head and tail to a robot chassis. Both use their legs to balance and approach a target.',
    alt: 'Small bipedal Compsognathus model with a narrow body, long hind legs and an extended tail',
    detailLabel: 'Anatomical tail', detail: 'Articulated',
    robot: {
      title: 'Walking with a fixed head and tail',
      anatomy: 'Twelve leg servos move the robot. Its head camera and inertial sensor ride on the chassis, while the tail stays fixed as the model balances and walks.',
      alt: 'Compsognathus robot model with a fixed head and tail and a pair of articulated legs',
      detailLabel: 'Head and tail', detail: 'Fixed to the chassis',
    },
    studies: {
      stance: {title: 'Settle onto both feet', description: 'The stance test checks that the foot pads and toes share the load without hopping or tipping onto an edge.'},
      recovery: {title: 'Regain balance', description: 'In this pilot task, a calibrated sideways or fore-and-aft push tests how the standing model responds.'},
      locomotion: {title: 'Walk upright', description: 'The model advances on its hind feet while keeping the rest of its body clear of the floor.'},
      behavior: {title: 'Arrive and slow down', description: 'The pelvis must enter a target zone at low speed with the body still upright. The target moves to a new position each episode.'},
    },
  },
};
