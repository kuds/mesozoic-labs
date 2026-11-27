// Unit tests for the animal pages' museum labels (`npm test`).  interpretation.ts
// is transpiled with the site's own typescript package and run in a node:vm
// context, as consent.test.mjs runs the consent code, and read against the
// generated catalog the pages render.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const SRC = new URL('../src/', import.meta.url);

/** Transpiles src/<path> (a module with no imports) to CommonJS and returns its exports. */
function load(path) {
  const source = readFileSync(new URL(path, SRC), 'utf8');
  const {outputText} = ts.transpileModule(source, {
    compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020},
  });
  const exports = {};
  vm.runInContext(`(function (exports) {${outputText}\n})`, vm.createContext({}))(exports);
  return exports;
}

const {INTERPRETATIONS, studyLabel} = load('components/AnimalPage/interpretation.ts');
const CATALOG = JSON.parse(readFileSync(new URL('data/species.generated.json', SRC), 'utf8'));
const modelOf = (species, embodiment) =>
  CATALOG.models.find((model) => model.species === species && model.embodiment === embodiment);
const COMPSOGNATHUS = INTERPRETATIONS.compsognathus_longipes;
const PAGE = readFileSync(new URL('components/AnimalPage/index.tsx', SRC), 'utf8');

test('a robot page shows the robot label of a stage and otherwise the species label', () => {
  assert.equal(studyLabel(COMPSOGNATHUS, 'robot', 'stance'), COMPSOGNATHUS.robot.studies.stance);
  assert.notEqual(COMPSOGNATHUS.robot.studies.stance, COMPSOGNATHUS.studies.stance);
  assert.equal(studyLabel(COMPSOGNATHUS, 'anatomical', 'stance'), COMPSOGNATHUS.studies.stance);
  // A stage the robot has no label for falls back to the species label, and a stage nobody labels to none
  // (the page then shows the stage's own title and description).
  assert.equal(COMPSOGNATHUS.robot.studies.recovery, undefined);
  assert.deepEqual(studyLabel(COMPSOGNATHUS, 'robot', 'recovery'), COMPSOGNATHUS.studies.recovery);
  assert.equal(studyLabel(COMPSOGNATHUS, 'robot', 'no_such_stage'), undefined);
  assert.deepEqual(studyLabel(INTERPRETATIONS.allosaurus_fragilis, 'robot', 'stance'), INTERPRETATIONS.allosaurus_fragilis.studies.stance);
  assert.equal(studyLabel(undefined, 'robot', 'stance'), undefined);
  // The page reads every stage's label through it, with the model's embodiment.
  assert.match(PAGE, /const label = studyLabel\(study, model\.embodiment, stage\.id\);/);
  assert.doesNotMatch(PAGE, /study\?\.studies\[/);
});

test('a species label shown on a robot page drops its reference recording of the anatomical model', () => {
  // The T. rex locomotion label carries the anatomical reference walk; a robot of the species with no locomotion
  // label of its own shows the label's text and the robot stage's own video instead.
  const walk = INTERPRETATIONS.tyrannosaurus_rex.studies.locomotion;
  assert.ok(walk.referenceRecording);
  assert.equal(studyLabel(INTERPRETATIONS.tyrannosaurus_rex, 'anatomical', 'locomotion'), walk);
  // (Spread into this context: the label objects come from the vm context, with its own Object prototype.)
  assert.deepEqual({...studyLabel(INTERPRETATIONS.tyrannosaurus_rex, 'robot', 'locomotion')}, {title: walk.title, description: walk.description});
});

test('the Compsognathus robot stance describes its soles and its current check, not the anatomical feet', () => {
  // The robot stands on one box sole per foot; its toe geoms are visual only, and its stance gate is
  // stance_quality/v1 on touch-sensor support, due to be replaced before its next stance run.
  const {title, description} = COMPSOGNATHUS.robot.studies.stance;
  assert.match(title + ' ' + description, /sole/);
  assert.doesNotMatch(description, /\bpads?\b|toes share/i);
  assert.match(description, /toes are surface detail and carry no weight/);
  assert.match(description, /touch sensor/);
  assert.match(description, /older check, due to be replaced by a stricter one/);
  assert.equal(modelOf('compsognathus_longipes', 'robot').stages[0].advancement_gate.gate_kind, 'stance_quality/v1');
  assert.doesNotMatch(COMPSOGNATHUS.robot.studies.locomotion.description, /hind/);
  // The robot has legs and no forelimbs: its page's support line says two legs, not two hind limbs.
  assert.match(PAGE, /model\.embodiment === 'robot' \? 'Two legs' : 'Two hind limbs'/);
});

test('the Compsognathus robot walk describes the command-driven walk it trains', () => {
  // configs/compsognathus_longipes/robot/locomotion.toml (decision D-D62): one whole-episode command per trial, a
  // forward cruise on a heading within 15 degrees of the spawn's or, with stop_probability 0.15, a stand, judged by
  // command_tracking/v1's walk profile.
  const walk = modelOf('compsognathus_longipes', 'robot').stages.find((stage) => stage.id === 'locomotion');
  assert.equal(walk.advancement_gate.gate_kind, 'command_tracking/v1');
  assert.equal(walk.advancement_gate.profile, 'walk');
  assert.match(walk.description, /follow a forward command with heading hold, and stand still on a zero command/);
  const {description} = COMPSOGNATHUS.robot.studies.locomotion;
  assert.match(description, /one command/);
  assert.match(description, /commanded speed, holding a heading close to the one it starts on/);
  assert.match(description, /about one trial in seven, stand still/);
  assert.doesNotMatch(description, /move forward on its two feet/);
});

test('the Compsognathus push-recovery label describes the pushes both models get', () => {
  // configs/compsognathus_longipes/{anatomical,robot}/recovery.toml: perturbation_direction = "uniform_horizontal"
  // (environments/shared/perturbation.py draws each push's heading uniformly on the horizontal circle) and
  // perturbation_interval = 2.0 s; the robot page shows this species label.
  const {description} = COMPSOGNATHUS.studies.recovery;
  assert.match(description, /random horizontal direction, about every two seconds/);
  assert.doesNotMatch(description, /sideways|fore-and-aft/);
  assert.equal(studyLabel(COMPSOGNATHUS, 'robot', 'recovery').description, description);
});

test('every robot model of the catalog has its own page text', () => {
  // Without it, the robot page shows the anatomical model's header (CONTRIBUTING.md, "Adding an embodiment").
  const robots = CATALOG.models.filter((model) => model.embodiment === 'robot');
  assert.ok(robots.length);
  for (const robot of robots) {
    const text = INTERPRETATIONS[robot.species]?.robot;
    assert.ok(text, `${robot.species} has a robot model but no robot entry in interpretation.ts`);
    for (const key of ['title', 'anatomy', 'alt', 'detailLabel', 'detail']) assert.ok(text[key], `${robot.species} robot: ${key}`);
  }
});

test('every label names a stage of a model of its species', () => {
  for (const [species, interpretation] of Object.entries(INTERPRETATIONS)) {
    const stages = CATALOG.models.filter((model) => model.species === species).flatMap((model) => model.stages);
    assert.ok(stages.length, `${species} has no model in the catalog`);
    for (const stageId of Object.keys(interpretation.studies)) {
      assert.ok(stages.some((stage) => stage.id === stageId), `${species}: no model has a stage ${stageId}`);
    }
    for (const stageId of Object.keys(interpretation.robot?.studies ?? {})) {
      const robot = modelOf(species, 'robot');
      assert.ok(robot, `${species} has robot labels but no robot model`);
      assert.ok(robot.stages.some((stage) => stage.id === stageId), `${species} robot: no stage ${stageId}`);
    }
  }
});

test('no robot text claims a result the robot has not published', () => {
  for (const [species, interpretation] of Object.entries(INTERPRETATIONS)) {
    const robot = modelOf(species, 'robot');
    if (!interpretation.robot || !robot || robot.historical_results.length) continue;
    const labels = Object.values(interpretation.robot.studies ?? {});
    for (const text of [interpretation.robot.title, interpretation.robot.anatomy, ...labels.flatMap((label) => [label.title, label.description])]) {
      assert.doesNotMatch(text, /certif|\bpass(es|ed)?\b|published|qualif/i, `${species} robot: ${text}`);
    }
    // The page header says what the robot is, not that it walks: no walk of it is published.
    for (const text of [interpretation.robot.title, interpretation.robot.anatomy]) {
      assert.doesNotMatch(text, /\bwalks\b|\bwalking\b/i, `${species} robot: ${text}`);
    }
  }
});
