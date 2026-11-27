// Unit tests for the animal pages' images (`npm test`).  animals.ts is transpiled with the
// site's own typescript package and run in a node:vm context, as consent.test.mjs runs the
// consent code, with its one import, ./species, built from the generated catalog the pages
// render.  A missing image does not fail the build: the page names it at run time.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {existsSync, readFileSync} from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const SRC = new URL('../src/', import.meta.url);
const STATIC = new URL('../static/', import.meta.url);
const CATALOG = JSON.parse(readFileSync(new URL('data/species.generated.json', SRC), 'utf8'));
// The fields of species.ts's adaptSpecies that animals.ts reads.
const ALL_MODELS = CATALOG.models.map((raw) => ({
  species: raw.species,
  embodiment: raw.embodiment,
  speciesName: raw.species_display_name,
  specialty: raw.specialty,
}));

/** Transpiles src/<path> to CommonJS and returns its exports, with each import taken from `modules`. */
function load(path, modules) {
  const source = readFileSync(new URL(path, SRC), 'utf8');
  const {outputText} = ts.transpileModule(source, {
    compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020},
  });
  const exports = {};
  vm.runInContext(`(function (exports, require) {${outputText}\n})`, vm.createContext({}))(exports, (name) => {
    assert.ok(name in modules, `unexpected import ${name}`);
    return modules[name];
  });
  return exports;
}

const {ANIMALS, animalModelImage} = load('data/animals.ts', {'./species': {ALL_MODELS}});
const TURNTABLE = readFileSync(new URL('components/AnimalPage/Turntable.tsx', SRC), 'utf8');
const PAGE = readFileSync(new URL('components/AnimalPage/index.tsx', SRC), 'utf8');

/** Whether the site serves `path` (a site path such as /img/animals/x.webp) from website/static/. */
const served = (path) => existsSync(new URL(path.replace(/^\//, ''), STATIC));

test('every model of the generated catalog has its still image', () => {
  assert.ok(ALL_MODELS.length > 0, 'the generated catalog lists no model: run python -m environments.shared.species_catalog');
  for (const model of ALL_MODELS) {
    const image = animalModelImage(model);
    assert.equal(
      image,
      `/img/animals/${model.species}-${model.embodiment}.webp`,
      'animalModelImage no longer names /img/animals/<species>-<embodiment>.webp: update this test and the ' +
        "'Adding an embodiment' checklist in CONTRIBUTING.md with it",
    );
    assert.ok(served(image), `${model.species}/${model.embodiment} has no still image: add website/static${image}`);
  }
  // The model page shows the image animalModelImage names.
  assert.match(
    PAGE,
    /image=\{animalModelImage\(model\)\}/,
    'AnimalPage/index.tsx no longer passes image={animalModelImage(model)}: update animal_images.test.mjs with it',
  );
});

test("every model's still image has the rotating view the Turntable derives from it", () => {
  // Turntable.tsx swaps /img/animals/ for /videos/turntables/ and drops .webp, then adds the poster
  // (.webp) and the light and dark orbits (.mp4, -dark.mp4); render_turntables.py writes all three.
  const derivation =
    'Turntable.tsx no longer derives the poster and videos from the still image this way: ' +
    'update animal_images.test.mjs with it';
  assert.match(TURNTABLE, /image\.replace\('\/img\/animals\/', '\/videos\/turntables\/'\)\.replace\(\/\\\.webp\$\/, ''\)/, derivation);
  assert.match(TURNTABLE, /const poster = asset \+ '\.webp';/, derivation);
  assert.match(TURNTABLE, /colorMode === 'dark' \? '-dark\.mp4' : '\.mp4'/, derivation);
  for (const model of ALL_MODELS) {
    const asset = animalModelImage(model).replace('/img/animals/', '/videos/turntables/').replace(/\.webp$/, '');
    for (const suffix of ['.webp', '.mp4', '-dark.mp4']) {
      assert.ok(
        served(asset + suffix),
        `${model.species}/${model.embodiment} has no website/static${asset}${suffix}: ` +
          `run python website/scripts/render_turntables.py ${model.species}-${model.embodiment}`,
      );
    }
  }
});

test('every animal of the collection has its image', () => {
  // The catalog's species, the Triceratops and the Stegosaurus.
  const species = new Set(ALL_MODELS.map((model) => model.species));
  assert.deepEqual(
    new Set(ANIMALS.filter((animal) => animal.models.length).map((animal) => animal.slug)),
    species,
    "the animals with models in website/src/data/animals.ts are not the catalog's species: " +
      'give each species of configs/species_manifest.toml an ANIMALS entry',
  );
  for (const animal of ANIMALS) {
    assert.ok(served(animal.image), `${animal.slug} has no image: add website/static${animal.image}`);
  }
});
