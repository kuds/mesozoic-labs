// Unit tests for the Google Analytics consent code (`npm test`).  The
// TypeScript sources are transpiled with the site's own typescript package
// and run against a minimal fake browser in a node:vm context.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const SRC = new URL('../src/', import.meta.url);
const ID = 'G-TEST123';
const KEY = 'mesozoic_cookie_consent';
const DENY_ALL = {
  ad_storage: 'denied',
  ad_user_data: 'denied',
  ad_personalization: 'denied',
  analytics_storage: 'denied',
};

/** Transpiles src/<path> to CommonJS and runs it in `context`. */
function load(path, context, modules = {}) {
  const source = readFileSync(new URL(path, SRC), 'utf8');
  const {outputText} = ts.transpileModule(source, {
    compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020},
  });
  const exports = {};
  const run = vm.runInContext(`(function (exports, require) {${outputText}\n})`, context);
  run(exports, (name) => {
    assert.ok(name in modules, `unexpected import ${name}`);
    return modules[name];
  });
  return exports;
}

const buildTime = load('components/CookieConsent/headScript.ts', vm.createContext({}), {
  './consent': load('components/CookieConsent/consent.ts', vm.createContext({})),
});

/**
 * A fake browser: localStorage (optionally throwing, as when site data is
 * blocked), a cookie jar keyed by name and domain ('' = host-only), the
 * scripts appended to <head>, and an event loop (`tick`) that, like a
 * browser, runs the tasks already queued before the next animation frame.
 * With `analytics`, the consent head script runs first, as on a real page.
 */
function browser({analytics = true, stored, storageThrows = false, hostname = 'www.mesozoiclabs.com', cookies = []} = {}) {
  const store = new Map(stored === undefined ? [] : [[KEY, stored]]);
  const blocked = () => {
    throw new Error('SecurityError: site data is blocked');
  };
  const localStorage = storageThrows
    ? {getItem: blocked, setItem: blocked, removeItem: blocked}
    : {
        getItem: (key) => store.get(key) ?? null,
        setItem: (key, value) => store.set(key, String(value)),
        removeItem: (key) => store.delete(key),
      };
  const jar = new Set(cookies.map(([name, domain]) => `${name}|${domain}`));
  const cookieWrites = [];
  const scripts = [];
  const tasks = [];
  const frames = [];
  const tick = () => {
    tasks.splice(0).forEach((fn) => fn());
    frames.splice(0).forEach((fn) => fn());
    tasks.splice(0).forEach((fn) => fn());
  };
  const document = {
    title: 'Home',
    head: {appendChild: (element) => scripts.push(element)},
    createElement: (tagName) => ({tagName}),
    get cookie() {
      return [...jar].map((entry) => `${entry.split('|')[0]}=1`).join('; ');
    },
    set cookie(value) {
      cookieWrites.push(value);
      const [pair, ...attributes] = value.split(';').map((part) => part.trim());
      const name = pair.split('=')[0];
      const domain = attributes.find((a) => /^domain=/i.test(a))?.slice('domain='.length) ?? '';
      if (attributes.some((a) => /^max-age=0$/i.test(a))) {
        jar.delete(`${name}|${domain}`);
      } else {
        jar.add(`${name}|${domain}`);
      }
    },
  };
  const window = {
    location: {hostname},
    localStorage,
    document,
    setTimeout: (fn) => tasks.push(fn),
    requestAnimationFrame: (fn) => frames.push(fn),
  };
  window.window = window;
  const context = vm.createContext(window);
  if (analytics) {
    vm.runInContext(buildTime.consentHeadScript(ID), context);
  }
  const consent = load('components/CookieConsent/consent.ts', context);
  const commands = () => JSON.parse(JSON.stringify((window.dataLayer ?? []).map((args) => Array.from(args))));
  return {window, context, consent, store, jar, cookieWrites, scripts, frames, tick, commands};
}

const GA_COOKIES = [
  ['_ga', '.mesozoiclabs.com'],
  ['_ga_TEST123', '.mesozoiclabs.com'],
  ['_ga_OLDID', '.mesozoiclabs.com'],
  ['_ga', '.www.mesozoiclabs.com'],
  ['_gid', ''],
  ['theme', ''],
];

test('measurementIdFor: env var, else production only in upstream Actions, else none', () => {
  const {measurementIdFor, PRODUCTION_MEASUREMENT_ID} = buildTime;
  const upstreamActions = {GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'kuds/mesozoic-labs'};
  assert.equal(measurementIdFor({GA_MEASUREMENT_ID: 'G-OTHER'}), 'G-OTHER');
  assert.equal(measurementIdFor({GA_MEASUREMENT_ID: 'G-OTHER', ...upstreamActions}), 'G-OTHER');
  assert.equal(measurementIdFor(upstreamActions), PRODUCTION_MEASUREMENT_ID);
  // An unset Actions secret arrives as an empty string.
  assert.equal(measurementIdFor({GA_MEASUREMENT_ID: '', ...upstreamActions}), PRODUCTION_MEASUREMENT_ID);
  assert.equal(measurementIdFor({GA_MEASUREMENT_ID: ''}), null);
  // A codespace of the upstream repository sets GITHUB_REPOSITORY but not GITHUB_ACTIONS.
  assert.equal(measurementIdFor({GITHUB_REPOSITORY: 'kuds/mesozoic-labs'}), null);
  assert.equal(measurementIdFor({GITHUB_ACTIONS: 'false', GITHUB_REPOSITORY: 'kuds/mesozoic-labs'}), null);
  assert.equal(measurementIdFor({GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: 'someone/mesozoic-labs'}), null);
  assert.equal(measurementIdFor({GITHUB_ACTIONS: 'true'}), null);
  assert.equal(measurementIdFor({}), null);
});

for (const [label, options] of [
  ['no stored choice', {}],
  ['a stored refusal', {stored: 'denied'}],
  ['blocked storage', {storageThrows: true}],
]) {
  test(`head script with ${label}: all signals denied, gtag.js not loaded`, () => {
    const page = browser(options);
    assert.deepEqual(page.commands(), [['consent', 'default', DENY_ALL]]);
    assert.equal(page.scripts.length, 0);
    assert.equal(page.consent.isAnalyticsEnabled(), true);
    assert.equal(page.consent.isAnalyticsActive(), false);
  });
}

test('head script with a stored acceptance loads gtag.js at startup', () => {
  const page = browser({stored: 'granted'});
  const commands = page.commands();
  assert.deepEqual(commands[0], ['consent', 'default', DENY_ALL]);
  assert.deepEqual(commands[1], ['consent', 'update', {analytics_storage: 'granted'}]);
  assert.equal(commands[2][0], 'js');
  assert.deepEqual(commands[3], ['config', ID, {anonymize_ip: true}]);
  assert.equal(commands.length, 4);
  assert.equal(page.scripts.length, 1);
  assert.equal(page.scripts[0].src, `https://www.googletagmanager.com/gtag/js?id=${ID}`);
  assert.equal(page.scripts[0].async, true);
  assert.equal(page.consent.isAnalyticsActive(), true);
});

test('accept stores the choice, grants analytics only and loads gtag.js once', () => {
  const page = browser();
  page.consent.acceptAnalytics();
  page.consent.acceptAnalytics();
  assert.equal(page.store.get(KEY), 'granted');
  const commands = page.commands();
  assert.deepEqual(commands[1], ['consent', 'update', {analytics_storage: 'granted'}]);
  assert.deepEqual(commands.filter((c) => c[0] === 'config'), [['config', ID, {anonymize_ip: true}]]);
  assert.equal(page.scripts.length, 1);
  assert.equal(page.consent.isAnalyticsActive(), true);
});

test('decline stores the choice, denies consent and deletes GA cookies on every domain', () => {
  const page = browser({cookies: GA_COOKIES});
  page.consent.declineAnalytics();
  assert.equal(page.store.get(KEY), 'denied');
  assert.deepEqual(page.commands().at(-1), ['consent', 'update', {analytics_storage: 'denied'}]);
  assert.equal(page.scripts.length, 0);
  // gtag.js was never loaded, so there is nothing to disable.
  assert.equal(page.window[`ga-disable-${ID}`], undefined);
  assert.deepEqual([...page.jar], ['theme|']);
  assert.ok(page.cookieWrites.every((w) => w.includes('Path=/') && w.includes('Max-Age=0')));
});

test('reset after accept forgets the choice, disables gtag.js and deletes GA cookies', () => {
  const page = browser({stored: 'granted', cookies: GA_COOKIES});
  page.consent.resetConsent();
  assert.equal(page.store.has(KEY), false);
  assert.deepEqual(page.commands().at(-1), ['consent', 'update', {analytics_storage: 'denied'}]);
  assert.equal(page.window[`ga-disable-${ID}`], true);
  assert.equal(page.consent.isAnalyticsActive(), false);
  assert.deepEqual([...page.jar], ['theme|']);
});

test('blocked storage: every choice still applies for the page view', () => {
  const page = browser({storageThrows: true, cookies: GA_COOKIES});
  assert.equal(page.consent.readStoredConsent(), null);
  page.consent.acceptAnalytics();
  assert.equal(page.scripts.length, 1);
  assert.equal(page.consent.isAnalyticsActive(), true);
  page.consent.declineAnalytics();
  page.consent.resetConsent();
  assert.equal(page.consent.isAnalyticsActive(), false);
  assert.deepEqual([...page.jar], ['theme|']);
});

test('a build without analytics: no banner, decline and reset are harmless', () => {
  const page = browser({analytics: false, hostname: 'localhost', cookies: [['_ga', ''], ['theme', '']]});
  assert.equal(page.consent.isAnalyticsEnabled(), false);
  page.consent.acceptAnalytics();
  page.consent.declineAnalytics();
  page.consent.resetConsent();
  assert.equal(page.window.dataLayer, undefined);
  assert.deepEqual([...page.jar], ['theme|']);
  // A single-label host has no parent domain to try.
  assert.ok(page.cookieWrites.every((w) => !/domain=/i.test(w)));
});

test('SPA page_view waits for the new title and needs consent', () => {
  // As react-helmet-async does while the new route renders: the title is
  // committed in an animation frame requested before onRouteDidUpdate runs.
  const navigate = (page, gtag) => {
    page.window.requestAnimationFrame(() => {
      page.window.document.title = 'Docs';
    });
    gtag.default.onRouteDidUpdate({
      location: {pathname: '/docs', search: '', hash: ''},
      previousLocation: {pathname: '/', search: '', hash: ''},
    });
  };
  const loadGtag = (page) =>
    load('clientModules/gtag.ts', page.context, {'@site/src/components/CookieConsent/consent': page.consent});

  const accepted = browser({stored: 'granted'});
  const gtag = loadGtag(accepted);
  const before = accepted.commands().length;
  navigate(accepted, gtag);
  assert.equal(accepted.commands().length, before, 'sent synchronously');
  accepted.tick();
  assert.deepEqual(accepted.commands().slice(before), [
    ['set', 'page_path', '/docs'],
    ['event', 'page_view', {page_title: 'Docs'}],
  ]);

  const undecided = browser();
  navigate(undecided, loadGtag(undecided));
  undecided.tick();
  assert.deepEqual(undecided.commands(), [['consent', 'default', DENY_ALL]]);

  // The first route of a page load has no previous location: config sends it.
  const first = browser({stored: 'granted'});
  loadGtag(first).default.onRouteDidUpdate({location: {pathname: '/', search: '', hash: ''}, previousLocation: null});
  assert.equal(first.frames.length, 0);
});
