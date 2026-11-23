/**
 * Google Analytics consent at runtime: the visitor's stored choice and what
 * Accept, Decline and Reset do.
 *
 * The inline <head> script built by ./headScript.ts runs before any of this:
 * it sets every Consent Mode signal to denied, defines `window.mzAnalytics`
 * and requests gtag.js at startup only when the visitor accepted on an
 * earlier visit.  A build without a measurement ID has no such script, so
 * `window.mzAnalytics` is undefined: no banner, and Decline or Reset only
 * clear the stored choice and any Google Analytics cookies.
 */

/** Where the choice is kept: a localStorage key, not a cookie. */
export const CONSENT_STORAGE_KEY = 'mesozoic_cookie_consent';

export type ConsentChoice = 'granted' | 'denied';

/** What the head script defines on `window.mzAnalytics`. */
export interface AnalyticsRuntime {
  /** The GA4 measurement ID this build reports to. */
  id: string;
  /** gtag.js has been requested and gtag('config') sent on this page load. */
  loaded: boolean;
  /** analytics_storage is currently granted. */
  granted: boolean;
  /** Grants analytics_storage; the first call also loads gtag.js and sends config. */
  grant: () => void;
}

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
    dataLayer?: unknown[];
    mzAnalytics?: AnalyticsRuntime;
  }
}

/**
 * The stored choice, or null when there is none.  Browsers that block site
 * data throw on any storage access; that also reads as no choice, so the
 * banner asks again on each page load.
 */
export function readStoredConsent(): ConsentChoice | null {
  try {
    const value = window.localStorage.getItem(CONSENT_STORAGE_KEY);
    return value === 'granted' || value === 'denied' ? value : null;
  } catch {
    return null;
  }
}

function storeConsent(choice: ConsentChoice | null): void {
  try {
    if (choice === null) {
      window.localStorage.removeItem(CONSENT_STORAGE_KEY);
    } else {
      window.localStorage.setItem(CONSENT_STORAGE_KEY, choice);
    }
  } catch {
    // Site data is blocked: the choice holds until the page is reloaded.
  }
}

/** Whether this build loads Google Analytics once the visitor accepts. */
export function isAnalyticsEnabled(): boolean {
  return typeof window !== 'undefined' && window.mzAnalytics !== undefined;
}

/** Whether gtag.js is loaded and analytics_storage is granted. */
export function isAnalyticsActive(): boolean {
  const runtime = typeof window === 'undefined' ? undefined : window.mzAnalytics;
  return runtime !== undefined && runtime.loaded && runtime.granted;
}

export function acceptAnalytics(): void {
  storeConsent('granted');
  window.mzAnalytics?.grant();
}

export function declineAnalytics(): void {
  storeConsent('denied');
  revokeAnalytics();
}

/** Forgets the choice, so the banner asks again on the next page load. */
export function resetConsent(): void {
  storeConsent(null);
  revokeAnalytics();
}

function revokeAnalytics(): void {
  const runtime = window.mzAnalytics;
  if (runtime !== undefined) {
    runtime.granted = false;
    window.gtag?.('consent', 'update', {analytics_storage: 'denied'});
    if (runtime.loaded) {
      // gtag.js cannot be unloaded; this stops it sending anything more.
      (window as unknown as Record<string, unknown>)[`ga-disable-${runtime.id}`] = true;
    }
  }
  deleteAnalyticsCookies(runtime?.id);
}

/**
 * Expires _ga, _ga_<container> (one per measurement ID) and any legacy _gid,
 * on the current host and on each parent domain: gtag.js sets its cookies
 * on the highest domain it can (e.g. .mesozoiclabs.com for
 * www.mesozoiclabs.com).
 */
function deleteAnalyticsCookies(measurementId: string | undefined): void {
  const names = new Set(['_ga', '_gid']);
  if (measurementId !== undefined) {
    names.add(`_ga_${measurementId.replace(/^G-/, '')}`);
  }
  for (const pair of document.cookie.split(';')) {
    const name = pair.split('=')[0].trim();
    if (name.startsWith('_ga_')) {
      names.add(name);
    }
  }
  const domains = cookieDomains(window.location.hostname);
  for (const name of names) {
    for (const domain of domains) {
      document.cookie =
        `${name}=; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/` +
        (domain ? `; Domain=${domain}` : '');
    }
  }
}

/**
 * '' (a host-only cookie) and every domain a cookie set from `hostname` may
 * be scoped to, the top-level domain excluded: for www.mesozoiclabs.com,
 * ['', '.www.mesozoiclabs.com', '.mesozoiclabs.com'].
 */
function cookieDomains(hostname: string): string[] {
  const domains = [''];
  // An IP address has no parent domains.
  if (/^[\d.]+$/.test(hostname) || hostname.includes(':')) {
    return domains;
  }
  const labels = hostname.split('.');
  for (let i = 0; i < labels.length - 1; i++) {
    domains.push(`.${labels.slice(i).join('.')}`);
  }
  return domains;
}
