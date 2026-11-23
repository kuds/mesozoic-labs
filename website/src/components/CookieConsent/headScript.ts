/**
 * Build-time half of the Google Analytics consent setup, used by
 * docusaurus.config.ts: which measurement ID a build reports to, and the
 * inline <head> script that loads Google Analytics only after consent.
 */

import {CONSENT_STORAGE_KEY} from './consent';

export const PRODUCTION_MEASUREMENT_ID = 'G-GGSQ47QSJN';
export const UPSTREAM_REPOSITORY = 'kuds/mesozoic-labs';

/**
 * GA_MEASUREMENT_ID when set (an empty value, which an unset Actions secret
 * gives, counts as unset); otherwise the production property, but only for
 * a build in the upstream repository's GitHub Actions; otherwise null, so
 * local, dev and fork builds load no analytics at all.  GITHUB_ACTIONS is
 * checked too because Codespaces also sets GITHUB_REPOSITORY.
 */
export function measurementIdFor(env: Record<string, string | undefined>): string | null {
  if (env.GA_MEASUREMENT_ID) {
    return env.GA_MEASUREMENT_ID;
  }
  const upstreamActions = env.GITHUB_ACTIONS === 'true' && env.GITHUB_REPOSITORY === UPSTREAM_REPOSITORY;
  return upstreamActions ? PRODUCTION_MEASUREMENT_ID : null;
}

/**
 * The inline <head> script (Consent Mode, basic).  It runs before gtag.js
 * could, so the first command on the dataLayer is the default that denies
 * all four signals.  gtag.js is neither requested nor configured until the
 * visitor accepts: at startup when the stored choice is 'granted' (a guarded
 * read, since browsers that block site data throw), otherwise from the
 * banner through window.mzAnalytics.grant().  Only analytics_storage is
 * ever granted; the ad signals stay denied.  ./consent.ts handles decline
 * and reset.
 */
export function consentHeadScript(measurementId: string): string {
  return (
    'window.dataLayer=window.dataLayer||[];' +
    'function gtag(){dataLayer.push(arguments);}' +
    'window.gtag=gtag;' +
    "gtag('consent','default',{ad_storage:'denied',ad_user_data:'denied'," +
    "ad_personalization:'denied',analytics_storage:'denied'});" +
    '(function(){' +
    `var mz=window.mzAnalytics={id:${JSON.stringify(measurementId)},loaded:false,granted:false,` +
    'grant:function(){' +
    'mz.granted=true;' +
    "gtag('consent','update',{analytics_storage:'granted'});" +
    "if(mz.loaded){window['ga-disable-'+mz.id]=false;return;}" +
    'mz.loaded=true;' +
    "gtag('js',new Date());" +
    "gtag('config',mz.id,{anonymize_ip:true});" +
    "var s=document.createElement('script');" +
    's.async=true;' +
    "s.src='https://www.googletagmanager.com/gtag/js?id='+encodeURIComponent(mz.id);" +
    'document.head.appendChild(s);' +
    '}};' +
    `var stored=null;try{stored=localStorage.getItem(${JSON.stringify(CONSENT_STORAGE_KEY)});}catch(e){}` +
    "if(stored==='granted'){mz.grant();}" +
    '})();'
  );
}
