import React from 'react';
import CookieConsent from '@site/src/components/CookieConsent';

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
    dataLayer?: unknown[];
  }
}

// NOTE: the Google Consent Mode default (analytics_storage denied until the
// visitor accepts) is set by an inline <head> script injected via
// `headTags` in docusaurus.config.ts.  It must run before gtag.js processes
// its command queue — a React effect here would fire long after the first
// page_view, setting analytics cookies before consent.

export default function Root({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <>
      {children}
      <CookieConsent />
    </>
  );
}
