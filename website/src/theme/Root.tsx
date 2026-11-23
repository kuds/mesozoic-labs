import React from 'react';
import CookieConsent from '@site/src/components/CookieConsent';

// NOTE: Google Analytics consent starts in an inline <head> script
// (src/components/CookieConsent/headScript.ts, injected via `headTags` in
// docusaurus.config.ts): it denies every Consent Mode signal and loads
// gtag.js only after the visitor accepts.  It must run before any React
// code; the banner below asks the visitor and applies the answer.

export default function Root({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <>
      {children}
      <CookieConsent />
    </>
  );
}
