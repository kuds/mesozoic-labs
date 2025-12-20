import React, { useEffect } from 'react';
import CookieConsent from '@site/src/components/CookieConsent';

const CONSENT_COOKIE_NAME = 'mesozoic_cookie_consent';

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
    dataLayer?: unknown[];
  }
}

// Initialize Google Analytics with consent mode
function initializeGtagConsent() {
  if (typeof window === 'undefined') return;

  // Set default consent to denied
  window.dataLayer = window.dataLayer || [];
  function gtag(...args: unknown[]) {
    window.dataLayer?.push(args);
  }
  window.gtag = gtag;

  // Check for existing consent
  const existingConsent = localStorage.getItem(CONSENT_COOKIE_NAME);

  // Set default consent state
  gtag('consent', 'default', {
    analytics_storage: existingConsent === 'granted' ? 'granted' : 'denied',
    wait_for_update: 500,
  });
}

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): React.JSX.Element {
  useEffect(() => {
    initializeGtagConsent();
  }, []);

  return (
    <>
      {children}
      <CookieConsent />
    </>
  );
}
