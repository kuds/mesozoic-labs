import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';

const CONSENT_COOKIE_NAME = 'mesozoic_cookie_consent';
const CONSENT_GRANTED = 'granted';
const CONSENT_DENIED = 'denied';

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
    dataLayer?: unknown[];
  }
}

export default function CookieConsent(): React.JSX.Element | null {
  const [isVisible, setIsVisible] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  useEffect(() => {
    // Check if user has already made a choice
    const consent = localStorage.getItem(CONSENT_COOKIE_NAME);

    if (!consent) {
      // No choice made yet, show the banner
      setIsVisible(true);
    } else if (consent === CONSENT_GRANTED) {
      // User previously granted consent, enable analytics
      enableAnalytics();
    }
    // If denied, do nothing - analytics stay disabled
  }, []);

  const enableAnalytics = () => {
    // Update Google Analytics consent
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('consent', 'update', {
        analytics_storage: 'granted',
      });
    }
  };

  const disableAnalytics = () => {
    // Revoke Google Analytics consent
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('consent', 'update', {
        analytics_storage: 'denied',
      });
    }
  };

  const handleAccept = () => {
    localStorage.setItem(CONSENT_COOKIE_NAME, CONSENT_GRANTED);
    enableAnalytics();
    closeBanner();
  };

  const handleDecline = () => {
    localStorage.setItem(CONSENT_COOKIE_NAME, CONSENT_DENIED);
    disableAnalytics();
    closeBanner();
  };

  const closeBanner = () => {
    setIsClosing(true);
    setTimeout(() => {
      setIsVisible(false);
      setIsClosing(false);
    }, 300);
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className={`${styles.cookieBanner} ${isClosing ? styles.closing : ''}`} role="dialog" aria-label="Cookie consent">
      <div className={styles.bannerContent}>
        <div className={styles.iconContainer} aria-hidden="true">
          <span className={styles.cookieIcon}>🦖</span>
        </div>

        <div className={styles.textContent}>
          <h3 className={styles.title}>Cookie Preferences</h3>
          <p className={styles.description}>
            We use cookies and analytics to understand how visitors interact with our site.
            This helps us improve the Mesozoic Labs experience. Your data is anonymized.
          </p>
        </div>

        <div className={styles.buttonContainer}>
          <button
            onClick={handleDecline}
            className={styles.declineButton}
            aria-label="Decline cookies"
          >
            Decline
          </button>
          <button
            onClick={handleAccept}
            className={styles.acceptButton}
            aria-label="Accept cookies"
          >
            Accept
          </button>
        </div>
      </div>

      <div className={styles.privacyLink}>
        <a href="/docs/privacy" className={styles.link}>
          Learn more about our privacy practices
        </a>
      </div>
    </div>
  );
}

export function resetConsentStatus(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(CONSENT_COOKIE_NAME);
}
