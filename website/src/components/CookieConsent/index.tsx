import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';
import {
  acceptAnalytics,
  declineAnalytics,
  isAnalyticsEnabled,
  readStoredConsent,
  resetConsent,
} from './consent';

export default function CookieConsent(): React.JSX.Element | null {
  const [isVisible, setIsVisible] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  useEffect(() => {
    // Ask only when this build loads analytics and no choice is stored (or
    // storage is blocked).  A stored acceptance was already applied by the
    // head script; a stored refusal needs nothing.
    if (isAnalyticsEnabled() && readStoredConsent() === null) {
      setIsVisible(true);
    }
  }, []);

  const handleAccept = () => {
    acceptAnalytics();
    closeBanner();
  };

  const handleDecline = () => {
    declineAnalytics();
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
        <div className={styles.textContent}>
          <h3 className={styles.title}>Cookie Preferences</h3>
          <p className={styles.description}>
            May we use analytics cookies to learn how people use the site?{' '}
            <a href="/docs/privacy" className={styles.link}>Privacy policy</a>
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
    </div>
  );
}

export function resetConsentStatus(): void {
  if (typeof window === 'undefined') return;
  resetConsent();
}
