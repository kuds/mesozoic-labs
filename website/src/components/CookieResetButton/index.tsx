import React, { useState } from 'react';
import { resetConsentStatus } from '../CookieConsent';

export default function CookieResetButton(): React.JSX.Element {
  const [isReset, setIsReset] = useState(false);

  const handleReset = () => {
    if (typeof window !== 'undefined') {
      resetConsentStatus();
      setIsReset(true);

      // Reload the page after a short delay
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    }
  };

  return (
    <button
      onClick={handleReset}
      className="button button--secondary"
      disabled={isReset}
      aria-label="Reset cookie preferences"
    >
      {isReset ? 'Preferences Reset! Reloading...' : 'Reset Cookie Preferences'}
    </button>
  );
}
