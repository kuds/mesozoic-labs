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

  const buttonStyle: React.CSSProperties = {
    padding: '0.75rem 1.5rem',
    fontSize: '0.875rem',
    fontWeight: 600,
    letterSpacing: '0.05em',
    border: '2px solid #00ff88',
    background: isReset ? '#00ff88' : 'transparent',
    color: isReset ? '#0a0a0f' : '#00ff88',
    cursor: isReset ? 'default' : 'pointer',
    transition: 'all 0.2s ease',
    clipPath: 'polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px)',
  };

  return (
    <button
      onClick={handleReset}
      style={buttonStyle}
      disabled={isReset}
      aria-label="Reset cookie preferences"
    >
      {isReset ? 'Preferences Reset! Reloading...' : 'Reset Cookie Preferences'}
    </button>
  );
}
