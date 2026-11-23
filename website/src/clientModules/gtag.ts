/**
 * Sends a page_view to Google Analytics on SPA route changes.
 *
 * A page load's first page_view comes from gtag('config'), which the consent
 * head script (src/components/CookieConsent/headScript.ts) sends only once
 * the visitor has accepted.  This module covers client-side navigations,
 * replacing the preset gtag plugin's client module, and sends nothing unless
 * gtag.js is loaded and consent is granted.
 */

import type {ClientModule} from '@docusaurus/types';
import {isAnalyticsActive} from '@site/src/components/CookieConsent/consent';

const module: ClientModule = {
  onRouteDidUpdate({location, previousLocation}) {
    if (
      previousLocation &&
      (location.pathname !== previousLocation.pathname ||
        location.search !== previousLocation.search ||
        location.hash !== previousLocation.hash)
    ) {
      // react-helmet-async sets the new page's document.title in an
      // animation frame it requested while the route rendered, so wait for
      // that frame and then one task.  Docusaurus's own gtag plugin waits
      // only the task (setTimeout, facebook/docusaurus#7420), which here
      // still runs before the frame and sends the previous page's title.
      requestAnimationFrame(() => {
        setTimeout(() => {
          if (!isAnalyticsActive()) {
            return;
          }
          window.gtag?.('set', 'page_path', location.pathname + location.search + location.hash);
          window.gtag?.('event', 'page_view', {page_title: document.title});
        });
      });
    }
  },
};

export default module;
