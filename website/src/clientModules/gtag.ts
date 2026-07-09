/**
 * Sends a page_view to Google Analytics on SPA route changes.
 *
 * The initial page_view comes from the inline gtag('config') in
 * docusaurus.config.ts headTags (which runs AFTER the Consent Mode
 * default — see the comment there).  This module covers client-side
 * navigations, replacing the preset gtag plugin's client module.
 */

import type {ClientModule} from '@docusaurus/types';

const module: ClientModule = {
  onRouteDidUpdate({location, previousLocation}) {
    if (
      previousLocation &&
      (location.pathname !== previousLocation.pathname ||
        location.search !== previousLocation.search ||
        location.hash !== previousLocation.hash)
    ) {
      window.gtag?.('set', 'page_path', location.pathname + location.search + location.hash);
      window.gtag?.('event', 'page_view');
    }
  },
};

export default module;
