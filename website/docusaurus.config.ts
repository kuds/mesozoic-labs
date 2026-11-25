import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import {consentHeadScript, measurementIdFor} from './src/components/CookieConsent/headScript';

// Google Analytics measurement ID, or null for a build without analytics:
// GA_MEASUREMENT_ID if set, else the production ID only in the upstream
// repository's GitHub Actions (see measurementIdFor).
const GA_MEASUREMENT_ID = measurementIdFor(process.env);

const config: Config = {
  title: 'Mesozoic Labs',
  tagline: 'Building robots inspired by Mesozoic animals',
  favicon: 'img/favicon.ico',

  // Production site URL - used for sitemap, canonical URLs, and social sharing
  url: 'https://mesozoiclabs.com',
  baseUrl: '/',

  // GitHub Pages config
  organizationName: 'kuds',
  projectName: 'mesozoic-labs',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Custom fields accessible in components
  customFields: {
    gaMeasurementId: GA_MEASUREMENT_ID,
  },

  // Google Analytics with Consent Mode (basic).  The inline script sets all
  // four consent signals to denied before anything else and requests
  // gtag.js only once the visitor has accepted (see consentHeadScript), so
  // nothing reaches Google before consent; that is also why there is no
  // preconnect to googletagmanager.com.  A build without a measurement ID
  // gets no analytics script, and the cookie banner stays hidden.
  headTags: GA_MEASUREMENT_ID
    ? [
        {
          tagName: 'script',
          attributes: {},
          innerHTML: consentHeadScript(GA_MEASUREMENT_ID),
        },
      ]
    : [],

  // Sends page_view on SPA route changes (replaces the preset gtag plugin's
  // client module).
  clientModules: ['./src/clientModules/gtag.ts'],

  plugins: [require.resolve('./plugins/animal-pages.cjs')],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/kuds/mesozoic-labs/tree/main/website/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/kuds/mesozoic-labs/tree/main/website/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        // NOTE: the preset's `gtag` plugin is intentionally NOT used — it
        // loads gtag.js for every visitor and injects its gtag('config')
        // snippet before any siteConfig headTags.  GA is loaded manually
        // above (headTags) only after consent, and SPA page views are sent
        // by src/clientModules/gtag.ts.
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Social card image for Open Graph - add 'img/mesozoic-labs-social-card.png' when available
    navbar: {
      title: 'Mesozoic Labs',
      items: [
        {to: '/animals', label: 'Animals', position: 'right'},
        {href: '/#journal', label: 'Build journal', position: 'right'},
        {to: '/timeline', label: 'Timeline', position: 'right'},
        {type: 'docSidebar', sidebarId: 'docsSidebar', label: 'Docs', position: 'right'},
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs',
            },
            {
              label: 'Animal collection',
              to: '/animals',
            },
            {
              label: 'Training',
              to: '/docs/training/recipes',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'Collaborate with the lab', to: '/collaborate'},
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/kuds/mesozoic-labs/discussions',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'Privacy and cookie settings', to: '/docs/privacy#cookie-settings'},
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/kuds/mesozoic-labs',
            },
            {
              label: 'Finding Theta',
              href: 'https://findingtheta.com',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Mesozoic Labs. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
