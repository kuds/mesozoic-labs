import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Google Analytics Measurement ID - set via environment variable or replace with your ID
const GA_MEASUREMENT_ID = process.env.GA_MEASUREMENT_ID || 'G-GGSQ47QSJN';

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

  // Google Analytics with Consent Mode.  Order inside this inline script is
  // what makes consent work: the consent DEFAULT must be pushed onto the
  // dataLayer before gtag('config') — gtag.js replays the queue in order,
  // so a default set any later (e.g. from a React effect) would let the
  // first page_view set analytics cookies before the visitor answered the
  // cookie banner.
  headTags: [
    {
      tagName: 'link',
      attributes: {rel: 'preconnect', href: 'https://www.googletagmanager.com'},
    },
    {
      tagName: 'script',
      attributes: {},
      innerHTML:
        'window.dataLayer = window.dataLayer || [];' +
        'function gtag(){dataLayer.push(arguments);}' +
        'window.gtag = gtag;' +
        "var mzConsent=null;try{mzConsent=localStorage.getItem('mesozoic_cookie_consent');}catch(e){}" +
        "gtag('consent','default',{analytics_storage:mzConsent==='granted'?'granted':'denied',wait_for_update:500});" +
        "gtag('js',new Date());" +
        `gtag('config','${GA_MEASUREMENT_ID}',{anonymize_ip:true});`,
    },
    {
      tagName: 'script',
      attributes: {
        async: 'true',
        src: `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`,
      },
    },
  ],

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
        // injects its gtag('config') snippet before any siteConfig headTags,
        // so a Consent Mode default could never precede it.  GA is loaded
        // manually below (headTags) with consent-default-first ordering,
        // and SPA page views are sent by src/clientModules/gtag.ts.
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
        {href: '/#roadmap', label: 'Roadmap', position: 'right'},
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
              to: '/docs/training/sac',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/kuds/mesozoic-labs/discussions',
            },
          ],
        },
        {
          title: 'More',
          items: [
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
