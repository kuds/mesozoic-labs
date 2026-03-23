import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Google Analytics Measurement ID - set via environment variable or replace with your ID
const GA_MEASUREMENT_ID = process.env.GA_MEASUREMENT_ID || 'G-GGSQ47QSJN';

const config: Config = {
  title: 'Mesozoic Labs',
  tagline: 'Building robotic dinosaurs through simulation and reinforcement learning',
  favicon: 'img/favicon.ico',

  // Production site URL - used for sitemap, canonical URLs, and social sharing
  url: 'https://mesozoiclabs.com',
  baseUrl: '/',

  // GitHub Pages config
  organizationName: 'mesozoic-labs',
  projectName: 'mesozoic-labs.github.io',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Custom fields accessible in components
  customFields: {
    gaMeasurementId: GA_MEASUREMENT_ID,
  },

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
        // Google Analytics - disabled by default, enabled via cookie consent
        gtag: {
          trackingID: GA_MEASUREMENT_ID,
          anonymizeIP: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Social card image for Open Graph - add 'img/mesozoic-labs-social-card.png' when available
    navbar: {
      title: 'Mesozoic Labs',
      logo: {
        alt: 'Mesozoic Labs Logo',
        src: 'img/logo.svg',
        srcDark: 'img/logo-dark.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/kuds/mesozoic-labs',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/',
            },
            {
              label: 'Models',
              to: '/docs/models/trex',
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
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
