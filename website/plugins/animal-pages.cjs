const fs = require('node:fs/promises');
const path = require('node:path');

module.exports = function animalPages({siteDir}) {
  const catalogPath = path.join(siteDir, 'src/data/species.generated.json');
  return {
    name: 'animal-pages',
    getPathsToWatch() {
      return [catalogPath];
    },
    async loadContent() {
      const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
      return [...new Set(catalog.models.map((model) => model.species))];
    },
    contentLoaded({content, actions}) {
      for (const slug of content) {
        actions.addRoute({
          path: '/animals/' + slug,
          component: '@site/src/components/AnimalPage',
          exact: true,
          props: {slug},
        });
      }
    },
  };
};
