# deep-dishes
Tools for datamining cooking recipes

## Workflow
Our workflow is separated into 3 main branches:
1. The data folder, where data is collected/processed and cleaned up
2. The features folder, where features are created and formatted to suit the correct task.
3. The models folder, where data science/ML models are created to perform analysis on the features.

## Models
These following models are used in the analysis of these recipes:
* CRF from NYT for processing ingredients into a more cohesive format.
* T-SNE for data clustering/visualization.
* A-priori for market basket analysis on the recipe ingredients
