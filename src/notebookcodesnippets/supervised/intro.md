# 05-Supervised learning

In this notebook we will train a model based on the previous 4 notebooks. This includes inspecting, finding the optimal hyperparameters (with many visualizations) and finally predict with the optimal model that has been found. This notebook is the same as every other supervised notebook in this project, but differs in training set, principal components and results. Therefore, to provide easy implementations, the notebook has been made modular to a certain degree, in particular for further implementation of other machine learning algorithms. It should be relatively straight forward to add more algorithms.

## Table of contents

- Imports and reading of data
- Algorithms
- Methods for finding optimal hyperparameters
- Optimal hyperparameter search
    - Visualizing the optimal parameters for dimensionality reduction
    - Visualizing confusion metrics
    - ROC-AUC  and precision recall curves
    - Visualizing the cross-validated trained models
    - Visualizing relevant features
    - Falsely predicted entries
- Predicting solid-state qubit candidates
    - Save the summary and models

The cell under provides the editorial difference between the different supervised notebooks in this project. 
