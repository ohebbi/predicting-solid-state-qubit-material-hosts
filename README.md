# Predicting solid state qubit candidates
==============================

This is the main repository behind predicting solid state qubit candidates

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── raw            <- The original, immutable data dump.
    │   └── 01-naive-approach
    │       └── processed  <- The final, canonical data sets for modeling.
    │
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── 01-naive-approach
    │       ├── summary
    │       └── trained-models
    │
    │
    ├── notebooks          <- Jupyter notebooks.
    │   ├── 01-generateDataset-notebook.ipynb
    │   ├── 02-buildFeatures-notebook.ipynb
    │   └── method-01-naive-approach
    │       ├── 03-dataMining-notebook.ipynb
    │       ├── 04-preprocessing-notebook.ipynb
    │       ├── 05-supervisedLearning-notebook.ipynb
    │       └── 06-postAnalysis-notebook.ipynb
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │   └── README.md       
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   ├── get_data_base.py
    │   │   ├── get_data_AFLOW.py
    │   │   ├── get_data_AFLOWML.py
    │   │   ├── get_data_Citrine.py
    │   │   ├── get_data_JARVIS.py
    │   │   ├── get_data_MP.py
    │   │   ├── get_data_OQMD.py
    │   │   └── utils.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │   └── featurizer.py
    │   │   └── featurizeAll.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
