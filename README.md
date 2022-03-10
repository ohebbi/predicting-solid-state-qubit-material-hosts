# Predicting solid state qubit candidates [![DOI](https://zenodo.org/badge/335907184.svg)](https://zenodo.org/badge/latestdoi/335907184)


This is the main repository behind predicting solid state qubit candidates for quantum technology, where the first release was the main work behind the master thesis in [this repository](https://github.com/ohebbi/master-thesis). 

## Run project / Reproduction

The application of this project is centered around an exploratory analysis using Jupyter notebooks. It is not neccessary to run anything to see result, only consult the notebooks either here on Github or [Jupyers` nbviewer project](https://nbviewer.jupyter.org/). However, if you intend running the notebooks, read the next section 'Development'. 

### Development

#### Jupyter notebooks

Clone the project, and run "python3 -m pip install -r ." and you are ready for development. Remember to get add your API-keys from Materials Project and Citrination to your environmental variables (e.g. use an '.env'-file). Run all notebooks chronologically.

Instead of running the notebooks to generate data, we've also made the development of tools and code available with make.

#### Make

"make features" will extract MP data based on 0.1eV and ICSD-entry, and start the featurization process based on the preset.py. This is the only way to run the featurizer at this stage.

"make data" is an easier method to apply for all data in this project, thus an easier method to run 01-generateDataset-notebook.ipynb.

## Is this repo up to date?
New data is added for Materials Project randomly and will make a new featurization process needed for every update. This is currenly a long and tedious process (for preset.py implemented). Therefore, data featurized for this repo only include December 2020 version of data from MP.


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── raw            <- The original, immutable data dump.
    │   └── <approaches>       <- The final, canonical data sets for modeling.
    │       
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── <approaches>    <- Similarly for the other approaches
    │       ├── summary
    │       └── trained-models
    │
    │
    ├── notebooks                                <- Jupyter notebooks.
    │   ├── 01-generateDataset-notebook.ipynb    <- Generate data notebooks.
    │   ├── 02-buildFeatures-notebook.ipynb      <- Construct features.
    │   ├── 03-preprocessing-notebook.ipynb      <- Clean and preprocess features.
    │   ├── method-01-Ferrenti-approach                    
    │   │   ├── 04-dataMining-notebook.ipynb                 <- Datamining approach 1.
    │   │   └── PCA-NUMBER-<insert pca number>
    │   │         ├── 05-supervisedLearning-notebook.ipynb   <- Machine learning and predictions
    │   │         └── 06-postAnalysis-notebook.ipynb         <- Analyse the predictions
    │   ├── method-02-Extended-Ferrenti-approach          
    │   │   ├── 04-dataMining-notebook.ipynb                 <- Datamining approach 2.
    │   │   └── PCA-NUMBER-<insert pca number>
    │   │         ├── 05-supervisedLearning-notebook.ipynb   <- Machine learning and predictions
    │   │         └── 06-postAnalysis-notebook.ipynb         <- Analyse the predictions
    │   └── method-03-Empirical-approach                  
    │       ├── 04-dataMining-notebook.ipynb                 <- Datamining approach 3.
    │       └── PCA-NUMBER-<insert pca number>
    │             ├── 05-supervisedLearning-notebook.ipynb   <- Machine learning and predictions
    │             └── 06-postAnalysis-notebook.ipynb         <- Analyse the predictions
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │   └── README.md       
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │                         
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   ├── make_dataset.py
        │   ├── get_data_base.py
        │   ├── get_data_AFLOW.py
        │   ├── get_data_AFLOWML.py
        │   ├── get_data_Citrine.py
        │   ├── get_data_JARVIS.py
        │   ├── get_data_MP.py
        │   ├── get_data_OQMD.py
        │   └── utils.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │   └── featurizer.py
        │   └── preset.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
     

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
