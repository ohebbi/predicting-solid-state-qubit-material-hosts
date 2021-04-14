from typing import Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

# Feature selections
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# ROC
from sklearn.metrics import confusion_matrix, auc, plot_roc_curve

# Scores
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import gridspec

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def chooseSampler(sampleMethod: Optional[str]):
    if sampleMethod == "under":
        return ("underSampler", RandomUnderSampler(sampling_strategy="majority"))

    elif sampleMethod == "over":
        return ("overSampler", SMOTE(sampling_strategy="minority"))

    elif sampleMethod == "both":
        return "overSampler", SMOTE(sampling_strategy="minority"),\
               "underSampler", RandomUnderSampler(sampling_strategy="majority")

    else:
        return None

def getPipe(model, sampleMethod: Optional[str]):
    sampler = chooseSampler(sampleMethod)
    if not (sampler):
        return Pipeline([
            ('scale', StandardScaler()),
            ("pca", PCA(svd_solver="randomized")),
            ('model', model)
        ])

    if len(sampler)==2:
        return Pipeline([
            ('scale', StandardScaler()),
            ("pca", PCA(svd_solver="randomized")),
            sampler,
            ('model', model)
        ])

    elif len(sampler)==4:
        return Pipeline([
            ('scale', StandardScaler()),
            ("pca", PCA(svd_solver="randomized")),
            sampler[0:2],
            sampler[2:4],
            ('model', model)
        ])

    else:
        raise ValueError("Wrong number of samplers: len(sampler)={}".format(len(sampler)))

def findParamGrid(model, numFeatures):
    typeModel = type(model)
    if typeModel == type(RandomForestClassifier()):
        return {#"model__n_estimators": [10, 100, 1000],
                "model__max_features": ['auto'],#, 'sqrt', 'log2'],#[1, 25,50, 75, 100], #
                "model__max_depth" : [2],#,4,6,8],
                #"model__criterion" :['gini', 'entropy'],
                "pca__n_components": range(1,numFeatures+1)
                }
    elif typeModel == type(GradientBoostingClassifier()):
        return {#"model__loss":["deviance", "exponential"],
                #"model__learning_rate": [0.01, 0.025, 0.1, 0.2],
                "model__max_depth":[2],#,4],6,8],
                "model__max_features":['auto'],#, 'sqrt', 'log2'],#[25,50, 75, 100], #['auto', 'sqrt', 'log2'],
                #"model__criterion": ["friedman_mse", "mse"],
                #"model__subsample":[0.5, 0.75, 1],
                #"model__n_estimators":[10,100,1000],
                "pca__n_components": range(1,numFeatures+1)
                }
    elif typeModel == type(LogisticRegression()):#penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
        return {"model__penalty":["l2"],# "l2", "elasticnet", "none"],
                #"model__learning_C": [0.001,0.01,0.1,1,10,100,1000],
                "model__max_iter":[200],
                "pca__n_components": range(1,numFeatures+1)
                }
    else:
        raise TypeError("No model has been specified: type(model):{}".format(typeModel))


def applyGridSearch(X: pd.DataFrame, y, model, cv, numPC: int, sampleMethod="None"):
    param_grid = findParamGrid(model, numFeatures=numPC)

    ## TODO: Insert these somehow in gridsearch (scoring=scoring,refit=False)
    scoring = {'accuracy':  make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall':    make_scorer(recall_score),
               'f1':        make_scorer(f1_score),
               }

    # Making a pipeline
    pipe = getPipe(model, sampleMethod)
    # Do a gridSearch
    grid = GridSearchCV(pipe, param_grid, scoring=scoring, refit="f1",
                        cv=cv,verbose=2,return_train_score=True, n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_estimator_)

    return grid.best_estimator_, grid


def fitAlgorithm(classifier, trainingData, trainingTarget):
    """
    Fits a given classifier / pipeline
    """
    #train the model
    return classifier.fit(trainingData, trainingTarget)
