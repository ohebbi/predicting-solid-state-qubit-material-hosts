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

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import matplotlib.pyplot as plt

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
            ('model', model)
        ])

    if len(sampler)==2:
        return Pipeline([
            sampler,
            ('model', model)
        ])

    elif len(sampler)==4:
        return Pipeline([
            sampler[0:2],
            sampler[2:4],
            ('model', model)
        ])

    else:
        raise ValueError("Wrong number of samplers: len(sampler)={}".format(len(sampler)))

def findParamGrid(model):
    typeModel = type(model)
    if typeModel == type(RandomForestClassifier()):
        return {"model__n_estimators": [10, 100, 1000],
                "model__max_features": ['auto', 'sqrt', 'log2'],
                "model__max_depth" : [2,4,6,8],
                "model__criterion" :['gini', 'entropy'],
                }
    elif typeModel == type(GradientBoostingClassifier()):
        return {"model__loss":["deviance", "exponential"],
                #"model__learning_rate": [0.01, 0.025, 0.1, 0.2],
                "model__max_features": ['auto', 'sqrt', 'log2'],
                "model__max_depth":[2,4,6,8],
                "model__max_features":["log2","sqrt"],
                "model__criterion": ["friedman_mse", "mse"],
                #"model__subsample":[0.5, 0.75, 1],
                "model__n_estimators":[10,100,1000]
                }
    else:
        raise TypeError("No model has been specified: type(model):{}".format(typeModel))


def applyGridSearch(X: pd.DataFrame, y, model, cv, sampleMethod="under"):
    param_grid = findParamGrid(model)

    ## TODO: Insert these somehow in gridsearch (scoring=scoring,refit=False)
    scoring = {'accuracy':  make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall':    make_scorer(recall_score),
               'f1':        make_scorer(f1_score)}

    # Making a pipeline
    pipe = getPipe(model, sampleMethod)
    # Do a gridSearch
    grid = GridSearchCV(pipe, param_grid, scoring=scoring, refit="f1",
                        cv=cv,verbose=2,return_train_score=True, n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_estimator_)

    return grid.best_estimator_, grid



def runSupervisedModel(classifier,
                       X: pd.DataFrame,
                       y,
                       k: int,
                       n: int,
                       cv,
                       featureImportance: Optional[bool] = False,
                       resamplingMethod: Optional[str] = "None"):
    print("hallo")
    def resampling(X, y, method = None, strategy = None):

        if method == "under":
            if strategy:
                underSample = RandomUnderSampler(sampling_strategy=strategy)
            else:
                underSample = RandomUnderSampler(sampling_strategy="majority")

            return underSample.fit_resample(X, y)

        elif method == "over":
            if strategy:
                overSample = RandomOverSampler(sampling_strategy=strategy)
            else:
                overSample = RandomOverSampler(sampling_strategy="minority")
            return overSample.fit_resample(X, y)

        elif method == "both":
            X, y = resampling(X, y, method = "over", strategy = 0.5)
            return resampling(X, y, method = "under", strategy = 1)
        else:
            #print("No resampling applied.")
            return X, y

    modelResults = {
        'trainAccuracy':   np.zeros(n*k),
        'testAccuracy':    np.zeros(n*k),
        'f1_score':        np.zeros(n*k),
        'std':             np.zeros(n*k),
        'importantKeys':   np.zeros(len(X.columns.values)),
        'numPredPero':     np.zeros(n*k),
        'confusionMatrix': np.zeros((len(y), len(y))),
        'falsePositives':  np.zeros(len(y)),
        'falseNegatives':  np.zeros(len(y)),
        'relativeImportance': np.zeros(len(X.columns.values))
        }
    """
    # ROC-curve stakeholders
    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    """
    # splitting into 50%/50% training and test data if n_splits = 2, or 90%/10% if n_splits=10
    #rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n, random_state=random_state)

    if (featureImportance):
        sel_classifier = SelectFromModel(classifier.named_steps["model"])

    for i, (train_index, test_index) in tqdm(enumerate(cv.split(X, y))):

        #partition the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ApplyResampling

        X_train, y_train = resampling(X_train, y_train,
                                method=resamplingMethod)

        #fit the model
        classifier.fit(X_train, y_train)
        if (featureImportance):
            sel_classifier.fit(X_train, y_train)

        #predict on test set
        y_pred      = classifier.predict(X_test)
        """
        # ROC-statistics
        y_pred_prob = classifier.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
        viz = plot_roc_curve(classifier, X_test, y_test,
                         alpha=0.3, lw=1, ax=ax)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        """
        #Finding predicted labels on all data based on training data.
        y_pred_full = classifier.predict(X)

        falsePositives = np.nonzero(y_pred_full.reshape((-1,)) > y)
        falseNegatives = np.nonzero(y_pred_full.reshape((-1,)) < y)

        #claim the scores
        modelResults['trainAccuracy'][i] = classifier.score(X_train, y_train)
        modelResults['testAccuracy'][i]  = classifier.score(X_test, y_test)
        modelResults['f1_score'][i]      = f1_score(y_true, y_pred)
        modelResults['std'][i]           = np.std(modelResults['testAccuracy'][0:i+1])
        modelResults['numPredPero'][i]   = np.sum(y_pred_full)
        modelResults['confusionMatrix']  = confusion_matrix(y_test, y_pred)
        modelResults['falsePositives'][falsePositives] += 1
        modelResults['falseNegatives'][falseNegatives] += 1

        if (featureImportance):
            modelResults['importantKeys'][sel_classifier.get_support()] += 1

    if (featureImportance):
        modelResults['relativeImportance'] = classifier.named_steps["model"].feature_importances_

    print ("Mean accuracy:{}".format(np.mean(modelResults['testAccuracy'])))
    print ("Standard deviation:{}".format(modelResults['std'][-1]))
    """
    # ROC-statistics
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    plt.show()
    """

    return modelResults



def fitAlgorithm(classifier, trainingData, trainingTarget):
    """
    Fits a given classifier / pipeline
    """
    #train the model
    return classifier.fit(trainingData, trainingTarget)
