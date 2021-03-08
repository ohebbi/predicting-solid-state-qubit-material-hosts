#Feature selections
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tqdm import tqdm
import numpy as np

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score,precision_score, recall_score, make_scorer

#Resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def chooseSampler(sampleMethod):
    if sampleMethod == "under":
        return ("underSampler", RandomUnderSampler(sampling_strategy="majority"))

    elif sampleMethod == "over":
        return ("overSampler", SMOTE(sampling_strategy="minority"))

    elif sampleMethod == "both":
        return "overSampler", SMOTE(sampling_strategy="minority"),\
               "underSampler", RandomUnderSampler(sampling_strategy="majority")

    else:
        return None

def getPipe(model, sampleMethod):
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
        return {"model__n_estimators": [10, 50, 100, 200],
                "model__max_features": ['auto', 'sqrt', 'log2'],
                "model__max_depth" : [2,3,4,5],
                "model__criterion" :['gini', 'entropy'],
                }
    elif typeModel == type(GradientBoostingClassifier()):
        return {"model__loss":["deviance"],
                "model__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "model__min_samples_split": np.linspace(0.1, 0.5, 3),
                "model__min_samples_leaf": np.linspace(0.1, 0.5, 3),
                "model__max_depth":[1,3,5,8],
                "model__max_features":["log2","sqrt"],
                "model__criterion": ["friedman_mse",  "mae", "mse"],
                "model__subsample":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "model__n_estimators":[5,10,15,20]
                }
    else:
        raise TypeError("No model has been specified: type(model):{}".format(typeModel))


def applyGridSearch(X, y, model, cv, sampleMethod="under"):
    param_grid = findParamGrid(model)

    ## TODO: Insert these somehow in gridsearch (scoring=scoring,refit=False)
    scoring = {'accuracy':  make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall':    make_scorer(recall_score),
               'f1':        make_scorer(f1_score),}

    # Making a pipeline
    pipe = getPipe(model, sampleMethod)

    # Do a gridSearch
    grid = GridSearchCV(pipe, param_grid, scoring=scoring, refit="f1",
                        cv=cv,verbose=2,return_train_score=True, n_jobs=-1)

    grid.fit(X, y)

    return grid.best_estimator_, grid



def runSupervisedModel(classifier, X, y, k, n,
                       featureImportance = False,
                       random_state = 481123480,
                       applyResampling = False,
                       resamplingMethod="under"):

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
        'std':             np.zeros(n*k),
        'importantKeys':   np.zeros(len(X.columns.values)),
        'numPredPero':     np.zeros(n*k),
        'confusionMatrix': np.zeros((len(y), len(y))),
        'falsePositives':  np.zeros(len(y)),
        'falseNegatives':  np.zeros(len(y)),
        'relativeImportance': np.zeros(len(X.columns.values))
        }


    # splitting into 50%/50% training and test data if n_splits = 2, or 90%/10% if n_splits=10
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n, random_state=random_state)

    if (featureImportance):
        sel_classifier = SelectFromModel(classifier.named_steps["model"])

    for i, (train_index, test_index) in tqdm(enumerate(rskf.split(X, y))):

        #partition the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ApplyResampling
        if (applyResampling):
            X_train, y_train = resampling(X_train, y_train,
                                method=resamplingMethod)

        #fit the model
        classifier.fit(X_train, y_train)
        if (featureImportance):
            sel_classifier.fit(X_train, y_train)

        #predict on test set
        y_pred = classifier.predict(X_test)

        #Finding predicted labels on all data based on training data.
        y_pred_full = classifier.predict(X)

        falsePositives = np.nonzero(y_pred_full.reshape((-1,)) > y)
        falseNegatives = np.nonzero(y_pred_full.reshape((-1,)) < y)

        #claim the scores
        modelResults['trainAccuracy'][i] = classifier.score(X_train, y_train)
        modelResults['testAccuracy'][i]  = classifier.score(X_test, y_test)
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

    return modelResults

def fitAlgorithm(classifier, trainingData, trainingTarget):
    """
    Fits a given classifier / pipeline
    """
    #train the model
    return classifier.fit(trainingData, trainingTarget)
