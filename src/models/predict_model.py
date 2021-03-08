from pathlib import Path
data_dir = Path.cwd().parent.parent / "data"

def runPredictions(fitted_classifier, testData):
    """
    Predicts qubit candidate label based on a fitted classifier / pipeline. 
    """
    #predict
    predictions = fitted_classifier.predict(testData)
    probability = fitted_classifier.predict_proba(testData)

    #the predicted perovskites
    return predictions, probability[:,1]
