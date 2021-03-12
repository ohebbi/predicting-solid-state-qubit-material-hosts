# plotting
import numpy as np

import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Linear Regression for bandgaps
from sklearn.linear_model import LinearRegression
def plotSimilarities(x, y, full_formulas, xlabel, ylabel, title=None):
    """
    A function used to plot band gaps.
    ...
    Args
    ----------
    x : list (dim:N)
        A list containing numeric values with np.nan as non-entries
    y : list (dim:N)
        A list containing numeric values with np.nan as non-entries
    xlabel: string
    ylabel: string
    title: string, default None

    Returns
    -------
    pd.DataFrame (dim:MxN)
        A DataFrame containing the resulting matching queries. This can result
        in several matching compounds
    """
    sim = 0
    for i, valx in enumerate(x):
        if float(valx)>0.:
            if (y[i]) == (valx):
                sim += 1
    print("The percentage of similarities is {}/{}={},".format(sim,len(x[x>0]),sim/len(x[x>0])))

    fig = go.Figure(data=go.Scattergl(x=x[(x>0)&(y>0)],
                                y=y[(x>0)&(y>0)],
                                mode='markers',
                                #marker_color=data['Perovskite'],
                                text=full_formulas,
                                line_width=1,
                                showlegend=False),
                layout = go.Layout (
                    autosize=False,
                    width=500,
                    height=500,
                    title=go.layout.Title(text=title),
                    xaxis=dict(title=xlabel),
                    yaxis=dict(title=ylabel, scaleanchor="x", scaleratio=1)))

    return fig

def plotBandGaps(x, y, full_formulas, xlabel, ylabel, title=None, addOLS = True):
    """
    A function used to plot band gaps.
    ...
    Args
    ----------
    x : list (dim:N)
        A list containing numeric values with np.nan as non-entries
    y : list (dim:N)
        A list containing numeric values with np.nan as non-entries
    df : pd.DataFrame(dim: NxN)

    xlabel: string
    ylabel: string
    title: string, default None
    addOLS: boolean, default True
        if True - fits an ordinary least square approximations to x and y,
        and adds the following model to the plot.

    Returns
    -------
    pd.DataFrame (dim:MxN)
        A DataFrame containing the resulting matching queries. This can result
        in several matching compounds
    """
    x = np.array(x)
    lowerBandGapLimit = 0.1
    x[x<lowerBandGapLimit] = np.nan
    y[y<lowerBandGapLimit] = np.nan

    fig = go.Figure(data=go.Scattergl(x=x[(x>0)&(y>0)],
                                y=y[(x>0)&(y>0)],
                                mode='markers',
                                #marker_color=data['Perovskite'],
                                text=full_formulas,
                                line_width=1,
                                showlegend=False),
                layout = go.Layout (
                    autosize=False,
                    width=500,
                    height=500,
                    title=go.layout.Title(text=title),
                    xaxis=dict(title=xlabel, range=[-0.1,8]),
                    yaxis=dict(title=ylabel, range=[-0.1,8], scaleanchor="x", scaleratio=1)))

    if addOLS:
        reg = LinearRegression().fit(x[(x>0)&(y>0)].reshape(-1,1),y[(x>0)&(y>0)])

        print("{}x+{}".format(reg.intercept_,reg.coef_))
        fig.add_trace(go.Scatter(y=[reg.intercept_,reg.intercept_+8*reg.coef_[0]], x=[0,8], mode="lines",showlegend=False))

    return fig


def plot_accuracy(models, names):
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['trainAccuracy'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Train accuracy',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Accuracy')
    fig.show()

    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['testAccuracy'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Test accuracy',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Accuracy')
    fig.show()

    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['std'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Standard deviation up to every iteration',
                   xaxis_title='Cross validation folds',
                   yaxis_title='std')
    fig.show()


    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['numPredPero'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Number of predicted candidates',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Counted candidates')
    fig.show()


def plot_important_features(models, names,X, k, n):
    """
    Plot features vs importance features
    """
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text='Features used in model (Nruns = {})'.format(k*n)),
                yaxis=dict(title="Number times"),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i], x=X.columns.values, y=model['importantKeys']))

    fig.show()

    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="Feature Importance for the 100th iteration".format(k*n)),
                yaxis=dict(title='Relative importance'),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i], x=X.columns.values, y=model['relativeImportance']))

    fig.show()

def plot_important_features_restricted_domain(models, names, trainingSet, k, n):
    """
    Only plot features that have been deemed important at least once.
    """

    threshold = 99
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="Features used in model (Nruns = {})".format(n*k)),
                yaxis=dict(title="Number times"),
                barmode="group"
            )
        )

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i],
                              x=trainingSet.columns[np.where(model["importantKeys"] > threshold)].values,
                              y=model["importantKeys"][model["importantKeys"] > threshold]))

    fig.show()
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="Feature Importance for the 100th iteration".format(n*k)),
                yaxis=dict(title="Relative importance"),
                barmode="group"
            )
        )

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i],
                              x=trainingSet.columns[np.where(model["relativeImportance"] > 0)].values,
                              y=model["relativeImportance"][model["relativeImportance"] > 0]))


    fig.show()



def plot_confusion_metrics(models, names, data,  k, n, abbreviations=[], cubicCase=False):
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="False positives (Nruns = {})".format(k*n)),
                yaxis=dict(title='Counts'),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        if cubicCase is not False:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data[abbreviations[i]]['Compound'][model['falsePositives'] > 0],
                                  y=model['falsePositives'][model['falsePositives'] > 0]))
        else:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data['Compound'][model['falsePositives'] > 0],
                                  y=model['falsePositives'][model['falsePositives'] > 0]))

    fig.show()

    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="False negatives (Nruns = {})".format(k*n)),
                yaxis=dict(title='Counts'),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        if cubicCase is not False:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data[abbreviations[i]]['Compound'][model['falseNegatives'] > 0],
                                  y=model['falseNegatives'][model['falseNegatives'] > 0]))
        else:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data['Compound'][model['falseNegatives'] > 0],
                                  y=model['falseNegatives'][model['falseNegatives'] > 0]))

    fig.show()


def plot_confusion_matrixQT(models, y, data, names, k, n):
    """
    Plot confusion matrix that lays the fundament of precision and recall metrics.
    """
    confidence = np.linspace(0,k*n,k*n)
    #confidence = 95 # % confidence. Put as 50 or more.
    mat = np.zeros((len(confidence),2,2))
    bigger_than = 100-confidence

    #Finding true positives and true negatives as a function of confidence
    for j, model in enumerate(models):

        for i, conf in enumerate(confidence):
            bigger_than = 100 - conf
            confidence[i] = bigger_than
            #print(conf)
            model["y_pred_full"] =  y.values.reshape(-1,).copy()

            model["y_pred_full"]\
                [data['material_id'][model['falseNegatives'] > bigger_than].index] = 0

            model["y_pred_full"]\
                [data['material_id'][model['falsePositives'] > bigger_than].index] = 1

            mat[i] = confusion_matrix(y.values.reshape(-1,), model["y_pred_full"])

        plt.plot(confidence, mat[:,0,1])
        plt.plot(confidence, mat[:,1,0])
        plt.plot([50,50],[-2,np.max(mat[:,0,1])], "--")
        #plt.plot(confidence, mat[:,1,1])
        #plt.plot(confidence, mat[:,0,0])

        plt.xlabel("Confidence / counts of wrongly predictions")
        plt.ylabel("Number of compounds")
        plt.title("Confusion matrix for predictions 100 times {}".format(names[j]))
        plt.legend(["False negatives", "False positives"])
        plt.show()

def confusion_matrixQT(models, y, names):
    #print(mat)
    for i, model in enumerate(models):
        mat = confusion_matrix(y.values.reshape(-1,), model["y_pred_full"])
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=[0,1],
                yticklabels=[0,1])
        plt.xlabel('true label')
        plt.ylabel('predicted label');
        plt.title("Confusion matrix {}".format(names[i]))
        plt.show()
