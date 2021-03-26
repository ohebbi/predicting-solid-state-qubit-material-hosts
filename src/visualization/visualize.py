# plotting
import plotly.graph_objs as go
import plotly.express as px
from plotly.graph_objs import *
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import auc, average_precision_score, roc_curve, precision_recall_curve, f1_score
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# textwidth in LateX
width = 411.14224

height = ((5**.5 - 1) / 2 )*width

width_plotly = 548.1896533333334 #pt to px
height_plotly = ((5**.5 - 0.75) / 2 )*width_plotly
tex_fonts = {
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.labelsize":12,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize":12
}
plt.rcParams.update(tex_fonts)


def set_size(width, fraction=1, subplots=(1,1)):
    """ Set fgure dimensions to avoid scaling in LateX.

    Args
    ---------
    width : float
            Document textwidth or columnwidth in pts
    fraction : float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and column of subplots
    Returns
    ---------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1/72.27

    # Golden ratio to set aeshetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


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
                    width=width_plotly,
                    height=height_plotly,
                    title=go.layout.Title(text=title),
                    xaxis=dict(title=xlabel),
                    yaxis=dict(title=ylabel, scaleanchor="x", scaleratio=1)))
    fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",
                   "paper_bgcolor": "rgba(0, 0, 0, 0)"})
    return fig

def matplotBandGaps(x, y, xlabel, ylabel, filename, title=None, addOLS = True):
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

    fig, ax = plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1)[0]))

    ax.plot(x[(x>0)&(y>0)], y[(x>0)&(y>0)], "o")
    ax.set(xlim=(0, 10), ylim=(0, 10))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if addOLS:
        reg = LinearRegression().fit(x[(x>0)&(y>0)].reshape(-1,1),y[(x>0)&(y>0)])

        linreg_y = [reg.intercept_,reg.intercept_+max(x)*reg.coef_[0]]
        linreg_x = [0,max(x)]

        ax.plot(linreg_x, linreg_y, color="red", label="")

        #some confidence interval
        ci = 1.96 * np.std(linreg_y)/np.mean(linreg_y)
        ax.fill_between(linreg_x, (linreg_y-ci), (linreg_y+ci), color='b', alpha=.1)

        print("label: {}. line = {:0.2f}x+{:0.2f}".format(ylabel,reg.coef_[0], reg.intercept_))
        print("CI: {:0.2f}".format(ci))

    fig.savefig(Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "bandgaps" \
                            / filename, format="pdf", bbox_inches="tight")
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
                    width=width_plotly,
                    height=height_plotly,
                    title=go.layout.Title(text=title),
                    xaxis=dict(title=xlabel, range=[-0.1,10]),
                    yaxis=dict(title=ylabel, range=[-0.1,10]),
                    font=dict(family="Palatino",
                              color="Black",
                              size=12),))

    if addOLS:
        reg = LinearRegression().fit(x[(x>0)&(y>0)].reshape(-1,1),y[(x>0)&(y>0)])

        print("{}x+{}".format(reg.intercept_,reg.coef_))
        fig.add_trace(go.Scatter(y=[reg.intercept_,reg.intercept_+10*reg.coef_[0]], x=[0,10], mode="lines",showlegend=False))
    fig.update_layout(font=dict(
                        family="Palatino",
                        color="Black",
                        size=12),
                      autosize=False,
                      width=width_plotly,
                      height=height_plotly,
                     )
    return fig

def plot_eigenvectors_principal_components(PCAcomponents, chosenNComponents:int = 10, NFeatures: int = 10):
    # plot
    fig, ax = plt.subplots(1,1,figsize=set_size(width, 1))

    c = range(0,chosenNComponents)

    cmap = plt.cm.get_cmap("cool", len(c))
    norm = mpl.colors.SymLogNorm(linthresh=2**(-4),vmin=c[-1], vmax=c[0])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i in range(chosenNComponents-1,0,-1):
        ax.plot(abs(PCAcomponents.iloc[i]).nlargest(NFeatures).values,"-", color=cmap(c[i]))

    ax.set_title("Sorted highest eigenvector for components")
    ax.set_xlabel("Top {} features".format(NFeatures))
    ax.set_ylabel("Eigenvectors")
    coloriarobaro=fig.colorbar(sm, ax=(ax), ticks=[0, c[-1]], shrink=1.0, fraction=0.05, format='%.0f')
    fig.show()

    # plot
    fig, ax = plt.subplots(1,1, figsize=set_size(width, 1))

    # Plot all eigenvectors
    #######################
    for i in range(chosenNComponents-1,0,-1):
        ax.plot(abs(PCAcomponents.iloc[i].values), ",",color=cmap(c[i]))

    # Plot first eigenvector
    #ax.plot(abs(PCAcomponents.iloc[0].values), "o", color=cmap(c[i]))

    ax.set_title("Principal component eigenvector per feature")
    ax.set_xlabel("Index of eigenvectors / Original feature number")
    ax.set_ylabel("Eigenvectors")
    coloriarobaro=fig.colorbar(sm, ax=(ax), ticks=[0, c[-1]], shrink=1.0, fraction=0.05, format='%.0f')

    fig.show()

def top_eigenvector_vs_features(PCAcomponents, whichComponent:int = 0, NFeatures: int = 10):

    # New figure
    fig, ax = plt.subplots(1,1,figsize=set_size(width, 1))

    #color
    c = range(0,NFeatures)
    cmap = plt.cm.get_cmap("cool", len(c))

    # Plot first pc
    ax.plot(abs(PCAcomponents.iloc[whichComponent].values), "o", color=cmap(c[0]))

    ax.set_title("Principal component eigenvector per feature")
    ax.set_xlabel("Index of eigenvectors / Original feature number")
    ax.set_ylabel("Eigenvectors")
    #coloriarobaro=fig.colorbar(sm, ax=(ax), ticks=[0, c[-1]], shrink=1.0, fraction=0.05, format='%.0f')

    fig.show()

    # Plot top features in first pc
    topFeaturesOfFirstComponent = abs(PCAcomponents.iloc[whichComponent]).nlargest(NFeatures).values.round(3)
    indices = abs(PCAcomponents.iloc[whichComponent]).nlargest(NFeatures).index

    fig = go.Figure(
            layout = go.Layout (
                autosize=False,
                width=width,
                height=height,
                title=go.layout.Title(text="Top {} features of PC[{}]".format(NFeatures, whichComponent)),
                yaxis=dict(title='Value of eigenvector',range=[0,topFeaturesOfFirstComponent[whichComponent]+topFeaturesOfFirstComponent[whichComponent]*0.5])
                #barmode='group'
            )
        )
    fig.add_traces(go.Scatter(x=indices.values, y=topFeaturesOfFirstComponent))

    fig.show()



def plot_accuracy(models, names):
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=set_size(width, 0.75, subplots=(3,1)))
    for i, model in enumerate(models):
        ax1.plot(model['trainAccuracy'], label=names[i])#, color = color[j])
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.set_title('Training accuracy')
    ax1.legend(loc='best')

    for i, model in enumerate(models):
        ax2.plot(model['testAccuracy'], label=names[i])#, color = color[j])
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.set_title('Test accuracy')

    for i, model in enumerate(models):
        ax3.plot(model['f1_score'], label=names[i])#, color = color[j])
    ax3.set_title('f1-score')

    ax3.set_xlabel("Cross validation folds")
    fig.tight_layout()

    plt.show()
    """
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['trainAccuracy'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(autosize=False,
                    width=width,
                    height=height,
                   title='Train accuracy',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Accuracy')
    fig.show()

    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['testAccuracy'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(
                    autosize=False,
                    width=width,
                    height=height,
                    title='Test accuracy',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Accuracy',
                   font=dict(family="Palatino",
                             color="Black",
                            size=12),)
    fig.show()

    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['f1_score'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(
                  autosize=False,
                  width=width,
                  height=height,
                  title='f1 score on test set',
                   xaxis_title='Cross validation folds',
                   yaxis_title='f1 score',
                   font=dict(family="Palatino",
                             color="Black",
                            size=12),)
    fig.show()
    """
    """
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['numPredPero'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Number of predicted candidates',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Counted candidates')
    fig.show()
    """

def plot_important_features(models, names,X, k, n):
    """
    Plot features vs importance features
    """
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
    """
    fig = go.Figure(
            layout = go.Layout (
                autosize=False,
                width=width_plotly,
                height=height_plotly,
                title=go.layout.Title(text="Feature Importance for the 100th iteration".format(k*n)),
                yaxis=dict(title='Relative importance'),
                barmode='group',
                font=dict(family="Palatino",
                          color="Black",
                         size=12),)
            )


    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i], x=X.columns.values, y=model['relativeImportance']))

    fig.show()

def plot_important_features_restricted_domain(models, names, trainingSet, k, n):
    """
    Only plot features that have been deemed important at least once.
    """

    threshold = 49
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="Features used in model (Nruns = {})".format(n*k)),
                yaxis=dict(title="Number times"),
                barmode="group",
                font=dict(family="Palatino",
                          color="Black",
                         size=12)))


    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i],
                              x=trainingSet.columns[np.where(model["importantKeys"] > threshold)].values,
                              y=model["importantKeys"][model["importantKeys"] > threshold]))

    fig.show()
    fig = go.Figure(
            layout = go.Layout (
                autosize=False,
                width=width*0.8,
                height=height*0.8,
                title=go.layout.Title(text="Feature Importance for the 100th iteration".format(n*k)),
                yaxis=dict(title="Relative importance"),
                barmode="group",
                font=dict(family="Palatino",
                          color="Black",
                         size=12)))


    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i],
                              x=trainingSet.columns[np.where(model["relativeImportance"] > 0)].values,
                              y=model["relativeImportance"][model["relativeImportance"] > 0]))


    fig.show()



def plot_confusion_metrics(models, names, data,  k, n, abbreviations=[]):
    """
    Plot false positives and false negatives for a given dataset.
    """
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="False positives (Nruns = {})".format(n*k)),
                yaxis=dict(title='Counts'),
                barmode='group',
                font=dict(family="Palatino",
                          color="Black",
                         size=12))
            )


    for i, model in enumerate(models):

        fig.add_traces(go.Bar(name=names[i],
                            x=data['full_formula'][model['falsePositives'] > 0],
                            y=model['falsePositives'][model['falsePositives'] > 0]))
    fig = go.Figure(
            layout = go.Layout (
                autosize=False,
                width=width_plotly,
                height=height_plotly,
                title=go.layout.Title(text="False positive (Nruns = {})".format(n*k)),
                yaxis=dict(title='Counts'),
                barmode='group',
                font=dict(family="Palatino",
                          color="Black",
                         size=12))
            )

    fig.show()

    fig = go.Figure(
            layout = go.Layout (
                autosize=False,
                width=width_plotly,
                height=height_plotly,
                title=go.layout.Title(text="False negative (Nruns = {})".format(n*k)),
                yaxis=dict(title='Counts'),
                barmode='group',
                font=dict(family="Palatino",
                          color="Black",
                         size=12),)
            )

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i],
                                x=data['full_formula'][model['falseNegatives'] > 0],
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
        plt.figure(figsize=set_size(width, 0.4))
        plt.show()

def draw_cv_roc_curve(classifier,
                       X: pd.DataFrame,
                       y,
                       k: int,
                       n: int,
                       cv,
                       title: str):
    """
    Draw a Cross Validated ROC Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object:
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1)[0]))
    i = 0
    for train, test in tqdm(cv.split(X, y)):
        probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, color='grey', alpha=0.4)
                # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)

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
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CV-ROC Curve " + str(title))
    ax.legend(loc="lower right")
    fig.tight_layout()

    plt.show()


def draw_cv_pr_curve(classifier,
                       X: pd.DataFrame,
                       y,
                       k: int,
                       n: int,
                       cv,
                       title: str):
    """
    Draw a Cross Validated PR Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Example adapted from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    # New figure
    fig, ax = plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1)[0]))

    y_real = []
    y_proba = []
    i = 0

    for train, test in tqdm(cv.split(X, y)):
        probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(y[test], probas_[:, 1])

        # Plotting each individual PR Curve
        ax.plot(recall, precision, lw=1, alpha=0.3, color='grey')
                 #label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

        y_real.append(y[test])
        y_proba.append(probas_[:, 1])

        i += 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)


    ax.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title("CV-PR Curve " + str(title))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()

def plot_parallel_coordinates(data, dimensions, color):
    fig = px.parallel_categories(data, dimensions=dimensions,
                                 color=color, color_continuous_scale=px.colors.sequential.Inferno)
    fig.update_layout(
                    {"plot_bgcolor": "rgba(0, 0, 0, 0)",
                       "paper_bgcolor": "rgba(0, 0, 0, 0)",
                      },
                      font=dict(
                        family="Palatino",
                        size=12),
                      autosize=False,
                      width=width_plotly,
                      height=height_plotly,
                     )
    fig.update_layout(
    font_family="Palatino",
    font_color="black",
    font_size=12
    )

    fig.show()

def plot_histogram_bg_nelements(entries):
    _nelements = {1: "Unary", 2: "Binary", 3: "Ternary", 4: "Quaternary", 5: "Quinary", 6: "Senary", 7: "Septenary", 8: "Octary"}
    fig = px.histogram(entries[entries["MP|band_gap"]<9], x="MP|band_gap", color="MP|nelements", nbins=20,
                       title='Band gaps and material phases in dataset',
                       labels={"MP|band_gap": "MP BG [ev]", 'MP|nelements':'Material phase'},
                       category_orders={"MP|nelements": list(_nelements.values())})

    fig.update_layout(
                    {"plot_bgcolor": "rgba(0, 0, 0, 0)",
                       "paper_bgcolor": "rgba(0, 0, 0, 0)",
                      },
                      font=dict(
                        family="Palatino",
                        color="Black",
                        size=12),
                      autosize=False,
                      width=width_plotly,
                      height=height_plotly,
                     )
    fig.write_image(str(Path(__file__).resolve().parents[2] / \
                                "reports" / "figures"  / "buildingFeatures"\
                                / "histogram_bg_nelements.pdf"))
    fig.show()

def plot_histogram_oxid_nelements(entries):
    _oxideType = {"None": 0, "Oxide":1, "Peroxide":2, "Hydroxide":3, "Superoxide":4, "Ozonide":5}
    _nelements = {1: "Unary", 2: "Binary", 3: "Ternary", 4: "Quaternary", 5: "Quinary", 6: "Senary", 7: "Septenary", 8: "Octary"}

    fig = px.histogram(entries, x="MP|nelements", color="MP|oxide_type", nbins=7,
                   title='Oxid types and material phases in dataset',
                   labels={'MP|nelements':'Material phase', "MP|oxide_type": "Oxid type"},
                   category_orders={"MP|nelements": list(_nelements.values()),
                                    "MP|oxide_type":list(_oxideType.keys())})
    fig.update_layout(
                    {"plot_bgcolor": "rgba(0, 0, 0, 0)",
                       "paper_bgcolor": "rgba(0, 0, 0, 0)",
                      },
                      font=dict(
                        family="Palatino",
                        color="Black",
                        size=12),
                      autosize=False,
                      width=width_plotly,
                      height=height_plotly,
                     )

    fig.write_image(str(Path(__file__).resolve().parents[2] / \
                                    "reports" / "figures"  / "buildingFeatures" \
                                    / "histogram_oxid_nelements.pdf"))
    fig.show()

def resampling(X, y, method = None, strategy = None):
    """
    Applies a given resampling technique to dataset.
    """
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

def runSupervisedModel(classifier,
                       X: pd.DataFrame,
                       y,
                       k: int,
                       n: int,
                       cv,
                       title: str,
                       featureImportance: Optional[bool] = False,
                       resamplingMethod: Optional[str] = "None"):


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

    # Initializing Creating ROC metrics
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 101)
    fig2, ax2 = plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1)[0]))


    #  Initializing precision recall metrics
    fig1, ax1= plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1)[0]))
    y_real = []
    y_proba = []

    # splitting into 50%/50% training and test data if n_splits = 2, or 90%/10% if n_splits=10
    #rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n, random_state=random_state)

    if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
        sel_classifier = SelectFromModel(classifier.named_steps["model"])

    for i, (train_index, test_index) in tqdm(enumerate(cv.split(X, y))):

        #partition the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]


        #fit the model
        classifier.fit(X_train, y_train)
        if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
            sel_classifier.fit(X_train, y_train)

        #predict on test set
        y_pred      = classifier.predict(X_test)
        probas_     = classifier.predict_proba(X_test)

        #Finding predicted labels on all data based on training data.
        y_pred_full = classifier.predict(X)

        ############################################
        ## Compute ROC curve and area under curve ##
        ############################################

        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax2.plot(fpr, tpr, lw=1, color='grey', alpha=0.4)
                # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        ######################################
        ## Finding precision recall metrics ##
        ######################################

        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])

        # Plotting each individual PR Curve
        ax1.plot(recall, precision, lw=1, alpha=0.3, color='grey')
                 #label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

        y_real.append(y_test)
        y_proba.append(probas_[:, 1])

        ######################################
        ##### Finding FP and FN metrics ######
        ######################################

        falsePositives = np.nonzero(y_pred_full.reshape((-1,)) > y)
        falseNegatives = np.nonzero(y_pred_full.reshape((-1,)) < y)

        #claim the scores
        modelResults['trainAccuracy'][i] = classifier.score(X_train, y_train)
        modelResults['testAccuracy'][i]  = classifier.score(X_test, y_test)
        modelResults['f1_score'][i]      = f1_score(y_test, y_pred)
        modelResults['std'][i]           = np.std(modelResults['testAccuracy'][0:i+1])
        modelResults['numPredPero'][i]   = np.sum(y_pred_full)
        modelResults['confusionMatrix']  = confusion_matrix(y_test, y_pred)
        modelResults['falsePositives'][falsePositives] += 1
        modelResults['falseNegatives'][falseNegatives] += 1

        if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
            modelResults['importantKeys'][sel_classifier.get_support()] += 1

    if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
        modelResults['relativeImportance'] = classifier.named_steps["model"].feature_importances_

    ######################################
    ## Finding precision recall metrics ##
    ######################################
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    ax1.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    ax1.set_title("CV-PR Curve " + str(title))
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")

    ax1.legend(loc="lower right")
    fig1.tight_layout()

    ######################################
    ######## ROC CURVE and AOG ###########
    ######################################

    ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax2.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax2.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                     label=r'$\pm$ 1 std. dev.')

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("CV-ROC Curve " + str(title))
    ax2.legend(loc="lower right")
    fig2.tight_layout()

    plt.show()


    print ("Mean accuracy:{:0.5f}".format(np.mean(modelResults['testAccuracy'])))
    print ("Standard deviation:{:0.5f}".format(modelResults['std'][-1]))
    print ("f1-score:{:0.5f}".format(modelResults['f1_score'][-1]))

    return modelResults
