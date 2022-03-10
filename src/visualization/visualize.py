# plotting
import plotly.graph_objs as go
import plotly.express as px
from plotly.graph_objs import *
from typing import Optional
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import auc, average_precision_score, roc_curve, precision_recall_curve, f1_score
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Linear Regression for bandgaps
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots

from matplotlib import gridspec
# textwidth in LateX
width = 411.14224

height = ((5**.5 - 1) / 2 )*width

width_plotly = 548.1896533333334 #pt to px
height_plotly = ((5**.5 - 0.75) / 2 )*width_plotly

#plt.rcParams.update(tex_fonts)

def set_size(width, fraction=1, subplots=(1,1), isTex=False):
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
    if isTex:
        return (fig_width_in*0.5, fig_height_in*1) #cm
    return (fig_width_in, fig_height_in)

import matplotlib.font_manager as font_manager

#font_dirs = ['/home/oliver/.local/share/fonts/', ]
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#font_list = font_manager.FontManager.addfont(font_files)
#font_manager.fontManager.ttflist.extend(font_list)

# uncomment the following lines to use with latex. Else, leave as is. 
pgf_with_latex = {                      # setup matplotlib to use latex for output
    #"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    #"text.usetex": True,                # use LaTeX to write all text
    "font.family": "Palatino Linotype",
    "font.serif": [],                   # blank entries should cause plots
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    #"axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    #"font.weight": "bold",
    "legend.fontsize": 8,               # Make the legend/label fonts
    "xtick.labelsize": 8,               # a little smaller
    "ytick.labelsize": 8,
    "figure.figsize": set_size(width, 0.9),     # default fig size of 0.9 textwidth
    #"pgf.preamble": r"\usepackage[detect-all,locale=DE]{siunitx} \usepackage[T1]{fontenc} \usepackage[utf8x]{inputenc}"
    }

mpl.rcParams.update(pgf_with_latex)

def save_matplot_fig(fig, dir_path, filename):

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    fig.savefig(dir_path / filename , format="pgf", bbox_inches="tight")


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

def matplotBandGaps(x1, y1, x2, y2, xlabel, ylabel, filename, title=None, addOLS = True, first=False):
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
    x1 = np.array(x1)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(set_size(width, 1, subplots=(1,2))[0], set_size(width, 0.45, subplots=(1,2))[0] ))

    ax1.plot(x1[(x1>0)&(y1>0)], y1[(x1>0)&(y1>0)], "o",color='k', markersize=3)
    ax1.set(xlim=(0, 10.5), ylim=(0, 10.5))
    ax1.plot([-5,15], [-5,15], "--",color='black')
    if first:
        ax1.set_title("Common db. entries")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if addOLS:
        reg = LinearRegression().fit(x1[(x1>0)&(y1>0)].reshape(-1,1),y1[(x1>0)&(y1>0)])

        linreg_y = [reg.intercept_,reg.intercept_+max(x1)*reg.coef_[0]]
        linreg_x = [0,max(x1)]

        ax1.plot(linreg_x, linreg_y, color="red", label=r"${:0.2f}x+{:0.2f}$".format(reg.coef_[0], reg.intercept_))

        #some confidence interval
        ci = 1.96 * np.std(linreg_y)/np.mean(linreg_y)
        ax1.fill_between(linreg_x, (linreg_y-ci), (linreg_y+ci), color='k', alpha=.1, label=r"$\pm {:0.2f}$".format(ci))

        print("label to the left: {}. line = {:0.2f}x+{:0.2f}".format(ylabel,reg.coef_[0], reg.intercept_))
        print("CI: {:0.2f}".format(ci))
    ax1.legend(loc="upper left")

    x2 = np.array(x2)
    y2 = np.array(y2)
    ax2.plot(x2[(x2>0)&(y2>0)], y2[(x2>0)&(y2>0)], "o", markersize=3)
    ax2.set(xlim=(0, 10.5), ylim=(0, 10.5))
    ax2.plot([-5,15], [-5,15], "--",color='black')
    if first:
        ax2.set_title("Common exp. entries")
    ax2.set_xlabel(xlabel)
    #ax2.set_ylabel(ylabel)

    if addOLS:
        reg = LinearRegression().fit(x2[(x2>0)&(y2>0)].reshape(-1,1),y2[(x2>0)&(y2>0)])

        linreg_y = [reg.intercept_,reg.intercept_+max(x2[x2>0])*reg.coef_[0]]
        linreg_x = [0,max(x2[x2>0])]

        ax2.plot(linreg_x, linreg_y, color="red", label=r"${:0.2f}x+{:0.2f}$".format(reg.coef_[0], reg.intercept_))

        #some confidence interval
        ci = 1.96 * np.std(linreg_y)/np.mean(linreg_y)
        ax2.fill_between(linreg_x, (linreg_y-ci), (linreg_y+ci), color='b', alpha=.1, label=r"$\pm {:0.2f}$".format(ci))

        print("label to the left: {}. line = {:0.2f}x+{:0.2f}".format(ylabel,reg.coef_[0], reg.intercept_))
        print("CI: {:0.2f}".format(ci))
    
    ax2.legend(loc="upper left")
    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "bandgaps"

    fig.savefig(dir_path / filename, format="pdf", bbox_inches="tight")

    fig.tight_layout()
    print(set_size(width, 1, subplots=(1,2), isTex=True)[0])

    tikzplotlib.save(dir_path / str(filename[:-4] + ".tex"),
                            axis_width = str(set_size(width, 0.9, subplots=(1,2), isTex=True)[0]) + "in",
                            axis_height  = str(set_size(width, 0.9, subplots=(1,2), isTex=True)[0]) + "in")
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



def plot_accuracy(models, names, prettyNames, numPC, approach, xlabel = "Cross validation folds"):
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

    ax3.set_xlabel(xlabel)
    fig.tight_layout()

    dir_path = Path(__file__).resolve().parents[2] / \
                                    "reports" / "figures"  / "cv-accuracy"

    #save_matplot_fig(fig, dir_path=dir_path, filename=Path(approach + "-" + str(numPC) + ".pgf"))

    plt.show()
def plot_important_features(models, X, k, n, prettyNames, numPC, approach, numFeat=25):
    scaledTrainingData = StandardScaler().fit_transform(X) # normalizing the features
    pca = PCA(0.955).fit(scaledTrainingData)
    dir_path = Path(__file__).resolve().parents[2] / "reports" / "figures" / "feature-importance"
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    fig = make_subplots(rows=models.shape[0], cols=1, shared_xaxes=True)
    fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",
                           "paper_bgcolor": "rgba(0, 0, 0, 0)",
                          },
                        barmode='group',
                        autosize=False,
                        width=width_plotly,
                        height=height_plotly,
                        margin=dict(l=0, r=0, t=25, b=0),
                        title=go.layout.Title(text="Mean feature importance for {} iterations".format(k*n)),
                        #xaxis=dict(title="Number principal components"),
                        #yaxis=dict(title="Relative importance"),
                        font=dict(family="Palatino",
                                  color="Black",
                                  size=12),)

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=prettyNames[i], x=np.arange(1,numPC+1),
                    y=np.mean(model["relativeImportance"], axis=0),
                    error_y=dict(type='data', array=np.std(model["relativeImportance"], axis=0))), cols = 1, rows=i+1)

    fig['layout']['xaxis']['title']='Number principal component'

    fig.write_image(str(dir_path / Path(approach + "-" + str(numPC) + "-" + prettyNames[i] +".pdf")))
    fig.show()

    fig, (ax0,ax1,ax2,ax3, ax4) = plt.subplots(5,1, figsize=(set_size(width, 1)[0], set_size(width, 1.5)[0]))

    scaledTrainingData = StandardScaler().fit_transform(X) # normalizing the features
    pca = PCA(0.955).fit(scaledTrainingData)
    #print(pca.explained_variance_ratio_)
    map_names={"01-ferrenti-approach": "Ferrenti approach", "02-augmented-ferrenti-approach": "Augmented Ferrenti approach", "03-insightful-approach": "Insightful approach"}
    #ax0.set_title(map_names[approach])

    mean_importance = np.mean(models[prettyNames[0]]["relativeImportance"], axis=0)[:numFeat]
    std_importance = np.std(models[prettyNames[0]]["relativeImportance"], axis=0)[:numFeat]
    ax0.bar(np.arange(1,len(mean_importance)+1,1), mean_importance, color="#88CCEE")
    ax0.errorbar(x=np.arange(1,len(mean_importance)+1,1), y=mean_importance,yerr=std_importance, fmt='none', capsize=4, color="#88CCEE")
    ax0.set_ylim([min(mean_importance-0.05),max(mean_importance+0.2)])
    ax0.set_xlim([0.5,numFeat+0.5])
    ax0.grid()
    #ax0.legend(loc="upper right")
    #ax0.set_title("Logistic regression coef.")

    mean_importance = np.mean(models[prettyNames[1]]["relativeImportance"], axis=0)[:numFeat]
    std_importance = np.std(models[prettyNames[1]]["relativeImportance"], axis=0)[:numFeat]
    ax1.bar(np.arange(1,len(mean_importance)+1,1), mean_importance, color="#CC6677")
    ax1.errorbar(x=np.arange(1,len(mean_importance)+1,1), y=mean_importance,yerr=std_importance, fmt='none', capsize=4, color="#CC6677")
    ax1.set_ylim([0,max(mean_importance+0.1)])
    ax1.set_xlim([0.5,numFeat+0.5])
    #ax1.legend()
    ax1.grid()
    #ax1.set_title("Decision tree f.i.")

    mean_importance = np.mean(models[prettyNames[2]]["relativeImportance"], axis=0)[:numFeat]
    std_importance = np.std(models[prettyNames[2]]["relativeImportance"], axis=0)[:numFeat]
    ax2.bar(np.arange(1,len(mean_importance)+1,1), mean_importance, color="#DDCC77")
    ax2.errorbar(x=np.arange(1,len(mean_importance)+1,1), y=mean_importance,yerr=std_importance, fmt='none', capsize=4, color="#DDCC77")
    ax2.set_ylim([0,max(mean_importance+0.1)])
    ax2.set_xlim([0.5,numFeat+0.5])
    ax2.grid()
    #ax2.set_title("Random forest f.i.")
    #ax2.legend()


    mean_importance = np.mean(models[prettyNames[3]]["relativeImportance"], axis=0)[:numFeat]
    std_importance = np.std(models[prettyNames[3]]["relativeImportance"], axis=0)[:numFeat]

    ax3.bar(np.arange(1,len(mean_importance)+1,1), mean_importance, color="#117733")
    ax3.errorbar(x=np.arange(1,len(mean_importance)+1,1), y=mean_importance,yerr=std_importance, fmt='none', capsize=4, color="#117733")
    ax3.set_ylim([0,max(mean_importance+0.1)])
    ax3.set_xlim([0.5,numFeat+0.5])
    ax3.grid()
    #ax3.legend()
    #ax3.set_xlabel("Principal components")
    #ax3.set_title("Gradient boost f.i.")

    ax4.bar( np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', color="#888888")
    ax4.set_ylim([0,max(pca.explained_variance_ratio_+0.01)])
    ax4.set_xlim([0.5,numFeat+0.5])
    ax4.grid()
    ax4.set_xlabel("Principal components")
    #ax4.legend()
    #ax4.set_title("Explained variance")

    chosenNComponents = np.where(pca.explained_variance_ratio_.cumsum()>0.95)[0][0]

    #ax4.set_title("Explained variance")

    fig.tight_layout()
    tikzplotlib.save(dir_path / Path(approach + ".tex"),
                                axis_height = str(set_size(width, 2, isTex=True)[0]) + "in",
                                axis_width  = str(set_size(width, 2, isTex=True)[0]) + "in")

    plt.show()

    colors = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#888888"]

    map_names={"01-ferrenti-approach": "Ferrenti approach", "02-augmented-ferrenti-approach": "Augmented Ferrenti approach", "03-insightful-approach": "Insightful approach"}
    map_legends = {"LOG ": "Logistic regression coef.", "DT ": "Decision tree f.i.", "RF ":"Random forest f.i.", "GB ": "Gradient boost f.i."}
    print(prettyNames)
    for i, name in enumerate(prettyNames):
        mean_importance = np.mean(models[name]["relativeImportance"], axis=0)[:numFeat]
        std_importance = np.std(models[name]["relativeImportance"], axis=0)[:numFeat]
        fig, ax = plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1.0)[0]))
        if i == 0:

            ax.set_ylim([min(mean_importance-0.4),max(mean_importance+0.2)])
        else:
            ax.set_ylim([0,max(mean_importance+0.05)])

        #ax.set_title(map_legends[name])
        ax.bar(np.arange(1,len(mean_importance)+1,1), mean_importance, color=colors[i])#, label=map_legends[name])
        ax.errorbar(x=np.arange(1,len(mean_importance)+1,1), y=mean_importance,yerr=std_importance, fmt='none', capsize=4, color=colors[i])
        #ax.xaxis.set_major_formatter(plt.NullFormatter())

        ax.set_xlim([0.5,numFeat+0.5])
        ax.grid()
        #ax.legend(loc="upper right")
        #fig.tight_layout()
        tikzplotlib.save(dir_path / Path(approach + name[:-1] + ".tex"),
                                    axis_height = str(set_size(width, 0.5, isTex=True)[0]) + "in",
                                    axis_width  = str(set_size(width, 1.0, isTex=True)[0]) + "in")
        fig.show()

    fig, ax = plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1.0)[0]))

    ax.bar( np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', color="#888888")
    ax.set_ylim([0,max(pca.explained_variance_ratio_+0.01)])
    ax.set_xlim([0.5,numFeat+0.5])
    ax.grid()
    #ax.set_xlabel("Principal components")
    #ax.legend(loc="upper right")
    tikzplotlib.save(dir_path / Path(approach + "PC" + ".tex"),
                                axis_height = str(set_size(width, 0.5, isTex=True)[0]) + "in",
                                axis_width  = str(set_size(width, 1.0, isTex=True)[0]) + "in")
    fig.show()
    #plt.title('Feature Importances')
    #plt.barh(importances, range(len(indices)), color='b', align='center')
    #plt.yticks(range(len(indices)), [features[i] for i in indices])
    #plt.xlabel('Relative Importance')
    #plt.show()


    #for i, model in enumerate(models):
        #ax.plot(abs(PCAcomponents.iloc[i]).nlargest(NFeatures).values,"-", color=cmap(c[i]))

def plot_important_features_restricted_domain(models, names, trainingSet, k, n):
    """
    Only plot features that have been deemed important at least once.
    """

    threshold = (k+n)/2
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



def plot_confusion_metrics(models, names, data,  k, n, prettyNames:[], numPC:int, approach:str):
    """
    Plot false positives and false negatives for a given dataset.
    """

    fig = go.Figure(
            layout = go.Layout (
                autosize=False,
                width=width_plotly*4,
                height=height_plotly,
                title=go.layout.Title(text="False positive (Nruns = {})".format(n*k)),
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

    fig.show()

    fig = go.Figure(
            layout = go.Layout (
                autosize=False,
                width=width_plotly*2,
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

    dir_path = Path(__file__).resolve().parents[2] / "reports" / "figures" / "confusion-metrics"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    fig.write_image(str(dir_path / Path(approach + "-" + str(numPC) +".pdf")))

    fig.show()


def confusion_matrixQT(models, y, prettyNames:str, numPC:int, approach:str):
    #print(mat)
    for i, model in enumerate(models):
        fig, ax = plt.subplots(1,1, figsize=set_size(width, 0.75))

        sns.heatmap(model["confusionMatrix"], square=True, annot=True, fmt='d', cbar=False,
                xticklabels=[0,1],
                yticklabels=[0,1])
        ax.set_xlabel('true label')
        ax.set_ylabel('predicted label');
        ax.set_title("Confusion matrix {}".format(prettyNames[i]))
        fig.tight_layout()

        dir_path = Path(__file__).resolve().parents[2] / \
                                    "reports" / "figures"  / "confusion-matrix"

        #save_matplot_fig(fig, dir_path=dir_path, filename=Path(approach + "-" + str(numPC) + "-" + prettyNames[i] +".pgf"))

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
    _nelements = {1: "Unary", 2: "Binary", 3: "Ternary", 4: "Quarternary", 5: "Quinary", 6: "Senary", 7: "Septenary", 8: "Octary"}
    fig = px.histogram(entries[entries["MP|band_gap"]<8], x="MP|band_gap", color="MP|nelements", nbins=20,
                       #title='Band gaps and material phases in dataset',
                       labels={"MP|band_gap": "Materials Project band gap [eV]", 'MP|nelements':'Compound type'},
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
                      height=height_plotly*0.8,
                     )
    fig.write_image(str(Path(__file__).resolve().parents[2] / \
                                "reports" / "figures"  / "buildingFeatures"\
                                / "histogram_bg_nelements.pdf"))
    fig.show()
def plot_histogram_bg_proba(entries, x = "MP BG", color = "RF "):
    #_nelements = {1: "Unary", 2: "Binary", 3: "Ternary", 4: "Quarternary", 5: "Quinary", 6: "Senary", 7: "Septenary", 8: "Octary"}
    fig = px.histogram(entries[entries["MP BG"]<8], x="MP BG", color="RF ", nbins=20)
                       #title='Band gaps and material phases in dataset',
                       #labels={"MP|band_gap": "Materials Project band gap [eV]", 'MP|nelements':'Compound type'},
                       #category_orders={"MP|nelements": list(_nelements.values())})

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
                      height=height_plotly*0.8,
                     )
    #fig.write_image(str(Path(__file__).resolve().parents[2] / \
    #                            "reports" / "figures"  / "buildingFeatures"\
    #                            / "histogram_bg_nelements.pdf"))
    fig.show()

def plot_histogram_oxid_nelements(entries):
    _oxideType = {"None": 0, "Oxide":1, "Peroxide":2, "Hydroxide":3, "Superoxide":4, "Ozonide":5}
    _nelements = {1: "Unary", 2: "Binary", 3: "Ternary", 4: "Quarternary", 5: "Quinary", 6: "Senary", 7: "Septenary", 8: "Octary"}

    fig = px.histogram(entries, x="MP|nelements", color="MP|oxide_type", nbins=7,
                   #title='Oxid types and material phases in dataset',
                   labels={'MP|nelements':'Compound type', "MP|oxide_type": "Oxide type"},
                   category_orders={"MP|nelements": list(_nelements.values()),
                                    "MP|oxide_type":list(_oxideType.keys())})
    fig.update_layout(
                    {"plot_bgcolor": "rgba(0, 0, 0, 0)",
                       "paper_bgcolor": "rgba(0, 0, 0, 0)",
                      },
                      font=dict(
                        family="Palatino, bold",
                        color="Black",
                        size=12),
                      autosize=False,
                      width=width_plotly,
                      height=height_plotly*0.75,
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

def evaluatePrecisionRecallMetrics(classifier,
                       X: pd.DataFrame,
                       y,
                       k: int,
                       n: int,
                       cv,
                       title: str,
                       numPC: int,
                       approach: str,
                       featureImportance: Optional[bool] = False,
                       resamplingMethod: Optional[str] = "None"):


    modelResults = {
        'trainAccuracy':   np.zeros(n*k),
        'testAccuracy':    np.zeros(n*k),
        'f1_score':        np.zeros(n*k),
        'std':             np.zeros(n*k),
        'importantKeys':   np.zeros(classifier.named_steps["pca"].n_components),
        'numPredPero':     np.zeros(n*k),
        'confusionMatrix': np.zeros((len(y), len(y))),
        'falsePositives':  np.zeros(len(y)),
        'falseNegatives':  np.zeros(len(y)),
        'relativeImportance': np.zeros((n*k,classifier.named_steps["pca"].n_components))
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
        #print(i)
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

        ######################################
        ############# SHAP metrics ###########
        ######################################

        #explainer = shap.Explainer(classifier["model"], masker=shap.maskers.Impute(data=X_train))
        #explainer = shap.KernelExplainer(classifier.named_steps['model'].predict_proba, )
        #shap_values = explainer.shap_values(X_test)

        #list_shap_values.append(shap_values)
        #list_test_sets.append(test_index)

        #claim the scores
        modelResults['trainAccuracy'][i] = classifier.score(X_train, y_train)
        modelResults['testAccuracy'][i]  = classifier.score(X_test, y_test)
        modelResults['f1_score'][i]      = f1_score(y_test, y_pred)
        modelResults['std'][i]           = np.std(modelResults['testAccuracy'][0:i+1])
        modelResults['numPredPero'][i]   = np.sum(y_pred_full)
        modelResults['confusionMatrix']  = confusion_matrix(y, y_pred_full)
        modelResults['falsePositives'][falsePositives] += 1
        modelResults['falseNegatives'][falseNegatives] += 1

        if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
            modelResults['relativeImportance'][i] = classifier.named_steps["model"].feature_importances_
        elif type(classifier["model"]) == type(LogisticRegression()):
            modelResults['relativeImportance'][i] = classifier.named_steps['model'].coef_

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

    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "roc-auc"
    #save_matplot_fig(fig1, dir_path=dir_path, filename = Path(approach + "-" + str(numPC) + "-" + title +".pgf"))

    dir_path = Path(__file__).resolve().parents[2] / \
                                "reports" / "figures"  / "recall-metrics"

    #save_matplot_fig(fig2, dir_path=dir_path, filename = Path(approach + "-" + str(numPC) + "-" + title +".pgf"))


    plt.show()

    """
    #combining results from all iterations
    test_set = list_test_sets[0]
    shap_values = np.array(list_shap_values[0])
    for i in range(0,len(list_test_sets)):
        test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
        shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
    #bringing back variable names
    X_test = pd.DataFrame(X[test_set],columns=columns)
    #creating explanation plot for the whole experiment
    shap.summary_plot(shap_values[1], X_test)
    """
    print ("Mean accuracy:{:0.5f}".format(np.mean(modelResults['testAccuracy'])))
    print ("Standard deviation:{:0.5f}".format(modelResults['std'][-1]))
    print ("f1-score:{:0.5f}".format(modelResults['f1_score'][-1]))

    return modelResults
def principalComponentsVSvariance(X: pd.DataFrame, approach:str):

    scaledTrainingData = StandardScaler().fit_transform(X) # normalizing the features
    pca = PCA(0.955).fit(scaledTrainingData)
    #print(pca.explained_variance_ratio_)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(set_size(width, 0.5)[0],set_size(width, 0.75)[0]))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax1.bar( np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    ax0.plot(np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum())

    chosenNComponents = np.where(pca.explained_variance_ratio_.cumsum()>0.95)[0][0]
    print("95% accumulated variance needs {} principal components".format(chosenNComponents))
    ax0.set_ylabel('Accumulated var ratio')
    ax1.set_ylabel('Var ratio')
    ax1.axvline(chosenNComponents,
                linestyle=':', label='$95\%$ accumulated variance')
    ax0.axvline(chosenNComponents,
                linestyle=':')

    #ax1.legend(loc="upper right")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    ax1.set_xlabel('Principal components')

    ax0.xaxis.set_major_formatter(plt.NullFormatter())
    #ax1.set_xlim([0.5,numPC+0.5])
    ax0.set_title("Explained variance for " + str(approach), wrap=True)
    ax1.set_ylim([0,max(pca.explained_variance_ratio_)])

    #ax1.set_xticks(range(1,numPC+1))
    fig.tight_layout()

    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "pca"
    #save_matplot_fig(fig, dir_path=dir_path, filename=Path(approach +".pgf"))

    plt.show()
    ########################

    fig, ax0 = plt.subplots(figsize=(set_size(width, 1)[0],set_size(width, 1)[0]))

    ax0.bar( np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')

    chosenNComponents = np.where(pca.explained_variance_ratio_.cumsum()>0.95)[0][0]
    ax0.set_ylabel('Accumulated var ratio')
    ax0.set_ylabel('Explained variance')
    ax0.axvline(chosenNComponents,
                linestyle='--', label='$95\%={}$'.format(chosenNComponents))

    ax0.legend(loc="best")
    #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #      ncol=3, fancybox=True, shadow=True)
    ax0.set_xlabel('Principal components')

    #ax0.xaxis.set_major_formatter(plt.NullFormatter())
    #ax1.set_xlim([0.5,numPC+0.5])
    ax0.set_title(str(approach), wrap=True)
    print(pca.explained_variance_ratio_[0])
    ax0.set_ylim([0,pca.explained_variance_ratio_[0]+0.01])

    #ax1.set_xticks(range(1,numPC+1))
    fig.tight_layout()

    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "pca"


    tikzplotlib.save(dir_path / Path(approach + ".tex"),
                        axis_height = str(set_size(width, 0.8, isTex=True)[0]) + "in",
                        axis_width  = str(set_size(width, 0.8, isTex=True)[0]) + "in")

    scaledGeneratedData = StandardScaler().fit_transform(X) # normalizing the features
    scaledGeneratedData = pd.DataFrame(scaledGeneratedData, columns=X.columns)
    pca = PCA(n_components=chosenNComponents)
    PCAGeneratedData = pd.DataFrame(pca.fit_transform(scaledGeneratedData))
    PCAcomponents = pd.DataFrame(pca.components_,columns=scaledGeneratedData.columns,index = [ "PCA-"+str(i) for i in range(0,chosenNComponents)])

    #plot_eigenvectors_principal_components(PCAcomponents, chosenNComponents=10, NFeatures=15)
    top_eigenvector_vs_features(PCAcomponents, whichComponent = 0, NFeatures=40)


def principalComponentsVSscores(X: pd.DataFrame, ModelsBestParams: pd.Series, prettyNames:str, numPC:int, approach:str):

    scaledTrainingData = StandardScaler().fit_transform(X) # normalizing the features
    pca = PCA().fit(scaledTrainingData)
    #print(pca.explained_variance_ratio_)
    for i, algorithm in enumerate(ModelsBestParams):

        fig, ax0 = plt.subplots(nrows=1, figsize=(set_size(width, 1)[0],set_size(width, 1)[0]))
        """
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        ax1.bar( np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
        ax1.step(np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), where='mid')

        ax1.set_ylabel('PCA var')
        ax1.axvline(algorithm.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='Optimal')

        ax1.legend(prop=dict(size=12))

        """

        # For each number of components, find the best classifier results
        results = pd.DataFrame(algorithm.cv_results_)
        components_col = 'param_pca__n_components'
        best_clfs = results.groupby(components_col).apply(
            lambda g: g.nlargest(1, 'mean_test_f1'))
        print(best_clfs)
        if (numPC > 100):

            best_clfs.plot(x=components_col, y='mean_train_accuracy',
                           label="Train", ax=ax0)

            best_clfs.plot(x=components_col, y='mean_test_accuracy',
                           label="Test", ax=ax0)

            best_clfs.plot(x=components_col, y='mean_test_precision',
                           label="Precision", ax=ax0)

            best_clfs.plot(x=components_col, y='mean_test_recall',
                           label="Recall", ax=ax0)

            best_clfs.plot(x=components_col, y='mean_test_f1',
                           label="f1", ax=ax0)

        else:
            best_clfs.plot(x=components_col, y='mean_train_accuracy', yerr='std_train_accuracy',
                           label="Train", ax=ax0, capsize=4)

            best_clfs.plot(x=components_col, y='mean_test_accuracy', yerr='std_test_accuracy',
                           label="Test", ax=ax0, capsize=4)

            best_clfs.plot(x=components_col, y='mean_test_precision', yerr='std_test_precision',
                           label="Precision", ax=ax0, capsize=4)

            best_clfs.plot(x=components_col, y='mean_test_recall', yerr='std_test_recall',
                           label="Recall", ax=ax0, capsize=4)

            best_clfs.plot(x=components_col, y='mean_test_f1', yerr='std_test_f1',
                           label="f1", ax=ax0, capsize=4)

            ax0.xaxis.set_major_formatter(plt.NullFormatter())

            ax0.set_xticks(range(1,numPC+1))

        ax0.axvline(algorithm.best_estimator_.named_steps['pca'].n_components, color="tab:purple",
                        linestyle='dashdot', label='Optimal')
        #display(pd.DataFrame(best_clfs[["mean_test_accuracy", "std_test_accuracy", "mean_test_f1", "std_test_f1"]]))
        #jepp = 93
        display(pd.DataFrame(best_clfs[best_clfs["param_pca__n_components"]==algorithm.best_estimator_.named_steps['pca'].n_components])[["mean_test_accuracy", "mean_test_precision", "mean_test_recall", "mean_test_f1"]])
        display(pd.DataFrame(best_clfs[best_clfs["param_pca__n_components"]==algorithm.best_estimator_.named_steps['pca'].n_components])[["std_test_accuracy", "std_test_precision", "std_test_recall", "std_test_f1"]])

        nameMapping = {"LOG ": "Logistic regression", "DT ": "Decision tree", "RF ": "Random forest", "GB ": "Gradient boost"}
        ax0.set_ylabel('Score')
        ax0.set_xlabel('Principal components')
        ax0.set_title(nameMapping[prettyNames[i]])

        ax0.set_xlim([0.5,numPC+0.5])
        #ax0.legend().set_visible(False)
        """
        if i==2:
            # Shrink current axis's height by 10% on the bottom
            box = ax0.get_position()
            ax0.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
                      fancybox=True, shadow=True, ncol=3)
        else:
            ax0.get_legend().remove()
        """
        #ax0.legend()

        fig.tight_layout()
        ax0.get_legend().remove()
        dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "pca-scores"
        #save_matplot_fig(fig, dir_path=dir_path, filename=Path(approach + "-" + str(numPC) + "-" + prettyNames[i][:-1] +".pgf"))
        #tikzplotlib.clean_figure()
        tikzplotlib.save(dir_path / str(approach + "-" + str(numPC) + "-" + prettyNames[i][:-1] +".tex"),
                        axis_height = str(set_size(width, 0.8, isTex=True)[0]) + "in",
                        axis_width  = str(set_size(width, 0.8, isTex=True)[0]) + "in")

        plt.show()

def gridsearchVSscores(X: pd.DataFrame, ModelsBestParams: pd.Series, prettyNames:str, approach:str):

    for i, algorithm in enumerate(ModelsBestParams):

        fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(set_size(width, 0.5)[0],set_size(width, 0.5)[0]))

        #print(algorithm.estimator.named_steps["model"])
        if type(algorithm.estimator.named_steps["model"]) == type(LogisticRegression()):
            components_col = 'param_model__C'
            xlabel = "Reg. strength"
            xscale = "log"
            best_param = algorithm.best_estimator_.named_steps['model'].C
        else:
            components_col = 'param_model__max_depth'
            xlabel = "Max depth"
            xscale = "linear"
            best_param = algorithm.best_estimator_.named_steps['model'].max_depth


        # For each number of components, find the best classifier results
        results = pd.DataFrame(algorithm.cv_results_)
        best_clfs = results.groupby(components_col).apply(
            lambda g: g.nlargest(1, 'mean_test_f1'))

        best_clfs.plot(x=components_col, y='mean_train_accuracy', yerr='std_train_accuracy',
                       label="Train", ax=ax0, capsize=4)

        best_clfs.plot(x=components_col, y='mean_test_accuracy', yerr='std_test_accuracy',
                       label="Test", ax=ax0, capsize=4)

        best_clfs.plot(x=components_col, y='mean_test_precision', yerr='std_test_precision',
                       label="Precision", ax=ax0, capsize=4)

        best_clfs.plot(x=components_col, y='mean_test_recall', yerr='std_test_recall',
                       label="Recall", ax=ax0, capsize=4)

        best_clfs.plot(x=components_col, y='mean_test_f1', yerr='std_test_f1',
                       label="f1", ax=ax0, capsize=4)


        ax0.axvline(best_param, linestyle='dashdot', label='Optimal', color="tab:purple")

        #ax1.legend(prop=dict(size=12))
        nameMapping = {"LOG ": "Logistic regression", "DT ": "Decision tree", "RF ": "Random forest", "GB ": "Gradient boost"}

        ax0.set_ylabel('Score')
        ax0.set_xlabel(xlabel)
        ax0.set_title(nameMapping[prettyNames[i]])
        ax0.set_xscale(xscale)
        #ax0.set_xlim([0.5,numPC+0.5])
        #ax1.set_xlim([0.5,numPC+0.5])

        #ax1.set_ylim([0,pca.explained_variance_ratio_.cumsum()[numPC+2]])
        #ax0.xaxis.set_major_formatter(plt.NullFormatter())

        #ax0.set_xticks(range(1,numPC+1))
        #ax1.set_xticks(range(1,numPC+1))
        # Put a legend below current axis
        #box = ax0.get_position()
        #ax0.set_position([box.x0, box.y0 + box.height * 0.1,
        #         box.width, box.height * 0.9])

        #ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
        ax0.get_legend().remove()
        fig.tight_layout()

        dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "grid-scores"

        Path(dir_path / "PR-RE").mkdir(parents=True, exist_ok=True)

        #fig.savefig(dir_path / Path(prettyNames[i] + approach + ".pgf") , format="pgf", bbox_inches="tight")

        tikzplotlib.save(dir_path / str(approach + "-" + prettyNames[i][:-1] +".tex"),
                        axis_height = str(set_size(width, 0.8, isTex=True)[0]) + "in",
                        axis_width  = str(set_size(width, 0.8, isTex=True)[0]) + "in")

        plt.show()

def make_parallel_coordinate_matplot(generatedData, insertApproach, title, applyLegend=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from matplotlib.colors import ListedColormap
    targetNames = ["Bad candidates", "Good candidates", "Unlabelled"]
    #iris = datasets.load_iris()
    interestingFeatures = {
    "MP|total_magnetization":"Mag",
    "MP|Polar SG": "Polar SG",
    "IonProperty|max ionic char":"Ionic char",
    #"AverageBondLength|mean Average bond length":"Avg bond length",
    "ElementProperty|MagpieData range CovalentRadius": "Cov range [pm]",# [\si{\pm}]",
    #"candidate":"Label",
    #"MP|oxide_type":"Oxid type",
    "MP|nelements": "Num elements",
    "MP_Eg":"Eg [eV]"# [\si{\eV}]
    }
    generatedData = generatedData.astype({"MP|Polar SG": int})
    generatedData = generatedData[generatedData["candidate"] != -1]
    df = generatedData.groupby('candidate').apply(lambda s: s.sample(min(len(s), 250)))
    #df = df[df["candidate"]!=-1]
    print(df[df["candidate"]==1].shape[0])
    ynames = interestingFeatures.values()
    ys = df[interestingFeatures.keys()].to_numpy()
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)

    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05

    #ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, :] = (ys[:, :] - ymins[:]) / dys[:] * dys[0] + ymins[0]

    if (applyLegend):
        fig, host = plt.subplots(figsize=(set_size(width, 1)[0],set_size(width, 0.7)[1]))
    else:
        fig, host = plt.subplots(figsize=(set_size(width, 1)[0],set_size(width, 0.65)[1]))

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=10)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title(title)
    colors = ['tomato', 'limegreen', 'grey']
    legend_handles = [None for _ in targetNames]
    for j in tqdm(range(ys.shape[0])):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [mpl.path.Path.MOVETO] + [mpl.path.Path.CURVE4 for _ in range(len(verts) - 1)]
        path = mpl.path.Path(verts, codes)
        #print(colors[int(generatedData["candidate"].values[j])])
        patch = patches.PathPatch(path, facecolor='none', lw=0.5, alpha=0.2, edgecolor=colors[int(df["candidate"].values[j])])
        legend_handles[int(df["candidate"].values[j])] = patch
        host.add_patch(patch)
    import matplotlib.lines as mlines
    if (applyLegend):

        legend_elements = [mlines.Line2D([0], [0], color="limegreen", label="Suitable candidates"),
                           mlines.Line2D([0], [0], color="tomato", label="Unsuitable candidates")]
        host.legend(handles=legend_elements,
                loc='lower center', bbox_to_anchor=(0.5, -0.18),
                ncol=len(targetNames), fancybox=False, shadow=False)


    plt.tight_layout()


    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "parallel_coordinates"

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    #fig.savefig(dir_path / Path(insertApproach + ".pgf") , format="pgf", bbox_inches="tight")


    plt.show()

def make_parallel_coordinate_matplot_summary(generatedData, insertApproach, title, applyLegend=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from matplotlib.colors import ListedColormap
    targetNames = ["Bad candidates", "Good candidates", "Unlabelled"]
    #iris = datasets.load_iris()
    interestingFeatures = {
    "LOG Prob":"LOG Prob",
    "DT Prob":"DT Prob",
    "RF Prob":"RF Prob",
    "GB Prob": "GB Prob",

    "MP BG":"Eg [eV]"# [\si{\eV}]
    }

    #print(generatedData.shape)
    generatedData = generatedData[generatedData["MP BG"] < 5]
    #df = generatedData.groupby('LOG ').apply(lambda s: s.sample(min(len(s), 500)))
    df = generatedData.sample(500)
    #print(df)

    ynames = interestingFeatures.values()
    ys = df[interestingFeatures.keys()].to_numpy()
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)

    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05

    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, :] = (ys[:, :] - ymins[:]) / dys[:] * dys[0] + ymins[0]

    if (applyLegend):
        fig, host = plt.subplots(figsize=(set_size(width, 1)[0],set_size(width, 0.7)[1]))
    else:
        fig, host = plt.subplots(figsize=(set_size(width, 1)[0],set_size(width, 0.65)[1]))

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=10)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title(title)

    from matplotlib.pyplot import cm
    colors=cm.jet(np.linspace(1,0,101))
    import math
    def round_down(x, a):
        return math.floor(x / a) * a

    legend_handles = [None for _ in targetNames]
    normalized_colors = (df["MP BG"].values-min(df["MP BG"].values))/(max(df["MP BG"].values)-min(df["MP BG"].values))


    #normalized_colors = df["MP BG"].values
    for j in tqdm(range(ys.shape[0])):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [mpl.path.Path.MOVETO] + [mpl.path.Path.CURVE4 for _ in range(len(verts) - 1)]
        path = mpl.path.Path(verts, codes)
        if df["MP BG"].values[j] < 0.5:
            patch = patches.PathPatch(path, facecolor='none', lw=0.5, alpha=0.8, edgecolor=colors[int(round_down(normalized_colors[j],0.01)*100)])
        else:
            patch = patches.PathPatch(path, facecolor='none', lw=0.5, alpha=0.2, edgecolor=colors[int(round_down(normalized_colors[j],0.01)*100)])

        #legend_handles[int(df["MP BG"].values[j])] = patch
        host.add_patch(patch)
    import matplotlib.lines as mlines
    if (applyLegend):

        legend_elements = [mlines.Line2D([0], [0], color="limegreen", label="Good candidates"),
                           mlines.Line2D([0], [0], color="tomato", label="Bad candidates")]
        host.legend(handles=legend_elements,
                loc='lower center', bbox_to_anchor=(0.5, -0.18),
                ncol=len(targetNames), fancybox=False, shadow=False)

    #fig.colorbar(cm.jet, colors)
    plt.tight_layout()


    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "parallel_coordinates"

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(dir_path / Path("summary-" + insertApproach + ".pgf"))
    #fig.savefig(dir_path / Path("summary-" + insertApproach + ".pgf") , format="pgf", bbox_inches="tight")


    plt.show()
def histogram_of_proba(Summary, insertApproach):
    import numpy as np                   # v 1.19.2
    import matplotlib.pyplot as plt      # v 3.3.2
    from matplotlib.lines import Line2D

    colors = ["#88CCEE", "#CC6677", "#DDCC77", "#117733"]

    # Create figure with 'step' type of histogram to improve plot readability
    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist([Summary["LOG Prob"], Summary["DT Prob"], Summary["RF Prob"], Summary["GB Prob"]], bins=15, histtype='step', linewidth=3,
            alpha=1.0, label=['Logistic regression','Decision tree', "Random forest", "Gradient boost"], color=colors)

    # Edit legend to get lines as legend keys instead of the default polygons
    # and sort the legend entries in alphanumeric order
    handles, labels = ax.get_legend_handles_labels()
    leg_entries = {}
    for h, label in zip(handles, labels):
        leg_entries[label] = Line2D([0], [0], color=h.get_facecolor()[:-1],
                                    alpha=h.get_alpha(), lw=h.get_linewidth())

    #lines = leg_entries.items()
    ax.legend(loc="upper center")

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotations
    ax.set_ylabel('Count')
    ax.set_xlabel('Probability of prediction')
    ax.set_xlim([0,1.05])
    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "histogram"

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    tikzplotlib.save(dir_path / Path("summary-"+ insertApproach + ".tex"),
                                axis_height = str(set_size(width, 0.9, isTex=True)[0]) + "in",
                                axis_width  = str(set_size(width, 0.9, isTex=False)[0])+ "in")

    plt.show()


def plot_2d_pca(trainingSet, trainingTarget, insertApproach, title, legend=False):

    X = trainingSet.drop(columns=["material_id", "full_formula"])

    scaler = StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)

    import pickle
    # We use here a pre-trained PCA model on the whole data for the purpose of visualizing and comparison of the different approaches.
    pca = pd.read_pickle(Path(__file__).resolve().parents[2] / \
                                "models" / "trained-models"  / "PCA-total" / "PCA-total.pkl")
    #pca = PCA()
    x_new = pca.fit_transform(X)

    def myplot(score,coeff, labels=None, y=None, showVec=None):

        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]

        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        
        ax.scatter(xs[y==4] * scalex,ys[y==4] * scaley, s=20, c = "orange", marker='x', label="Si")
        ax.scatter(xs[y==3] * scalex,ys[y==3] * scaley, s=20, c = "black", marker='+', label="SiC")
        ax.scatter(xs[y==2] * scalex,ys[y==2] * scaley, s=20, c = "blue", marker='*', label="Diamond")
        ax.scatter(xs[y==1] * scalex,ys[y==1] * scaley, s=20, c = "limegreen", marker='s', label="Other suitables")
        ax.scatter(xs[y==0] * scalex,ys[y==0] * scaley, s=20, c = "tomato", marker="^", label = "Unsuitable")

        if legend:
            ax.legend()
        if showVec:
            for i in tqdm(range(2)):
                ax.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
                if labels is None:
                    ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
                else:
                    ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(set_size(width, 0.7)[0],set_size(width, 0.7)[0]))
    plt.grid()


    ax.set_xlabel("The first principal component")
    ax.set_ylabel("The second principal component")
    ax.set_title(title)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.55, 0.7)
    #Call the function. Use only the 2 PCs.
    myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]), y=trainingTarget.to_numpy())

    dir_path = Path(__file__).resolve().parents[2] / \
                            "reports" / "figures"  / "pca-2d-plots"

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_path / Path(insertApproach + ".pdf") , format="pdf", bbox_inches="tight")
    #tikzplotlib.save(dir_path / Path(insertApproach + ".tex"))

    tikzplotlib.save(dir_path / Path(insertApproach + ".tex"),
                            axis_height = str(set_size(width, 0.45, isTex=True)[0]) + "in",
                            axis_width  = str(set_size(width, 0.45, isTex=True)[0]) + "in")

    plt.show()

def visualize_heatmap_of_combinations(Summary):
    abbreviations = ["LOG ", "DT ", "RF ", "GB "]

    dictionary = {}
    for i in range(len(abbreviations)):

        lists = []
        for j in range(len(abbreviations)):
            #if i!=j:
            lists.append( Summary[(Summary[abbreviations[i]] == 1) & (Summary[abbreviations[j]] == 1)].shape[0] / Summary[(Summary[abbreviations[i]] == 1)].shape[0])

        dictionary[abbreviations[i]] = lists
    df = pd.DataFrame(dictionary, index= abbreviations, columns= abbreviations)

    import seaborn as sns


    sns.heatmap(df, annot=True)
    plt.show()

    permutations = [["LOG ", "LOG ", "LOG ", "LOG "], ["LOG ", "LOG ", "LOG ", "DT "], ["LOG ", "LOG ", "RF ", "DT "], ["LOG ", "RF ", "GB ", "DT "],
                    ["DT ", "GB ", "DT ", "DT "], ["DT ",  "RF ", "DT ", "DT "], ["RF ", "RF ", "GB ", "GB "],
                    ["RF ", "RF ", "RF ", "RF "], ["GB ", "GB ", "GB ", "GB "],
                    ["GB ", "LOG ", "LOG ", "LOG "], ["RF ", "LOG ", "LOG ", "LOG "]]


    #permutations = [["LOG ","LOG ","LOG ","LOG "],["LOG ","LOG ","LOG ","DT "],["LOG ","LOG ","LOG ","RF "],["LOG ","LOG ","LOG ","GB "],["LOG ","LOG ","DT ","DT "],["LOG ","LOG ","DT ","RF "],["LOG ","LOG ","DT ","GB "],["LOG ","LOG ","RF ","RF "],["LOG ","LOG ","RF ","GB "],["LOG ","LOG ","GB ","GB "],["LOG ","DT ","DT ","DT "],["LOG ","DT ","DT ","RF "],["LOG ","DT ","DT ","GB "],["LOG ","DT ","RF ","RF "],["LOG ","DT ","RF ","GB "],["LOG ","DT ","GB ","GB "],["LOG ","RF ","RF ","RF "],["LOG ","RF ","RF ","GB "],["LOG ","RF ","GB ","GB "],["LOG ","GB ","GB ","GB "],["DT ","DT ","DT ","DT "],["DT ","DT ","DT ","RF "],["DT ","DT ","DT ","GB "],["DT ","DT ","RF ","RF "],["DT ","DT ","RF ","GB "],["DT ","DT ","GB ","GB "],["DT ","RF ","RF ","RF "],["DT ","RF ","RF ","GB "],["DT ","RF ","GB ","GB "],["DT ","GB ","GB ","GB "],["RF ","RF ","RF ","RF "],["RF ","RF ","RF ","GB "],["RF ","RF ","GB ","GB "],["RF ","GB ","GB ","GB "],["GB ","GB ","GB ","GB "]]
    new_abbreviations = []
    dictionary = {}
    lists = []
    for i in range(len(permutations)):
        #print(i)
        #tmp =
        record=0
        for j in range(len(permutations[0])):
            if Summary[Summary[permutations[i][j]] == 1].shape[0] > record:
                record = Summary[Summary[permutations[i][j]] == 1].shape[0]

        tmp = Summary[(Summary[permutations[i][0]] == 1) &
                      (Summary[permutations[i][1]] == 1) &
                      (Summary[permutations[i][2]] == 1) &
                      (Summary[permutations[i][3]] == 1) ].shape[0] / record

        lists.append(tmp)

        new_abbreviations.append(permutations[i][0]+permutations[i][1]+permutations[i][2]+permutations[i][3])
        dictionary[permutations[i][0]+permutations[i][1]+permutations[i][2]+permutations[i][3]] = tmp

    print(lists)
    df = pd.DataFrame(dictionary, index=new_abbreviations, columns=new_abbreviations)
    display(df)
    import seaborn as sns

    # Getting the Upper Triangle of the co-relation matrix
    matrix = np.triu(df)

    # using the upper triangle matrix as mask
    sns.heatmap(df, annot=False)#, mask=matrix)
    plt.show()

def plot_2D3Dcontours(trainingSet, y, Summary, prettyNames, insertApproach,numberOfPrincipalComponents):

    formulas_in_trainingset = trainingSet["full_formula"]
    #print(trainingSet[["material_id", "full_formula"]])

    #ZnO_index = trainingSet[trainingSet["full_formula"]=="Zn2O2"].index
    #ZnO_index = trainingSet[trainingSet["full_formula"]=="Zn2O2"].index


    #print(ZnO_index)
    X = trainingSet.drop(["material_id", "full_formula"], axis=1)

    #scaler = StandardScaler()
    #X = StandardScaler().fit_transform(X)
    #X = PCA(0.95).fit_transform(X)
    #X = X[:, :3]
    #print(formulas_in_trainingset)
    import joblib

    classifier = joblib.load(Path(__file__).resolve().parents[2] / "models" / insertApproach / "trained-models" / Path("PCA-" + str(numberOfPrincipalComponents) + "-" + prettyNames[1] + ".pkl"))



    #y = X.pop("candidate", axis=1)
    X = classifier["scale"].transform(X.drop("candidate", axis=1))
    X = classifier["pca"].transform(X)

    testSet = pd.read_pickle(Path(__file__).resolve().parents[2] / "data" /insertApproach / "processed" / "testSet.pkl")
    testSet = classifier["scale"].transform(testSet.drop(["candidate", "material_id", "full_formula", "pretty_formula"], axis=1))
    testSet = classifier["pca"].transform(testSet)

    clf = classifier["model"]
    from dtreeviz.trees import dtreeviz
    #fig = plt.figure(figsize=(40,40))
    viz = dtreeviz(clf, X, y,
                target_name="candidate",
                feature_names=["PC0", "PCA1", "PC2"])


    viz
    from IPython.display import display
    display(viz)
    for i in range(len(prettyNames)):
        classifier = joblib.load(Path(__file__).resolve().parents[2] / "models" / insertApproach / "trained-models" / Path("PCA-" + str(numberOfPrincipalComponents) + "-" + prettyNames[i] + ".pkl"))
        clf = classifier["model"]
        print(clf)
        points = 10
        feature_1 = np.linspace(X[:, 0].min(), X[:, 0].max(), points)
        feature_2 = np.linspace(X[:, 1].min(), X[:, 1].max(), points)
        feature_3 = np.linspace(X[:, 2].min(), X[:, 2].max(), points)

        XX1, XX2, XX3 = np.array(np.meshgrid(feature_1, feature_2, feature_3))
        df = pd.DataFrame(np.array([XX1.ravel(), XX2.ravel(), XX3.ravel()]).T, columns=["PC1", "PC2", "PC3"])
        #df.columns = ["PC1", "PC2", "PC3"]

        Z_grid = np.array(clf.predict_proba(df)[:,1]).reshape(points,points,points)

        fig = go.Figure(data=go.Volume(
            x=XX1.ravel(),
            y=XX2.ravel(),
            z=XX3.ravel(),
            value=Z_grid.flatten(),
            isomin=0.05,
            isomax=1.0,
            opacity=0.5, # needs to be small to see through all surfaces
            surface_count=20,
            colorscale=[(0,"tomato"), (1,"limegreen")],#
            ),
            layout = Layout(
            #title=go.layout.Title(text="Probability for qubit material host"),
            showlegend=True,
            scene=layout.Scene(
                xaxis=dict(title='PC1'),
                yaxis=dict(title='PC2'),
                zaxis=dict(title='PC3')
            )
        ))
        fig.update_layout(
            font_family="Palatino",
            font_color="black",
            font_size=12
        )


        interval = [0.00, 1.0]


        ############## Test set with probability ##############
        fig.add_trace(
            go.Scatter3d(x= testSet[:,0][Summary["DT Prob"].between(interval[0], interval[1], inclusive=False)],
                          y=testSet[:,1][Summary["DT Prob"].between(interval[0], interval[1], inclusive=False)],
                          z=testSet[:,2][Summary["DT Prob"].between(interval[0], interval[1], inclusive=False)],
                          mode='markers',
                          marker=dict(
                          size=4,
                            color=Summary["DT "][Summary["DT Prob"].between(interval[0], interval[1], inclusive=False)],                # set color to an array/list of desired values
                            colorscale=[(0,"tomato"), (1,"limegreen")],   # choose a colorscale
                            opacity=0.8
                          ),
                          showlegend=False,
                          hovertext=Summary["full_formula"][Summary["DT Prob"].between(interval[0],interval[1], inclusive=False)]),
        )

        """
        #mpids =     ["mp-1070", "mp-1639", "mp-2691", "mp-20305", "mp-22895", "mp-569346", "mp-629458", "mp-1008523", "mp-1078597"]
        mpids = ["mp-3829", "mp-3839", "mp-4008", "mp-4524", "mp-4899", "mp-5213", "mp-5238", "mp-20554"]
        for i in range(len(mpids)):
            fig.add_trace(
                go.Scatter3d(x= testSet[:,0][Summary["material_id"]==mpids[i]],
                              y=testSet[:,1][Summary["material_id"]==mpids[i]],
                              z=testSet[:,2][Summary["material_id"]==mpids[i]],
                              mode='markers',
                              marker=dict(
                              size=4,
                                color="turquoise",                # set color to an array/list of desired values
                                colorscale='Plasma',   # choose a colorscale
                                opacity=0.8
                              ),
                              showlegend=False,
                              hovertext=Summary["full_formula"][Summary["material_id"]==mpids[i]]),
            )
        #fig.show()
        """
        """
        fig.add_trace(
            go.Scatter3d(x= X[:,0],#[y==1],
                          y=X[:,1],#[y==1],
                          z=X[:,2],#[y==1],
                          mode='markers',
                          marker=dict(
                            size=4,
                            color=y,                # set color to an array/list of desired values
                            colorscale=[(0,"tomato"), (1,"limegreen")],   # choose a colorscale
                            opacity=0.8
                          ),
                          showlegend=False,
                          hovertext=formulas_in_trainingset,)
        )


        #test
        Summary = Summary.drop(Summary.loc[Summary['full_formula']=="Ho2V2O8"].index)

        mpids = Summary[(Summary["DT Prob"] > 0.75) &
                            (Summary["GB Prob"] > 0.75) &
                            (Summary["LOG Prob"] > 0.75) &
                            (Summary["RF Prob"] > 0.75)]["material_id"].to_list()

        #mpids =     ["mp-4524", "mp-629458", "mp-1008523", "mp-1009792", "mp-1198022"]
        #for i in range(len(mpids)):
        #    print(mpids[i])

        #print(testSet[:,0][Summary["material_id"].isin(mpids)])
        fig.add_trace(
                go.Scatter3d(x= testSet[:,0][Summary[Summary["material_id"].isin(mpids)].index],
                              y=testSet[:,1][Summary[Summary["material_id"].isin(mpids)].index],
                              z=testSet[:,2][Summary[Summary["material_id"].isin(mpids)].index],
                              mode='markers',
                              marker=dict(
                              size=4,
                                color="turquoise",                # set color to an array/list of desired values
                                colorscale='Plasma',   # choose a colorscale
                                opacity=0.8
                              ),
                              showlegend=False,
                              hovertext=Summary["full_formula"][Summary[Summary["material_id"].isin(mpids)].index]),
            )

        #fig.show()
        """
        dir_path = Path(__file__).resolve().parents[2] / \
                                "reports" / "figures"  / "pca-3d-plots"

        Path(dir_path).mkdir(parents=True, exist_ok=True)

        #fig.write_image(str(dir_path / str(prettyNames[i][:-1] + "-3d-iso-train.pdf")))
        #viz.save(Path(__file__).resolve().parents[2] / "reports" / "figures" / "decision tree" / "hallo.svg")
        fig.show()
    #display(graphviz.Source(export_graphviz(clf)))

""" TODO: Add calibration of classifiers
# #############################################################################
# Plot calibration plots
from sklearn.calibration import CalibratedClassifierCV,calibration_curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")



# Randomly sample 70% of your dataframe
X_train = trainingData.sample(frac=0.3).drop(["material_id", "full_formula"], axis=1)
y_train = trainingTarget[X_train.index]
print(X_train.shape, y_train.shape)

X_test = trainingData.loc[~trainingData.index.isin(X_train.index)].drop(["material_id", "full_formula"], axis=1)
y_test = trainingTarget[X_test.index]

#print(X_test.shape,y_t)
#print(Algorithms[0])
lr = Algorithms[0]
dt = Algorithms[1]
rfc = Algorithms[2]
gb  = Algorithms[3]

rf_isotonic = CalibratedClassifierCV(rfc, cv=rskfold, method='isotonic')
print(lr)
for clf, name in [(lr, 'Logistic regression'),
                  (dt, 'Decision tree'),
                  (rfc, 'Random forest'),
                  (rf_isotonic, "Random forest calibrated"),
                  (gb, 'Gradient boost')]:

    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()
"""
