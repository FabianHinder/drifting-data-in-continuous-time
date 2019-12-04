# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_classification_dataset(X, y, colors=['r', 'b', 'g', 'y'], xticks=[], yticks=[], title="", show_legend=True, savefig_path=None, show=True):
    """
    Plot a 1d or 2d classification data set.
    Parameters
    ----------
    X : numpy array
        1d or 2d data set. Data points are expected to be stored row-wise.
    y : numpy array
        1d array containing the labels (integers).
    """
    
    if X.shape[1] != 1 and X.shape[1] != 2:
        raise "Invalid number of features! Expect either 1 or 2 features (1d or 2d data only)"
    
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
   
    plt.figure()
    for c in colors:
        plt.scatter([], [], c=c)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    if X.shape[1] == 2:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        plt.scatter(X[:, 0], X[:, 1], c=list(map(lambda i: colors[i], y)), cmap=cmap_bold, edgecolor='k', s=20)

        plt.ylim(y_min, y_max)
    else:
        plt.scatter(X[:, 0], np.zeros(X.shape[0]), c=list(map(lambda i: colors[i], y)), cmap=cmap_bold, edgecolor='k', s=20)

    plt.xlim(x_min, x_max)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title(title)

    if show_legend is True:
        plt.legend(['Class {0}'.format(i) for i in np.unique(y)])
    
    if show:
        if savefig_path is None:
            plt.show()
        else:
            plt.savefig(savefig_path)


def plot_2d_decisionboundary(model, X, y, grid_resolution=0.1, color_data={'colors': ['r', 'b', 'g', 'y'], 'c_light': ['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFFFAA'],
    'c_bold': ['#FF0000', '#0000FF', '#00FF00', '#FFFF00']}, xticks=[], yticks=[], title="", show_legend=True, savefig_path=None, show=True):
    """
    Plot the decision boundary of a classifier for a 2d data set.
    Parameters
    ----------
    model: object
        Model/Classifier must implement a `predict` method.
    X : numpy array
        1d or 2d data set. Data points are expected to be stored row-wise.
    y : numpy array
        1d array containing the labels (integers).
    """
    
    h = grid_resolution
    cmap_light = ListedColormap(color_data['c_light'][:len(np.unique(y))])
    cmap_bold = ListedColormap(color_data['c_bold'][:len(np.unique(y))])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    # Compute the prediction for each grid cell

    # Plot
    plt.figure()
    for c in color_data['colors']:
        plt.scatter([], [], c=c)

    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=list(map(lambda i: color_data['colors'][i], y)), cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title(title)
    if show_legend is True:
        plt.legend(['Class {0}'.format(i) for i in np.unique(y)])
    
    if show:
        if savefig_path is None:
            plt.show()
        else:
           plt.savefig(savefig_path, dpi=500, bbox_inches="tight", pad_inches=0.1)
