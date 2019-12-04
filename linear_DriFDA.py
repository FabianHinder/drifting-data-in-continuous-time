#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import FastICA

def linearDriFDA(T,X,n_components=None,mi_threashold=None):
    X0 = np.concatenate( (T.reshape(-1,1),X), axis=1)
    if n_components is None:
        n_components = X0.shape[1]
    ica = FastICA(n_components=n_components, max_iter=500)
    S = ica.fit_transform(X0)
    mi = mutual_info_regression(S, X0[:,0])
    mi /= mi.sum()
    if mi_threashold is None:
        mi_threashold = mi.mean()
    
    X_ = (ica.mixing_[:,mi >= mi_threashold] @ (S[:,mi >= mi_threashold]).T).T
    for i in range(X_.shape[1]):
        X_[:,i] = X_[:,i] + X0[:,i].mean()
    ##X_[:,0] = X[:,0]
    return X_[:,1:]

    
