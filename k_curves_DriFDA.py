#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import MiniBatchKMeans

def cdist_(a,b):
    return (a[:,None]-b[None,:])

def rbf(T, c,s,W):
    X = np.exp(-cdist_(T,c)**2/s)
    X = X/X.sum(axis=1)[:,None]
    return np.dot(X, W)
def rbfs(X, c,s,Ws):
    Y = np.empty( shape=(Ws.shape[0],X.shape[0],Ws.shape[2]) )
    for i,W in enumerate(Ws):
        Y[i] = rbf(X,c,s,W)
    return Y

def rbf_train(T,Y, c,s,l):
    X = np.exp(-cdist_(T,c)**2/s)
    X = X/X.sum(axis=1)[:,None]
    return np.linalg.solve( np.dot(X.T,X) + l*np.eye(c.shape[0]) , np.dot(X.T,Y) )

def k_curves(T,D,n_curves, sigma=0.01,lambd=1E-4,n_prototypes=20,n_chunks=50,n_max_itr=500):
    min_,max_ = T.min(),T.max()
    T = (T-min_)/(max_-min_)
    n_dim = D.shape[1]
    
    prototypes = np.linspace(0,1,n_prototypes)
    Ws = 0.*np.random.normal( size=(n_curves,n_prototypes,n_dim) )
    Ws = Ws+MiniBatchKMeans(n_clusters=n_curves).fit(D).cluster_centers_[:,None,:]
    
    for j in range(n_chunks):
        J = int(T.shape[0]*(j+1)/n_chunks)
        T_J, D_J = T[:J],D[:J]
        for _ in range(n_max_itr):
            E = rbfs(T_J, prototypes,sigma,Ws)
            I = np.linalg.norm(D_J[None,:,:]-E, axis=2).argmin(axis=0)
            Ws_ = np.copy(Ws)
            for i in range(Ws.shape[0]):
                Ws[i] = rbf_train( T_J[I == i] ,D_J[I == i], prototypes,sigma,lambd)
            if np.linalg.norm(Ws_-Ws)/n_curves < 0.1:
                break
    
    E = rbfs(T, prototypes,sigma,Ws)
    I = np.linalg.norm(D[None,:,:]-E, axis=2).argmin(axis=0)
    
    out = np.zeros( (D.shape) )
    for i in range(I.shape[0]):
        out[i,:] = E[I[i],i]
    return out

