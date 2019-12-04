#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linear_DriFDA import linearDriFDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

if __name__ == '__main__':
    n_samples = 750
    T = np.random.random(size=(n_samples))
    T.sort()
    T = T.reshape(-1,1)
    
    X = np.concatenate( ((50*np.maximum(T-0.66,0)*np.random.choice([-1,1], (n_samples,1) ) +0.5*np.random.random(size=(n_samples,1))) , np.random.random(size=n_samples).reshape(-1,1) ), axis=1 )
    T = T.reshape(1,-1)[0]
    
    Tmin,Tmax = T.min(),T.max()
    
    out = linearDriFDA(T,X)
    
    err = X-out
    
    
    fig = plt.figure(figsize=(3*5,1*5))
    ax = fig.add_subplot(231, projection='3d')
    ax.scatter(X[:,0],X[:,1],T)
    ax = fig.add_subplot(232, projection='3d')
    ax.scatter(out[:,0],out[:,1],T)
    ax = fig.add_subplot(233, projection='3d')
    ax.scatter(err[:,0],err[:,1],T)
    
    sel = np.random.randint(0,X.shape[0],250)
    ax = fig.add_subplot(245)
    ax.scatter(X[sel,0],X[sel,1], c=plt.get_cmap()(T[sel]) )
    ax = fig.add_subplot(246)
    ax.scatter(out[sel,0],out[sel,1], c=plt.get_cmap()(T[sel]) )
    ax = fig.add_subplot(247)
    ax.scatter(err[sel,0],err[sel,1], c=plt.get_cmap()(T[sel]) )
    
    neigh = KNeighborsRegressor(n_neighbors=8,weights='distance')
    neigh.fit(err, T)
    
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(err[:,1].min(), err[:,1].max() + dy, dy), slice(err[:,0].min(), err[:,0].max() + dx, dx)]
    z =  neigh.predict( np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1) ).reshape(x.shape) 
    #z = ( z - T.mean() )**2
    print(z.mean())
    ax = fig.add_subplot(248)
    ax.pcolor(x, y, z)
    
    plt.show()
