import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

from sklearn import neighbors

from parsers import KwargParser

import numpy as np
from utils import make_points

##########################
# Animate a rank 3 tensor along the first dimension

def imshow(X,**kwargs):
    # Hate specifying no interpolation each time.
    ax = plt.gca()
    if 'interpolation' in kwargs:
        ax.imshow(X,**kwargs)
    else:
        ax.imshow(X,interpolation='none',**kwargs)

def cdf_points(X):
    (N,) = X.shape    
    S = np.sort(X)
    return (S,np.linspace(0,1,N))

def animate_cdf(X,**kwargs):
    # Parse input
    parser = KwargParser()
    parser.add_optional('save_file',str)
    parser.add('title','No title',str)
    parser.add('xlabel','x',str)
    parser.add('ylabel','y',str)
    args = parser.parse(kwargs)

    assert(2 == len(X.shape)) # Frame index, index
    (I,J) = X.shape
    
    fig = plt.figure()
    low = np.min(X)
    hi = np.max(X)

    print 'Starting animation...'        
    Plotters = []
    
    for i in xrange(I):
        S = np.sort(X[i,:])
        cdf = plt.plot(S,np.linspace(0,1,J),'-b',lw=2.0)
        plt.ylim([0,1])
        plt.xlim([low,hi])
        Plotters.append(cdf)
    im_ani = animation.ArtistAnimation(fig,Plotters,\
                                       interval=50,\
                                       repeat_delay=3000)
    plt.xlabel(args['xlabel'])
    plt.ylabel(args['ylabel'])
    plt.title(args['title'])

    if 'save_file' in args:
        im_ani.save(args['save_file'])
    plt.show()

def animate_frames(frames):
    (T,X,Y) = frames.shape
    lo = np.min(frames)-1e-3
    hi = np.max(frames)+1e-3

    fig = plt.figure()
    Plotters = []
    for t in xrange(T):
        p = plt.pcolormesh(frames[t,...],
                           vmin=lo,vmax=hi,
                           cmap='viridis')
        Plotters.append([p])
        
        im_ani = animation.ArtistAnimation(fig,Plotters,\
                                           interval=50,\
                                           repeat_delay=3000,\
                                           blit=True)
    plt.show()

def scatter_knn(y,X,K,G):
    # K = number of neighbors
    # G = grid points per dim
    
    (N,D) = X.shape
    assert(2 == D)
    assert(N == y.size)
    
    # Make mesh
    cuts = []
    for d in xrange(D):
        cuts.append(np.linspace(np.min(X[:,d]),np.max(X[:,d]),G))
    P = make_points(cuts)
    XI,YI = np.meshgrid(*cuts)

    # Build model and eval on mesh
    knn = neighbors.KNeighborsRegressor(K,weights='distance')
    Z = knn.fit(X,y).predict(P)
    Z = np.reshape(Z,(G,G),order='F')

    return (XI,YI,Z)
