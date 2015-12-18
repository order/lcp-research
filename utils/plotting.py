import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

from parsers import KwargParser

import numpy as np

##########################
# Animate a rank 3 tensor along the first dimension

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

def animate_frames(Frames,**kwargs):
    """
    Turn frames into a movie
    """
    # Parse input
    parser = KwargParser()
    parser.add_optional('save_file',str)
    parser.add('title','No title',str)
    parser.add('xlabel','x',str)
    parser.add('ylabel','y',str)
    parser.add('cmap','jet',str)
    args = parser.parse(kwargs)
    
    assert(3 == len(Frames.shape)) # Frame index, x, y
    (I,X,Y) = Frames.shape
    
    fig = plt.figure()
    low = np.min(Frames)
    hi = np.max(Frames)

    print 'Starting animation...'        
    Plotters = []
    cmap = plt.get_cmap(args['cmap'])
    for i in xrange(I):
        img = plt.pcolor(Frames[i,:,:],
                         vmin = low,
                         vmax = hi,
                         cmap=cmap)
        Plotters.append([img])
    im_ani = animation.ArtistAnimation(fig,Plotters,\
                                       interval=50,\
                                       repeat_delay=3000,\
                                       blit=True)
    plt.xlabel(args['xlabel'])
    plt.ylabel(args['ylabel'])
    plt.title(args['title'])

    if 'save_file' in args:
        im_ani.save(args['save_file'])
    plt.show()

def plot(data,**kwargs):
    parser = KwargParser()
    parser.add_optional('save_file',str)
    parser.add('title','No title',str)
    parser.add('xlabel','x',str)
    parser.add('ylabel','y',str)
    args = parser.parse(kwargs)
    
    plt.plot(data)
    plt.xlabel(args['xlabel'])
    plt.ylabel(args['ylabel'])
    plt.title(args['title'])

    if 'save_file' in args:
        plt.savefig(args['save_file'], bbox_inches='tight')
    plt.show()

