import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

from parsers import KwargParser

import numpy as np

##########################
# Animate a rank 3 tensor along the first dimension
def animate_frames(Frames,**kwargs):
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

def plot(fn,**kwargs):
    parser = KwargParser()
    parser.add('title','No title',str)
    parser.add('xlabel','x',str)
    parser.add('ylabel','y',str)
    args = parser.parse(kwargs)
    
    plt.plot(fn)
    plt.xlabel(args['xlabel'])
    plt.ylabel(args['ylabel'])
    plt.title(args['title'])

    plot.show()

