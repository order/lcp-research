import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

from kwargparser import KwargParser

import numpy as np

##########################
# Animate a rank 3 tensor along the first dimension
def animate_frames(Frames,**kwargs):
    # Parse input
    parser = KwargParser()
    parser.add_optional('save_file',str)
    args = parser.parse(kwargs)
    
    assert(3 == len(Frames.shape)) # Frame index, x, y
    (I,X,Y) = Frames.shape
    
    fig = plt.figure()

    print 'Starting animation...'        
    Plotters = []
    for i in xrange(I):
        Plotters.append([plt.pcolor(Frames[i,:,:])])

    im_ani = animation.ArtistAnimation(fig,Plotters,\
                                       interval=50,\
                                       repeat_delay=3000,
                                   blit=True)
    if 'save_file' in args:
        save_file = args['save_file']
        im_ani.save(save_file)
    else:
        plt.show()


def split_into_frames(Data,A,n,x,y):
    """
    Take a I x N matrix where the columns are either
    primal or dual block-structured vectors from an 
    LCP'd MDP. The MDP should be based on 2 continuous
    dimensions (e.g. the double integrator)

    Convert into an I x (A+1) x X x Y rank-4 tensor where
    T[i,a,:,:] is a 2D image.
    """
    (I,N) = Data.shape
    assert(N == (A+1)*n)
    assert(x*y <= n )

    # Reshape into I x n x (A+1)
    Frames = np.reshape(Data,(I,n,(A+1)),order='F')

    # Crop out non-physical states
    Frames = Frames[:,:(x*y),:]

    # Reshape into  I x X x Y x (A+1)
    Frames = np.reshape(Frames,(I,x,y,(A+1)),order='C')

    # Swap axes to be I x (A+1) x Y x X
    Frames = np.swapaxes(Frames,1,3)
    return Frames
