import numpy as np
import utils.plotting as plotting
from utils.kwargparser import KwargParser

import matplotlib.pyplot as plt
import time

import pickle

import sys

def read_in_args(filename):
    params = pickle.load(open(param_filename,'rb'))
    parser = KwargParser()
    parser.add('x_nodes',None,int)
    parser.add('v_nodes',None,int)
    parser.add('actions',None,int)
    parser.add('size',None,int)
    params = parser.parse(params)

    A = params['actions']
    n = params['size']
    x = params['x_nodes']
    v = params['v_nodes']

    return (A,n,x,v)

def plot_value(Frames):
    plotting.animate_frames(Frames[:,0,:,:])

def plot_flow(Frames):
    policy = np.argmax(Frames[:,1:,:,:],axis=1)
    plotting.animate_frames(policy)

def plot_value_complement(PrimalFrames,DualFrames):
    complement = PrimalFrames[:,0,:,:] * DualFrames[:,0,:,:]
    plotting.animate_frames(complement)

def plot(fn):
    plt.plot(fn)
    plt.show()
    

if __name__ == '__main__':
    filename = sys.argv[1]
    assert('.npz' == filename[-4:])
    param_filename = filename[:-4] + '.pickle'

    # Load primal and dual data
    data = np.load(filename)

    # Load parameters
    print 'Loading params from', param_filename
    params = read_in_args(param_filename)

    Primal = data['primal']
    Dual = data['dual']
    PrimalDir = data['primal_dir']
    DualDir = data['dual_dir']
    Steplen = data['steplen']
    
    # Split    
    PrimalFrames = plotting.split_into_frames(data['primal'],*params)
    DualFrames = plotting.split_into_frames(data['dual'],*params)
    PrimalDirFrames = plotting.split_into_frames(data['primal_dir'],*params)
    DualDirFrames = plotting.split_into_frames(data['dual_dir'],*params)

    # Animate
    #plot_flow(PrimalFrames)
    #plot_value(PrimalDirFrames)
    #plot_value_complement(PrimalFrames,DualFrames)
    #plot(np.log(Steplen))
    plt.imshow(np.log(PrimalDir))
    plt.show()
