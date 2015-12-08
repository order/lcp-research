import numpy as np
import utils.plotting as plotting
from utils.parsers import KwargParser

import matplotlib.pyplot as plt
import time

import pickle

import sys

def read_in_args(filename):
    params = pickle.load(open(param_filename,'rb'))
    parser = KwargParser()
    parser.add('discretizer',None)
    parser.add_optional('solver')
    parser.add_optional('objects')
    params = parser.parse(params)

    discretizer = params['discretizer']
    assert(2 == discretizer.get_dimension())
    
    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()
    (x,y) = discretizer.get_basic_lengths()
    
    return (A,n,x,y)

def plot_value(Frames,**kwargs):
    plotting.animate_frames(Frames[:,0,:,:],**kwargs)

def plot_flow(Frames,**kwargs):
    policy = np.argmax(Frames[:,1:,:,:],axis=1)
    plotting.animate_frames(policy,**kwargs)

def plot_log_advantage(Frames,**kwargs):
    SFrames = np.sort(Frames[:,1:,:,:],axis=1)
    adv = np.log(SFrames[:,-1,:,:] - SFrames[:,-2,:,:] + 1e-12)
    plotting.animate_frames(adv,**kwargs)    

def plot_value_complement(PrimalFrames,DualFrames,**kwargs):
    complement = PrimalFrames[:,0,:,:] * DualFrames[:,0,:,:]
    plotting.animate_frames(complement,**kwargs)
    

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
    #plot_log_advantage(PrimalFrames)
    plot_value(PrimalDirFrames,cmap='rainbow')
    #plot_value_complement(PrimalFrames,DualFrames)
    #plot(np.log(Steplen))
