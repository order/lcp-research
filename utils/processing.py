from parsers import KwargParser
import numpy as np

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

def img_movie(data,params,**kwargs):
    """
    Turns data into movie frames based on provided inputs
    """
    def val(F):
        return F[:,0,:,:]

    def flow(F):
        return np.argmax(F[:,1:,:,:],axis=1)

    def adv(F):
        SF = np.sort(F[:,1:,:,:],axis=1)
    return np.log(SF[:,-1,:,:] - SF[:,-2,:,:] + 1e-22)

    def complement(P,D):
        return P * D
    
    data_field = data[kwargs['field']] # Hardcoded
    fn = eval(kwargs['fn']) # Hardcoded
    
    discretizer = params['discretizer']
    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()
    (x,y) = discretizer.get_basic_lengths()    
    frames = split_into_frames(data_field,A,n,x,y)

    return fn(frames)

def cdf_movie(data,params,**kwargs):
    """
    Returns a subset of the data based on provided inputs
    """
    def val(F,n):
        return F[:,:n]

    def flow(F,n):
        return F[:,n:]
    
    data_field = data[kwargs['field']]
    fn = eval(kwargs['fn'])
    
    discretizer = params['discretizer']

    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()

    return fn(data_field,n)
