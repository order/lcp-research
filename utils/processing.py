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

def final_frame(data,params,**kwargs):
    """
    Like frame processing, but only the final frame
    """
    
    # Inner functions for different processing options
    def value(F):
        return F[0,:,:]

    def flow(F):
        return np.argmax(F[1:,:,:],axis=0)

    def agg_flow(F):
        return np.sum(F[1:,:,:],axis=0)

    def adv(F):
        SF = np.sort(F[1:,:,:],axis=0)
        return np.log(SF[-1,:,:] - SF[-2,:,:] + 1e-22)

    parser = KwargParser()
    parser.add('field','primal') #primal/dual
    parser.add('option','value')
    args = parser.parse(kwargs)

    if 'comp' == args['field']:
        data_field = data['primal'] * data['dual']
    else:
        data_field = data[args['field']]
    fn = eval(args['option'])
    
    discretizer = params['discretizer']
    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()
    (x,y) = discretizer.get_basic_lengths()    
    frames = split_into_frames(data_field,A,n,x,y)

    return fn(frames[-1,:,:,:])
    

def frame_processing(data,params,**kwargs):
    """
    Turns data into movie frames based on provided inputs
    """

    # Inner functions for different processing options
    def value(F):
        return F[:,0,:,:]

    def flow(F):
        return np.argmax(F[:,1:,:,:],axis=1)

    def agg_flow(F):
        return np.sum(F[:,1:,:,:],axis=0)

    def adv(F):
        SF = np.sort(F[:,1:,:,:],axis=1)
        return np.log(SF[:,-1,:,:] - SF[:,-2,:,:] + 1e-22)

    parser = KwargParser()
    parser.add('field','primal') #primal/dual
    parser.add('option','value')
    args = parser.parse(kwargs)

    if 'comp' == args['field']:
        data_field = data['primal'] * data['dual']
    else:
        data_field = data[args['field']]
    fn = eval(args['option'])
    
    discretizer = params['discretizer']
    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()
    (x,y) = discretizer.get_basic_lengths()    
    frames = split_into_frames(data_field,A,n,x,y)

    return fn(frames)

def vector_processing(data,params,**kwargs):
    """
    Returns a subset of the data based on provided inputs
    """
    def val(F,n):
        return F[:,:n]

    def flow(F,n):
        return F[:,n:]

    def log_flow(F,n):
        return np.log10(flow(F,n))

    parser = KwargParser()
    parser.add('field','primal') #primal/dual
    parser.add('option','val')
    args = parser.parse(kwargs)

    if 'comp' == args['field']:
        data_field = data['primal'] * data['dual']
    else:
        data_field = data[args['field']]
        
    fn = eval(args['option'])
        
    discretizer = params['discretizer']

    A = discretizer.get_num_actions()
    n = discretizer.get_num_nodes()

    return fn(data_field,n)

def time_series(data,params,**kwargs):
    
    def identity(x):
        return x

    def log(x):
        return np.log10(x)

    def log_vector_cond(X):
        """
        Assumes X > 0, and it's of the form (I,N) = X.shape
        where the rows are iteration numbers and columns are
        components of the solution vector for that iteration.

        The "vector condition number" is the ratio between the
        max and the min values of X.

        Log is taken because the condition numbers can blow up.
        """
        assert (2 == len(X.shape))
        assert(not np.any(X <= 0))
        (I,N) = X.shape
        ret = np.amax(X,axis=1) / np.amin(X,axis=1)
        assert((I,) == ret.shape)
        return np.log10(ret)

    parser = KwargParser()
    parser.add('field')
    parser.add('option','identity')
    args = parser.parse(kwargs)

    fn = eval(args['option'])
    
    ret = fn(data[args['field']])
    assert(1 == len(ret.shape)) # Should be a vector

    return ret
