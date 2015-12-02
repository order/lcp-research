import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
A function that splits a I x N numpy array
of MDP primal or dual information into 2D frames
First block is the value information,
Rest is flow
"""
def split_into_frames(Data,A,n,x,y):
    # Data is a single vector
    if 1 == len(Data.shape):
        Data = Data[np.newaxis,:]

    # Size & sanity
    (I,N) = Data.shape
    assert(N == (A+1)*n)
    assert(x*y <= n)
    
    Frames = []
    for i in xrange(I):
        Imgs = []
        Comps = mdp_vector_into_components(Data[i,:],A,n)
        for a in xrange(A+1):
            Img = vector_to_2d_image(Comps[:(x*y),a],x,y)
            Imgs.append(Img)
        Frames.append(Imgs)
    return np.array(Frames)

def mdp_vector_into_components(V,A,n):
    assert(1 == len(V.shape))
    assert((A+1)*n == V.size)

    return np.reshape(V,(n,A+1),order='F')

def vector_to_2d_image(V,x,y):
    assert(1 == len(V.shape))
    N = V.size
    assert(x*y <= N)

    return np.reshape(V[:(x*y)],(y,x),order='f')

def split_into_frames_tensor(Data,A,n,x,y):
    """
    Kind of opaque tensor way of doing the above
    """
    (I,N) = Data.shape
    assert(N == (A+1)*n)
    assert(x*y <= n )

    Frames = np.reshape(Data,(I,n,(A+1)),order='F')
    Frames = Frames[:,:(x*y),:]
    print Frames.shape
    Frames = np.reshape(Frames,(I,x,y,(A+1)),order='C')
    Frames = np.swapaxes(Frames,1,3)
    return Frames

def animate_frames(Frames):
    assert(3 == len(Frames.shape)) # Frame index, x, y
    (I,X,Y) = Frames.shape
    
    fig = plt.figure()
    
    Plotters = []
    for i in xrange(I):
        Plotters.append([plt.pcolor(Frames[i,:,:])])
    
    im_ani = animation.ArtistAnimation(fig,Plotters,\
                                       interval=50,\
                                       repeat_delay=3000,
                                   blit=True)
    #im_ani.save('im.mp4')
    plt.show()    

def plot_costs(discretizer,action):
    (x_n,v_n) = discretizer.get_basic_len()
  
    states = discretizer.basic_mapper.get_node_states()
    costs = discretizer.cost_obj.cost(states,action)
    CostImg = np.reshape(costs,(x_n,v_n))
    plt.imshow(CostImg,interpolation = 'nearest')
    plt.title('Cost function')

    plt.show()       

def plot_soln_evol(A):
    plt.semilogy(A)
    plt.title('Solution component evolution')
    plt.show()

def plot_soln_movie(X,A,n,x,y):
    fig = plt.figure()
    (I,N) = X.shape
    assert(N == (A+1)*n)
    assert(x*y <= n)
    
    Frames = []
    splitter = TwoDSplitter(A,n,x,y)
    Splits = splitter.split(X)
    assert(I == len(Splits))
    for i in xrange(I):
        Frames.append([plt.pcolor(Splits[i][0].T)])
    
    #im_ani = animation.ArtistAnimation(fig,Frames,\
    #                                   interval=50,\
    #                                   repeat_delay=3000,
    #                               blit=True)
    #im_ani.save('im.mp4')
    plt.pcolor(Splits[-1][2].T)
    plt.show()

def plot_trajectory(discretizer,policy):
    boundary = discretizer.get_basic_boundary()
    (x_lo,x_hi) = boundary[0]
    (v_lo,v_hi) = boundary[1]

    shade = 0.5
    x_rand = shade*random.uniform(x_lo,x_hi)
    v_rand = shade*random.uniform(v_lo,v_hi)
    init_state = np.array([[x_rand,v_rand]])
    assert((1,2) == init_state.shape)
    
    # Basic sanity
    discretizer.physics.remap(init_state,action=0) # Pre-animation sanity
    
    sim = DoubleIntegratorSimulator(discretizer)
    sim.simulate(init_state,policy,1000)    
    
    
def plot_value_function(discretizer,value_fun_eval,**kwargs):
    boundary = discretizer.get_basic_boundary()    
    (x_lo,x_hi) = boundary[0]
    (v_lo,v_hi) = boundary[1]
    
    grid = kwargs.get('grid_size',51)
    [x_mesh,v_mesh] = np.meshgrid(np.linspace(x_lo,x_hi,grid),\
                                  np.linspace(v_lo,v_hi,grid),indexing='ij')
    Pts = np.column_stack([x_mesh.flatten(),v_mesh.flatten()])
    
    vals = value_fun_eval.evaluate(Pts)
    ValueImg = np.reshape(vals,(grid,grid))
    
    plt.pcolor(x_mesh,v_mesh,ValueImg)
        
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Cost-to-go function')
    plt.show()

def plot_value_vector(discretizer,value_fun_eval,**kwargs):
    (x_n,v_n) = discretizer.get_basic_len()
  
    J = value_fun_eval.node_to_cost[:(x_n*v_n)]
    J = np.reshape(J,(x_n,v_n))
    plt.pcolor(J.T)
    plt.title('Cost-to-go function')

    plt.show()   
    
def plot_flow(discretizer,value_fun_eval,**kwargs):
    boundary = discretizer.get_basic_boundary()    
    (x_lo,x_hi) = boundary[0]
    (v_lo,v_hi) = boundary[1]
    
    grid = kwargs.get('grid_size',51)
    [x_mesh,v_mesh] = np.meshgrid(np.linspace(x_lo,x_hi,grid),\
                                  np.linspace(v_lo,v_hi,grid),indexing='ij')
    Pts = np.column_stack([x_mesh.flatten(),v_mesh.flatten()])
    
    vals = value_fun_eval.evaluate(Pts)
    ValueImg = np.reshape(vals,(grid,grid))
    
    plt.pcolor(x_mesh,v_mesh,ValueImg)
        
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Cost-to-go function')
    plt.show()       
    
def plot_policy(discretizer,policy,**kwargs):
    boundary = discretizer.get_basic_boundary()
    (x_lo,x_hi) = boundary[0]
    (v_lo,v_hi) = boundary[1]
    
    grid = kwargs.get('grid_size',50)
    [x_mesh,y_mesh] = np.meshgrid(np.linspace(x_lo,x_hi,grid),\
                                  np.linspace(v_lo,v_hi,grid),indexing='ij')
    Pts = np.column_stack([x_mesh.flatten(),y_mesh.flatten()])
    
    
    PolicyImg = np.reshape(policy.get_decisions(Pts),(grid,grid))

    plt.pcolor(PolicyImg.T)
    plt.title('Policy map')

    plt.show()   
    
def plot_advantage(discretizer,value_fun_eval,action1,action2):
    (x_n,v_n) = discretizer.get_basic_len()

    states = discretizer.basic_mapper.get_node_states()
    next_states1 = discretizer.physics.remap(states,action=action1)
    next_states2 = discretizer.physics.remap(states,action=action2)

    adv = value_fun_eval.evaluate(next_states1)\
          - value_fun_eval.evaluate(next_states2)
    AdvImg = np.reshape(adv, (x_n,v_n))
    x_mesh = np.reshape(states[:,0], (x_n,v_n))
    v_mesh = np.reshape(states[:,1], (x_n,v_n))

    plt.pcolor(x_mesh,v_mesh,AdvImg)
        
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Advantage function')
    plt.show() 
