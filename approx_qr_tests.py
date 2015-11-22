import bases
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


import numpy as np
import scipy as sp
import scipy.interpolate

def coherence(A):
    return np.max(abs((A.T).dot(A)))
    
def unorthogonality(A):
    (N,M) = A.shape
    return np.linalg.norm((A.T).dot(A) - np.eye(M))
    
def plot_2d_interp(points,vals):
    [N,D] = points.shape
    assert(D == 2)
    
    # Build grid
    G = 25 # num grid points
    [x_lo,x_hi] = [np.min(points[:,0]),np.max(points[:,0])]
    [y_lo,y_hi] = [np.min(points[:,1]),np.max(points[:,1])]
    [grid_x,grid_y] = np.meshgrid(np.linspace(x_lo,x_hi,G),np.linspace(y_lo,y_hi,G))
    
    grid_z = sp.interpolate.griddata(points,vals,(grid_x,grid_y),method='nearest')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x,grid_y,grid_z,cmap=cm.jet,rstride=1, cstride=1,linewidth=0)
    plt.show()

def generate_points(n,D):
    grid = np.linspace(0,2*np.pi,n)
    meshes = np.meshgrid(*([grid]*D))
    flat_meshes = [x.flatten() for x in meshes]
    points = np.column_stack(flat_meshes)  
    assert((n**D,D) == points.shape)
    return points

def approx_ortho(Phi,Samples):
    (N,K) = Phi.shape

    idx = np.random.choice(xrange(N),Samples)
    A = Phi[idx,:]
    assert((Samples,K) == A.shape)
    R = np.linalg.qr(A,mode='r')
    assert((K,K) == R.shape)
    
    # Solve for Q:
    # Phi = QR -> R^T Q^T = Phi^T
    Q = np.linalg.solve(R.T,Phi.T).T
    return Q

def sampling_vs_coherence_experiment():
    n = 15
    D = 3
    N = n**D
    K = int(np.sqrt(N))
    T = 100

    # Generate points
    points = generate_points(n,D)
    assert((N,D) == points.shape)

    # Generate the basis for those points
    fourier = bases.RandomFourierBasis(num_basis=K,
                scale=10.0)
    basis_gen = bases.BasisGenerator(fourier)
    Phi = basis_gen.generate(points)
    assert((N,K) == Phi.shape)
    

    # Run the experiment
    Data = np.empty((T,2))
    Range = [int(1.1*K),N]

    Metric = lambda A: np.max(abs((A.T).dot(A)))
    #Metric = lambda A: np.linalg.norm((A.T).dot(A) - np.eye(K))

    for t in xrange(T):
        S = np.random.randint(*Range)
        print 'Trial {0} with {1} samples'.format(t,S)
        # Get QR for subsampled system
        Q = approx_ortho(Phi,S)
        assert(Q.shape == Phi.shape)
        
        # Report something:
        res = Metric(Q)
        print '\tResidual:',res
        Data[t,0] = S
        Data[t,1] = res
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log')
        
    # Scatter plot
    ax.scatter(Data[:,0] / float(N),Data[:,1])

    # Original Phi
    base_metric = Metric(Phi)
    print base_metric
    baseline = [base_metric]*2
    ax.plot(np.array(Range) / float(N),baseline,'ro-')
    plt.xlabel('Sampling proportion')
    plt.ylabel('Coherence')
    plt.title('Coherence of approximate QR decomposition')
    plt.legend(['Sampled Rows','Original'],loc='best')

    plt.show()
    
def coherence_vs_scaling():
    D = 3
    T = 100
    SizeRange = [10,20]
    ScalingRange = [1e-9,10]
    BasisProportion = lambda x: int(np.sqrt(x))
    
    Data = np.empty((T,3))
    
    for t in xrange(T):
        n = np.random.randint(*SizeRange)
        N = n**D
        K = BasisProportion(N)
        points = generate_points(n,D)    

        scale = np.random.uniform(*ScalingRange)
        fourier = bases.RandomFourierBasis(num_basis=K,
                    scale=scale)
        basis_gen = bases.BasisGenerator(fourier)
        Phi = basis_gen.generate(points)
        
        Data[t,0] = np.log(N)
        Data[t,1] = scale
        Data[t,2] = unorthogonality(Phi)
        
    plot_2d_interp(Data[:,:2],Data[:,2])

        
def size_vs_coherence_experiment():

    D = 3
    Repeat = 7
    SizeRange = range(10,30)
    TotalRuns = Repeat * len(SizeRange)
    Data = np.empty((TotalRuns,2))
    
    BasisProportion = lambda x: int(np.sqrt(x))
    SampleProportion = lambda x: 3.0*BasisProportion(x)
    Metric = lambda X: unorthogonality(X)
    
    I = 0
    for n in SizeRange:
        points = generate_points(n,D)
        N = n**D
        K = BasisProportion(N)
        S = SampleProportion(N)

        # Generate points
        assert((N,D) == points.shape)
        
        for _ in xrange(Repeat):    


            # Generate the basis for those points
            fourier = bases.RandomFourierBasis(num_basis=K,
                        scale=10.0)
            basis_gen = bases.BasisGenerator(fourier)
            Phi = basis_gen.generate(points)
            assert((N,K) == Phi.shape)

            # Run the experiment
            Q = approx_ortho(Phi,S)
                
            # Report something:
            res = Metric(Q)
            original = Metric(Phi)
            Data[I,0] = N
            Data[I,1] = res / original
            I += 1
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # Original Phi
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Scatter plot
    ax.scatter(Data[:,0],Data[:,1])



    plt.xlabel('Instance size')
    plt.ylabel('Coherence Ratio')
    plt.title('Coherence of approximate QR decomposition; 3.0 x K sampling')
    plt.legend(['Sampled Rows','Original'],loc='best')

    plt.show()
    
if __name__ == '__main__':
    #sampling_vs_coherence_experiment()
    size_vs_coherence_experiment()
    #coherence_vs_scaling()