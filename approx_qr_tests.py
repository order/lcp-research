import bases
import matplotlib.pyplot as plt

import numpy as np

n = 15
D = 3
N = n**D
K = int(np.sqrt(N))
T = 150

# Generate points
grid = np.linspace(0,1,n)
meshes = np.meshgrid(*([grid]*D))
flat_meshes = [x.flatten() for x in meshes]
points = np.column_stack(flat_meshes)
assert((N,D) == points.shape)

# Generate the basis for those points
fourier = bases.RandomFourierBasis(num_basis=K,
            scale=10.0)
basis_gen = bases.BasisGenerator(fourier)
Phi = basis_gen.generate(points)
assert((N,K) == Phi.shape)

# Run the experiment
Data = np.empty((T,2))
Range = [int(1.1*K),int(10*K)]

Metric = lambda A: np.max(abs((A.T).dot(A)))
#Metric = lambda A: np.linalg.norm((A.T).dot(A) - np.eye(K),ord=2)

for t in xrange(T):
    S = np.random.randint(*Range)
    print 'Trial {0} with {1} samples'.format(t,S)
    # Get QR for subsampled system
    idx = np.random.choice(xrange(N),S)
    A = Phi[idx,:]
    assert((S,K) == A.shape)
    R = np.linalg.qr(A,mode='r')
    assert((K,K) == R.shape)
    
    # Solve for Q:
    # Phi = QR -> R^T Q^T = Phi^T
    Q = np.linalg.solve(R.T,Phi.T).T
    assert(Q.shape == Phi.shape)
    
    # Report something:
    res = Metric(Q)
    print '\tResidual:',res
    Data[t,0] = S
    Data[t,1] = res
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
    
# Scatter plot
ax.scatter(Data[:,0] / N,Data[:,1])

# Original Phi
base_metric = Metric(Phi)
baseline = [base_metric]*2
ax.plot(np.array(Range) / float(N),baseline,'r-')
ax.set_yscale('log')
plt.xlabel('Sampling proportion')
plt.ylabel('Coherence')
plt.title('Coherence of approximate QR decomposition')
plt.legend('Sampled Rows','Original')

plt.show()