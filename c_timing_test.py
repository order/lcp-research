import discrete
import numpy as np
import time
import discrete

N = 25
D = 4
M = 1
R = 2500

P = np.random.rand(M,D);

grid_desc = [(0,1,N) for _ in xrange(D)]
disc = discrete.RegularGridInterpolator(grid_desc)

start = time.time()
for _ in xrange(R):
    W = disc.points_to_index_distributions(P)
print time.time() - start
