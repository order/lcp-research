import lcp.gen as gen
import lcp.solvers as solvers
import lcp.util as util

import functools
import numpy as np
import scipy
import scipy.sparse.linalg
import pylab as pl
import time

# Generate instance
N = 500
m = 25
(M,q) = gen.rand_psd_lcp(N)
#(M,q) = gen.rand_lpish(N,m)
#M = M + 8*np.eye(N)
L = scipy.sparse.linalg.eigs(M,k=1,which='LM',return_eigenvectors=False,tol=1e-8)

# Set up solver
solver = solvers.iter_solver()
solver.record_fn = util.residual_recorder
solver.term_fn = functools.partial(util.max_iter_res_thesh_term, 1000, 1e-9)

solver.params['step'] = 1/L[0]
solver.params['omega'] = 1.4
solver.params['restart'] = 0.01
solver.params['scale'] = 0.125/L[0]

# start = time.time()
# solver.iter_fn = solvers.projected_jacobi_iter
# (record_jacobi,state) = solver.solve(M,q);
# elapsed = time.time() - start
# print 'Jacobi elapsed', elapsed
start = time.time()
solver.iter_fn = solvers.euler_iter
(record_euler,state) = solver.solve(M,q);
elapsed = time.time() - start
print 'Euler elapsed', elapsed

start = time.time()
solver.iter_fn = solvers.euler_linesearch_iter
(record_eulerls,state) = solver.solve(M,q);
elapsed = time.time() - start
print 'Euler LS elapsed', elapsed

start = time.time()
solver.iter_fn = solvers.accelerated_prox_iter
(record_accel,state) = solver.solve(M,q);
elapsed = time.time() - start
print 'Accelerated elapsed', elapsed

start = time.time()
solver.iter_fn = solvers.psor_iter
(record_psor,state) = solver.solve(M,q);
elapsed = time.time() - start
print 'PSOR elapsed', elapsed

pl.semilogy(record_euler.residual)
pl.semilogy(record_eulerls.residual)
pl.semilogy(record_accel.residual)
pl.semilogy(record_psor.residual)
pl.legend(['Euler','Euler LS','Accelerated','PSOR'])

pl.show()