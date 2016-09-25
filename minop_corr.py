import subprocess
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.csgraph as csgraph

import matplotlib.pyplot as plt

import tri_mesh_viewer as tmv
from utils.archiver import Unarchiver,read_shewchuk
from projective_solver import solve_plcp_file

BASE_FILENAME = 'min'

def run_approx_gen(filename,F,V,generate):
    filename = base_file
    cmd = 'cdiscrete/minop_approx -o {0} -F {1} -V {2} -e {3}'.format(filename,
                                                                      F,V,0.15)
    if not generate:
        cmd += ' -m ' + filename + '.tri'
    
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    return subprocess.check_call([cmd],
                                 shell=True,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr)

def read_solution(filename):
    plcp = Unarchiver(filename + '.plcp')
    psol = Unarchiver(filename + '.psol')

    p = psol.p
    assert 0 == p.size % 3
    V = p.size / 3
    P = np.reshape(p,(V,3),order='F')
    psol_value = P[:,0]

    b = plcp.b
    c = plcp.c
    true_value = np.minimum(b,c)

    residual = psol_value - true_value
    jitter = plcp.jitter
    basis = plcp.flow_basis

    return residual,jitter,basis


if __name__ == "__main__":
    base_file = '/home/epz/data/minop/minop'
    iters = 150
    # INITIAL MESH AND LCP GENERATION
    jitter = []
    residual = []

    basis = None
    Fourier = 35
    Voronoi = 4
    for i in xrange(iters):
        if 0 == i:
            rc = run_approx_gen(base_file,Fourier,Voronoi,True)
        else:
            rc = run_approx_gen(base_file,Fourier,Voronoi,False)
        assert(0==rc)
        solve_plcp_file(base_file + '.plcp')
        (res,jit,new_basis) = read_solution(base_file)
        if basis is not None:
            assert np.linalg.norm(basis - new_basis) < 1e-9
        else:
            basis = new_basis
            
        residual.append(res)
        jitter.append(jit)

    
    group = np.argmax(np.abs(basis),1)
    idx = np.argsort(group)
    residual = np.array(residual).T
    jitter = np.array(jitter).T

    np.savez("tmp.npz",
             residual=residual,
             jitter=jitter)
    
    Cor1 = np.cov(residual[idx,:])
    Cor2 = np.cov(residual[idx,:]*jitter[idx,:])
    cmap = plt.get_cmap('jet')
    (nodes,faces) = read_shewchuk(base_file)
    for (i,c) in enumerate([Cor1,Cor2]):
        plt.figure(1);
        plt.subplot(1,2,i+1)
        tmv.plot_vertices(nodes,faces,np.diag(c),
                          cmap=cmap,interp='nearest');
        plt.colorbar()
        
        plt.figure(2);
        plt.subplot(1,2,i+1)
        plt.imshow(c,
                   cmap=cmap,interpolation='nearest');
        plt.colorbar()


    plt.show()
