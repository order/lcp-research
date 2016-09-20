import numpy as np
import scipy as sp
import scipy.stats as stats

from utils.archiver import Unarchiver, read_medit_mesh
from utils import standardize,make_points

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse
import sys
import os.path



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description='Display the faces of a tetrahedral mesh.')
    parser.add_argument('mesh', metavar='F', type=str,
                        help='Mesh input file (INRIA .mesh)')
    args = parser.parse_args()

    meshfile = args.mesh
    (base,ext) = meshfile.rsplit('.',1)
    assert ext == 'mesh'
    
    tet_mesh = read_medit_mesh(meshfile)
    (vertices,edges,triangles,tetrahedra) = tet_mesh
    
    vertices = tet_mesh[0]
    V = vertices.shape[0]
    T = tetrahedra.shape[0]
    print 'Reading INRIA .mesh file',meshfile
    print '\tFound', V, 'vertices'
    print '\tFound', T, 'tetrahedra'

    bbox = np.empty((3,2))
    for i in xrange(3):
        bbox[i,0] = np.min(vertices[:,i])
        bbox[i,1] = np.max(vertices[:,i])
        
    kernel = stats.gaussian_kde(vertices.T,0.1)

    G = 40
    grids = [np.linspace(bbox[i,0],bbox[i,1],G) for i in xrange(3)]
    P = make_points(grids)
    P += 0.1*np.random.randn(*P.shape)

    Z = kernel(P.T)
    stdZ = standardize(Z)
    cmap = plt.get_cmap('spectral')
    C = cmap(stdZ)
    C[:,3] = (stdZ)**1.5

    mask = (C[:,3] > 0.025)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[mask,0],P[mask,1],P[mask,2],c=C[mask,:],
               s=125,lw=0)
    plt.show()
