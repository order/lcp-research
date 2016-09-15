import numpy as np

import itertools

from utils.archiver import *
from utils import standardize,make_points

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import time
import argparse
import sys
import os.path
import subprocess

def plot_mesh_slice(f,bound,meshfile,**kwargs):
    G = kwargs.get('grid_points',64)
    flat = kwargs.get('flat',True)
    
    assert((3,2) == bound.shape)
    idx = np.where(bound[:,0] == bound[:,1])[0]
    nidx = np.where(bound[:,0] != bound[:,1])[0]
    if 2 != nidx.size:
        print "Check slice bounds, need exactly 2 non-trivial dimensions"
    assert 1 == idx.size

    bound = np.hstack([bound,G*np.ones((3,1))])
    bound[idx,2] = 1
    
    grids = [np.linspace(*list(bound[i,:])) for i in xrange(3)]
    (points,meshes) = make_points(grids,True)

    timestamp = str(time.time())
    point_file = "/tmp/points." + timestamp
    value_file = "/tmp/value." + timestamp
    out_file = "/tmp/out." + timestamp
    arch = Archiver(points=points)
    arch.write(point_file)
    arch.clear()
    arch.add(values=f)
    arch.write(value_file)

    (base,ext) = os.path.splitext(meshfile)
    assert '.mesh' == ext
    
    
    cmd = ['cdiscrete/tet_interp',
           '--mesh',base + '.ctri',
           '--points',point_file,
           '--values',value_file,
           '--out',out_file]
    cmd = ' '.join(cmd)
    print cmd
    try:
        subprocess.check_call(cmd,shell=True)
    except Exception:
        print "Interpolation failed; check .ctri file?"
        quit()
    unarch = Unarchiver(out_file)
    F = np.reshape(unarch.interp,(G,G))
    Fm = np.ma.masked_where(np.isnan(F),F)

    if flat:
        plt.gcf()
        [X,Y] = [meshes[i].squeeze() for i in nidx]
        plt.pcolormesh(X,Y,Fm)
    else:
        Fm = standardize(Fm)
        [X,Y,Z] = [mesh.squeeze() for mesh in meshes]
        fig = plt.gcf()
        ax = fig.gca(projection='3d')
        cmap = plt.get_cmap('jet')
        colors = cmap(Fm)
        colors[...,3]= 0.25*(1-Fm)**1.5
        p = ax.plot_surface(X,Y,Z,
                            rstride=1,cstride=1,
                            facecolors=colors,
                            shade=False)    

if __name__ == "__main__":
    # Try to factor out the commonalities
    # between this and stained_glass
    # * Read in faces, vertices, tets, or solutions
    parser = argparse.ArgumentParser(
        description='Display the faces of a tetrahedral mesh.')
    parser.add_argument('mesh', metavar='F', type=str,
                        help='Mesh input file (INRIA .mesh)')
    parser.add_argument('-s','--solution', metavar='F',
                        help='LCP solution file; archive with p and d')
    parser.add_argument('-a','--action', metavar='N',type=int,
                        help='Which action to use (value=0)')
    parser.add_argument('-d','--dual',action="store_true",
                        help='Use dual variables')
    parser.add_argument('-v','--vertex', metavar='F',
                        help='File with vertex values')
    parser.add_argument('-f','--face', metavar='F', 
                        help='File with face values')
    parser.add_argument('-t','--tetra', metavar='F',
                        help='File with tetrahedral values')
    parser.add_argument('-L','--large',action="store_true",
                        help="Make large values more visible.")
    parser.add_argument('-l','--log', action="store_true",
                        help="Plot the abs log of function")
    parser.add_argument('-p','--policy',action="store_true",
                        help="Plot flow policy")
    parser.add_argument('-i','--ignore',type=int, nargs='+',
                        help="Ignore actions in policy plot")
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

    if not args.solution:
        assert not args.action
        assert not args.dual

    if args.policy:
        assert args.solution
        assert not args.action
        assert not args.dual
        assert not args.log
        assert not args.large
    else:
        assert not args.ignore
    
    if args.vertex:
        assert not args.face
        assert not args.tetra
        assert not args.solution
        print 'Reading vertex archive',args.vertex
        unarch = Unarchiver(args.vertex)
        assert(1 == len(unarch.data))
        f = unarch.data.values()[0]
        (N,) = f.shape
        if N == V+1:
            print '\tCropping OOB node'
            f = f[:-1]
        assert (V,) == f.shape
        
    elif args.solution:
        assert not args.vertex
        assert not args.face
        assert not args.tetra
        
        print 'Reading LCP solution archive',args.solution
        unarch = Unarchiver(args.solution)
        assert 'p' in unarch.data
        assert 'd' in unarch.data
        if args.dual:
            v = unarch.d
        else:
            v = unarch.p
        # Should be a multiple of (V+1)
        r = v.size % (V+1)
        assert 0 == r
        A = v.size / (V+1)
        print '\tFound',A-1,'actions'
        F = np.reshape(v,((V+1),A),order='F')

        if args.policy:
            f = np.argmax(F[:-1,1:],1)
            if args.ignore:
                f = f.astype(np.double)
                for i in args.ignore:
                    f[f == i] = np.nan
        else:
            assert args.action is not None
            assert 0 <= args.action < A
            f = F[:-1,args.action] # Crop oob
        assert (V,) == f.shape
    else:
        print "Mode not supported yet"
        quit()

    if args.log:
        f = np.log(np.abs(f) + 1e-25)

    if args.large:
        alpha_fn = lambda x: 0.25*(x)**2
        cmap = 'plasma'
    elif args.policy:
        alpha_fn = lambda x: 0.05
        cmap = 'jet'
    else:
        alpha_fn = lambda x: 0.25*(1-x)**2
        cmap = 'jet'

    bound = np.array([[-5,5],[-5,5],[-np.pi,np.pi]])
    for i in xrange(3):
        cut = np.array(bound)
        cut[i,:] = 0
        plot_mesh_slice(f,cut,
                        meshfile,
                        flat=True,
                        grid_points=320,
                        cmap=cmap)
        plt.title("Slice along D[{0}]=0.0".format(i))
        plt.show()
