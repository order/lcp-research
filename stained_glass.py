import numpy as np

import itertools

from utils.archiver import Unarchiver
from utils import standardize

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import time
import argparse
import sys
import os.path

def prep_lines(lines):
    lines = [line.split('#',1)[0].strip() for line in lines]
    lines = [line for line in lines if line]
    return list(reversed(lines))

# Move to another file
def read_medit_mesh(filename):
    # Read in file in INRIA's .medit format
    FH = open(filename,"r")
    lines = FH.readlines()
    lines = prep_lines(lines)

    names = ['vertices','edges','triangles','tetrahedra']
    for name in names:
        exec(name + ' = []')
    while lines:
        line = lines.pop()
        while lines and line.lower() not in names:
            line = lines.pop()
        if not lines:
            break
        name = line.lower()
        assert name in names
        
        n = int(lines.pop())
        objs = []
        for _ in xrange(n):
            line = lines.pop()
            tokens = line.split()
            obj = map(float,tokens[:-1]) # Ignore boundary marker information
            objs.append(obj)
        objs = np.array(objs) 
        exec(name + ' = objs')

    return vertices,edges,triangles,tetrahedra

def plot_mesh(F,vertices,edges,triangles,tetrahedra,**kwargs):
    no_function = (F is None)
    if not no_function:      
        std_F = standardize(F) # Between 0 and 1
        assert F.size == vertices.shape[0]
    V = vertices.shape[0]

    cmap = plt.get_cmap(kwargs.get('cmap','jet'))
    no_nodes = kwargs.get('no_nodes',False)
    no_mesh = kwargs.get('no_mesh',False)
    alpha_fn = kwargs.get('alpha_fn',lambda x : 0.1)
    
    # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
         
    p = ax.scatter(vertices[:,0],
                   vertices[:,1],
                   vertices[:,2],
                   s=25,
                   c = F,
                   alpha=0.25,
                   lw=0,
                   cmap=cmap)
    fig.colorbar(p)

    # Build line collection
    if not no_mesh:
        segs = []
        seg_set = set()
        obj_groups = [x.astype(np.integer) for x in [edges,triangles,tetrahedra]]
        for objs in obj_groups:
            if objs is None:
                continue
            (N,D) = objs.shape
            for i in xrange(N):
                for verts in itertools.combinations(objs[i,:],2):
                    verts = [int(v) - 1 for v in verts]
                    for v in verts:
                        assert 0 <= v < V
                    key = tuple(verts)
                    if key in seg_set:
                        continue
                    seg_set.add(key)
                    segs.append([vertices[x,:] for x in verts])
        S = len(segs)
        linecolors = [0.5,0.5,0.5,0.05] # Dark gray
        print 'Plotting {0} line segments'.format(S)
        seg_collection = Line3DCollection(segs,colors=linecolors)
        ax.add_collection3d(seg_collection)

    # Build a poly collection of faces
    # This makes for a "stained glass" look
    if not no_function:
        poly = []
        poly_set = set()
        obj_groups = [x.astype(np.integer) for x in [triangles,tetrahedra]]
        facecolors = []
        for (I,objs) in enumerate(obj_groups):
            if objs is None or no_function:
                continue
            (N,D) = objs.shape
            for i in xrange(N):
                for verts in itertools.combinations(objs[i,:],3):
                    verts = [int(v) - 1 for v in verts]
                    for v in verts:
                        assert 0 <= v < V
                    key = tuple(verts)
                    if key in poly_set:
                        continue
                    poly_set.add(key)

                    if np.any(np.isnan(std_F[verts])):
                        continue
                    
                    mean_F = np.mean(std_F[verts])
                    alpha = alpha_fn(mean_F)
                    
                    if alpha < 0.025:
                        # Skip if all vertices are greater
                        # than cutoff
                        continue                             
                    triangle = [vertices[x,:] for x in verts]
                    poly.append(triangle)
                    # Color with the mean vertex color
                    color = list(cmap(mean_F))
                    color[3] = alpha
                    facecolors.append(color)
        P = len(poly)
        print 'Plotting {0} triangles'.format(P)
        edgecolors = np.zeros((P,4))
        poly_collection = Poly3DCollection(poly,
                                           facecolors=facecolors,
                                           edgecolors=edgecolors)
        ax.add_collection3d(poly_collection)

def get_macros():
    macros = {}
    macros['value']

if __name__ == "__main__":    
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
    
    plot_mesh(f,*tet_mesh,cmap=cmap,
              no_mesh=True,
              alpha_fn=alpha_fn)
    plt.show()

    
