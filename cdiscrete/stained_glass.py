import numpy as np
import sys

import itertools

from utils.archiver import Unarchiver
from utils import standardize

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def prep_lines(lines):
    lines = [line.split('#',1)[0].strip() for line in lines]
    lines = [line for line in lines if line]
    return list(reversed(lines))

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
    if no_function:      
        F = 'b'
    else:
        F = standardize(F) # Between 0 and 1
        assert F.size == vertices.shape[0]
    V = vertices.shape[0]

    percentile = kwargs.get('percentile',5)
    cmap = plt.get_cmap(kwargs.get('cmap','jet'))
    no_nodes = kwargs.get('no_nodes',False)
    no_mesh = kwargs.get('no_mesh',False)
    
    # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

             
    if not no_nodes:
        ax.scatter(vertices[:,0],
                   vertices[:,1],
                   vertices[:,2],
                   s=25,
                   c = F,
                   alpha=0.1,
                   lw=0,
                   cmap=cmap)
    else:
        #Need to manually set limits
        for (i,d) in zip(range(3),['x','y','z']):
            l = np.min(vertices[:,i])
            u = np.max(vertices[:,i])
            exec('ax.set_{0}lim3d({1},{2})'.format(d,l,u)) 

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
        cutoff = np.percentile(F,percentile)
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

                    if np.all(cutoff <= F[verts]):
                        # Skip if all vertices are greater
                        # than cutoff
                        continue
                
                    mean_F = np.mean(F[verts])
                    alpha = 0.15*(1-mean_F)**5
                
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

if __name__ == "__main__":    
    (_,meshfile,solnfile,field) = sys.argv
    print "#"*30
    print "##       STAINED GLASS      ##"
    print "## tetrahedral mesh viewer  ##"
    print "#"*30
    print ""
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

    print 'Reading in',field,'from archive',solnfile
    unarch = Unarchiver(solnfile)
    f = eval('unarch.' + field)
    print '\tFound vector:',f.shape
    (N,) = f.shape
    if N == V+1:
        print '\tCropping OOB node'
        f = f[:-1]
    assert((V,) == f.shape)
    
    plot_mesh(f,*tet_mesh,cmap='jet',
              no_nodes=True,
              no_mesh=True,
              percentile=10)
    plt.show()

    
