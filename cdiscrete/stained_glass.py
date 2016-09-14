import numpy as np
import sys

import itertools

from utils.archiver import Unarchiver
from utils import standardize

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def strip_comments(lines):
    lines = [line.split('#',1)[0].strip() for line in lines]
    lines = [line for line in lines if line]
    return lines

def read_medit_mesh(filename):
    
    # Read in file in INRIA's .medit format
    FH = open(filename + '.mesh',"r")
    lines = FH.readlines()
    lines = strip_comments(lines)

    I = 0
    names = ['vertices','edges','triangles','tetrahedra']
    objects = {}
    for name in names:
        while name not in lines[I].lower():
            I += 1
        assert name in lines[I].lower()
        I += 1
        n = int(lines[I])
        I += 1
        # Ignore boundary marker information
        objs = [map(float,x.split()[:-1]) for x in lines[I:(I+n)]]
        objs = np.array(objs) 
        exec(name + ' = objs')
        I += n

    return vertices,edges,triangles,tetrahedra

def plot_mesh(F,vertices,edges,triangles,tetrahedra,**kwargs):
    no_function = (F is None)
    if no_function:      
        F = 'b'
    else:
        F = standardize(F) # Between 0 and 1
        assert F.size == vertices.shape[0]

    cmap = plt.get_cmap(kwargs.get('cmap','jet'))
    
    # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0],
               vertices[:,1],
               vertices[:,2],
               s=35,
               c = F,
               cmap=cmap)

    # Build line collection
    segs = []
    seg_set = set()
    V = vertices.shape[0]
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
    linecolors = 0.1 * np.ones((S,4)) # Dark gray
    print 'Found {0} line segments'.format(S)
    seg_collection = Line3DCollection(segs,colors=linecolors)
    ax.add_collection3d(seg_collection)

    # Build a poly collection of faces
    # This makes for a "stained glass" look
    poly = []
    poly_set = set()
    obj_groups = [x.astype(np.integer) for x in [triangles,tetrahedra]]
    facecolors = []
    for (I,objs) in enumerate(obj_groups):
        if objs is None:
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
                triangle = [vertices[x,:] for x in verts]
                poly.append(triangle)
                # Color with the mean vertex color
                if no_function:
                    color = np.zeros(4)
                else:
                    mean_F = np.mean(F[verts])
                    color = list(cmap(mean_F))
                    color[3] = 0.5*(1 - mean_F)**2
                facecolors.append(color)
    P = len(poly)
    print 'Found {0} triangles'.format(P)
    edgecolors = np.zeros((P,4))
    poly_collection = Poly3DCollection(poly,
                                       facecolors=facecolors,
                                       edgecolors=edgecolors)
    ax.add_collection3d(poly_collection)

if __name__ == "__main__":    
    (_,filename) = sys.argv

    (base,ext) = filename.rsplit('.',1)
    assert ext == 'mesh'
    tet_mesh = read_medit_mesh(base)


    unarch = Unarchiver("test.out")
    dist = unarch.dist
    (NN,G) = dist.shape
    v = np.random.rand(NN)

    v[-1] = 0 # OOB notes
    V = dist.T.dot(v)

    plot_mesh(v[:-1],*tet_mesh,cmap='plasma')
    #plot_mesh(None,*tet_mesh)
    plt.show()

    
