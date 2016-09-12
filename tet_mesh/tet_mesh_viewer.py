import numpy as np
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
        # Ignore boundary information
        objs = [map(float,x.split()[:-1]) for x in lines[I:(I+n)]]
        objs = np.array(objs)
        exec(name + ' = objs')
        I += n

    return vertices,edges,triangles,tetrahedra

def read_tetgen_mesh(base_name):
    nodes = read_tetgen_file(base_name + '.node',np.double)
    exts = ['edge','face','elem']
    out = [nodes]
    for ext in exts:
        ret = read_tetgen_file(base_name + '.' + ext,np.integer)
        out.append(ret)
    return tuple(out)

def read_tetgen_file(filename,dtype):
    try:
        FH = open(filename,'r')
        lines = FH.readlines()
        FH.close()
    except Exception:
        return None    

    lines = strip_comments(lines)
    (N,D) = map(lines[0].split())[:2]
    assert(N+1 == len(lines))
    
    ret = np.empty((N,D),dtype=dtype)
    for i in xrange(1,N+1):
        tokens = map(float,lines[i].split())
        assert int(tokens[0]) == i
        ret[i-1,:] = tokens[1:(D+1)]
    return ret

def plot_mesh(vertices,edges,triangles,tetrahedra):
    
    # Color map
    cmap = plt.get_cmap('plasma')
    
    # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = np.sqrt(np.sum(vertices[:,:3]**2,1))
    alpha = (np.max(norm) - norm)
    alpha = (alpha / np.max(alpha))
    colors = cmap(alpha)
    #colors[:,3] = 0.5 + 0.5*alpha
    ax.scatter(vertices[:,0],
               vertices[:,1],
               vertices[:,2],
               s=35,c=colors)

    # Build line collection
    segs = []
    colors = []
    # Edges are black, faces are red, cells are green.
    color_set = [[0,0,0,0.5],
                 [0.5,0,0,0.5],
                 [0,0.5,0,0.5]]
    seg_set = set()
    V = vertices.shape[0]
    obj_groups = [x.astype(np.integer) for x in [edges,triangles,tetrahedra]]
    for (I,objs) in enumerate(obj_groups):
        if objs is None:
            continue

        (N,D) = objs.shape
        for i in xrange(N):
            for j in xrange(D-1):
                idx_a = int(objs[i][j]) - 1 # Vertex index A
                assert(0 <= idx_a < V)
                for k in xrange(j+1,D):
                    idx_b = int(objs[i][k]) - 1 # Vertex index B
                    assert(0 <= idx_b < V)
                    
                    key = tuple(sorted([idx_a,idx_b]))
                    if key in seg_set:
                        continue
                    seg_set.add(key)
                    
                    segs.append((vertices[idx_a,:],
                                 vertices[idx_b,:]))
                    colors.append(color_set[I])
               
    S = len(segs)
    print 'Found {0} line segments'.format(S)
    seg_collection = Line3DCollection(segs,colors=colors)
    ax.add_collection3d(seg_collection)
    plt.show()

if __name__ == "__main__":
    (_,filename) = sys.argv

    (base,ext) = filename.rsplit('.',1)
    
    if ext == 'mesh':
        tet_mesh = read_medit_mesh(base)
    else:
        assert ext in ['ele','node']
        tet_mesh = read_tetgen_mesh(base)
    plot_mesh(*tet_mesh)
