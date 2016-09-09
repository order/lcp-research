import numpy as np
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

def read_tet_mesh(filename):
    # Read in file in INRIA's .mesh format
    FH = open(filename,"r")
    lines = FH.readlines()

    I = 0
    names = ['vertices','triangles','tetrahedra']
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

    V = vertices.shape[0]
    Tr = triangles.shape[0]
    Te = tetrahedra.shape[0]
    
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

    # Build tetrahedral line collection
    segs = []
    colors = []
    seg_set = set()
    for tri in triangles:
        for i in xrange(0,2):
            vi = int(tri[i])-1
            assert(0 <= vi < V)
            for j in xrange(i,3):
                vj = int(tri[j])-1
                assert(0 <= vj < V)
                
                key = tuple(sorted([vi,vj]))
                if key in seg_set:
                    continue
                seg_set.add(key)
                
                segs.append((vertices[vi,:],
                            vertices[vj,:]))
                colors.append([0.75,0,0,0.5])
    for tet in tetrahedra:
        for i in xrange(0,3):
            vi = int(tet[i])-1
            assert(0 <= vi < V)
            for j in xrange(i,4):
                vj = int(tet[j])-1
                assert(0 <= vj < V)
                key = tuple(sorted([vi,vj]))
                if key in seg_set:
                    continue
                seg_set.add(key)
                
                segs.append((vertices[vi,:],
                            vertices[vj,:]))
                colors.append([0,0.75,0,0.25])                
    S = len(segs)
    print 'Found {0} line segments'.format(S)
    ln_coll = Line3DCollection(segs,colors=colors)
    ax.add_collection3d(ln_coll)
    plt.show()

if __name__ == "__main__":
    (_,filename) = sys.argv
    tet_mesh = read_tet_mesh(filename)
