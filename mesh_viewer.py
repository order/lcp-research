import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys

from utils import make_points
from utils.archiver import Unarchiver

def remove_comments(lines):
    cleaned = []
    for line in lines:
        clean = line.partition('#')[0]
        clean = clean.strip()
        if len(clean) > 0:
            cleaned.append(clean)
    return cleaned        

def read_node(node_file):
    with open(node_file) as FH:
        lines = FH.readlines()
    lines = remove_comments(lines)

    (n_vert,D,A,B) = map(int,lines[0].split())
    assert 2 == D # Dimension
    assert 0 == A # Attributes
    assert 0 == B # Boundary markers
    assert (n_vert + 1 ) == len(lines)
    
    vertices = np.empty((n_vert,2))
    for i in xrange(1,len(lines)):
        (v_id,x,y) = map(float,lines[i].split())
        vertices[i-1,0] = x
        vertices[i-1,1] = y
    return vertices

def read_ele(ele_file):
    with open(ele_file) as FH:
        lines = FH.readlines()
    lines = remove_comments(lines)    

    (n_faces,vert_per_face,A) = map(int,lines[0].split())
    assert 3 == vert_per_face
    assert 0 == A # Attributes
    assert (n_faces + 1 ) == len(lines)

    faces = np.empty((n_faces,vert_per_face))
    for i in xrange(1,len(lines)):
        sp_line = map(float,lines[i].split())
        assert(4 == len(sp_line))
        faces[i-1,:] = np.array(sp_line[1:])
    return faces

def read_sp_mat(sp_mat_file):
    data = np.fromfile(sp_mat_file)
    R = int(data[0])
    C = int(data[1])
    NNZ = int(data[2])
    assert (3 + 3*NNZ,) == data.shape

    data = np.reshape(data[3:],(NNZ,3))
    
    S = sps.coo_matrix((data[:,2],(data[:,0], data[:,1])),shape=(R,C))
    return S

if __name__ == "__main__":
    # Replace with argparse
    assert len(sys.argv) <= 3
    if 2 == len(sys.argv):
        (_,base_file) = sys.argv
    elif 3 == len(sys.argv):
        (_,base_file,soln_file) = sys.argv
        unarch = Unarchiver(soln_file)
        p = unarch.p
    else:
        print "Unknown arguments"

    flat = True
    mode = 'mesh' # Mesh only, value, log agg flow, policy
    G = 640 # Grid size for flat image
       
    nodes = read_node(base_file + ".node")
    faces = read_ele(base_file + ".ele")

    (N,d) = nodes.shape
    assert 2 == d
    M = p.size
    assert(0 == M % N)
    assert(M / N >= 3)

    F = np.reshape(p,(N,M/N),order='F')
    
    if mode == 'value':
        f = F[:,0]
        cmap = plt.get_cmap('jet')
    elif mode == 'policy':
        f = np.argmin(F[:,1:],axis=1)
        cmap = plt.get_cmap('veridis')
    elif mode == 'agg':
        f = np.log(np.sum(F[:,1:],axis=1))
        cmap = plt.get_cmap('plasma')
    elif mode == 'mesh':
        f = np.zeros(N)
        cmap = plt.get_cmap('Blues')
    else:
        print "Mode not recognized"
    assert(N == f.size)

    fig = plt.figure()
    if flat:
        (P,(X,Y)) = make_points([np.linspace(np.min(nodes[:,i]),
                                             np.max(nodes[:,i]),G)
                                 for i in [0,1]],True)
        Z = griddata(nodes,f,P,method='linear')
        Z = np.reshape(Z,(G,G))
        plt.pcolormesh(X,Y,Z,lw=0,cmap=cmap)
        if mode == 'mesh':
            plt.triplot(nodes[:,0],
                        nodes[:,1],
                        faces,'-k')
            plt.plot(nodes[:,0],nodes[:,1],'.k');

        else:
            plt.triplot(nodes[:,0],
                        nodes[:,1],
                        faces,'-k',alpha=0.2)
            plt.plot(nodes[:,0],nodes[:,1],'.k',alpha=0.2);

    else:
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(nodes[:,0],nodes[:,1],f,
                        triangles=faces,
                        cmap=cmap,
                        alpha=0.75)

    plt.show()
    
