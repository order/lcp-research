import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import sys

from utils import make_points

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
    (_,base_file) = sys.argv
    nodes = read_node(base_file + ".node")
    faces = read_ele(base_file + ".ele")
    dist = read_sp_mat(base_file + ".grid")

    (R,C) = dist.shape
    print dist.shape
    print nodes.shape
    
    G = 150
    (P,(X,Y)) = make_points([np.linspace(-1,1,G)]*2,True)

    (N,D) = P.shape
    assert C == N
    assert R == (nodes.shape[0])

    nodes = nodes[:-1,:] # Strip OOB node

    # Random values
    #v = np.random.rand(R)
    v = np.hstack([np.sum(np.abs(nodes),1),2])
    Z = dist.T.dot(v);

    #plt.scatter(P[:,0],P[:,1],c=Z,s=15,lw=0,alpha=0.25);
    plt.pcolormesh(X,Y,np.reshape(Z,(G,G)),lw=0)
    
    plt.triplot(nodes[:,0],
                nodes[:,1],
                '-k',
                faces,alpha=0.25)
    plt.plot(nodes[:,0],
             nodes[:,1],
             '.k')
    
    plt.show()
    