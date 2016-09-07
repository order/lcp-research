import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import argparse

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

def get_solution(nodes,faces,fn,mode):
    (N,_) = nodes.shape
    (M,) = fn.shape
    assert(0 == M % N)
    assert(M / N >= 3)
    
    F = np.reshape(fn,(N,M/N),order='F')
    
    if mode == 'value':
        f = F[:,0]
        cmap = plt.get_cmap('spectral')
    elif mode == 'policy':
        f = np.argmin(F[:,1:],axis=1)
        cmap = plt.get_cmap('Paired')
    elif mode == 'agg':
        f = np.sum(F[:,1:],axis=1)
        cmap = plt.get_cmap('plasma')
    else:
        print "Mode not recognized"
    assert(N == f.size)
    return (f,cmap)

def plot_vertices(nodes,faces,fn,cmap=None,G=640):
    fn = np.ma.array(fn,mask=~np.isfinite(fn))
    assert(fn.size == nodes.shape[0])
    if cmap is None:
        cmap = plt.get_cmap('jet')
    cmap.set_bad('w',1.)

    (P,(X,Y)) = make_points([np.linspace(np.min(nodes[:,i]),
                                         np.max(nodes[:,i]),G)
                             for i in [0,1]],True)
    Z = griddata(nodes,fn,P,method='linear')
    Z = np.reshape(Z,(G,G))
    plt.pcolormesh(X,Y,Z,lw=0,cmap=cmap)
    plt.triplot(nodes[:,0],nodes[:,1],faces,'-k',alpha=0.25)
    plt.colorbar()
    plt.show()
    
def plot_faces(nodes,faces,fn,cmap=None):
    fn = np.ma.array(fn,mask=~np.isfinite(fn))
    assert(fn.size == faces.shape[0])
    if cmap is None:
        cmap = plt.get_cmap('jet')
    cmap.set_bad('w',1.)
    plt.tripcolor(nodes[:,0],nodes[:,1],faces,facecolors=fn,edgecolor='k')
    plt.colorbar()
    plt.show()

def plot_vertices_3d(nodes,faces,fn,cmap=None):
    assert(fn.size == nodes.shape[0])
    if cmap is None:
        cmap = plt.get_cmap('jet')
    fig = plt.gcf()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(nodes[:,0],nodes[:,1], fn,
                    triangles=faces, cmap=cmap, alpha=0.75)
    plt.show()

def read_shewchuk(filename):
    nodes = read_node(filename + ".node")
    faces = read_ele(filename + ".ele")   
    (N,nd) = nodes.shape
    (V,vd) = faces.shape
    assert 2 == nd
    assert 3 == vd

    return nodes,faces

def plot_bare_mesh(filename):
    fig = plt.gca()
    (nodes,faces) = read_shewchuk(filename)
    plt.triplot(nodes[:,0],nodes[:,1],faces,'-k')
    plt.plot(nodes[:,0],nodes[:,1],'.k')   


if __name__ == "__main__":
    # Replace with argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('base_file', metavar='F', type=str,
                        help='base file name for .node and .ele files')
    parser.add_argument('-s','--solution', default=None,
                        help='solution file')
    parser.add_argument('-m','--mode',default='value',
                        help="value,agg, or policy")
    parser.add_argument('-r','--raw_binary', default=None,
                        help='raw binary function file')
    parser.add_argument('-l','--log',action="store_true",
                        help='apply log(abs(.)) transform')
    args = parser.parse_args()

    ############################################
    # Read in option function information
    # Make sure that solution and binary are both selected
    assert (args.solution is None) or (args.raw_binary is None)
    if (args.solution is None) and (args.raw_binary is None):
        plot_bare_mesh(args.base_file)
        plt.title('Mesh')
        plt.show()
        quit()

    if args.solution is None:
        plt.figure()
        plt.title(args.raw_binary)
        # Raw binary information
        fn_data = np.fromfile(args.raw_binary)
        if args.log:
            fn_data = np.log(np.abs(fn_data))
            fn_data[~np.isfinite(fn_data)] = np.nan
        assert (N == fn_data.size) or (V == fn_data.size)
        if (N == fn_data.size):
            plot_vertices(nodes,faces,fn_data)
        else:
            plot_faces(nodes,faces,fn_data)
        quit()

    if args.raw_binary is None:
        # Read primal from archive
        unarch = Unarchiver(args.solution)
        (fn_data,cmap) = get_solution(nodes,faces,unarch.p,args.mode)
        if args.log:
            fn_data = np.log(np.abs(fn_data))
        plt.figure()
        plt.title(args.solution + ', ' + args.mode)
        
        assert(N == fn_data.size)
        plot_vertices(nodes,faces,fn_data,cmap)
        quit()


    
