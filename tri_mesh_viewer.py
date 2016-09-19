import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.archiver import read_shewchuk

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


def get_solution(nodes,faces,fn,mode):
    (N,_) = nodes.shape
    (M,) = fn.shape
    if 0 == M % (N+1):
        assert M / (N+1) >= 3
        F = np.reshape(fn,(N+1,M/(N+1)),order='F')
        F = F[:-1,:]
    else:
        assert 0 == M % N
        assert M / N >= 3
        F = np.reshape(fn,(N,M/N),order='F')

    if mode == 'value':
        f = F[:,0]
        cmap = plt.get_cmap('spectral_r')
    elif mode == 'policy':
        f = np.argmin(F[:,1:],axis=1)
        cmap = plt.get_cmap('spectral_r')
    elif mode == 'agg':
        f = np.sum(F[:,1:],axis=1)
        cmap = plt.get_cmap('jet')
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
    plt.gca()
    plt.pcolormesh(X,Y,Z,lw=0,cmap=cmap)
    plt.triplot(nodes[:,0],nodes[:,1],faces,'-k',alpha=0.25)
    plt.colorbar()
    
def plot_faces(nodes,faces,fn,cmap=None):
    fn = np.ma.array(fn,mask=~np.isfinite(fn))
    assert(fn.size == faces.shape[0])
    if cmap is None:
        cmap = plt.get_cmap('jet')
    cmap.set_bad('w',1.)
    plt.gca()
    plt.tripcolor(nodes[:,0],nodes[:,1],faces,facecolors=fn,
                  edgecolor='k',cmap=cmap)
    plt.colorbar()

def plot_vertices_3d(nodes,faces,fn,cmap=None):
    assert(fn.size == nodes.shape[0])
    if cmap is None:
        cmap = plt.get_cmap('jet')
    ax = plt.gca(projection='3d')
    ax.plot_trisurf(nodes[:,0],nodes[:,1], fn,
                    triangles=faces, cmap=cmap, alpha=0.75)

def plot_bare_mesh(filename):
    fig = plt.gca()
    (nodes,faces) = read_shewchuk(filename)
    plt.triplot(nodes[:,0],nodes[:,1],faces,'-k')
    plt.plot(nodes[:,0],nodes[:,1],'.k')

def plot_solution_mesh(mesh_filename,soln_filename,mode,log):
    (nodes,faces) = read_shewchuk(mesh_filename)
    (N,nd) = nodes.shape

    unarch = Unarchiver(soln_filename)
    (fn_data,cmap) = get_solution(nodes,faces,unarch.p,mode)
    if log:
        fn_data = np.log(np.abs(fn_data) + 1e-15)        
    assert(N == fn_data.size)

    fig = plt.gca()
    plot_vertices(nodes,faces,fn_data,cmap)

def plot_raw_binary_mesh(mesh_filename,raw_bin_name,log=False):
    (nodes,faces) = read_shewchuk(mesh_filename)
    (N,nd) = nodes.shape
    (V,vd) = faces.shape

    # Read in file
    ext = raw_bin_name.split('.')[-1]
    if ext == 'uvec':
        fn_data = np.fromfile(raw_bin_name,dtype=np.integer)
    else:
        fn_data = np.fromfile(raw_bin_name,dtype=np.double)
        
    if log:
        fn_data = np.log(np.abs(fn_data))
        fn_data[~np.isfinite(fn_data)] = np.nan
    assert (N == fn_data.size) or (V == fn_data.size)
    
    plt.gca()
    if (N == fn_data.size):
        plot_vertices(nodes,faces,fn_data)
    else:
        plot_faces(nodes,faces,fn_data)

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
        # Raw binary information
        plt.title(args.raw_binary)
        plot_raw_binary_mesh(args.base_file,
                             args.raw_binary,args.log)
        plt.show()
        quit()

    if args.raw_binary is None:
        # Read primal from archive
        plot_solution_mesh(args.base_file,
                           args.solution,
                           args.mode,args.log)
        plt.title(args.solution + ', ' + args.mode)
        plt.show()
        quit()


    
