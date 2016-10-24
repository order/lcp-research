import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.archiver import *
from utils import make_points
from utils.neighbours import *

import os,re


if __name__ == '__main__':
    # Make sure that
    filedir = "/home/epz/data/minop/vf_bumpy_free/"
    summary_file = filedir + "summary.npy"

    

    if os.path.isfile(summary_file):
        print "Loading summary file"
        data = np.load(summary_file)
    else:
        print "Reading data from instance files"
        filenames = os.listdir(filedir)
        pattern = re.compile("vf_(\d+)\.exp_res")
        data = []
        for filename in filenames:
            res = pattern.match(filename)
            if res is None:
                continue
            i = int(res.group(1))
            unarch = Unarchiver(filedir+filename)
            data.append(unarch.data)

        data = np.array(data)
        np.save(filedir + "summary",data)


    grid = [np.linspace(np.min(data[:,i]),np.max(data[:,i]),128)
            for i in xrange(2)]
    (P,(X,Y)) = make_points(grid,True)

    
    #Z = smooth(P,data[:,:2],data[:,2],3)
    fig = plt.figure()
    for (i,p) in enumerate([5,50,95]):
        plt.subplot(2,2,i+1)
        fn = lambda x: np.percentile(x,p)

        # 2 is residual, 3 is iter count
        Q = local_fn(fn,P,data[:,:2],data[:,2],1.5)
        Q = np.reshape(Q,X.shape)
        Qm = ma.masked_where(np.isnan(Q),Q)
        plt.pcolormesh(X,Y,Qm)
        plt.colorbar()
        plt.title("Percentile " + str(p))
        plt.xlabel('# Value Basis')
        plt.ylabel('# Flow Basis')
    plt.suptitle('Basis number vs. Residual')

    fig = plt.figure()
    for (i,p) in enumerate([5,50,95]):
        plt.subplot(2,2,i+1)
        fn = lambda x: np.percentile(x,p)

        # 2 is residual, 3 is iter count
        Q = local_fn(fn,P,data[:,:2],data[:,3],1.5)
        Q = np.reshape(Q,X.shape)
        Qm = ma.masked_where(np.isnan(Q),Q)
        plt.pcolormesh(X,Y,Qm)
        plt.colorbar()
        plt.title("Percentile " + str(p))
        plt.xlabel('# Value Basis')
        plt.ylabel('# Flow Basis')
    plt.suptitle('Basis number vs. Iteration')

    """
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    for (i,p) in enumerate([5,50,95]):
        fn = lambda x: np.percentile(x,p)
        # 2 is residual, 3 is iter count
        Q = local_fn(fn,P,data[:,:2],np.log(data[:,2]),1.5)
        Q = np.reshape(Q,X.shape)
        Qm = ma.masked_where(np.isnan(Q),Q)
        if p != 50:
            ax.plot_surface(X,Y,Qm,alpha=0.25,
                            cstride=2,rstride=2,lw=0)
        else:
            ax.plot_surface(X,Y,Qm,alpha=0.5,
                            cstride=2,rstride=2,lw=0)
    """     
    plt.show()
    

