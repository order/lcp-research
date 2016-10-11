import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.archiver import *
from utils import make_points
from utils.neighbours import *

import os,re


if __name__ == '__main__':
    filedir = "/home/epz/data/minop_value_vs_flow_free/"
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


    grid = [np.linspace(np.min(data[:,i]),np.max(data[:,i]),256)
            for i in xrange(2)]
    (P,(X,Y)) = make_points(grid,True)

    
    #Z = smooth(P,data[:,:2],data[:,2],3)
    fig = plt.figure()
    for (i,p) in enumerate([5,50,95]):
        plt.subplot(2,2,i+1)
        fn = lambda x: np.percentile(x,p)

        # 2 is residual, 3 is iter count
        Q = local_fn(fn,P,data[:,:2],data[:,2],2)
        Q = np.reshape(Q,X.shape)
        Qm = ma.masked_where(np.isnan(Q),Q)
        plt.pcolormesh(X,Y,Qm)
        plt.colorbar()
        plt.title("Percentile " + str(p))
        plt.xlabel('# Value Basis')
        plt.ylabel('# Flow Basis')
    plt.suptitle('Basis number vs. Residual')
    plt.show()
    

