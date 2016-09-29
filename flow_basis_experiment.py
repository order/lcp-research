import numpy as np
import matplotlib.pyplot as plt

from utils import make_points
from utils.archiver import Unarchiver
import sys,os

smoother = [0.05,0.2,0.5,0.2,0.05]

def plot_data(idx,data):
    (N,R) = data.shape

    IDX = np.tile(np.reshape(idx,(N,1)),(1,R))

    plt.plot(IDX.ravel(),data.ravel(),'.')

    assert(idx.size >=5)

    p25 = np.convolve(np.percentile(data,25,1),smoother,'valid')
    p50 = np.convolve(np.median(data,1),smoother,'valid')
    p75 = np.convolve(np.percentile(data,75,1),smoother,'valid')
    plt.plot(idx[2:-2],p25,'r--',lw=2)
    plt.plot(idx[2:-2],p50,'r-',lw=2)
    plt.plot(idx[2:-2],p75,'r--',lw=2)
    return p50

def plot_trends(idx,trends):
    for t in trends.values():
        plt.plot(idx[2:-2],t)
    plt.legend(trends.keys())

if __name__ == "__main__":

    (_,dir_name) = sys.argv
    files = os.listdir(dir_name)
    files = filter(lambda x: ".exp_res" in x, files)

    trends = {}
    num_basis = None
    for f in files:
        unarch = Unarchiver(dir_name + f)
        assert hasattr(unarch,"res_data")
        assert hasattr(unarch,"num_basis")
        if num_basis is None:
            num_basis = unarch.num_basis
        else:
            assert np.all(num_basis == unarch.num_basis)
            
        data = unarch.res_data
        plt.figure()
        plt.title(f)
        trends[f] = plot_data(num_basis,data)
    plt.figure()
    plot_trends(num_basis,trends)
    plt.show()
