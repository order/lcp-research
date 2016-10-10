import numpy as np
import matplotlib.pyplot as plt
from utils.archiver import *
import os,re

if __name__ == '__main__':
    filedir = "/home/epz/data/minop/"
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

    plt.figure()
    plt.scatter(data[:,0],
                data[:,1],
                c=data[:,2],
                s=50,lw=0,alpha=0.25)
    plt.xlabel('Value basis size')
    plt.ylabel('Flow basis size')
    plt.colorbar()
    plt.show()
    
