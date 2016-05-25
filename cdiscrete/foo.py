import numpy as np
import h5py

M = np.random.rand(500,15);
f = h5py.File("test.h5",'w')
dset = f.create_dataset("foo", data=M)
f.close()
