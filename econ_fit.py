import csv
import numpy as np
import scipy.sparse as sps
import scipy
import scipy.sparse
import scipy.sparse.linalg as spsl
import sys

import matplotlib.pyplot as plt

FIELD = 'BID'  # Field name to base series on

if __name__ == "__main__":
    assert 2 == len(sys.argv), "Please provide input CRSP stock data CSV file"
    (_,filename) = sys.argv


    # Read in the data
    data = []
    with open(filename,'r') as fh:
        reader = csv.reader(fh)

        # Get the index of the appropriate field
        fields = reader.next()
        print "Found {} fields".format(len(fields))
        print "Fields: {}".format(fields)
        assert FIELD in fields, "{} not found in fields".format(FIELD)
        idx = fields.index(FIELD)
        
        for row in reader:
            data.append(float(row[idx]))
    data = np.array(data)
    n = data.size
    
    b = data[:-1] - data[1:] # Difference; shape: (n-1,)

    num_points = 512
    grid_search = np.zeros((num_points,2))
    grid_search[:,0] = np.linspace(-1, 1, num=num_points)
    for i in xrange(num_points):
        theta = grid_search[i,0]
        A = sps.eye(n-1) + sps.diags(theta * np.ones(n-2), -1)
        ret = spsl.lsqr(A,b)
        grid_search[i,1] = np.linalg.norm(ret[0])

        print grid_search[i,:]

    
    plt.figure()
    plt.semilogy(grid_search[:,0],grid_search[:,1])
    plt.legend(['l1','l2'])
    plt.show()
            
