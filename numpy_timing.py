import time
import numpy as np

X = np.random.rand(10000,10000)

start = time.time()
Y = np.matrix(X)
print 'Elapsed:', time.time() - start