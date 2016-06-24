import numpy as np
import matplotlib.pyplot as plt
import linalg

X = np.random.rand(5,3)
X = np.hstack([X, np.zeros((5,1))])
print X

B = linalg.orthonorm(X)

print B.T.dot(B)

