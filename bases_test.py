import bases
import numpy as np

import matplotlib.pyplot as plt

def test():
    N = 10
    K = 3
    
    X = np.linspace(0,2*np.pi,N)
    P = np.hstack([X,np.zeros(K)])[:,np.newaxis]

    fourier = bases.RandomFourierBasis()
    basis_gen = bases.BasisGenerator(fourier)
    B = basis_gen.generate(P,N+K,special_points=range(N,N+K))

    
    plt.imshow(B,interpolation='none')
    plt.show()


if __name__ == '__main__':
    test()
