import bases
import matplotlib.pyplot as plt

import numpy as np

n = 25
D = 3
K = int(np.sqrt(n**D))

grid = np.linspace(0,1,n)
meshes = np.meshgrid(*([grid]*D))
flat_meshes = [x.flatten() for x in meshes]
points = np.column_stack(flat_meshes)

fourier = bases.RandomFourierBasis(num_basis=K,
            scale=1.0)
basis_gen = bases.BasisGenerator(fourier)
Phi = basis_gen.generate(points)


