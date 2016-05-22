import numpy as np
import cdiscrete as cd
from discrete import make_points
import matplotlib.pyplot as plt

vals = np.random.rand(5,2)

low = np.zeros(2)
high = np.ones(2)
num = np.ones(2,dtype='uint64')

G = 50
cuts = np.linspace(0,1,G)
XI,YI = np.meshgrid(cuts,cuts)
P = make_points([cuts]*2)

I = cd.argmax_interpolate(vals,P,low,high,num)

ZI = np.reshape(I,(G,G))
plt.pcolor(XI,YI,ZI)
plt.colorbar()
plt.show()
