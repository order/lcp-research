import numpy as np
import cdiscrete as cd
from discrete import make_points
import matplotlib.pyplot as plt


N = 10
low = -np.ones(2,dtype='double')
high = np.ones(2,dtype='double')
num = N*np.ones(2,dtype='uint64')


cuts = np.linspace(-1,1,N+1)
P = make_points([cuts]*2)
c = np.array([1,-0.2])
v = np.array((P.dot(c)) < 0.5,dtype='double')
v = np.hstack([v,-1])


E = 51
cuts = np.linspace(-1.1,1.1,E)
P = make_points([cuts]*2)
XI,YI = np.meshgrid(cuts,cuts,indexing='ij')

I = cd.interpolate(v,P,low,high,num)
Img = np.reshape(I,(E,E))

plt.pcolor(XI,YI,Img)
plt.scatter(P[:,0],P[:,1],c=I);
plt.show()
