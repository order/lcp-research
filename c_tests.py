import cdiscrete as cd
import matplotlib.pyplot as plt

out = cd.simulate()

plt.plot(out.points[0,0,:],
         out.points[0,1,:],'.-b')
plt.show()
