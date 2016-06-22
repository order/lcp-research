import numpy as np
import matplotlib.pyplot as plt
from utils import Marshaller

marsh = Marshaller()
(x,f) = marsh.load('cdiscrete/foo.dat')

plt.plot(x,f)
plt.show()
