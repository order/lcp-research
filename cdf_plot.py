import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import cdf_points


if __name__ == '__main__':
    files = sys.argv[1:]
    for filename in files:
        data = np.load(filename)
        (x,F) = cdf_points(data)
        plt.plot(x,F)
plt.legend(files,loc='best')
plt.show()
