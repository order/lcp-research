from utils.archiver import *
import matplotlib.pyplot as plt

x = np.arange(15,dtype=np.double)

arch = Archiver(x=x);
arch.write('test.foo')

unarch = Unarchiver('test.foo')
print unarch.x
