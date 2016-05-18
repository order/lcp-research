import cdiscrete as cd
import numpy as np

x = np.arange(6,dtype=np.int).reshape(2,3)
print 'PY Original:\n', x
y = cd.incr(x)
print 'PY Original (redux):\n', x
print 'PY Incr:\n', y
