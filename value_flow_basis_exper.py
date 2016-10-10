import sys
import numpy as np
from utils.multi_core import *

# Task generating function
def minop_tgf(I,**kwargs):
    T = task('./cdiscrete/minop_approx',**kwargs)
    T.params['o'] = '/home/epz/data/minop/vf_' + str(I)
    return T


if __name__ == '__main__':
    (_,N) = sys.argv
    N = int(N)
    max_cores = mp.cpu_count()    
    print "Using",max_cores,"cores"

    params = ['v','f']
    settings = []
    for i in xrange(N):
        setting = (np.random.randint(5,100),np.random.randint(5,100))
        settings.append(setting)
    
    queue = build_task_queue(minop_tgf, params, settings)
    print "Generated tasks:",len(queue)
    print "Running..."
    #print '\n'.join([t.get_cmd() for t in queue])
    run_task_queue(queue,max_cores-1)
    print "Done."
