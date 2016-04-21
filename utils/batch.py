import multiprocessing as mp
from multiprocessing import Pool
import time
import numpy as np

def sleep_dummy(N):
    time.sleep(N)

def batch_process(command,args,num_workers):
    start = time.time()
    pool = Pool(processes=num_workers)
    res = pool.map(command,args)
    print 'Processed {0} jobs in {1}s'.format(len(args),
                                              time.time()-start)
    pool.close()
    return res
