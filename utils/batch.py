import multiprocessing as mp
from multiprocessing.dummy import Pool
import time
import numpy as np

def sleep_dummy(N):
    time.sleep(N)

def break_ndarray(A,n):
    """
    Break Nx? array A into nx? chunks
    """
    N = A.shape[0]
    chunks = []

    B = int(N / n)
    for i in xrange(B):
        start = i * n
        chunks.append(A[start:(start+n)])
    if B*n < N:
        # Leftovers
        assert(N - B*n < n)
        chunks.append(A[B*n:])
    return chunks

def batch_process(command,args,num_workers):
    start = time.time()
    pool = Pool(processes=num_workers)
    res = pool.map(command,args)
    res = None
    print 'Processed {0} jobs in {1}s'.format(len(args),
                                              time.time()-start)
    return res
