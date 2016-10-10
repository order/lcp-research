import multiprocessing as mp
import subprocess
import itertools
import sys,time,os

class task(object):
    def __init__(self,prog,**kwargs):
        self.base = prog
        self.params = kwargs
        
    def get_cmd(self):
        cmd = [self.base]
        for (k,v) in self.params.items():
            if 1 == len(k):
                cmd.append('-{0}{1}'.format(k,v))
            else:
                cmd.append('--{0} {1}'.format(k,v))
    
        return ' '.join(cmd)
    
    def run(self):
        fnull = open(os.devnull,'w')
        return subprocess.call(self.get_cmd(),
                               stdout=fnull,
                               stderr=subprocess.STDOUT,
                               shell=True,)

def build_task_queue(tgf,params,settings):
    P = len(params)
    queue = []
    for (I,setting) in enumerate(settings):
        queue.append(tgf(I,**dict(zip(params,setting))))
    return queue
    
def build_grid_task_queue(tgf,**kwargs):
    # tgf = task generating function
    params = kwargs.keys()
    queue = []
    for (I,x) in enumerate(itertools.product(*kwargs.values())):
        queue.append(tgf(I,**dict(zip(params,x))))
    return queue

def task_processer(T):
    assert isinstance(T,task)
    return T.run()

def run_task_queue(queue,cores):
    pool = mp.Pool(cores)
    N = len(queue)
    for (i,res) in enumerate(pool.imap_unordered(task_processer,queue)):
        sys.stderr.write('\rFinished {0} tasks ({1:.1f}%)'.format(i+1,100.0*float(i+1)/float(N)))
    print


