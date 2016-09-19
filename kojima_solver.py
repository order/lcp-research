import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from lcp import LCPObj
from utils.archiver import Unarchiver,Archiver

from solvers import augment_lcp,KojimaIPIterator,IterativeSolver
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

import time
import argparse
import sys
import os.path


###############################################
# Solvers
def kojima_solve(lcp,**kwargs):
    thresh   = kwargs.get('thresh',1e-8)
    max_iter = kwargs.get('max_iter',1000)
    val_reg  = kwargs.get('val_reg',1e-12)
    flow_reg = kwargs.get('flow_reg',1e-10)
    
    (N,) = lcp.q.shape
    x0 = kwargs.get('x0',np.ones(N))
    y0 = kwargs.get('y0',np.ones(N))

    iterator = KojimaIPIterator(lcp,x0=x0,y0=y0)
    solver = IterativeSolver(iterator)

    term_conds = [InnerProductTerminationCondition(thresh),
                  MaxIterTerminationCondition(max_iter),                  
                  SteplenTerminationCondition(1e-20)]
    announce = [IterAnnounce(),
                PotentialAnnounce(),
                PotentialDiffAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)    

    solver.solve()
    (p,d) = (iterator.get_primal_vector(),iterator.get_dual_vector())

    return (p,d,iterator.data)

def solve_lcp_file(lcp_file):
    unarch = Unarchiver(lcp_file)
    lcp = LCPObj(unarch.M,unarch.q)
    
    # Augment
    alcp,x0,y0 = augment_lcp(lcp,1e3)
    (N,) = alcp.q.shape
    assert(N == lcp.q.shape[0] + 1) # Augmented
    
    (p,d,data) = kojima_solve(alcp,
                              x0=np.ones(N),
                              y0=np.ones(N))

    print '#'*20
    print 'FINISHED'
    print 'Slack variables', p[-1],d[-1]
        
    # Strip augmented variable and expand omitted nodes
    p = p[:-1]
    d = d[:-1]

    filename, file_extension = os.path.splitext(lcp_file)
    print "Writing solution to ",  filename + '.sol'
    arch = Archiver(p=p,d=d)
    arch.write(filename + '.sol')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Solve an LCP using Kojima.')
    parser.add_argument('lcp_file_name', metavar='F', type=str,
                        help='LCP input file')
    args = parser.parse_args()
    
    #Read LCP file
    solve_lcp_file(args.lcp_file_name)
