import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from lcp import ProjectiveLCPObj
from utils.archiver import Unarchiver,Archiver

from solvers import augment_plcp,ProjectiveIPIterator,IterativeSolver
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

import time
import argparse
import sys
import os.path


###############################################
# Solvers
def projective_solve(plcp,**kwargs):
    thresh   = kwargs.get('thresh',1e-12)
    max_iter = kwargs.get('max_iter',1000)
    reg  = kwargs.get('reg',1e-7)
    
    (N,K) = plcp.Phi.shape
    x0 = kwargs.get('x0',np.ones(N))
    y0 = kwargs.get('y0',np.ones(N))
    w0 = kwargs.get('w0',plcp.Phi.T.dot(plcp.q))

    iterator = ProjectiveIPIterator(plcp,x0=x0,y0=y0,w0=w0)
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

def solve_plcp_file(plcp_file):
    unarch = Unarchiver(plcp_file)
    (P,U,r) = (unarch.Phi,unarch.U,unarch.r)
    q = P.dot(r)

    a = 1e-4
    plcp = ProjectiveLCPObj(P,a*U,a*U,a*q)
    
    # Augment
    alcp,x0,y0,w0 = augment_plcp(plcp,1e5)
    (N,) = alcp.q.shape
    assert(N == plcp.q.shape[0] + 1) # Augmented
    
    (p,d,data) = projective_solve(alcp,
                                  x0=x0,
                                  y0=y0,
                                  w0=w0)

    print '#'*20
    print 'FINISHED'
    print 'Slack variables', p[-1],d[-1]
        
    # Strip augmented variable and expand omitted nodes
    p = p[:-1]
    d = d[:-1]

    filename, file_extension = os.path.splitext(plcp_file)
    sol_file =  filename + '.psol'
    print "Writing solution to ", sol_file
    arch = Archiver(p=p,d=d)
    arch.write(sol_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Solve an LCP using Projective.')
    parser.add_argument('plcp_file_name', metavar='F', type=str,
                        help='PLCP input file')
    args = parser.parse_args()
    
    #Read LCP file
    solve_plcp_file(args.plcp_file_name)
