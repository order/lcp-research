import numpy as np
import scipy.sparse as sps
import itertools
from pulp import *

import time
from collections import deque

def brute_force_dependent_sets_from_frame(F):
    (n,N) = F.shape
    
    Indices = []
    for idx in itertools.combinations(range(N),n):
        if n != np.linalg.matrix_rank(F[:,idx]):
            Indices.append(idx)
    return Indices

def minimal_dependent_set_from_frame(F):
    return help_dep(F,0,set([]))

def help_dep(F,pos,cols):
    (n,N) = F.shape
    k = len(cols)

    if pos == N:
        return []
    
    assert k <= n
    
    if k > 1 and np.linalg.matrix_rank(F[:, list(cols)]) < k:
        return [tuple(cols)] # Any extension also dependent

    if k == n:
        return [] # No interesting dependent sets

    dep_sets = help_dep(F,pos+1,cols) # Move on w/o adding
    cols.add(pos)
    dep_sets += help_dep(F,pos+1,cols) # w/ adding
    cols.remove(pos)
    
    return dep_sets

def dependent_set_from_frame(F):
    (n,N) = F.shape
    
    fringe = deque([list(x) for x in itertools.combinations(range(N),2)])
    sets=[]

    while fringe:
        cand = fringe.popleft()
        assert len(cand) <= n
        scand = set(cand)

        eclipsed = False
        for s in sets:
            assert s != scand
            assert len(s) <= len(scand)
            if s.issubset(scand):
                eclipsed = True
                break

        if eclipsed:
            continue

        if np.linalg.matrix_rank(F[:,list(cand)]) < len(cand):
            sets.append(scand)
            continue

        if len(cand) == n:
             continue
         
        for i in xrange(max(cand)+1,N):
            fringe.append(cand + [i])

    return sets
                   
    
  
def mangasarian_mu(F,dep_sets,linf_comp):
    """
    Solve for the theoretical perturbation constant in
    "Error bounds for non-degenerate monotone LCPs" by
    Mangasarian.
    """
    
    prob = LpProblem("Mangasarian Mu problem",LpMaximize)
    (n,N) = F.shape
    
    U = 1e9-1 # Magnitude bound

    # Variables and their support twiddles
    # So (T[i] = 0) -> (X[i] = 0)
    X = []
    T = []

    for i in xrange(N):
        X.append(LpVariable("x_{0}".format(i),lowBound=0))
        T.append(LpVariable("t_{0}".format(i),0,1,LpInteger))
        prob += (X[i] <= U*T[i]),"Toggle " + str(i)

    # Add objective value
    if linf_comp is None:
        prob += LpAffineExpression(zip(X,[1]*N)) # L1 objective
    else:
        assert 0 <= linf_comp < N
        prob += X[linf_comp] #Single out component, max over all problems.

    # Row abs. value
    # Used for ensuring that the l1-norm is bounded by 1
    AV = []
    for i in xrange(n):
        AV.append(LpVariable("av_{0}".format(i),lowBound=0))

        col_idx = np.where(F[i,:] != 0)[0]
        coef_list = [(X[j],F[i,j]) for j in col_idx]
        
        prob += (LpAffineExpression(coef_list) <= AV[i]), "AbsVal_{0}_{1}".format(i,0)
        prob += (-AV[i] <= LpAffineExpression(coef_list)), "AbsVal_{0}_{1}".format(i,1)
    prob += (LpAffineExpression(zip(AV,[1]*n))) <= 1, "l1 norm"

    prob += (LpAffineExpression(zip(T,[1]*N))) <= n, "toggle"
    for (C,s) in enumerate(dep_sets):
        prob += (LpAffineExpression([(T[i],1) for i in s]) <= len(s)-1), "dep_set_"+str(C)

    #print prob
        
    prob.solve()
    #print "\tStatus:", LpStatus[prob.status]
    #print "\tMangasarian mu:", value(prob.objective)

    return value(prob.objective)

    
if __name__ == "__main__":
    for D in xrange(2,10):
        # Form MC evaluation LCP (A=1)
        print "D:",D
        g = 0.98
        P = sps.spdiags(0.25*np.ones(D),0,D,D) + sps.spdiags(0.75*np.ones(D),-1,D,D)
        P = P.tolil()
        P[0,-1] = 0.75
        
        B = sps.eye(D) - g * P
        M = sps.bmat([[None,B],[-B.T,None]])
        q = np.empty(2*D)
        q[:D] = -np.ones(D) / D
        q[D:] = np.ones(D)
        q[-1] = 0

        n = 2*D
        N = 2*n+1

        # Build the frame [-q,-M,I]
        F = sps.hstack([-np.reshape(q,(n,1)),-M,sps.eye(n)]).toarray()
        assert (n,N) == F.shape

        # Find all minimal dependent sets (expensive!)
        t2 = time.time()
        dep_sets = dependent_set_from_frame(F)
        t3 = time.time()

        #print "Found",len(dep_sets),"dependent sets in", (t3 - t2), "seconds"

        linf_obj = -np.inf
        for i in xrange(N):
            #print "\tComponent LP for",i
            obj = mangasarian_mu(F,dep_sets,i)
            linf_obj = max(linf_obj,obj)

        print "Linf:",linf_obj
