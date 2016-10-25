import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

import itertools
from pulp import *

import time
from collections import deque

def lin_dep(F,cols):
    k = len(cols)
    rank = np.linalg.matrix_rank(F[:,cols])
    return rank < k

def minimal_dependent_set_from_frame(F):
    return help_dep(F,0,set([]))

def help_dep(F,pos,cols):
    (n,N) = F.shape
    k = len(cols)

    if pos == N:
        return []
    
    assert k <= n
    
    if k > 1 and lin_dep(F,cols):
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

        if lin_dep(F,cand):
            sets.append(scand)
            continue

        if len(cand) == n:
             continue
         
        for i in xrange(max(cand)+1,N):
            fringe.append(cand + [i])

    return sets
                   
def dep_sets_from_identity(F):
    (n,N) = F.shape
    assert np.linalg.norm(F[:,-n:] - np.eye(n)) < 1e-15

    dep_sets = []

    for i in xrange(N-n):
        idx = np.where(F[:,i] != 0)[0]
        if len(idx) >= (n/2):
            continue
        new_set = [i] + [j+n+1 for j in idx]
        assert lin_dep(F,new_set)
        dep_sets.append(tuple(new_set))
    return dep_sets

def dep_sets_from_combination(F,k):
    (n,N) = F.shape
    
    Indices = []
    for idx in itertools.combinations(range(N),k):
        if lin_dep(F,idx):
            Indices.append(idx)
    return Indices   
  
def mangasarian_mu(F,dep_sets,linf_comp=None):
    """
    Solve for the theoretical perturbation constant in
    "Error bounds for non-degenerate monotone LCPs" by
    Mangasarian.
    """
    
    prob = LpProblem("Mangasarian Mu problem",LpMaximize)
    (n,N) = F.shape
    
    U = 1e3 # Magnitude bound

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

    prob += (LpAffineExpression(zip(T,[1]*N))) == n, "toggle"
    for (C,s) in enumerate(dep_sets):
        prob += (LpAffineExpression([(T[i],1) for i in s]) <= len(s)-1), "dep_set_"+str(C)
        
    prob.solve()
    #print "\tMangasarian mu:", value(prob.objective)
    #print "Status:", LpStatus[prob.status]

    x_array = np.empty(N)
    for (i,x) in enumerate(X):
        x_array[i] = x.varValue
    av_array = np.empty(n)
    for (i,av) in enumerate(AV):
        av_array[i] = av.varValue
    t_array = np.empty(N)
    for (i,t) in enumerate(T):
        t_array[i] = t.varValue
    
    return value(prob.objective),x_array

def build_hallway(N,g,p):
    
        P = sps.spdiags((1-p)*np.ones(N),0,N,N) + sps.spdiags(p*np.ones(N),-1,N,N)
        P = P.tolil()
        P[0,-1] = p

        Q = sps.spdiags((1-p)*np.ones(N),0,N,N) + sps.spdiags(p*np.ones(N),1,N,N)
        Q = Q.tolil()
        Q[-1,0] = p
        
        I = sps.eye(N)
        B = I - g * P
        C = I - g*Q
        M = sps.bmat([[None,B,C],
                      [-B,None,None],
                      [-C,None,None]])

        c = np.ones(N)
        c[int(N/2)] = 0
        
        q = np.empty(3*N)
        q[:D] = -np.ones(N) / N
        q[D:(2*D)] = c
        q[(2*D):] = c

        return (M,q)

def build_smallway(N,g,p):    
        P = sps.spdiags((1-p)*np.ones(N),0,N,N) + sps.spdiags(p*np.ones(N),-1,N,N)
        P = P.tolil()
        P[0,-1] = p
        
        I = sps.eye(N)
        B = I - g * P
        M = sps.bmat([[None,B],
                      [-B,None]])

        c = np.ones(N)
        c[int(N/2)] = 0
        
        q = np.empty(2*N)
        q[:D] = -np.ones(N) / N
        q[D:] = c

        return (M,q)

def check_solution(F,x):
    idx = list(np.where(x > 0)[0])
    if not lin_dep(F,idx):
        return True

    # Refine
    repeat = True
    while repeat:
        repeat = False
        for i in xrange(len(idx)):
            cand = idx[:i] + idx[(i+1):] # Everything other than i
            
            if lin_dep(F,cand):
                repeat = True
                idx = cand
                break
    return idx

def constraint_gen(F,dep_sets,i):
    I = 0
    while True:
        I+=1
        #print "\tCGR:",I
        obj,x = mangasarian_mu(F,dep_sets,i)
        res = check_solution(F,x)
        if res is True:
            return obj
        else:
            assert len(res) <= n
            dep_sets.append(res)

def linf_mu(F,dep_sets):
    (n,N) = F.shape
    linf = -np.inf
    for i in xrange(N):
        #print "Component ",i
        obj = constraint_gen(F,dep_sets,i)
        linf = max(linf,obj)
    return linf

def l1_mu(F,dep_sets):
    return constraint_gen(F,dep_sets,None)
    
if __name__ == "__main__":

    dims = []
    mu = []
    for D in range(2,15):
        # Form MC evaluation LCP (A=1)
        n = 2*D
        (M,q) = build_smallway(D,0.99,0.75)
        assert (n,n) == M.shape
        N = 2*n+1

        # Build the frame [-q,-M,I]
        F = sps.hstack([-np.reshape(q,(n,1)),-M,sps.eye(n)]).toarray()
        assert (n,N) == F.shape

        # Find dep sets using the identity matrix
        t2 = time.time()
        dep_sets = []
        #dep_sets = dep_sets_from_identity(F)
        #dep_sets += dep_sets_from_combination(F,2)
        dep_sets += dep_sets_from_combination(F,3)
        #dep_sets += dep_sets_from_combination(F,4)
        t3 = time.time()

        #print "Found",len(dep_sets),"dependent sets in", (t3 - t2), "seconds"
        #print "\tComponent LP for",i
        obj = linf_mu(F,dep_sets)
        #obj = l1_mu(F,dep_sets)
        
        dims.append(n)
        mu.append(obj)
        print "mu("+str(n)+") = ",obj
        np.savez('test',dims=dims,mu=mu)
