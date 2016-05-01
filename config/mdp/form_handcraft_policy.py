import numpy as np
import pickle
import sys

from mdp.policy import MaxFunPolicy,IndexPolicyWrapper,RandomDiscretePolicy
from mdp.policy import BangBangPolicy
from mdp.solution_process import *

if __name__ == '__main__':
    (_,outfile) = sys.argv


    hand_policy = BangBangPolicy()
    
    FH = open(outfile,'w')
    pickle.dump(hand_policy,FH)
    FH.close() 
