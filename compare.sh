#!/bin/bash

CHEAPN=20
GOODN=80
A=3
ROOT=data/di
CHEAP=${ROOT}_${CHEAPN}
GOOD=${ROOT}_${GOODN}

#python generate_di_problem.py $ROOT.prob
#python generate_mdp.py $ROOT.prob $CHEAPN $A $CHEAP.mdp $CHEAP.disc
#python generate_mdp.py $ROOT.prob $GOODN $A $GOOD.mdp $GOOD.disc
#python solve_mdp_kojima.py $CHEAP.mdp $CHEAP.sol
#python solve_mdp_kojima.py $GOOD.mdp $GOOD.sol
python form_q_policy.py $CHEAP.sol $CHEAP.disc $CHEAP.mdp $CHEAP.q.policy
python form_v_function.py $CHEAP.sol $CHEAP.disc $CHEAP.mdp $CHEAP.vfn
python simulate_policy_vs_value_estimate.py $CHEAP.vfn\
       $GOOD.q.policy $ROOT.prob foo bar
