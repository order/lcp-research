#!/bin/bash

CHEAPN=20
GOODN=80
A=3
ROOT=data/di
CHEAP=${ROOT}_${CHEAPN}
GOOD=${ROOT}_${GOODN}
MCTS=${ROOT}_mcts

python generate_di_problem.py $ROOT.prob
python generate_mdp.py $ROOT.prob $CHEAPN $A $CHEAP.mdp $CHEAP.disc
python generate_mdp.py $ROOT.prob $GOODN $A $GOOD.mdp $GOOD.disc
python solve_mdp_kojima.py $CHEAP.mdp $CHEAP.sol
python solve_mdp_kojima.py $GOOD.mdp $GOOD.sol

# Value Functions
python form_v_function.py $CHEAP.sol $CHEAP.disc $CHEAP.mdp $CHEAP.vfn
python form_v_function.py $GOOD.sol $GOOD.disc $GOOD.mdp $GOOD.vfn

# Cheap Q
python form_q_policy.py $CHEAP.sol $CHEAP.disc $CHEAP.mdp $CHEAP.q.policy
python simulate_policy_vs_value_estimate.py $GOOD.vfn $CHEAP.q.policy $ROOT.prob $CHEAP.q.sim
python compare_plot.py $CHEAP.q.sim $ROOT.prob $CHEAP.q.png -t"20x20 Q-policy Return vs Value"

# Good Q
python form_q_policy.py $GOOD.sol $GOOD.disc $GOOD.mdp $GOOD.q.policy
python simulate_policy_vs_value_estimate.py $GOOD.vfn $GOOD.q.policy $ROOT.prob $GOOD.q.sim
python compare_plot.py $GOOD.q.sim $ROOT.prob $GOOD.q.png -t"80x80 Q-policy Return vs Value"


# Cheap Flow
python form_flow_policy.py $CHEAP.sol $CHEAP.disc $CHEAP.mdp $CHEAP.flow.policy
python simulate_policy_vs_value_estimate.py $GOOD.vfn $CHEAP.flow.policy $ROOT.prob $CHEAP.flow.sim
python compare_plot.py $CHEAP.flow.sim $ROOT.prob $CHEAP.flow.png -t"20x20 Flow policy Return vs Value"

# Good Flow
python form_flow_policy.py $GOOD.sol $GOOD.disc $GOOD.mdp $GOOD.flow.policy
python simulate_policy_vs_value_estimate.py $GOOD.vfn $GOOD.flow.policy $ROOT.prob $GOOD.flow.sim
python compare_plot.py $GOOD.flow.sim $ROOT.prob $GOOD.flow.png -t"80x80 Flow policy Return vs Value"

# MCTS
#python form_mcts_policy.py $CHEAP.sol $CHEAP.disc $CHEAP.mdp $ROOT.prob $MCTS.policy -z10 -b50
#python simulate_policy_vs_value_estimate.py $GOOD.vfn $MCTS.policy $ROOT.prob $MCTS.sim
#python compare_plot.py $MCTS.sim $ROOT.prob $MCTS.png -t"MCTS Return vs Value"


python cdf_plot_sim.py $ROOT.cdf.png $GOOD.flow.sim $CHEAP.flow.sim $GOOD.q.sim $CHEAP.q.sim $MCTS.sim
