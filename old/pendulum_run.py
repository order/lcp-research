import bases
import solvers
import lcp
import mdp

import numpy as np
import time
import pickle

# Choices
from gens.pendulum_discretizer import PendulumGenerator as InstanceGenerator
#from di.value_iter import build_solver,extract_data
from gens.kojima import KojimaGenerator as SolverGenerator

#################
# Main function

if __name__ == '__main__':
    qn = 70
    dqn = 50
    an = 3

    #time_str = datetime.datetime.now().isoformat()
    save_file = 'data/pendulum_kojima_traj' #.format(time_str) # no extension
    param_save_file = save_file + '.pickle'

    # Build the discretizer
    params = {'q_n':qn,\
              'dq_desc':(-5,5,dqn),\
              'a_desc':(-1,1,an)}
    inst_gen = InstanceGenerator()
    discretizer = inst_gen.generate(**params)
    assert(issubclass(type(discretizer), mdp.MDPDiscretizer))

    # Build the solver
    # May build intermediate objects (MDP, LCP, projective LCP)
    sol_gen = SolverGenerator()
    [solver,objs] = sol_gen.build(discretizer=discretizer)
    assert(issubclass(type(solver), solvers.IterativeSolver))
    
    # Solve; return primal and dual trajectories
    solver.solve()
    data = sol_gen.extract(solver)
    print 'Final iteration:',solver.get_iteration()

    # Save the trajectories for analysis
    np.savez(save_file,**data) # Extension auto-added

    #Save experiment parameters
    params = {'x_nodes':qn+1,\
              'y_nodes':dqn+1,\
              'actions':an,\
              'size':int(discretizer.get_num_nodes())}
    pickle.dump(params,open(param_save_file,'wb'))
