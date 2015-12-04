import bases
import solvers
import lcp
import mdp

import numpy as np
import time
import pickle

# Choices
from di.discretizer import DIGenerator as InstanceGenerator
#from di.value_iter import build_solver,extract_data
from di.kojima import KojimaGenerator as SolverGenerator

#################
# Main function

if __name__ == '__main__':
    xn = 30
    vn = 30
    an = 3

    #time_str = datetime.datetime.now().isoformat()
    save_file = 'data/di_traj' #.format(time_str) # no extension
    param_save_file = save_file + '.pickle'

    # Build the discretizer
    params = {'x_desc':(-4,4,xn),\
              'v_desc':(-6,6,vn),\
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
    params = {'x_nodes':xn+1,\
              'v_nodes':vn+1,\
              'actions':an,\
              'size':int(discretizer.get_num_nodes())}
    pickle.dump(params,open(param_save_file,'wb'))
