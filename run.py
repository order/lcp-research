import sys
import importlib
import pickle

import numpy as np

import mdp,gens,solvers
from utils.parsers import ConfigParser, hier_key_dict
from utils import load_str

import os


#################
# Main function

def build_generator(conf_file):
    path = conf_file.split(os.sep)
    file_string = path[-1]
    assert(file_string.endswith('.py'))
    base_string = file_string[:-3]
    
    mod_string = '.'.join(path[:-1] + [base_string])
    module = importlib.import_module(mod_string)
    print module
    quit()

if __name__ == '__main__':
    # Parse command line
    if 4 != len(sys.argv):
        print 'Usage: python {0} <inst file> <solver file> <save_file>'\
            .format(sys.argv[0])
        quit()
    (_,inst_conf_file,solver_conf_file,save_file) = sys.argv
    param_save_file = save_file + '.pickle'
    print 'Note: saving internal info to', param_save_file

    discretizer = build_generator(inst_conf_file)
    assert(issubclass(type(discretizer), mdp.MDPDiscretizer))
    
    # Build the solver
    # May build intermediate objects (MDP, LCP, projective LCP)
    [solver,objs] = build_generator(solver_conf_file)
    assert(issubclass(type(solver), solvers.IterativeSolver))
    
    # Solve; return primal and dual trajectories
    solver.solve()
    data = sol_gen.extract(solver)

    print 'Final iteration:',solver.get_iteration()

    # Save the trajectories for analysis
    np.savez(save_file,**data) # Extension auto-added
    if 'primal' in data:
        # Save the final primal iteration as the solution
        np.save(save_file + '.sol',data['primal'][-1,:])

    #Save experiment parameters
    params = {'discretizer':discretizer,
              'inst_name':inst_name,
              'solver':solver,
              'solver_name':sol_name,
              'objects':objs}
    pickle.dump(params,open(param_save_file,'wb'))
