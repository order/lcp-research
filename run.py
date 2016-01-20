import sys
import pickle

import numpy as np

import mdp,solvers
from utils.parsers import ConfigParser, hier_key_dict
import utils

import os


#################
# Main function

def build_generator(conf_file):
    module = utils.load_module_from_filename(conf_file)
    print module
    classlist = utils.list_module_classes(module)
    assert(1 == len(classlist))

    instance = classlist[0][1]() # Instantiate the factory
    return instance.build() # Build object
    

if __name__ == '__main__':
    # Parse command line
    if 4 != len(sys.argv):
        print 'Usage: python {0} <inst file> <solver file> <save_file>'\
            .format(sys.argv[0])
        quit()
    (_,inst_conf_file,solver_conf_file,save_file) = sys.argv
    param_save_file = save_file + '.pickle'
    print 'Note: saving internal info to', param_save_file

    inst_gen = build_generator(inst_conf_file)
    discretizer = inst_gen.generate()
    assert(issubclass(type(discretizer), mdp.MDPDiscretizer))
    
    # Build the solver
    # May build intermediate objects (MDP, LCP, projective LCP)
    solver_gen = build_generator(solver_conf_file)
    [solver,objs] = solver_gen.generate(discretizer)
    assert(issubclass(type(solver), solvers.IterativeSolver))
    
    # Solve; return primal and dual trajectories
    solver.solve()
    data = solver_gen.extract(solver)

    print 'Final iteration:',solver.get_iteration()

    # Save the trajectories for analysis
    np.savez(save_file,**data) # Extension auto-added
    if 'primal' in data:
        # Save the final primal iteration as the solution
        np.save(save_file + '.sol',data['primal'][-1,:])

    #Save experiment parameters
    params = {'discretizer':discretizer,
              'solver':solver,
              'objects':objs}
    pickle.dump(params,open(param_save_file,'wb'))
