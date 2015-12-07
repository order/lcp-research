import sys
import importlib

import numpy as np
from utils.parsers import ConfigParser
from utils import load_class
#################
# Main function

if __name__ == '__main__':

    # Parse command line
    if 4 != len(sys.argv):
        print 'Usage: python {0} <inst file> <solver file> <save_file>'\
            .format(sys.argv[0])
        quit()
    (_,inst_conf_file,solver_conf_file,save_file) = sys.argv
    param_save_file = save_file + '.pickle'
    print 'Saving temp info to', param_save_file

    # Build the discretizer
    print 'Building discretizer'
    parser = ConfigParser(inst_conf_file)
    parser.add_handler('gen_fn',load_class)
    args = parser.parse()
    
    print args
    quit()
    inst_gen = load_class(inst_gen_str)()
    discretizer = inst_gen.generate()
    assert(issubclass(type(discretizer), mdp.MDPDiscretizer))

    # Build the solver
    # May build intermediate objects (MDP, LCP, projective LCP)
    sol_gen = importlib_import_module(solver_gen_file)
    [solver,objs] = sol_gen.build(discretizer=discretizer)
    assert(issubclass(type(solver), solvers.IterativeSolver))
    
    # Solve; return primal and dual trajectories
    solver.solve()
    data = sol_gen.extract(solver)
    print 'Final iteration:',solver.get_iteration()

    # Save the trajectories for analysis
    np.savez(save_file,**data) # Extension auto-added

    #Save experiment parameters
    params = {'discretizer':discretizer,\
              'solver':solver,\
              'objects':objects}
    pickle.dump(params,open(param_save_file,'wb'))
