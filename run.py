import sys
import pickle

import numpy as np

import mdp,solvers
from utils.parsers import ConfigParser, hier_key_dict
import utils

import os


#################
# Main function

def get_instance_from_file(conf_file):
    """
    Loads a class from file string
    So if the string is 'foo/bar/baz.py' then load the UNIQUE class in
    that file.
    """
    module = utils.load_module_from_filename(conf_file)
    classlist = utils.list_module_classes(module)

    assert(1 == len(classlist)) # Class is UNIQUE.
    return classlist[0][1]() # Instantiate too
    

if __name__ == '__main__':
    # Parse command line
    if 4 != len(sys.argv):
        print 'Usage: python {0} <inst file> <solver file> <save_file>'\
            .format(sys.argv[0])
        quit()
    (_,inst_conf_file,solver_conf_file,save_file) = sys.argv
    param_save_file = save_file + '.pickle'
    print 'Note: saving internal info to', param_save_file

    # Get the instance generator and builder from file
    instance_generator = get_instance_from_file(inst_conf_file)
    instance_builder = instance_generator.generate()
    
    # Get the solver generator
    solver_generator = get_instance_from_file(solver_conf_file)
    assert(issubclass(type(solver_generator),
                      config.generator.SolverGenerator))
    # May build intermediate objects (MDP, LCP, projective LCP)
    [solver,intermediate_objects] = solver_generator.generate(
        object_builder=instance_builder)
    assert(issubclass(type(solver), solvers.IterativeSolver))
    
    # Solve; return primal and dual trajectories
    solver.solve()
    data = solver_generator.extract(solver)

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
