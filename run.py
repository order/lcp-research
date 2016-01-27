import sys
import pickle

import numpy as np

import solvers
import utils
import config

def get_instance_from_file(conf_file):
    """
    Loads a class from file string
    So if the string is 'foo/bar/baz.py' then it loads the UNIQUE
    class in that file.
    """
    module = utils.load_module_from_filename(conf_file)
    classlist = utils.list_module_classes(module)

    assert(1 == len(classlist)) # Class is UNIQUE.
    return classlist[0][1]() # Instantiate too

def get_instance_builder(inst_conf_file):
    """
    Gets an InstanceConfig from file.
    This object configures the instance builder (e.g. Discretizer)
    The builder is used in setting up the solver
    """
    instance_config = get_instance_from_file(inst_conf_file)    
    assert(issubclass(type(instance_config),
                      config.InstanceConfig))
    return instance_config.configure_instance_builder()  

def get_solver_generator(solver_conf_file):
    """
    Gets a SolverConfig from file.
    This object configures the solver generator (e.g. 
    ValueIterGenerator)
    The generator uses the above builder to set up the solver.
    It also extracts data from the solver for reporting.
    """
    solver_config = get_instance_from_file(solver_conf_file)
    assert(issubclass(type(solver_config),
                      config.SolverConfig))
    return solver_config.configure_solver_generator()

def build_solver(solver_generator,instance_builder):
    # May build intermediate objects (MDP, LCP, projective LCP)
    [solver,intermediate_objects] = solver_generator.generate(
        instance_builder)
    assert(issubclass(type(solver), solvers.IterativeSolver))
    return [solver,intermediate_objects]


def save_data(save_file,data):
    # Save the trajectories for analysis
    np.savez(save_file,**data) # Extension auto-added
    if 'primal' in data:
        # Save the final primal iteration as the solution
        np.save(save_file + '.sol',data['primal'][-1,:])

def pickle_objects(param_save_file,instance_builder,objs):
    #Save experiment's objects
    params = {'instance_builder':instance_builder,
              'objects':objs}
    pickle.dump(params,open(param_save_file,'wb'))


###############
# Entry point #
###############
    
if __name__ == '__main__':
    # Parse command line
    if 4 != len(sys.argv):
        print 'Usage: run.py <inst file> <solver file> <save_file>'
        quit()
        
    (_,inst_conf_file,solver_conf_file,save_file) = sys.argv
    param_save_file = save_file + '.pickle'
    print 'Note: saving internal info to', param_save_file

    # Configure and build
    instance_builder = get_instance_builder(inst_conf_file)
    solver_generator = get_solver_generator(solver_conf_file)
    [solver,interm_objects] = build_solver(solver_generator,
                                         instance_builder)
    
    # Solve; return primal and dual trajectories
    solver.solve()
    data = solver_generator.extract(solver)

    # Save
    save_data(save_file,data)
    pickle_objects(param_save_file,
                   instance_builder,
                   interm_objects)

