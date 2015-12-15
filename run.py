import sys
import importlib
import pickle

import numpy as np

import mdp,gens,solvers
from utils.parsers import ConfigParser, hier_key_dict
from utils import load_class


#################
# Main function

def build_generator(conf_file):
    parser = ConfigParser(conf_file)
    parser.add_handler('gen_fn',load_class)
    args = parser.parse()    

    # "generator" and "name" are special keywords
    gen_class = args['generator'] # Don't instantiate yet
    name = args.get('name',conf_file)
    del args['generator']
    del args['name']

    gen_opts = hier_key_dict(args,':')
    
    gen_inst = gen_class(**gen_opts) # Rest of args for the gen instantiation
    
    return (gen_inst,name)  

if __name__ == '__main__':

    # Parse command line
    if 4 != len(sys.argv):
        print 'Usage: python {0} <inst file> <solver file> <save_file>'\
            .format(sys.argv[0])
        quit()
    (_,inst_conf_file,solver_conf_file,save_file) = sys.argv
    param_save_file = save_file + '.pickle'
    print 'Note: saving internal info to', param_save_file

    (inst_gen,inst_name,inst_args) = build_generator(inst_conf_file)
    assert(issubclass(type(inst_gen), gens.Generator))
    discretizer = inst_gen.generate(**inst_args)
    assert(issubclass(type(discretizer), mdp.MDPDiscretizer))
    
    # Build the solver
    # May build intermediate objects (MDP, LCP, projective LCP)
    (sol_gen,sol_name,sol_args) = build_generator(solver_conf_file)
    assert(issubclass(type(sol_gen), gens.SolverGenerator))    
    [solver,objs] = sol_gen.generate(discretizer=discretizer,\
                                  **sol_args)
    assert(issubclass(type(solver), solvers.IterativeSolver))
    
    # Solve; return primal and dual trajectories
    solver.solve()
    data = sol_gen.extract(solver)
    print 'Final iteration:',solver.get_iteration()

    # Save the trajectories for analysis
    np.savez(save_file,**data) # Extension auto-added

    #Save experiment parameters
    params = {'discretizer':discretizer,
              'inst_name':inst_name,
              'solver':solver,
              'solver_name':sol_name,
              'objects':objs}
    pickle.dump(params,open(param_save_file,'wb'))
