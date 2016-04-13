import numpy as np
import discrete
from discrete.regular_interpolate import RegularGridInterpolator
from mdp.mdp_builder import MDPBuilder

from argparse import ArgumentParser
import pickle

def make_uniform_mdp(problem,N,A):
    boundary = problem.gen_model.boundary.boundary
    grid = [(l,u,N) for (l,u) in boundary]
    discretizer = RegularGridInterpolator(grid)

    action_limits = problem.action_limits
    action_cuts = [np.linspace(l,u,A) for (l,u) in action_limits]
    actions = discrete.make_points(action_cuts)

    num_samples = 10
    
    builder = MDPBuilder(problem,
                         discretizer,
                         actions,
                         num_samples)
    mdp_obj = builder.build_mdp()
  
    return (mdp_obj,discretizer)

if __name__ == '__main__':
    parser = ArgumentParser(__file__,'Generates an MDP from continuous problem')
    parser.add_argument('problem_in_file',
                        metavar='FILE',
                        help='problem file')
    parser.add_argument('num_states',
                        metavar='N',
                        type=int,
                        help='number of states per dimension')
    parser.add_argument('num_actions',
                        metavar='A',
                        type=int,
                        help='number of actions per dimension')
    parser.add_argument('mdp_out_file',
                        metavar='FILE',
                        help='mdp save file')
    parser.add_argument('disc_out_file',
                        metavar='FILE',
                        help='discretizer save file')
    args = parser.parse_args()

    FH = open(args.problem_in_file,'r')
    problem = pickle.load(FH)
    FH.close()

    (mdp,disc) = make_uniform_mdp(problem,
                                  args.num_states,
                                  args.num_actions)

    FH = open(args.mdp_out_file,'w')
    pickle.dump(mdp,FH)
    FH.close()
                        
 
    FH = open(args.disc_out_file,'w')
    pickle.dump(disc,FH)
    FH.close()
