import config
import solvers
import bases
import bases.rbf as rbf
import config.solver.gens.projective_gen as gen
import numpy as np

class ProjectiveBasicConfig(config.SolverConfig):
    def __init__(self):
        params = {}
        params['value_regularization'] = 0.0
        params['flow_regularization'] = 1e-12

        term_conds = {'max_iter':solvers.MaxIterTerminationCondition(100),
                      'primal':solvers.PrimalChangeTerminationCondition(1e-6)}
        recorders = {'primal':solvers.PrimalRecorder(),
                     'dual':solvers.DualRecorder(),
                     'steplen':solvers.StepLenRecorder()}
        notify = {'res':solvers.ResidualAnnounce(),
                  'primal':solvers.PrimalDiffAnnounce(),
                  'potential':solvers.PotentialAnnounce()}
        params['termination_conditions'] = term_conds
        params['recorders'] = recorders
        params['notifications'] = notify

        self.center_grid = (5,5) # how many points along each dimension

        self.params = params

    def configure_solver_generator(self,instance_builder):

        # Get boundary information        
        boundary = instance_builder.problem.boundary
        assert(2 == len(boundary))
        [(x_low,x_high),(v_low,v_high)] = boundary
        (xn,vn) = self.center_grid

        # Build the mesh of centers 
        [X,V] = np.meshgrid(np.linspace(x_low,x_high,xn),
                            np.linspace(v_low,v_high,vn))
        
        centers = np.column_stack([X.flatten(),V.flatten()])

        # Add standard delta / constant bases
        Zero = rbf.RadialBasis(centers=np.array([[0,0]]),
                               covariance=0.01*np.eye(2))
        One = bases.ConstBasis()
        
        # Add RBF component
        RBFs = rbf.RadialBasis(centers=centers,
                               covariance=np.array([[1,-0.5],
                                                    [-0.5,1]]))

        # Build value generator
        value_generator = bases.BasisWrapper([Zero,One,RBFs])
        value_generator.add_special_modifiers(instance_builder)
        generators = [value_generator]

        # Add flow basis from kojima solution file
        A = instance_builder.num_actions
        N = instance_builder.get_num_nodes()
        sol_file = 'data/kojima.sol.npy'
        for a in xrange(A):
            file_gen = bases.FromSolFileBasis(sol_file, a+1, N)
            generators.append(file_gen)
        
        self.params['basis_generators'] = generators
        return gen.ProjectiveGenerator(**self.params)
