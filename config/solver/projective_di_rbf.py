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

        term_conds = {'max_iter':solvers.MaxIterTerminationCondition(1000),
                      'primal':solvers.PrimalChangeTerminationCondition(1e-8)}
        recorders = {'primal':solvers.PrimalRecorder(),
                     'dual':solvers.DualRecorder(),
                     'steplen':solvers.StepLenRecorder()}
        notify = {'res':solvers.ResidualAnnounce(),
                  'primal':solvers.PrimalDiffAnnounce(),
                  'potential':solvers.PotentialAnnounce()}
        params['termination_conditions'] = term_conds
        params['recorders'] = recorders
        params['notifications'] = notify

        # Experimental parameter
        params['x_dual_bases'] = False

        self.center_grid = (5,5)

        self.params = params

    def configure_solver_generator(self,instance_builder):
        boundary = instance_builder.problem.boundary
        assert(2 == len(boundary))
        [(x_low,x_high),(v_low,v_high)] = boundary
        (xn,vn) = self.center_grid
        
        [X,V] = np.meshgrid(np.linspace(x_low,x_high,xn),
                            np.linspace(v_low,v_high,vn))
        centers = np.column_stack([X.flatten(),V.flatten()])

        Zero = rbf.RadialBasis(centers=np.array([[0,0]]),
                               covariance=0.01*np.eye(2))
        One = bases.ConstBasis()
        RBFs = rbf.RadialBasis(centers=centers,
                               covariance=np.array([[1,-0.5],
                                                    [-0.5,1]]))
        generator = bases.BasisWrapper([Zero,One,RBFs])
        
        self.params['basis_generator'] = generator
        return gen.ProjectiveGenerator(**self.params)
