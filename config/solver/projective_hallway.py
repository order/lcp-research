import config
import solvers
import bases
import bases.rbf as rbf
import bases.one_d as one_d
import config.solver.gens.projective_gen as gen
import numpy as np

import matplotlib.pyplot as plt

class ProjectiveBasicConfig(config.SolverConfig):
    def __init__(self):
        params = {}
        params['value_regularization'] = 0.0
        params['flow_regularization'] = 1e-15

        term_conds = {'max_iter':solvers.MaxIterTerminationCondition(1000),
                      'primal':solvers.PrimalChangeTerminationCondition(1e-12)}
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

        # This is the basic part of the basis generation
        # It's wrapped by BasisGenerator

        
        centers = np.linspace(-1,1,10)[:,np.newaxis]

        Zero = rbf.RadialBasis(centers=np.array([[0]]),
                               covariance=0.001)
        #One = bases.ConstBasis()
        #RBFs = rbf.RadialBasis(centers=centers,
        #                       covariance=0.05)
        #generator = bases.BasisGenerator([Zero,One,RBFs])
        Cheby = one_d.ChebyshevBasis(K=10)
        generator = bases.BasisGenerator([Zero,Cheby])
        
        params['basis_generator'] = generator
        self.params = params

    def configure_solver_generator(self):
        return gen.ProjectiveGenerator(**self.params)
