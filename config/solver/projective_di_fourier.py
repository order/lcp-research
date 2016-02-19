import config
import solvers

import bases
import bases.fourier as fourier
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
        params['x_dual_bases'] = True

        # This is the basic part of the basis generation
        # It's wrapped by BasisGenerator
        
        [X,V] = np.meshgrid(np.linspace(-5,5,5),np.linspace(-6,6,5))
        centers = np.column_stack([X.flatten(),V.flatten()])

        Zero = rbf.RadialBasis(centers=np.array([[0,0]]),
                               covariance=0.01*np.eye(2))
        One = bases.ConstBasis()

        random = False
        if random:
            Fourier = fourier.RandomFourierBasis(scale=1.0,
                                                 num_basis=5,
                                                 dim=2)
        else:
            (W,Phi) = fourier.make_regular_frequencies(3,3)

            Fourier = fourier.FourierBasis(frequencies=W,
                                           shifts=Phi)
            
        generator = bases.BasisGenerator([Zero,One,Fourier])
        
        params['basis_generator'] = generator
        self.params = params

    def configure_solver_generator(self):
        return gen.ProjectiveGenerator(**self.params)