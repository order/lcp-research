import numpy as np
import config
import config.solver.gens.td_gen as gen
import solvers
import mdp.policy as policy

class TabularTDIterationConfig(config.SolverConfig):
    def __init__(self):
        params = {}
        term_conds = {'max_iter':solvers.MaxIterTerminationCondition(10000)}
        recorders = {'value':solvers.ValueRecorder()}
        notify = {'value':solvers.ValueDiffAnnounce()}
        params['termination_conditions'] = term_conds
        params['recorders'] = recorders
        params['notifications'] = notify
        params['num_samples'] = 100
        params['step_size'] = 0.05
        
        params['policy'] = policy.ConstantDiscretePolicy(1)

        self.params = params

    def configure_solver_generator(self,discretizer):
        return gen.TabularTDGenerator(**self.params)

