import config
import config.solver.gens.value_iter_gen as gen
import solvers

class ValueIterationConfig(config.SolverConfig):
    def __init__(self):
        params = {}
        term_conds = {'max_iter':solvers.MaxIterTerminationCondition(2500),
                      'value':solvers.ValueChangeTerminationCondition(1e-12)}
        recorders = {'value':solvers.ValueRecorder()}
        notify = {'value':solvers.ValueDiffAnnounce()}
        params['termination_conditions'] = term_conds
        params['recorders'] = recorders
        params['notifications'] = notify
        self.params = params

    def configure_solver_generator(self,discretizer):
        return gen.ValueIterGenerator(**self.params)

