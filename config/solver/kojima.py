import config
import solvers
import config.solver.gens.kojima_gen as gen

class KojimaBasicConfig(config.Config):
    def __init__(self):
        params = {}
        params['value_regularization'] = 0.0
        params['flow_regularization'] = 1e-12

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
        self.params = params

    def build(self):
        return gen.KojimaGenerator(**self.params)
