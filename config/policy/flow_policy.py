import numpy as np

import config.generator

class FlowPolicyGenerator(config.generator.PolicyGenerator):
    def generate_policy(self,data,params):
        f = np.reshape(data['primal'][-1,N:],(N,A),order='F')
        f_fns = []
        
        for a in xrange(A):
            f_fn = InterpolatedFunction(discretizer,f[:,a])
            f_fns.append(f_fn)
        flow_policy = MaxFunPolicy(mdp_obj.actions,
                                   f_fns)

        return flow_policy
