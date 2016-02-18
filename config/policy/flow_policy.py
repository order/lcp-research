import numpy as np
import mdp.state_functions as state_fns
import mdp.policy

import config.generator

class FlowPolicyGenerator(config.generator.PolicyGenerator):
    def generate_policy(self,data,params):
        discretizer = params['instance_builder']
        mdp_obj = params['objects']['mdp']


        N = discretizer.get_num_nodes()
        A = discretizer.num_actions

        f = np.reshape(data['primal'][-1,N:],(N,A),order='F')
        f_fns = []
        
        for a in xrange(A):
            f_fn = state_fns.InterpolatedFunction(discretizer,
                                                  f[:,a])
            f_fns.append(f_fn)
        flow_policy = mdp.policy.MaxFunPolicy(mdp_obj.actions,
                                              f_fns)

        return flow_policy
