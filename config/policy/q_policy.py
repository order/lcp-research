import config
import mdp.q_estimation
import mdp.state_functions as state_fns
import mdp.policy

class QPolicyGenerator(config.PolicyGenerator):
    def generate_policy(self,data,params):
        discretizer = params['instance_builder']
        mdp_obj = params['objects']['mdp']

        N = discretizer.get_num_nodes()
        A = discretizer.num_actions
        
        v = data['primal'][-1,:N]
        
        q = mdp.q_estimation.get_q_vectors(mdp_obj,v)
        q_fns = []
        for a in xrange(A):
            q_fn = state_fns.InterpolatedFunction(discretizer,
                                                  q[:,a])
            q_fns.append(q_fn)
        q_policy = mdp.policy.MinFunPolicy(mdp_obj.actions,
                                q_fns)

        return q_policy
