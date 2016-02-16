class QEstimator(object):
    def get_q_values(self,states):
        raise NotImplementedError()

class BasicQEstimator(QEstimator):
    def __init__(self,discretizer,v_fn):
        self.discretizer = discretizer
        self.v_fn = v_fn
        
    def get_q_values(self,states):
        (N,d) = states.shape
        A = self.discretizer.get_num_actions()
        Actions = self.discretizer.get_actions()

        Q = np.empty((N,A))

        gamma = self.discretizer.discount
        for a in actions:
            action = actions[a,:]
            costs = discretizer.cost_obj.evaluate(states,action=action)
            xnext = discretizer.remap_states(states,action)
            value = self.v_fn.evaluate(xnext)
            
            Q[:,a] = costs + gamma * value

        return Q
