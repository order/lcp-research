class Problem(object):
    def __init__(self,gen_model,
                 action_limits,
                 discount):
        self.gen_model = gen_model
        assert(gen_model.action_dim == len(action_limits))
        self.action_limits = action_limits
        assert(0 < discount < 1)
        self.discount = discount
