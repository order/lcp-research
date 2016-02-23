import numpy as np

from solver import MDPIterator
from utils.parsers import KwargParser

class TDIterator(MDPIterator):
    def __init__(self,discretizer,**kwargs):
        parser = KwargParser()
        parser.add('policy')
        parser.add('num_samples')
        parser.add('state_distribution')
        parser.add('step_size')
        args = parser.parser(kwargs)
        self.__dict__.update(args)
        
        self.discretizer = discretizer

        N = discretizer.get_num_nodes()
        self.v = np.zeros(N) # Assume tabular form
        self.iteration = 0

    def next_iteration(self):
        # 1) Sample S
        # 2) Get action from policy
        # 3) Get successor state S' and cost C
        # 4) V(S) <- V(S) + alpha[C + gamma V(S') - V(S)]
        alpha = self.step_size
        gamma = self.discretizer.discount

        S = self.state_distribution.sample(self.num_samples)
        A = self.policy.get_decisions(S)
        R = self.problem.cost_obj.evaluate(S,actions=A)
        S_next = self.discretizer.problem.next_states(S,A)

        # TODO
        # What is the right way of assuming the tabular form?
        # Suppose that we're doing LSTD using an 'interpolation basis'
        

    def get_value_vector(self):
        return self.v
        
    def get_iteration(self):
        return self.iteration
