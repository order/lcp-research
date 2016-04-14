import numpy as np

from mdp.policy import IndexPolicy
from mcts import MonteCarloTree

class MCTSPolicy(IndexPolicy):
    def __init__(self,problem,
                 actions,
                 rollout_policy,
                 value_fn,
                 horizon,
                 budget):
        self.trans_fn = problem.gen_model.trans_fn
        self.cost_fn = problem.gen_model.cost_fn
        self.discount = problem.discount
        self.actions = actions
        self.policy = rollout_policy
        self.val_fn = value_fn
        self.horizon = horizon
        self.budget = budget

        self.action_dim = actions.shape[1]

    def get_single_decision_index(self,point):
        tree = MonteCarloTree(self.trans_fn,
                              self.cost_fn,
                              self.discount,
                              self.actions,
                              self.policy,
                              self.val_fn,
                              point,
                              self.horizon)
        tree.grow_tree(self.budget)
        a_id = np.argmax(tree.root_node.action_visits)
        return a_id
        
    def get_decision_indices(self,points):
        (N,D) = points.shape
        actions = np.empty(N)
        for i in xrange(N):
            actions[i] = self.get_single_decision_index(
                points[i,:])
        return actions
    
    def get_single_decision(self,point):
        aid = self.get_single_decision_index(point)
        return self.actions[aid,:]
    
    def get_decisions(self,points):
        aids = self.get_decision_indices(points).astype('i')
        return self.actions[aids,:]
