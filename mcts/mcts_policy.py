import numpy as np

from mdp.policies import IndexPolicy
from mcts import MonteCarloTree

class MCTSPolicy(IndexPolicy):
    def __init__(self,problem,
                 actions,
                 rollout_policy,
                 initial_prob,
                 value_fn,
                 horizon,
                 prob_scale,
                 budget):
        self.trans_fn = problem.gen_model.trans_fn
        self.cost_fn = problem.gen_model.cost_fn
        self.discount = problem.discount
        self.actions = actions
        self.rollout_policy = rollout_policy
        self.initial_prob = initial_prob
        self.val_fn = value_fn
        self.horizon = horizon
        self.prob_scale = prob_scale
        self.budget = budget

        self.action_dim = actions.shape[1]

        self.reuse_tree = False
        self.sim_thresh = 1e-15

        self.tree = None

    def get_single_decision_index(self,point):
        subtree_desc = None
        if self.reuse_tree and self.tree:
            subtree_desc = self.tree.find_subtree(point,
                                               self.sim_thresh)
            
        if subtree_desc:
            (a_id,c_id) = subtree_desc
            self.tree.crop_subtree(a_id,c_id)
            explore_budget = self.budget\
                             - self.tree.num_visits()
        else:
            self.tree = MonteCarloTree(self.trans_fn,
                                       self.cost_fn,
                                       self.discount,
                                       self.actions,
                                       self.rollout_policy,
                                       self.initial_prob,
                                       self.val_fn,
                                       point,
                                       self.horizon,
                                       self.prob_scale)
            explore_budget = self.budget
            
            
        self.tree.grow_tree(int(explore_budget))
        a_id = np.argmax(self.tree.root_node.action_visits)
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

    def get_action_dim(self):
        return self.actions.shape[1]
