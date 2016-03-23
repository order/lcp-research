import heapq
import defaultdict from collections
import numpy as np
from mdp.policy import UniformDiscretePolicy
from utils.discretizer import IrregularSplit

def partition_samples(S,K):
    """
    Parition samples into D^K partitions by order statistic
    """
    (N,D) = S.shape

    Prcnt = np.linspace(0,100,K+1) # Percentiles
    Breaks = np.empty((K+1,D))
    Cutpnts[0,:] = -np.inf
    Cutpnts[-1,:] = np.inf
    
    for d in xrange(D):
        Cutpnts[1:-1,D] = np.percentile(S[:,D],Prcnt)[1:-1]
    
    return IrregularSplit(Cutpnts)
        

class MCTSChanceNode(object):
    def __init__(self,state,action):
        self.state = state
        self.action = action

        self.children = set()
        self.visits = 0

    def total_val(self):
        assert(len(self.children)>0)
        child_agg_value = [child.value_agg for child in self.children]
        return np.sum(child_agg_value)
        
    def expected_value(self):        
        return self.total_val() / self.visits

    def 

class MCTSDecisionNode(object):
    """
    Decision node for the Monte Carlo Tree Search
    """
    def __init__(self,state):
        self.state = state
        self.children = {} # Action -> chance nodes

        self.visits = 0.0 # visit counter
        self.value_agg = 0.0 # Total value

    def best_action(self,A):
        """
        Get the action index of the best action from a UCB1 
        perspective
        """

        # First check if all of the actions have been visited
        # at least once
        if len(self.children) < A-1:
            missing = set(range(A)) - set(self.children.keys())
            return list(missing)[0] # Return any unvisited action

        # Then calculate the UCB1 score for all actions
        # Any non-determinism is marginalized out
        UCBs = []
        for a in xrange(A):
            assert(a in self.children)

            # Compact notation
            Q = self.child_value[a]
            Nc = self.child_visits[a]
            N = self.visits
                
            # UCB1
            ucb = (Q / Nc) + np.sqrt(2.0*np.log(N) / Nc)

            UCBs.append(ucb)
        a_id = np.argmax(UCBs)
        
        return a_id

    def is_child(self,a_id,state):
        """
        Check if the state is a known success of the node
        and action
        """
        for child in self.children[a_id]:
            if child.state == state:
                return True
        return False

    def get_child(self,a_id,state):
        """
        Check if the state is a known success of the node
        and action
        """
        for child in self.children[a_id]:
            if child.state == state:
                return child
        return None        
    
    def add_child(self,a_id,state):
        assert(not self.is_child(a_id,state))
        child = MCTSNode(state)
        self.children[a_id].append(child)
        return child

    def update(self,G):
        # Update the node with a new visit and return
        self.vists += 1.0
        self.value += G


class MonteCarloTreeSearch(object):
    def __init__(self,mdp_obj,root_state):
        self.mdp_obj = mdp_obj
        self.discount = mdp_obj.discount
        
        self.root_node = MCTSNode(root_state,None,None)
        self.leaf_heap = heapq.heapify([root_node])

        self.horizon = 100 # TODO: pick rationally
        self.num_expansions = 500

        # Use the uniform discrete policy unless overriden
        A = mdp_obj.num_actions
        self.rollout_policy = UniformDiscretePolicy(A)

    def build_tree(self):
        for i in xrange(self.num_expansions):
            self.expand()

    def get_path_to_next_node(self):
        """
        Follows the following tree policy:
        take the action that maximizes expected UCB1.
        Expand the first node that runs off the tree.
        """
        v = self.root_node
        Path = []

        Depth = 0
        while v.has_children():
            # Get the best action so far (UCB1)
            a_id = v.best_action()

            # Simulate the action
            s = v.state
            (new_s,r) = self.mdp_obj.next_state(s,a_id)
            Path.append((v,a_id,r))

            # Get the child node corresponding to the
            # simulated next state
            child = v.get_child(a_id,new_s)
            if not child:
                # Ran off the tree, this is a new node; stop
                child = v.add_child(a_id,new_s)
                return (child,Path) # Must eventually reach here.
            
            v = child
            Depth += 1
            assert(Depth < 1e3) # Primative loop check.

    def rollout(self,node,H):
        """
        Simulate out from given node for H steps
        """
        
        G = 0.0
        s = node.state
        policy = self.rollout_policy
        gamma = self.discount
        D = 1.0
        for h in xrange(H):
            a = policy.get_action_indices(np.array([s]))[0]
            (s,r) = self.mdp_obj.next_state_index(s,a)
            G += D * r
            D *= gamma
        return G
            

    def expand(self):
        # Find the next node to expand
        (node,path) = self.get_path_to_next_node()

        # Play out from that node using the default_policy
        G = self.rollout(node,self.horizon) # returns the returns

        # Update the node
        node.update(G)

        # Update the path
        gamma = self.discount
        while len(path) > 0:
            (v,a,r) = path.pop()
            G = r + gamma * G
            v.update(G)
        

        
