import heapq
import defaultdict from collections
import numpy as np

class MCTSNode(object):
    def __init__(self,state,parent):
        self.state = state
        self.children = defaultdict(list) # Action -> list of nodes
        self.child_value = {}
        self.child_visits = {}

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
    
    def add_child(self,a_id,state):
        assert(not self.is_child(a_id,state))
        child = MCTSNode(state)
        self.children[a_id].append(child)

    def update(self,G):
        # Update the node with a new visit and return
        self.vists += 1.0
        self.value += G


class MonteCarloTreeSearch(object):
    def __init__(self,mdp_obj,root_state):
        self.mdp_obj = mdp_obj
        
        self.root_node = MCTSNode(root_state,None,None)
        self.leaf_heap = heapq.heapify([root_node])

        self.horizon = 100 # TODO: pick rationally

    def get_next_node(self):
        v = self.root_node
        Vs = [v]
        As = []
        
        while v.has_children():
            a_id = v.best_action()
            state = # TODO
            if 
            Vs.append(v)
            As.append(a_id)

        assert(len(Vs) == len(As) + 1)

    def expand(self):
        (Vs,As) = self.get_next_node()
        
        Rs = self.playout(path,self.horizon)
        

        
