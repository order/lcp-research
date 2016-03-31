import numpy as np
from mdp.policy import UniformDiscretePolicy
import discrete       

###########################
# CHANCE NODE

class MCTSChanceNode(object):
    def __init__(self, state, action):
        self.state = state
        self.action = action

        self.value = 0.0
        self.visits = 0.0
        
        self.children = {} # State -> decision node
        
    def next_node(self,trans_fn):
        """
        Currently assuming small branching factor
        """

        # Sample a state
        state = trans_fn.transition(self.state,
                                    self.action)[0,:,:]
        assert(state.shape == self.state.shape)

        # Seen it before
        if state in self.children:
            return self.children[state]

        # New node
        new_node = MCTSDecisionNode(state)
        self.children[state] = new_node
        return new_node

    def update(self,G):
        self.visits += 1.0
        self.value += (1.0 / self.visits) * G
        
###############################
# DECISION NODE
class MCTSDecisionNode(object):
    """
    Decision node for the Monte Carlo Tree Search
    """
    def __init__(self,state):
        self.state = state
        
        self.children = {} # Action id -> chance node
        self.costs = {} # Action id -> costs

        self.visits = 0 # visit counter
        self.value = 0 # Total value

    def is_leaf(self):
        return 0 == len(self.children)

    def best_node(self, actions, trans_fn, cost_fn, discount):
        """
        Find the best action w.r.t. UCT
        """
        A = actions.shape[0]

        # Unexplored action; just pick it
        # TODO: initialize with value estimates?
        if len(self.children) < A:
            a_id = len(self.children) # pick in numerical order
            action = actions[a_id,:]
            
            chance = MCTSChanceNode(self.state, action)
            cost = cost_fn.cost(self.state, action)
            decision = chance.next_node(trans_fn)
            
            self.children[action] = chance
            self.costs[action] = cost
            
            return (chance,cost,decision)

        # Pick best child w.r.t UCT
        best_child = None
        best_cost = None
        best_uct = np.inf # Minimize cost
        for action in self.children.keys():
            cost = self.costs[action]
            child = self.children[action]
            
            exploit = cost + discount*child.value
            explore =  np.sqrt(np.log(self.visits)
                               / child.visits)
            uct = exploit - explore
            
            if best_uct > uct:
                best_uct = uct
                best_child = child
                best_cost = cost
                
        decision = best_child.next_node(trans_fn)
        return (best_child,best_cost,decision)

    def update(self,G):
        self.visits += 1.0
        self.value += (1.0 / self.visits) * G

###################################
# MONTE CARLO TREE
class MonteCarloTree(object):
    def __init__(self,transition_function,
                 cost_function,
                 discount,
                 actions,
                 root_state,
                 horizon=100):
        """
        Expand later to include rollout policies and
        value functions
        """
        
        self.trans_fn = transition_function
        self.costs = cost_function
        self.discount = discount
        self.actions = actions
        self.horizon = horizon
        
        self.root_node = MCTSDecisionNode(root_state)

    def path_to_leaf(self):
        A = len(self.actions)
        curr = self.root_node

        # Path only contains decision nodes
        path = [curr]
        while True:
            res = curr.best_node(self.actions,
                                 self.trans_fn)
            (chance,cost,decision) = res
            assert(isinstance(chance,MCTSChanceNode))
            assert(isinstance(decision,MCTSDecisionNode))
            path.append(res)
            if decision.is_leaf():
                return path
            assert(1e3 > len(path))

    def rollout(self,path):
        A = self.actions.shape[0]
        (_,_,decision) = path[-1]
        state = decision.state
        
        G = 0.0
        for t in xrange(self.horizon):
            a_id = np.random.randint(A)
            action = self.actions[a_id]
            state = self.trans_fn(state, action)[0,:,:]
            cost = self.costs.cost(state, action)
            G += (self.discount**t)*cost
        return G

    def backup(self,path,G):
        while path:
            (chance,cost,decision) = path.pop()
            chance.update(G)
            decision.update(G)
            G = curr.reward + self.discount*G
            
