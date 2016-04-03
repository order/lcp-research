import numpy as np
from mdp.policy import UniformDiscretePolicy
import discrete       
from utils import hash_ndarray

from graphviz import Digraph

###########################
# CHANCE NODE

class MCTSNode(object):
    def __init__(self,state):
        self.state=state
        self.value=0.0
        self.visits=0.0

        self.children # Action_id -> set of next states

    def is_leaf(self):
        return (0 == len(self.children))

    def best_action_uct(self):
        """
        Return the best action according to the
        UCT criterion
        """
        assert(not self.is_leaf())
        raise NotImplementedError()

    def get_child(self,action):
        """
        Get an existing child
        """
        raise NotImplementedError()
     

    def sample_child(self,action):
        """
        Get a (possibly) new child
        """
        raise NotImplementedError()


    def update(self,G):
        
        

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
        Return (decision_node,from_cache)
        """

        # Sample a state
        state = trans_fn.single_transition(self.state,
                                           self.action)
        assert(state.shape == self.state.shape)

        state_hash = hash_ndarray(state) # Get the hash

        # Seen it before
        if state_hash in self.children:
            print 'Hashed state'
            return (self.children[state_hash],True)

        # New node
        print 'Found new state'
        new_node = MCTSDecisionNode(state)
        self.children[state_hash] = new_node
        return (new_node,False)

    def update(self,G):
        self.visits += 1.0
        alpha = (1.0 / self.visits)
        self.value = (1.0 - alpha)*self.value +  alpha * G

    def __str__(self):
        return '{0},{1},{2},{3}'.format(self.state,
                                         self.action,
                                         self.value,
                                         self.visits)
        
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
            cost = cost_fn.single_cost(self.state,
                                       action)
            decision = chance.next_node(trans_fn)

            # Use a_id as the hash
            self.children[a_id] = chance
            self.costs[a_id] = cost
            
            return (chance,cost,decision)

        # Pick best child w.r.t UCT
        best_child = None
        best_cost = None
        best_uct = np.inf # Minimize cost
        for a_id in self.children:
            # Using a_id as hash
            cost = self.costs[a_id]
            child = self.children[a_id]
            
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
        alpha = (1.0 / self.visits)
        self.value = (1.0 - alpha)*self.value +  alpha * G

    def __str__(self):
        return '{0},{1},{2}'.format(self.state,
                                     self.value,
                                     self.visits)
        

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

        assert(1 == len(root_state.shape)) # vector
        self.root_node = MCTSDecisionNode(root_state)

    def path_to_leaf(self):
        A = len(self.actions)
        curr = self.root_node

        path = []
        while True:
            # From the decision get the best
            res = curr.best_node(self.actions,
                                 self.trans_fn,
                                 self.costs,
                                 self.discount)
            # From current decision get a chance node,
            # a cost, and the next decision node
            (chance,cost,decision) = res
            assert(isinstance(chance,MCTSChanceNode))
            assert(isinstance(decision,MCTSDecisionNode))
            path.append((curr,cost,chance))
            print (curr,cost,chance)
            
            if decision.is_leaf():
                res = decision.best_node(self.actions,
                                         self.trans_fn,
                                         self.costs,
                                         self.discount)
                (chance,cost,new_leaf) = res     
                path.append((decision,cost,chance))
                path.append((new_leaf,None,None))
                return path
            if(len(path) >= 6):
                display_tree(self.root_node)
            assert(len(path) < 6)


    def rollout(self,path):
        A = self.actions.shape[0]
        (decision,_,_) = path[-1]
        state = decision.state
        
        G = 0.0
        for t in xrange(self.horizon):
            a_id = np.random.randint(A)

            # Transition
            action = self.actions[a_id]
            states = self.trans_fn.single_transition(state,
                                                     action)
            cost = self.costs.single_cost(state,
                                          action)
                
            G += (self.discount**t)*cost
        return G

    def backup(self,path,G):
        while path:
            (decision,cost,chance) = path.pop()
            if chance:
                # Chance node exists, so update
                # the chance node and return
                chance.update(G)
                G = cost + self.discount*G
            decision.update(G)
            
def print_path(path):
    for (i,(dec,cost,chance)) in enumerate(path):
        print '{0}: [{1}] -> [{3}] (R: {2})'.format(i,
                                                    dec,
                                                    cost,
                                                    chance)


def display_tree(root):
    dot = Digraph(comment='MCTS Tree')
    fringe = [(None,root)]
    while fringe:
        (parent,child) = fringe.pop()
        child_str = str(child)
        child_hash = str(hash(child))
        if isinstance(child,MCTSDecisionNode):
            dot.attr('node',shape='box',style='')
            dot.node(child_hash,child_str)
        else:
            assert(isinstance(child,MCTSChanceNode))
            dot.attr('node',shape='ellipse',style='filled')
            dot.node(child_hash,child_str)
            
        if parent:
            parent_hash = str(hash(parent))
            dot.edge(parent_hash,child_hash)
            
        for grandchild in child.children.values():
            fringe.append((child,grandchild))
    dot.render('data/test.gv', view=True)
