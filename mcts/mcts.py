import numpy as np
from mdp.policy import UniformDiscretePolicy
import discrete       
from utils import hash_ndarray

from collections import defaultdict
from graphviz import Digraph
import heapq

###########################
# CHANCE NODE

class MCTSNode(object):
    def __init__(self,state,num_actions):
        A = num_actions
        self.state=state
        self.num_actions = A

        self.V=0.0 # State value
        self.total_visits=0.0 # Total visits

        self.children = defaultdict(list) # a_id -> Node list
        self.costs = np.full(A,np.nan) # Costs
        self.Q = np.full(A,np.inf)# Q-values
        self.action_visits = np.zeros(A)

        # Init the UCT heap
        self.U = [(-np.inf,a) for a in xrange(num_actions)]

    def is_leaf(self):
        return (0 == len(self.children))

    def best_action_by_uct(self):
        """
        Return the best action according to the
        UCT criterion
        """
        assert(not self.is_leaf())
        (uct,a_id) = self.U[0] # Top of heap
        return a_id
    
    def get_child(self,a_id):
        """
        Get an existing child
        """
        C = len(self.children[a_id])
        P = self.actions_visits / self.total_visits
        assert(np.abs(np.sum(P) - 1.0) < 1e-15)
        
        i = np.random(C,p=P)
        child = self.children[a_id][i]
        assert(isinstance(child,MCTSNode))
        return child

    def add_child(self,a_id,
                  state,
                  cost):
        # Add cost, or ensure same as old value
        # (assumes deterministic cost)
        if len(self.children[a_id]) > 0:
            assert(np.abs(self.costs[a_id] - cost) < 1e-12)
        else:
            self.costs[a_id] = cost
        
        # Add the new node
        new_node = MCTSNode(state,self.num_actions)  
        self.children[a_id].append(new_node)

        return new_node

    def find_node(self,a_id,target):
        C = len(self.children[a_id])
        for i in xrange(C):
            child_state = self.children[a_id][i].state
            if np.sum(np.abs(target - child_state)) < 1e-15:
                # Already exists...
                return i
        return None

    def sample_and_add_child(self,a_id,
                             actions,
                             trans_fn,
                             cost_fn):
        """
        Get a (possibly) new child
        """
        action = actions[a_id]
        next_state = trans_fn.single_transition(self.state,
                                                action)
        cost = cost_fn.single_cost(self.state,
                                   action)
        
        node_idx = self.find_node(a_id,next_state)
        if node_idx:
            return self.children[a_id][state_idx]

        # create and return new node
        new_node = self.add_child(a_id,next_state,cost)
        return new_node

    def update(self,
               a_id,
               discount,
               G):
        """
        We did action a_id, and saw a return of G.
        Q(s,a) = E[V(s')]
        """
        
        # Update visits
        self.total_visits += 1.0
        self.action_visits[a_id] += 1.0

        # Update Q
        c = self.costs[a_id]
        t = 1.0 / self.action_visits[a_id]
        Q_new = (1 - t) * self.Q[a_id] \
                + t * (c + discount * G)
        self.Q[a_id] = Q_new

        # Update V
        self.V = min(self.V,Q_new)


        # Update UCT heap
        new_uct = ucb1(Q_new,
       

        # Ensure that the a_id was the top of the heap
        (top_uct,top_a) = self.U[0]
        assert(top_a==a_id)
        heapq.heapreplace(self.U,(new_uct,a_id))
        
        return self.V
        

    def __str__(self):
        return 'N<{0},{1},{2}>'.format(self.state,
                                      self.V,
                                      self.total_visits)

        

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
        self.cost_fn = cost_function
        self.discount = discount
        self.actions = actions
        self.num_actions = actions.shape[0]
        self.horizon = horizon
        
        assert(1 == len(root_state.shape)) # vector
        self.root_node = MCTSNode(root_state,self.num_actions)

    def path_to_leaf(self):
        curr = self.root_node

        path = []
        action_list = []
        while True:
            path.append(curr)
            if curr.is_leaf():
                assert(len(action_list) == len(path)-1)
                return (path,action_list)
            best_action = curr.best_action_by_uct()
            action_list.append(best_action)
            curr = curr.sample_and_add_child(best_action,
                                             self.actions,
                                             self.trans_fn,
                                             self.cost_fn)
            assert(len(path) < 1e2)

    def expand_leaf(self,leaf):
        assert(leaf.isleaf())
        a_id = 0
        assert(leaf.U[0][1] == a_id)
        action_0 = self.actions[a_id]
        
        child_state = self.trans_fn.single_transition(leaf.state,
                                                      action_0)
        child_cost = self.cost_fn.single_cost(leaf.state,
                                              action_0)
        child_node = leaf.add_child(a_id,child_state,
                                    child_cost)
        return node


    def rollout(self,state):
        A = self.num_actions
        
        G = 0.0
        asc = None # action,state,cost
        for t in xrange(self.horizon):
            a_id = np.random.randint(A)

            # Transition
            action = self.actions[a_id]
            state = self.trans_fn.single_transition(state,
                                                     action)
            cost = self.cost_fn.single_cost(state,
                                            action)
            if t == 0:
                assert(not asc)
                asc = (a_id,state,cost)
                
            G += (self.discount**t)*cost
        assert(asc)
        return (G,asc[0],asc[1],asc[2])

    def backup(self,path,action_list,G):
        while path:
            a_id = action_list.pop()
            G = path.pop().update(a_id,self.discount,G)

def display_tree(root,no_label=False):
    dot = Digraph(comment='MCTS Tree')
    fringe = [(None,root)]
    while fringe:
        (parent_hash,child) = fringe.pop()
        if no_label:
            child_str = ''
        else:
            child_str = str(child)
        child_hash = str(hash(child))

        if child.is_leaf():
            dot.attr('node',shape='box',style='')
        else:
            dot.attr('node',shape='ellipse',style='')
        dot.node(child_hash,child_str)
            
        if parent_hash:
            if no_label:
                label = ''
            else:
                label = str(child.state)
            dot.edge(parent_hash,
                     child_hash,
                     label=label)

        # Add new nodes
        for a_id in child.children:
            dot.attr('node',shape='diamond',style='filled')
            child_aid_hash = child_hash + str(a_id)
            dot.node(child_aid_hash,'')

            if no_label:
                label = ''
            else:
                label = str(a_id)
            dot.edge(child_hash,
                     child_aid_hash,
                     label=label)
            for gc in child.children[a_id]:
                fringe.append((child_aid_hash,gc))
                
    dot.render('data/test.gv', view=True)


def ucb1(Q, # Value of node
         T, # Visits
         n):
    assert(T >= n)
    assert(n > 0)
    explore = np.sqrt(np.log(T) / n)
    c = np.sqrt(2.0) # Tradeoff between explore and exloit
    return Q - c * explore # minus because costs
