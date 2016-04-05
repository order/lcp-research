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
        self.state=state
        self.num_actions = num_actions
        
        self.value=0.0
        self.visits=0.0

        self.children = defaultdict(list)
        self.costs = defaultdict(float)

        # Init the heap
        self.ucts = [(-np.inf,a) for a in xrange(num_actions)]

    def is_leaf(self):
        assert(len(self.costs) == len(self.children))
        return (0 == len(self.children))

    def best_action_by_uct(self):
        """
        Return the best action according to the
        UCT criterion
        """
        assert(not self.is_leaf())
        (uct,a_id) = self.ucts[0]
        return a_id
    
    def get_child(self,action):
        """
        Get an existing child
        """
        C = len(self.children[action])
        P = [float(child.visits) / float(self.visits)
             for child in self.children[action]]
        assert(np.abs(np.sum(P) - 1.0) < 1e-15)
        
        i = np.random(C,p=P)
        child = self.children[action][i]
        assert(isinstance(child,MCTSNode))
        return child

    def add_child(self,action,
                  state,
                  cost):
        # Add the new node
        new_node = MCTSNode(state,self.num_actions)
        self.children[action].append(new_node)

        # Add the cost
        if action in self.costs:
            assert(np.abs(self.costs[action] - cost) < 1e-12)
        self.costs[action] = cost

        return new_node

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
        
        C = len(self.children[a_id])
        for i in xrange(C):
            child_state = self.children[a_id][i].state
            if np.sum(np.abs(next_state - child_state)) < 1e-15:
                return self.children[a_id][i]
            
        new_node = self.add_child(a_id,next_state,cost)            
        return new_node

    def update(self,action,
               discount,
               G):
        # Update visits
        self.visits += 1.0

        # Update return based on immediate cost
        G = self.costs[action] + discount*G

        # Update value
        a = 1.0 / self.visits
        self.values *= (1 - a)
        self.values += a * G

        # Ensure that the a_id was the top of the heap
        (old_uct,a_id) = self.ucts[0]
        assert(action == a_id) # Used best

        # Aggregate expected value and visits
        children = self.children[action]
        C = len(children)
        exp_value = 0.0
        total_visits = 0.0
        for child in children:            
            total_visits += child.visits
            exp_value += child_visits * child.value
        exp_value /= total_visits
        # TODO: cache this rather than recalculate
            
        explore = np.sqrt(np.log(self.visits) / total_visits)
        c = np.sqrt(2.0) # Tradeoff between explore and exloit
        new_uct = exp_value - c * explore

        heapq.heapreplace(self.ucts,(new_uct,a_id))

        return G
        

    def __str__(self):
        return 'N<{0},{1},{2}>'.format(self.state,
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

    def backup(self,path,G):
        while path:
            G = path.pop().update(G)

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
