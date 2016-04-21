import numpy as np
from mdp.policy import UniformDiscretePolicy
import discrete       
from utils import hash_ndarray

from collections import defaultdict
from graphviz import Digraph
import heapq

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

###########################
# CHANCE NODE

class MCTSNode(object):
    NODE_ID = 0
    def __init__(self,
                 state,
                 num_actions,
                 init_prob_fn,
                 prob_scale):
        A = num_actions
        self.state=state
        self.num_actions = A
        self.id = MCTSNode.NODE_ID
        MCTSNode.NODE_ID += 1

        self.V=np.inf # State value
        self.total_visits=0 # Total visits

        self.children = defaultdict(list) # a_id -> Node list
        self.costs = np.full(A,np.nan) # Costs
        
        self.Q = np.full(A,np.inf)# Q-values
        self.QVar = np.full(A,np.inf)
        
        self.action_visits = np.zeros(A,dtype='i') # N values

        self.init_prob_fn = init_prob_fn # Probability generating functions
        self.P = init_prob_fn.get_single_prob(state) # Initial probabilities 
        self.B = prob_scale # Relative weight of exploration terms

    def is_leaf(self):
        return (0 == len(self.children))

    def has_unexplored(self):
        return (len(self.children) != self.num_actions)

    def get_ucb(self,a):
        return ucb1(self.Q[a],
                    self.total_visits,
                    self.action_visits[a])

    def get_silver_score(self,a):
        """
        Score used in Silver's "Mastering Go" paper
        """
        N = self.action_visits[a]
        P = self.P[a]
        Q = self.Q[a]
        B = self.B
        return  Q + B * P / (1.0 + N)

    def best_action(self):
        """
        Return the best action
        """
        A = self.num_actions
        best_U = np.inf
        best_a = -1
        for a in xrange(A):
            U = self.get_silver_score(a)
            #U = self.get_ucb(a)
            if U < best_U:
                best_U = U
                best_a = a
        return best_a
    
    def get_child(self,a_id):
        """
        Get an existing child
        """
        C = len(self.children[a_id])
        P = self.actions_visits / self.total_visits
        assert(np.abs(np.sum(P) - 1.0) < 1e-15)
        
        i = np.random(C,p=P) # Pick a random child
        child = self.children[a_id][i]
        assert(isinstance(child,MCTSNode))
        return child

    def add_child(self, a_id, state, cost):
        # Add cost, or ensure same as old value
        # (assumes deterministic cost)
        if len(self.children[a_id]) > 0:
            assert(np.abs(self.costs[a_id] - cost) < 1e-12)
        else:
            self.costs[a_id] = cost
        
        # Add the new node
        new_node = MCTSNode(state,
                            self.num_actions,
                            self.init_prob_fn,
                            self.B)  
        self.children[a_id].append(new_node)

        return new_node

    def find_node(self,a_id,target):
        C = len(self.children[a_id])
        for i in xrange(C):
            child_state = self.children[a_id][i].state
            if np.sum(np.abs(target - child_state)) < 1e-12:
                # Already exists...
                return i
        return None

    def get_next_child(self, actions,
                             trans_fn,
                             cost_fn):
        best_aid = self.best_action()
        C = len(self.children[best_aid])
        if C == 0 or np.random.rand() < 1.0 / float(C):
            # Randomly add a new state
            return self.sample_and_add_child(best_aid,
                                             actions,
                                             trans_fn,
                                             cost_fn)
        node = np.random.choice(self.children[best_aid])
        return (best_aid,node,False)

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
        if node_idx != None:
            return (a_id,self.children[a_id][node_idx],False)
        
        # create and return new node
        new_node = self.add_child(a_id,next_state,cost)
        return (a_id,new_node,True)

    def init_q(self,a_id,G):
        assert(self.total_visits == 0.0)

    def update(self, a_id, discount, G):
        """
        We did action a_id, and saw a return of G.
        Q(s,a) = E[V(s')]
        """
        
        # Update visits
        self.total_visits += 1.0
        self.action_visits[a_id] += 1.0

        # Update Q
        c = self.costs[a_id]
        G_new = (c + discount * G)
        
        if self.action_visits[a_id] == 1:
            Q_new = G_new
        else:
            mode = 'momentum'
            if mode == 'average':
                t = 1.0 / float(self.action_visits[a_id])
            elif mode == 'momentum':
                t = min(0.05,
                        1.0 / float(self.action_visits[a_id]))
            elif mode == 'best':
                if G_new < self.Q[a_id]:
                    t = 1.0
                else:
                    t = 0.0
            else:
                raise NotImplementedError()
            Q_new = (1.0 - t) * self.Q[a_id] \
                    + t * G_new
        self.Q[a_id] = Q_new

        # Update V
        self.V = np.min(self.Q)

        return self.V
        

    def __str__(self):
        return 'N{0}<{1},{2:0.3f},{3}>'.format(self.id,
                                          self.state,
                                          self.V,
                                          self.total_visits)

    def print_children(self):
        for a_id in self.children:
            print a_id, ':', map(str,self.children[a_id])

###################################
# MONTE CARLO TREE

class MonteCarloTree(object):
    def __init__(self,transition_function,
                 cost_function,
                 discount,
                 actions,
                 rollout_policy,
                 init_prob,
                 value_fn,
                 root_state,
                 horizon,
                 prob_scale):
        """
        Expand later to include rollout policies and
        value functions
        """
        
        self.trans_fn = transition_function
        self.cost_fn = cost_function
        self.discount = discount
        self.actions = actions
        self.rollout_policy = rollout_policy
        self.init_prob = init_prob
        self.value_fn = value_fn
        self.num_actions = actions.shape[0]
        self.horizon = horizon
        
        assert(1 == len(root_state.shape)) # vector
        self.root_node = MCTSNode(root_state,
                                  self.num_actions,
                                  init_prob,
                                  prob_scale)
        
    def grow_tree(self,budget):
        for i in xrange(budget):
            (path,a_list) = self.path_to_leaf()
            leaf = path[-1]
            (G,a_id,state,cost) = self.rollout(leaf.state)
            a_list.append(a_id)

            self.backup(path,a_list,G)        

    def path_to_leaf(self):
        curr = self.root_node

        path = []
        action_list = []
        while True:
            # Add curr node and best action
            path.append(curr)

            # Sample next node
            ret = curr.get_next_child(self.actions,
                                      self.trans_fn,
                                      self.cost_fn)
            (best_aid,next_node,added) = ret
            action_list.append(best_aid)

            if added:
                # New node
                path.append(next_node)
                break
            else:
                # Been here before; continue down path
                curr = next_node
                
            assert(len(path) < 1e4) # Cheap infinite loop check
        return (path,action_list)

    def expand_leaf(self,leaf):
        assert(leaf.is_leaf())
        a_id = 0
        action_0 = self.actions[a_id]
        
        child_state = self.trans_fn.single_transition(leaf.state,
                                                      action_0)
        child_cost = self.cost_fn.single_cost(leaf.state,
                                              action_0)
        child_node = leaf.add_child(a_id,child_state,
                                    child_cost)
        return child_node


    def rollout(self,state):
        A = self.num_actions
        
        G = 0.0
        asc = None # action,state,cost
        rollout  = self.rollout_policy.get_single_decision_index
        for t in xrange(self.horizon):
            # Policy
            a_id = rollout(state)
            action = self.actions[a_id,:]
            
            # Cost
            cost = self.cost_fn.single_cost(state,
                                            action)
            # Transition
            state = self.trans_fn.single_transition(state,
                                                     action)

            if t == 0:
                assert(not asc)
                asc = (a_id,state,cost)
                
            G += (self.discount**t)*cost
        # Use value function for the rest
        v = self.value_fn.evaluate(state[np.newaxis,:])[0]
        G += (self.discount**self.horizon)*v
        assert(asc)
        return (G,asc[0],asc[1],asc[2])

    def backup(self,path,action_list,G):
        assert(len(path) == len(action_list))
        leaf = path.pop()
        rollout_a_id = action_list.pop()
        leaf.init_q(rollout_a_id,G)
        
        while path:
            a_id = action_list.pop()
            G = path.pop().update(a_id,self.discount,G)

##########################################
# Display path
            
def display_path(path,a_list):
    assert(len(path) == len(a_list))
    for (i,node) in enumerate(path):
        if i > 0:
            print '\t->',a_list[i-1]
        print i,node,node.__repr__()

############################################
# Display tree
        
def display_tree(root,**kwargs):
    max_depth = 5
    dot = Digraph(comment='MCTS Tree')
    fringe = [(None,0,root)]
    while fringe:
        (parent_hash,d,child) = fringe.pop()
        child_str = str(child)
        child_hash = str(hash(child))

        if child.is_leaf():
            dot.attr('node',shape='box',style='')
        else:
            dot.attr('node',shape='ellipse',style='')
        dot.node(child_hash,child_str)
            
        if parent_hash:
            label = str(child.state)
            dot.edge(parent_hash,
                     child_hash,
                     label=label)        

        # Add new nodes
        for a_id in child.children:
            if d < max_depth:
                dot.attr('node',shape='diamond',style='filled')
                child_aid_hash = child_hash + str(a_id)
                dot.node(child_aid_hash,'')
            else:
                dot.attr('node',shape='box',style='filled')
                child_aid_hash = child_hash + str(a_id)
                dot.node(child_aid_hash,'...')

            label = '{0},{1:0.2f},{2}'.format(a_id,
                                         child.get_ucb(a_id),
                                         child.costs[a_id])
            dot.edge(child_hash,
                     child_aid_hash,
                     label=label)
            if d < max_depth:
                for gc in child.children[a_id]:
                    fringe.append((child_aid_hash,d+1,gc))
    dot.format='png'
    dot.render('data/test.gv')
    img = mpimg.imread('data/test.gv.png')
    plt.imshow(img)
    plt.title(kwargs.get('title',''))
    plt.show()

#####################################
# UCB1 function

def ucb1(Q, # Value of node
         T, # Visits
         n):
    assert(T >= n)
    if n == 0:
        return -np.inf
    explore = np.sqrt(np.log(T) / float(n))
    B = 1.0 / (1.0 - 0.99)
    c = np.sqrt(2) # Tradeoff between explore and exloit
    return Q - c * explore # minus because costs
