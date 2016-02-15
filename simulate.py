
import numpy as np
from config.instance.hallway_continuous import HallwayConfig as inst_conf

class Policy(object):
    def decide(self,points):
        raise NotImplementedError()

class ConstantPolicy(Policy):
    def __init__(self,action):
        assert(1 == len(action.shape))
        self.action = action
    def decide(self,points):
        (N,d) = points.shape
        (u,) = self.action.shape
        decision = np.tile(self.action,(N,1))
        assert((N,u) == decision.shape)
        return decision
        
    
class SimulationObject(object):
    def __init__(self,discretizer,policy):
        self.policy = policy
        self.physics = discretizer.physics
        self.cost_obj = discretizer.cost_obj
        self.exception_state_remappers = discretizer.exception_state_remappers
        self.exception_node_mappers = discretizer.exception_node_mappers

    def next(self,x):
        (N,d) = x.shape
        assert(not np.any(x == np.nan))
        assert(1 == N) # (1,d) array

        # Get the actions
        action = self.policy.decide(x)

        # Run the physics
        x_next = self.physics.remap(x,action=action)

        # Fix some boundary violations
        for remapper in self.exception_state_remappers:
            x_next = remapper.remap(x_next)

        # Report unfixable oobs
        oob = {}
        for mapper in self.exception_node_mappers:
            oob = mapper.states_to_node_dists(next_states,ignore)
            if len(oob) > 0:
                x_next[:,:] = np.nan
                break
        terminate = len(oob) > 0
        cost = self.cost_obj.eval(x_next)

        return (x_next,cost,terminate)       

def uniform(boundaries):
    D = len(boundaries)
    x = np.empty((1,D))
    for (i,(low,high)) in enumerate(boundaries):        
        x[0,i] = np.random.uniform(low,high)
        
    return x 

def simulate(discretizer,runs,iters):
    D = discretizer.get_dimension()
    boundaries = discretizer.get_basic_boundary()
    
    costs = np.empty((runs,iters))
    states = np.empty((runs,iters,D))

    policy = ConstantPolicy(np.zeros(1))
    sim_obj = SimulationObject(discretizer,policy)
    for r in xrange(runs):
        print 'Run',r
        x = uniform(boundaries)
        for i in xrange(iters):
            (x,cost,terminal) = sim_obj.next(x)

            if terminal:
                sl = slice(i,iters)
            else:
                sl = slice(i,i+1)

            states[r,sl,:] = x
            costs[r,sl] = cost

            if terminal:
                break

    return (costs,states)


discretizer = inst_conf().configure_instance_builder()

(costs,states) = simulate(discretizer,1,10)
print states
print costs
