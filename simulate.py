import numpy as np
from config.instance.double_integrator import DoubleIntegratorConfig as inst_conf
from mdp.policy import ConstantPolicy
    
class SimulationObject(object):
    def __init__(self,problem,policy):
        self.policy = policy
        self.problem = problem
        self.cost_obj = problem.cost_obj

    def next(self,x):
        (N,d) = x.shape
        assert(not np.any(x == np.nan))

        # Get the actions
        actions = self.policy.get_decisions(x)

        # Run the physics
        x_next = self.problem.next_states(x,actions)
        assert(x.shape == x_next.shape)
        assert(not np.any(x == np.nan))
        # NAN-out unfixed oobs
        
        oobs = self.problem.out_of_bounds(x_next)
        x_next[oobs,:] = np.nan
        
        costs = self.cost_obj.evaluate(x_next) # Should have NAN handling
        
        return (x_next,costs,oobs)       

def uniform(N,boundaries):
    D = len(boundaries)
    x = np.empty((N,D))
    for (i,(low,high)) in enumerate(boundaries):        
        x[:,i] = np.random.uniform(low,high,N)
    return x 

def simulate(problem,R,I):
    D = problem.get_dimension()
    boundaries = problem.get_boundary()

    oob_cost = problem.cost_obj.evaluate(np.full((1,D),np.nan))
    costs = np.full((R,I),oob_cost) # Default to oob_cost
    states = np.full((R,I,D),np.nan) # Default to nan.

    policy = ConstantPolicy(0 * np.ones(2))
    sim_obj = SimulationObject(problem,policy)
    x = uniform(R,boundaries)
    in_bounds = np.ones(R,dtype=bool)
    for i in xrange(I):
        (x,c,new_oob) = sim_obj.next(x[in_bounds,:])

        states[in_bounds,i,:] = x
        costs[in_bounds,i] = c

        in_bounds[in_bounds] = 1 - new_oob # Update what is `inbounds'


    return (costs,states)


problem = inst_conf().configure_problem_instance()

(costs,states) = simulate(problem,2,10)
print states
print costs
