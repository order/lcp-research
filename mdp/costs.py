class QuadraticCost(object):
    def __init__(self,coeff,**kwargs):
        self.coeff = coeff
        self.override = kwargs
    
    def cost(self,ids,points,action):
        assert(len(ids) == points.shape[0])
        
        costs = (points**2).dot(self.coeff)
        for (node_id,cost_override) in self.override.items():
            costs[node_id] = cost_override
        return costs