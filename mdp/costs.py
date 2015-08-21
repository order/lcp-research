class QuadraticCost(object):
    def __init__(self,coeff):
        self.coeff = coeff
    
    def cost(self,points,action):
        return (points**2).dot(self.coeff)