import utils

class BasisImproverWrapper(object):
    def __init__(self,
                 solver,
                 basis_improver):
        self.solver = solver
        self.basis_improver = basis_improver
        self.termination_conditions = []

    def solve(self):
        iterator = self.solver.iterator
        
        done = False
        utils.banner('Replace with actual termination conditions')
        for i in xrange(2):
            utils.banner('Basis outerloop iteration')
            # Inner loop solve
            self.solver.solve()

            # Generate a new basis function
            # (N,1) ndarray or sparse matrix
            (basis_fn,block_id) = self.basis_improver.\
                                  improve_basis(iterator)
            iterator.update_basis(basis_fn,block_id) # update


class BasisImprover(object):
    def improve_basis(self,iterator):
        raise NotImplementedError()

class BellmanValueResidualBasisImprover(BasisImprover):
    def __init__(self):
        pass
    def improve_basis(self,iterator):
        utils.banner('Will not always have access to MDP object')
        mdp_obj = iterator.mdp_obj
        v = iterator.get_value_vector()
        res = mdp_obj.get_value_residual(v)
        return (res,0) # Value is 0
        
