class BasisImproverWrapper(object):
    def __init__(self,
                 solver,
                 basis_improver):
        self.solver = solver
        self.basis_improver = basis_improver
        self.termination_conditions = []

    def solve(self):
        iterator = solver.iterator()
        
        done = False
        while not done:
            # Check for termination
            for term_cond in self.termination_conditions:
                if term_cond.isdone(iterator):
                    done = True
                    break
            if done:
                break

            # Inner loop solve
            solver.solve()

            # Generate a new basis function
            # (N,1) ndarray or sparse matrix
            (basis_fn,block_id) = self.basis_improver.\
                                  improve_basis(iterator)
            iterator.update_basis(new_basis_fn,block_id) # update


class BasisImprover(object):
    def improve_basis(self,iterator):
        raise NotImplementedError()

class BellmanValueResidualBasisImprover(BasisImprover):
    def __init__(self):
        pass
    def improve_basis(self,iterator):
        mdp_obj = iterator.mdp_obj
        v = iterator.get_value_vector()
        res = mdp_obj.get_value_residual(v)
        return (res,0) # Value is 0
        
