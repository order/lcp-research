from utils.parsers import KwargParser

import solvers
from solvers.kojima import KojimaIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *
import config
import bases

import time

class ProjectiveGenerator(config.SolverGenerator):
    def __init__(self,**kwargs):
        # Parsing
        parser = KwargParser()
        parser.add('value_regularization',1e-12)
        parser.add('flow_regularization',1e-12)
        parser.add('termination_conditions')
        parser.add('recorders')
        parser.add_optional('notifications')
        parser.add('basic_basis_generator')
        args = parser.parse(kwargs)

        # Dump into self namespace
        self.__dict__.update(args)

    def generate(self,discretizer):
        mdp_obj = discretizer.build_mdp()

        # Get sizes
        A = mdp_obj.num_actions
        n = mdp_obj.num_states
        N = (A+1)*n
        gamma = mdp_obj.discount

        # Actual basis generator wraps the
        # basic basis generator
        # (Which doesn't deal with non-phyiscal states)
        basis_generator = bases.BasisGenerator(self.basic_basis_generator)

        points = discretizer.get_node_states()

        Phi = basis_generator.generate_basis(points)
        assert((n,n) == Phi.shape)
        Phi_other = np.eye(n)
        assert(np.linalg.norm(Phi - Phi_other) / n < 1e-12)
        (d,k) = Phi.shape
        assert(d == n)
        assert(k <= n)
        K = (A+1)*k

        #if basis_gen.isortho():
        Q = Phi
        #else:
            #Orthogonalize using QR decomposition
        #    [Q,R] = sp.linalg.qr(Phi,mode='economic')
        #    assert(Q.shape == Phi.shape)


        # Find the in-basis approximations for E = I - gamma* P
        # Done with least-squares:
        # E   ~= Phi U
        # E.T ~= Phi W
        
        BigU_blocks = [[None]]
        for a in xrange(A):
            P = mdp_obj.transitions[a]
            E = sps.eye(n) - gamma * P
            U = linalg.lsmr_matrix(Q,E)
            W = linalg.lsmr_matrix(Q,E.T)

            BigU_blocks[0].append(U)
            BigU_blocks.append([-W] + [None]*A)

        # Construct the block basis:
        # Phi 0   ... 0
        # 0   Phi ... 0
        #     ...
        # 0   ...     Phi
        BigQ = sps.block_diag([Q]*(A+1))
        assert((N,K) == BigQ.shape)

        # Construct the block coefficients:
        # 0 U_1 ... U_A
        # -W_1  0 ... 0
        #         ...
        # -W_A  0 ... 0
        BigU = sps.bmat(BigU_blocks)
        assert((K,N) == BigU.shape)

        lcp_obj = mdp_obj.build_lcp()
        M = lcp_obj.M
        approx_M = BigQ.dot(BigU)
        print 'Approximation residual norm:', linalg.norm(M - approx_M)
        quit()  


###########################################
# Build a PROJECTIVE LCP

def build_projective_lcp(mdp_obj,discretizer,**kwargs):
    print 'Building projected LCP object...'
    start = time.time()

    # Get sizes
    A = mdp_obj.num_actions
    n = mdp_obj.num_states
    N = (A+1)*n

    basis = kwargs.get('basis','fourier')
    if basis == 'identity':
        K = n
        assert((A+1)*K == N)
    else:
        K = kwargs.get('K',100)    
    print '\tUsing {0} {1} vectors as basis'.format(
        K, basis)
    
    # Generate the LCP
    lcp_obj = mdp_obj.tolcp()
    M = lcp_obj.M.todense()

    print '\tLCP {0}x{0}'.format(N)
    assert((N,N) == M.shape)
    
    # Get the points in the state-space
    points = discretizer.get_node_states()
    (P,_) = points.shape
    assert(P == n)

    # Generate the basis
    nan_idx = np.where(np.any(np.isnan(points),axis=1))[0]
    if basis == 'random_fourier':
        fourier = bases.RandomFourierBasis(num_basis=K,
            scale=1.0)
        basis_gen = bases.BasisGenerator(fourier)
        Phi = basis_gen.generate_block_diag(points,(A+1),\
                                            special_points=nan_idx)
    elif basis == 'regular_rbf':
        # Generate grid of frequencies
        x_grid = np.linspace(-4,4,12)
        v_grid = np.linspace(-6,6,12)        
        [X,Y] = np.meshgrid(x_grid,v_grid)
        centers = np.column_stack([X.flatten(),Y.flatten()])
        
        RBF = bases.RegularRadialBasis(centers=centers,
            bandwidth=0.75)
        basis_gen = bases.BasisGenerator(RBF)
        Phi = basis_gen.generate_block_diag(points,(A+1),\
                                            special_points=nan_idx)        
    elif basis == 'identity':
        # Identity basis; mostly for testing
        Phi = np.eye(N)
    else:
        raise NotImplmentedError()
        
    ortho = True
    if ortho:
        [Q,R] = sp.linalg.qr(Phi,mode='economic')
        assert(Q.shape == Phi.shape)
        Phi = Q

    U = np.linalg.lstsq(Phi,M)[0]
    Mhat = Phi.dot(U)
    assert((N,N) == Mhat.shape)

    proj_lcp_obj = lcp.ProjectiveLCPObj(sps.csr_matrix(Phi),\
                                        sps.csr_matrix(U),lcp_obj.q)
    print 'Build projected LCP object. ({0:.2f}s)'\
        .format(time.time() - start)  

    return proj_lcp_obj
###########################################
# Solve for the VALUE FUNCTION
    
def solve(discretizer,mdp_obj,**kwargs):
    discount = kwargs.get('discount',0.99)
    max_iter = kwargs.get('max_iter',1000)
    thresh = kwargs.get('thresh',1e-6)
    method = kwargs.get('method','projective')
    
    # Build the LCP object
    if method in ['kojima']:
        print 'Building LCP object...'
        start = time.time()
        lcp_obj = mdp_obj.tolcp()        
        print 'Built LCP object. ({0:.2f}s)'.format(time.time() - start)

    # Build the Projective LCP object
    if method in ['projective']:
        proj_lcp_obj = build_projective_lcp(mdp_obj,discretizer,**kwargs)
        proj_lcp_obj.to_csv('test_proj_lcp')

    # Select the iterator
    if method == 'value':
        iter = ValueIterator(mdp_obj)
    elif method == 'kojima':
        iter = KojimaIPIterator(lcp_obj)
    elif method == 'projective':
        print "Initializing Projected interior point iterator..."
        start = time.time()
        iter = ProjectiveIPIterator(proj_lcp_obj)
        print "Finished initialization {0}s".format(time.time() - start)
    else:
        raise NotImplmentedError()

    # Set up the solver object
    solver = solvers.IterativeSolver(iter)

    # Add termination conditions
    max_iter_cond = MaxIterTerminationCondition(max_iter)
    primal_diff_cond = PrimalChangeTerminationCondition(thresh)
    solver.termination_conditions.append(max_iter_cond)    
    solver.termination_conditions.append(primal_diff_cond)

    # Set up notifications
    if method in ['value']:
        val_change_term = ValueChangeTerminationCondition(thresh)
        solver.termination_conditions.append(val_change_term)
        solver.notifications.append(ValueChangeAnnounce())        
    elif method in ['kojima','projective']:
        res_change_term = ResidualTerminationCondition(thresh)
        solver.termination_conditions.append(res_change_term)
        solver.notifications.append(ResidualAnnounce())
        solver.notifications.append(PotentialAnnounce())
        num_states = mdp_obj.num_states
        solver.notifications.append(PrimalDiffAnnounce())
        #sl = slice(0,num_states)
        #solver.notifications.append(PrimalDiffAnnounce(indices=sl))

        solver.recorders.append(PrimalRecorder())

    # Actually do the solve
    print 'Starting {0} solve...'.format(type(iter))
    start = time.time()
    solver.solve()
    print 'Done. ({0:.2f}s)'.format(time.time() - start)

    
    # Extract the cost-to-go function
    if method == 'value':
        J = iter.get_value_vector()
    elif method in ['kojima','projective']:
        N = mdp_obj.num_states
        J = iter.get_primal_vector()[:N]

    # Place in an interpolating evaluator
    value_fun_eval = mdp.BasicValueFunctionEvaluator(discretizer,J)
    # TODO: use the low-dimension weights and basis if projective

    primals = np.array(solver.recorders[0].data)

    return [value_fun_eval,primals]
