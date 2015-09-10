from mdp.node_mapper import PiecewiseConstRegularGridNodeMapper,InterpolatedGridNodeMapper
import numpy as np
import matplotlib.pyplot as plt


def piecewiseconst_regular_test():
    """
    Takes random 2D points, maps them to the nearest grid node, and then maps these
    nodes to their canonical state. Should look regular.
    """
    x_n = 5
    y_n = 8
    eps = 1e-8
    gridder = PiecewiseConstRegularGridNodeMapper((0,1,x_n),(0,1,y_n))

    P = np.random.rand(2500,2)

    (N,d) = P.shape
    assert(d == 2)
    for i in xrange(N):
        state = np.reshape(P[i,:],(1,2))
        node_dist_map = gridder.states_to_node_dists(state)
        if 0 == len(node_dist_map):
            # TODO: check and fix
            break
        assert(0 in node_dist_map)
        node_id = node_dist_map[0].get_unique_node_id()
        canon_state = gridder.nodes_to_states([node_id])
        plt.plot([state[0,0],canon_state[0,0]],[state[0,1],canon_state[0,1]])
    plt.show()
    
def piecewiseconst_regular_test2():
    """
    Check that 
    """
    x_n = 2
    y_n = 3
    z_n = 4
    eps = 1e-8
    gridder = PiecewiseConstRegularGridNodeMapper((0,1,x_n),(0,1,y_n),(0,1,z_n))

    P = gridder.get_node_states()
    Q = gridder.nodes_to_states(gridder.get_node_ids())
    
    print 'Norm Difference', np.linalg.norm(P - Q) / np.linalg.norm(P)
    plt.imshow(np.hstack([P,Q]),interpolation='none')
    plt.show()
    
def interp_grid_test():
    """
    Takes random 2D points, maps them to their node dists, and then maps these
    nodes to their canonical state. Should look regular.
    """
    x_n = 4
    y_n = 5
    eps = 1e-8
    gridder = InterpolatedGridNodeMapper(np.linspace(0,1,x_n),np.linspace(0,1,y_n))

    P = np.random.rand(500,2)

    (N,d) = P.shape
    assert(d == 2)
    for i in xrange(N):
        state = np.reshape(P[i,:],(1,2))
        node_dist_map = gridder.states_to_node_dists(state)
        if 0 == len(node_dist_map):
            # TODO: check and fix
            break
        assert(0 in node_dist_map)
        for (node_id,w) in node_dist_map[0].items():
            canon_state = gridder.nodes_to_states([node_id])
            plt.plot([state[0,0],canon_state[0,0]],[state[0,1],canon_state[0,1]],alpha=w)
    plt.show()
    
def interp_grid_test2():
    """
    Check that 
    """
    x_n = 2
    y_n = 3
    z_n = 4

    eps = 1e-8
    gridder = InterpolatedGridNodeMapper(np.linspace(0,1,x_n),np.linspace(0,1,y_n),np.linspace(0,1,z_n))

    P = gridder.get_node_states()
    Q = gridder.nodes_to_states(gridder.get_node_ids())
    
    print np.linalg.norm(P - Q) / np.linalg.norm(P)
    plt.imshow(np.hstack([P,Q]),interpolation='none')
    plt.show()

#piecewiseconst_regular_test()
piecewiseconst_regular_test2()
#interp_grid_test2()