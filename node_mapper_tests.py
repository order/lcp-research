from mdp.node_mapper import PiecewiseConstRegularGridNodeMapper,InterpolatedGridNodeMapper
import numpy as np
import matplotlib.pyplot as plt


def piecewiseconst_regular_test():
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
    
def interp_grid_test():
    x_n = 3
    y_n = 4
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

interp_grid_test()


