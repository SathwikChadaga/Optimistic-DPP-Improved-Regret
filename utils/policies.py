import numpy as np
from scipy.optimize import linprog

def get_static_policy(node_edge_adjacency, source_node, destination_node, true_costs, edge_capacities, arrival_rate):

    bounds_list = []
    for edge_capacity in edge_capacities:
        bounds_list.append((0, edge_capacity))

    A_matrix = np.delete(node_edge_adjacency, (destination_node), axis=0)
    b_matrix = np.zeros([A_matrix.shape[0],])
    b_matrix[source_node] = -arrival_rate

    result = linprog(c = true_costs, A_ub = A_matrix, b_ub = b_matrix, bounds = bounds_list)

    if result.success == False:
        print("No optimal solution found.")
    
    return result.x

def max_weight_policy(queue_state, node_edge_adjacency, cost_matrix, edge_capacities, nu):
    N_runs = queue_state.shape[0]
    N_nodes = node_edge_adjacency.shape[0]
    N_edges = node_edge_adjacency.shape[1]

    # get Q_i and duplicate the contents along edge axis 
    # (need not worry about whether the edge is connected or not, as disconnected edges will anyways have infinite cost)
    queue_ii = np.zeros([N_runs, N_nodes, N_edges])
    queue_ii += np.expand_dims(queue_state, axis=-1)

    # collect backlogs at jj for each edge (ii,jj), and bring it to ii
    backlog_at_tail_of_edge = queue_state@(node_edge_adjacency == 1)
    queue_jj = np.zeros([N_runs, N_nodes, N_edges])
    queue_jj += (np.expand_dims(node_edge_adjacency==-1, axis=0))*(np.expand_dims(backlog_at_tail_of_edge, axis=1))

    # calculate backlog differential
    queue_differentials = (queue_ii - queue_jj)
    queue_differentials[queue_differentials < 0] = 0

    # get edge weights adding the penalty term (adds -\infty for non-existent edges) 
    edge_weights = queue_differentials - nu*cost_matrix

    # send max rate on edges with positive edge weights
    scheduled_edges = (edge_weights > 0)
    planned_edge_rates_per_node = scheduled_edges*edge_capacities[np.newaxis, np.newaxis, :]
    planned_edge_rates = np.sum(planned_edge_rates_per_node, axis=1)

    return planned_edge_rates