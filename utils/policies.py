import numpy as np
from scipy.optimize import linprog

# static optimization
def get_static_policy(simulation_params):

    node_edge_adjacency = simulation_params.node_edge_adjacency
    N_nodes = node_edge_adjacency.shape[0]
    M_edges = node_edge_adjacency.shape[1]
    K_commodites = len(simulation_params.arrival_rate_list)

    # cost vector (repeated to account for multipple commodities)
    all_costs = np.tile(simulation_params.true_edge_costs, K_commodites)

    # initialize variables for constraint inequalites
    A_matrix = np.empty(shape=[0, M_edges*K_commodites])
    b_vector = np.empty(shape=[0,])

    # add flow constraints one-by-one for each commodity kk
    for kk, destination_node in enumerate(simulation_params.destination_list): 

        # create node-edge adjacency matrix for commodity kk excluding its destination node
        A_block_kk = np.zeros([N_nodes-1, M_edges*K_commodites])
        A_block_kk[:, kk*M_edges:(kk+1)*M_edges] = np.delete(node_edge_adjacency, destination_node, axis=0)

        # add corresponding arrival rates for the RHS contraint vector
        b_block_kk = np.zeros([N_nodes,])
        b_block_kk[simulation_params.source_list[kk]] = -simulation_params.arrival_rate_list[kk]
        b_block_kk = np.delete(b_block_kk, destination_node, axis=0)
        
        # append them to the main constraint variables
        A_matrix = np.concat((A_matrix, A_block_kk), axis=0)
        b_vector = np.concat((b_vector, b_block_kk), axis=0)

    # add capacity constraints to the constraints set
    A_matrix = np.concat((A_matrix, np.tile(np.eye(node_edge_adjacency.shape[1]), K_commodites)), axis=0)
    b_vector = np.concat((b_vector, simulation_params.edge_capacities), axis=0)

    # solve the linear optimization problem and return results
    result = linprog(c = all_costs, A_ub = A_matrix, b_ub = b_vector, bounds = (0,None))
    if result.success == False: print("No optimal solution found.")

    # see Theorem 1 of the paper, C_B >= l-infinity norm of dual optimal solution
    dual = linprog(c = b_vector, A_ub = -A_matrix.T, b_ub = all_costs, bounds = (0,None))
    if dual.success == False: print("No dual optimal solution found.")
    backlog_cost_C_L = np.sum(np.abs(dual.x))
    # backlog_cost_C_L = np.sum(simulation_params.true_edge_costs)*K_commodites

    return result.x, backlog_cost_C_L

# max weight policy
def max_weight_policy(queue_state, node_edge_adjacency, edge_costs, edge_capacities, nu):
    # calculate weights using queue differentials and edge costs
    edge_weights = -queue_state@node_edge_adjacency - nu*edge_costs[:,None,:]

    # pick commodity that has max weight
    scheduled_edges = (edge_weights == np.max(edge_weights, axis=1, keepdims=True))
    
    # normalize so that if multiple commodities have same max weight, their sum does not exceed 1
    normalization = np.sum(scheduled_edges, axis=1, keepdims=True)
    scheduled_edges = scheduled_edges/normalization

    # allocate maximum rates to all edges where weight is positive 
    scheduled_edges[edge_weights <= 0] = 0
    planned_edge_rates = scheduled_edges*edge_capacities[None, None, :]

    return planned_edge_rates