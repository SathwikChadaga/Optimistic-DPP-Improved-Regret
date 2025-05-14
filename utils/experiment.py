import numpy as np
import utils.network as qnet
import utils.policies as polc


def run_experiment(simulation_params, custom_seed = None, queueing_network = None, store_extra_info = False):
    # store algorithm parameters
    beta  = simulation_params.beta
    delta = simulation_params.delta
    nu    = simulation_params.nu

    # define network from the given parameters
    if(queueing_network == None): 
        queueing_network = qnet.OnlineQueueNetwork(simulation_params, custom_seed)

    queueing_network.store_extra_info = store_extra_info
    network_status = 0
    while(network_status == 0):
        # get queue state
        queue_state = queueing_network.queues

        # estimate edge costs from previous observation
        estimated_edge_costs = queueing_network.edge_cost_means - np.sqrt(beta*np.log((queueing_network.tt+1)/delta)/queueing_network.edge_num_pulls)

        # get planned transmissions from the policy
        planned_edge_rates = polc.max_weight_policy(queue_state, queueing_network.node_edge_adjacency, estimated_edge_costs, queueing_network.edge_capacities, nu)
        
        # take action and update the network state 
        network_status = queueing_network.step(planned_edge_rates)

    return queueing_network

# function to calculate costs of a network after running the simulation
def calculate_total_costs(queueing_network, cost_type = 'planned'):  
    if(cost_type == 'planned'): tran_cost_till_T = np.sum(queueing_network.planned_tran_cost_at_tt)
    else: tran_cost_till_T = np.sum(queueing_network.actual_tran_cost_at_tt)
    
    backlog_at_T = queueing_network.backlog_at_tt[queueing_network.T_horizon-1]

    return tran_cost_till_T, backlog_at_T

# function to calculate costs of a network after running the simulation
def calculate_per_time_metrics(queueing_network, backlog_cost_C_B, cost_type = 'planned'):
    if(cost_type == 'planned'): tran_cost_at_tt = queueing_network.planned_tran_cost_at_tt
    else: tran_cost_at_tt = queueing_network.actual_tran_cost_at_tt
    tran_cost_till_tt = np.cumsum(tran_cost_at_tt)

    backlog_at_tt = queueing_network.backlog_at_tt
    backlog_cost_at_tt = queueing_network.backlog_at_tt*backlog_cost_C_B

    return tran_cost_at_tt, tran_cost_till_tt, backlog_at_tt, backlog_cost_at_tt

# converts cost array into matrix
def prepare_cost_matrix(node_edge_adjacency, costs, N_commodities):
    if(len(costs.shape) > 1): 
        cost_matrix = np.expand_dims(node_edge_adjacency == -1, axis=0)*costs[:, np.newaxis, :] + 0.0
        cost_matrix[:, node_edge_adjacency != -1] = np.inf

    else: 
        cost_matrix = (node_edge_adjacency == -1)*costs + 0.0
        cost_matrix[node_edge_adjacency != -1] = np.inf
    
    cost_matrix = np.repeat(cost_matrix[:,None,:,:], N_commodities, axis=1)
    return cost_matrix
