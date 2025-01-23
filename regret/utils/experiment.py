import numpy as np
import utils.misc as mutil
import utils.network as qnet
import utils.policies as polc

def set_simulation_params(simulation_params, T_horizon):
    # set T
    simulation_params.T_horizon = T_horizon

    # set nu
    simulation_params.nu = np.sqrt(T_horizon) # T_horizon**(1/3)

    # set delta
    noise_variance = simulation_params.noise_variance
    if(noise_variance == 0): simulation_params.delta = 1
    else: simulation_params.delta = T_horizon**(-2*noise_variance/(simulation_params.beta-2*noise_variance))

    return simulation_params

def run_experiment(simulation_params, custom_seed = None, queueing_network = None):
    # store algorithm parameters
    beta  = simulation_params.beta
    delta = simulation_params.delta
    nu    = simulation_params.nu

    # define network from the given parameters
    if(queueing_network == None): 
        queueing_network = qnet.OnlineQueueNetwork(simulation_params, custom_seed)

    network_status = 0
    while(network_status == 0):
        # get queue state
        queue_state = queueing_network.queues

        # estimate edge costs from previous observation
        estimated_edge_costs = queueing_network.edge_cost_means - np.sqrt(beta*np.log((queueing_network.tt+1)/delta)/queueing_network.edge_num_pulls)
        estimated_cost_matrix = mutil.prepare_cost_matrix(queueing_network.node_edge_adjacency, estimated_edge_costs)

        # get planned transmissions from the policy
        planned_edge_rates = polc.max_weight_policy(queue_state, queueing_network.node_edge_adjacency, estimated_cost_matrix, queueing_network.edge_capacities, nu)
        
        # take action and update the network state 
        network_status = queueing_network.step(planned_edge_rates)

    return queueing_network


def calculate_total_costs(queueing_network, cost_type = 'planned'):
    if(cost_type == 'planned'):
        tran_cost_per_time_per_run = queueing_network.planned_edge_rates.transpose((0,2,1))@queueing_network.true_edge_costs
    else: # cost_type == 'actual'
        tran_cost_per_time_per_run = queueing_network.actual_edge_rates.transpose((0,2,1))@queueing_network.true_edge_costs
        
    tran_cost_till_T = np.mean(np.sum(tran_cost_per_time_per_run, axis=1), axis=0)
    backlog_at_T = np.mean(np.sum(queueing_network.queues, axis=1), axis=0)
    backlog_cost_at_T = backlog_at_T*np.sum(queueing_network.true_edge_costs) # C_B = sum_{ij} c_{ij}

    return tran_cost_till_T, backlog_cost_at_T
