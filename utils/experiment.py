import numpy as np
import utils.network as qnet
import utils.policies as polc

def set_simulation_params(simulation_params, T_horizon):
    # set T
    simulation_params.T_horizon = T_horizon

    # set nu
    simulation_params.nu = np.sqrt(T_horizon) 

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
        # estimated_cost_matrix = prepare_cost_matrix(queueing_network.node_edge_adjacency, estimated_edge_costs, queueing_network.N_commodities)

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
    backlog_cost_at_T = backlog_at_T*np.sum(queueing_network.true_edge_costs) # C_B = sum_{ij} c_{ij}

    return tran_cost_till_T, backlog_cost_at_T

# function to calculate costs of a network after running the simulation
def calculate_per_time_metrics(queueing_network, cost_type = 'planned'):
    if(cost_type == 'planned'): tran_cost_at_tt = queueing_network.planned_tran_cost_at_tt
    else: tran_cost_at_tt = queueing_network.actual_tran_cost_at_tt
    tran_cost_till_tt = np.cumsum(tran_cost_at_tt)

    backlog_at_tt = queueing_network.backlog_at_tt
    backlog_cost_at_tt = queueing_network.backlog_at_tt*np.sum(queueing_network.true_edge_costs)

    return tran_cost_at_tt, tran_cost_till_tt, backlog_at_tt, backlog_cost_at_tt

# class to hold all simulaiton parameters
class SimulationParameters:
    def __init__(self, 
                 node_edge_adjacency, 
                 true_edge_costs, edge_capacities, 
                 source_list, destination_list, 
                 noise_variance, noise_distribution,
                 arrival_rate_list, 
                 N_runs, T_horizon, 
                 beta, delta, nu):
        self.N_runs              = N_runs
        self.T_horizon           = T_horizon
        self.arrival_rate_list   = arrival_rate_list
        self.noise_variance      = noise_variance
        self.noise_distribution  = noise_distribution
        self.nu                  = nu
        self.beta                = beta         
        self.delta               = delta
        self.node_edge_adjacency = node_edge_adjacency
        self.source_list         = source_list
        self.destination_list    = destination_list
        self.true_edge_costs     = true_edge_costs
        self.edge_capacities     = edge_capacities

# prepares node to edge adjacency
def prepare_adjacency(edges, N_nodes):
    N_edges = len(edges)
    node_edge_adjacency = np.zeros([N_nodes, N_edges]) # node_edge_adjacency_(v,e) = {-1 if e = Out(v), 1 if e = In(v), 0 otherwise}
    for ll, vv in enumerate(edges):
        node_edge_adjacency[vv[0], ll] = -1
        node_edge_adjacency[vv[1], ll] = 1

    return node_edge_adjacency

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
