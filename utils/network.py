import numpy as np

class OnlineQueueNetwork:
    def __init__(self, simulation_params, custom_seed = None):
        # topology information
        self.node_edge_adjacency = simulation_params.node_edge_adjacency 
        self.source_list = simulation_params.source_list
        self.destination_list = simulation_params.destination_list

        # sizes and lenghts
        self.N_nodes = self.node_edge_adjacency.shape[0]
        self.N_edges = self.node_edge_adjacency.shape[1]
        self.N_commodities = len(self.source_list)
        self.N_runs = simulation_params.N_runs        
        self.T_horizon = simulation_params.T_horizon

        # edge costs and capacities
        self.edge_capacities = simulation_params.edge_capacities
        self.true_edge_costs = simulation_params.true_edge_costs
        self.noise_variance = simulation_params.noise_variance
        self.noise_distribution = simulation_params.noise_distribution
        
        # traffic dynamics and queue state
        self.arrival_rate_list = simulation_params.arrival_rate_list
        self.queues = np.zeros([self.N_runs, self.N_commodities, self.N_nodes])

        # variables to store results
        self.backlog_at_tt = np.zeros([self.T_horizon])
        self.planned_tran_cost_at_tt = np.zeros([self.T_horizon])
        self.actual_tran_cost_at_tt = np.zeros([self.T_horizon])
        
        # initializations
        self.tt = -1
        self.edge_cost_means = np.zeros([self.N_runs, self.N_edges])
        self.edge_num_pulls = np.zeros([self.N_runs, self.N_edges])

        self.arrival_matrix, self.destination_matrix = self.prepare_commodity_traffic_matrices()
        
        # initial exploration 
        if(custom_seed != None): np.random.seed(custom_seed)
        self.initial_exploration()

    
    def step(self, planned_edge_rates):
        # check if time horizon has reached and return -1
        if(self.tt == self.T_horizon): 
            return -1
            
        # new arrivals at each commodity's source nodes
        new_arrivals = np.random.poisson(lam = self.arrival_matrix, size = [self.N_runs, self.N_commodities, self.N_nodes])

        # get actual rates and store rates
        actual_edge_rates = self.get_actual_edge_rates(planned_edge_rates)
        self.store_metrics(self.queues, planned_edge_rates, actual_edge_rates)
    
        # queue evolution 
        internal_arrivals_departures = actual_edge_rates@self.node_edge_adjacency.T
        self.queues = self.queues + new_arrivals + internal_arrivals_departures
        if(np.any(self.queues < -1e-10)): print('Warning: Negative queues; something is wrong.')
        self.queues[self.queues < 0] = 0

        # packets at destination node exit the network immediately
        self.queues[np.repeat(self.destination_matrix[None,:,:], self.N_runs, axis=0)] = 0 

        # get observed costs and update cost estimates
        activated_edges = np.any(planned_edge_rates > 0, axis=1)
        self.edge_num_pulls += activated_edges
        self.edge_cost_means[activated_edges] += (self.get_observation() - self.edge_cost_means)[activated_edges]/self.edge_num_pulls[activated_edges]
        
        # update current time step and return 0
        self.tt += 1
        return 0

    def get_actual_edge_rates(self, planned_edge_rates):
        # calulate total excess departures planned from each node
        total_planned_queue_departure = planned_edge_rates@(self.node_edge_adjacency == -1).T
        excess_planned_queue_departure  = total_planned_queue_departure - self.queues
        queues_affected = (excess_planned_queue_departure > 0)

        # for every node with excess planned departures, back off the planned departures to be within queue sizes
        back_off_ratio_per_queue = np.ones(queues_affected.shape)
        back_off_ratio_per_queue[queues_affected] = self.queues[queues_affected]/total_planned_queue_departure[queues_affected]

        # back-off applied evenly to each outgoing edge
        back_off_ratio_per_edge = back_off_ratio_per_queue@(self.node_edge_adjacency == -1)
        actual_edge_rates = back_off_ratio_per_edge*planned_edge_rates

        return actual_edge_rates
    
    def initial_exploration(self):
        # get noisy observation for each edge (by sending null packets)
        # and update edge means and number of observations
        self.edge_cost_means = self.get_observation()
        self.edge_num_pulls[:,:] = 1
        self.tt += 1
        return 
    
    def prepare_commodity_traffic_matrices(self):
        arrival_matrix = np.zeros([self.N_commodities, self.N_nodes])
        destination_matrix = np.zeros([self.N_commodities, self.N_nodes])
        for kk, arrival_rate in enumerate(self.arrival_rate_list):
            arrival_matrix[kk, self.source_list[kk]] = arrival_rate
            destination_matrix[kk, self.destination_list[kk]] = 1
        return arrival_matrix, (destination_matrix == 1)
    
    def get_observation(self):
        observation_noise = np.sqrt(self.noise_variance)*self.noise_distribution(size=[self.N_runs, self.N_edges])
        # observation_noise = np.sqrt(self.noise_variance)*np.random.standard_normal(size=[self.N_runs, self.N_edges])
        return self.true_edge_costs[np.newaxis, :] + observation_noise
    
    def store_metrics(self, queues, planned_edge_rates, actual_edge_rates):
        # store current total queue backlog and current total transmission cost
        self.backlog_at_tt[self.tt] = np.mean(np.sum(queues, axis=(1,2)), axis=0)
        self.planned_tran_cost_at_tt[self.tt] = np.mean(np.sum(planned_edge_rates, axis=1)@self.true_edge_costs, axis=0)
        self.actual_tran_cost_at_tt[self.tt] = np.mean(np.sum(actual_edge_rates, axis=1)@self.true_edge_costs, axis=0)
    
