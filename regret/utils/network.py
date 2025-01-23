import numpy as np

class OnlineQueueNetwork:
    def __init__(self, simulation_params, custom_seed = None):
        # topology information
        self.node_edge_adjacency = simulation_params.node_edge_adjacency 
        self.source_node = simulation_params.source_node
        self.destination_node = simulation_params.destination_node

        # sizes and lenghts
        self.N_nodes = self.node_edge_adjacency.shape[0]
        self.N_edges = self.node_edge_adjacency.shape[1]
        self.N_runs = simulation_params.N_runs        
        self.T_horizon = simulation_params.T_horizon

        # edge costs and capacities
        self.edge_capacities = simulation_params.edge_capacities
        self.true_edge_costs = simulation_params.true_edge_costs
        self.noise_variance = simulation_params.noise_variance
        self.noise_distribution = simulation_params.noise_distribution
        
        # variables to store traffic dynamics and transmission rates
        self.queues = np.zeros([self.N_runs, self.N_nodes, self.T_horizon+1])
        self.planned_edge_rates = np.zeros([self.N_runs, self.N_edges, self.T_horizon])
        self.actual_edge_rates = np.zeros([self.N_runs, self.N_edges, self.T_horizon])
        self.arrivals = np.random.poisson(lam = simulation_params.arrival_rate, size = [self.N_runs, self.T_horizon])
        
        # initializations
        self.tt = -1
        self.edge_cost_means = np.zeros([self.N_runs, self.N_edges])
        self.edge_num_pulls = np.zeros([self.N_runs, self.N_edges])
        
        # initial exploration 
        if(custom_seed != None): np.random.seed(custom_seed)
        self.initial_exploration()
    
    def step(self, planned_edge_rates):
        # check if time horizon has reached and return -1
        if(self.tt == self.T_horizon): 
            return -1
            
        # new arrivals at the single source node
        new_arrivals = np.zeros([self.N_runs, self.N_nodes])
        new_arrivals[:, self.source_node] = self.arrivals[:, self.tt]

        # get actual rates and store rates
        actual_edge_rates = self.get_actual_edge_rates(planned_edge_rates)
        self.planned_edge_rates[:, :, self.tt] = planned_edge_rates
        self.actual_edge_rates[:, :, self.tt] = actual_edge_rates
    
        # queue evolution 
        internal_arrivals_departures = actual_edge_rates@self.node_edge_adjacency.T
        self.queues[:,:, self.tt+1] = self.queues[:,:, self.tt] + new_arrivals + internal_arrivals_departures
        if(np.any(self.queues < -1e-10)): print('Something is wrong.')
        self.queues[self.queues < 0] = 0

        # packets at destination node exit the network immediately
        self.queues[:, self.destination_node, :] = 0 

        # get observed costs and update cost estimates
        self.edge_num_pulls += (planned_edge_rates > 0)
        self.edge_cost_means[planned_edge_rates > 0] += (self.get_observation() - self.edge_cost_means)[planned_edge_rates > 0]/self.edge_num_pulls[planned_edge_rates > 0]
        
        # update current time step and return 0
        self.tt += 1
        return 0

    def get_actual_edge_rates(self, planned_edge_rates):
        # calulate total excess departures planned from each node
        total_planned_node_departure = planned_edge_rates@(self.node_edge_adjacency == -1).T
        excess_planned_node_departure  = total_planned_node_departure - self.queues[:, :, self.tt]
        nodes_affected = (excess_planned_node_departure > 0)

        # for every node with excess planned departures, back off the planned departures to be within queue sizes
        back_off_ratio_per_node = np.ones(nodes_affected.shape)
        back_off_ratio_per_node[nodes_affected] = self.queues[nodes_affected, self.tt]/total_planned_node_departure[nodes_affected]

        # back-off applied evenly to each outgoing edge
        back_off_ratio_per_edge = back_off_ratio_per_node@(self.node_edge_adjacency == -1)
        actual_edge_rates = back_off_ratio_per_edge*planned_edge_rates

        return actual_edge_rates
    
    def initial_exploration(self):
        # get noisy observation for each edge (by sending null packets)
        # and update edge means and number of observations
        self.edge_cost_means = self.get_observation()
        self.edge_num_pulls[:,:] = 1
        self.tt += 1
        return 
    
    def get_observation(self):
        observation_noise = np.sqrt(self.noise_variance)*self.noise_distribution(size=[self.N_runs, self.N_edges])
        # observation_noise = np.sqrt(self.noise_variance)*np.random.standard_normal(size=[self.N_runs, self.N_edges])
        return self.true_edge_costs[np.newaxis, :] + observation_noise