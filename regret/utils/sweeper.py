import numpy as np
import utils.policies as polc
import utils.experiment as expt
import pickle as pkl

# helper function to sweep parallely
def perform_regret_experiment(args):
    arrival_rate, noise_variance, save_directory = args

    # simulation lengths
    T_horizon = None # time horizon (to be set later)
    T_horizon_list = np.linspace(10000,100000,19, dtype=int)
    N_runs = 500 # number of simulations

    # noise and arrival rates
    def random_uniform(size = []):
        return 2*np.random.uniform(size = size)-1
    noise_distribution = random_uniform # np.random.standard_normal

    # algorithm parameters
    beta  = 4.5*noise_variance # exploration tuner (should theoretically be > 4 sigma^2)
    nu    = None # backlog-cost tradeoff tuner (should theoretically be T^{1/3}) (to be set later)
    delta = None # exploration tuner (should theoretically be T^{(-2 sigma^2)/(beta - 2 sigma^2)}) (to be set later)

    # topology
    N_nodes          = 9
    source_node      = 0
    destination_node = 8
    edges_list       = [[0,1], [0,4], [0,2], [1,3], [1,4], [2,5], [3,6], [6,4], [4,6], [4,7], [5,4], [5,7], [6,8], [4,8], [7,8]]
    node_edge_adjacency = expt.prepare_adjacency(edges_list, N_nodes)

    # edge properties
    edge_capacities = np.array([4,2,2,2,2,2,2,1,1,1,1,1,2,5,2]) # max-flow = 8
    true_edge_costs = np.array([2,5,1,1,2,1,1,1,1,1,1,3,3,1,1])/10

    # pack parameters
    simulation_params = expt.SimulationParameters(node_edge_adjacency, 
                    true_edge_costs, edge_capacities, 
                    source_node, destination_node, 
                    noise_variance, noise_distribution,
                    arrival_rate, 
                    N_runs, T_horizon, 
                    beta, delta, nu)

    # visualize topology
    # pltutils.visualize_network(edges_list, N_nodes)

    # get solution to static optimization problem
    stat_edge_rates = polc.get_static_policy(node_edge_adjacency, source_node, destination_node, true_edge_costs, edge_capacities, arrival_rate)
    total_stat_cost_per_time = stat_edge_rates@true_edge_costs
    stat_costs = T_horizon_list*total_stat_cost_per_time

    # intialization
    tran_cost_till_T_dpop = np.zeros(T_horizon_list.shape)
    backlog_cost_at_T_dpop = np.zeros(T_horizon_list.shape)

    # iterate over given T values
    for jj in range(T_horizon_list.shape[0]):
        ii = T_horizon_list.shape[0] - jj - 1

        # change policy parameters for this value of T
        simulation_params = expt.set_simulation_params(simulation_params, T_horizon_list[ii])

        # run experiment for this value of T
        queueing_network = expt.run_experiment(simulation_params, custom_seed = 0)
        
        # save cost and backlog values
        tran_cost_till_T_dpop[ii], backlog_cost_at_T_dpop[ii] = expt.calculate_total_costs(queueing_network)

    # calculate total cost
    dpop_costs = tran_cost_till_T_dpop + backlog_cost_at_T_dpop

    # save results
    save_result = {'T_horizon_list': T_horizon_list, 
                    'dpop_costs': dpop_costs, 
                    'stat_costs':stat_costs, 
                    'example_edge_cost_means': queueing_network.edge_cost_means[0,:], 
                    'true_edge_costs': true_edge_costs}
    save_file = 'regret-lambda-' + str(arrival_rate).replace('.','_') + '-var-' + str(noise_variance).replace('.','_') + '.pkl'
    if(save_directory is not None): 
        with open(save_directory + '/' + save_file, 'wb') as f: pkl.dump(save_result, f)