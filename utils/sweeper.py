import numpy as np
import utils.params as pars
import utils.policies as polc
import utils.experiment as expt
import pickle as pkl

# helper function to sweep parallely
def perform_regret_experiment(args):
    arrival_rate_scaling, noise_variance, save_directory = args

    simulation_params = pars.get_settings(arrival_rate_scaling, noise_variance, visualize_network=False)

    # store some variables locally for ease of use
    node_edge_adjacency = simulation_params.node_edge_adjacency
    N_commodities  = len(simulation_params.destination_list)
    N_edges    = node_edge_adjacency.shape[1]
    true_edge_costs = simulation_params.true_edge_costs
    T_horizon_list = simulation_params.T_horizon_list

    # get solution to static optimization problem
    stat_edge_rates = polc.get_static_policy(node_edge_adjacency, simulation_params)
    stat_cost_at_tt = np.sum(stat_edge_rates.reshape([N_commodities, N_edges])@true_edge_costs)
    stat_costs = T_horizon_list*stat_cost_at_tt

    # intialization
    tran_cost_till_T_dpop = np.zeros(T_horizon_list.shape)
    backlog_cost_at_T_dpop = np.zeros(T_horizon_list.shape)

    # iterate over given T values
    for jj in range(T_horizon_list.shape[0]):
        ii = T_horizon_list.shape[0] - jj - 1

        # change policy parameters for this value of T
        simulation_params = pars.set_simulation_params(simulation_params, T_horizon_list[ii])

        # run experiment for this value of T
        queueing_network = expt.run_experiment(simulation_params, custom_seed = None)
        
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
    save_file = 'regret-lambda-' + str(arrival_rate_scaling).replace('.','_') + '-var-' + str(noise_variance).replace('.','_') + '.pkl'
    if(save_directory is not None): 
        with open(save_directory + '/' + save_file, 'wb') as f: pkl.dump(save_result, f)