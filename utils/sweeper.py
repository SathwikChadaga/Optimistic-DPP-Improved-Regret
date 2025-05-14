import numpy as np
import utils.params as pars
import utils.policies as polc
import utils.experiment as expt
import pickle as pkl

# helper function to sweep parallely
def perform_regret_experiment(args):
    # get the topology and simulation parameters
    network_type, arrival_rate_scaling, noise_variance, save_directory = args
    simulation_params = pars.get_simulation_params(network_type, arrival_rate_scaling, noise_variance, is_regret_sim = True, visualize_network = False)

    # store some variables locally for ease of use
    T_horizon_list = simulation_params.T_horizon_list

    # get solution to static optimization problem
    stat_edge_rates, backlog_cost_C_L = polc.get_static_policy(simulation_params)
    N_commodities   = len(simulation_params.destination_list)
    N_edges         = len(simulation_params.edge_capacities)
    stat_cost_at_tt = np.sum(stat_edge_rates.reshape([N_commodities, N_edges])@simulation_params.true_edge_costs)
    stat_costs = T_horizon_list*stat_cost_at_tt

    # intialization
    dpop_tran_costs = np.zeros(T_horizon_list.shape)
    dpop_backlogs = np.zeros(T_horizon_list.shape)

    # iterate over given T values
    for ii in range(T_horizon_list.shape[0]):
        # change policy parameters for this value of T
        simulation_params = pars.set_simulation_params(simulation_params, T_horizon_list[ii])

        # run experiment for this value of T
        queueing_network = expt.run_experiment(simulation_params, custom_seed = None)
        
        # save cost and backlog values
        dpop_tran_costs[ii], dpop_backlogs[ii] = expt.calculate_total_costs(queueing_network)

    # save results
    save_result = {'T_horizon_list': T_horizon_list,
                    'dpop_tran_costs': dpop_tran_costs, 
                    'dpop_backlogs': dpop_backlogs, 
                    'C_L': backlog_cost_C_L,
                    'stat_costs': stat_costs, 
                    'example_edge_cost_means': queueing_network.edge_cost_means[0,:], 
                    'true_edge_costs': simulation_params.true_edge_costs}
    save_file = 'regret-lambda-' + ('%.3f'%(arrival_rate_scaling)).replace('.','_') + '-var-' + str(noise_variance).replace('.','_') + '.pkl'
    if(save_directory is not None): 
        with open(save_directory + '/' + network_type + '/' + save_file, 'wb') as f: 
            pkl.dump(save_result, f)