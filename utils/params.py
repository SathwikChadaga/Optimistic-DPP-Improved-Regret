import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# class to hold all simulaiton parameters
class SimulationParameters:
    def __init__(self, network_type,
                 node_edge_adjacency, 
                 true_edge_costs, edge_capacities, 
                 source_list, destination_list, 
                 noise_variance, noise_distribution,
                 arrival_rate_list, arrival_rate_scaling,
                 N_runs, T_horizon, T_horizon_list,
                 beta, delta, nu):
        self.network_type   = network_type
        self.N_runs         = N_runs
        self.T_horizon      = T_horizon
        self.T_horizon_list = T_horizon_list
        self.arrival_rate_list   = arrival_rate_list
        self.arrival_rate_scaling = arrival_rate_scaling
        self.noise_variance      = noise_variance
        self.noise_distribution  = noise_distribution
        self.nu    = nu
        self.beta  = beta         
        self.delta = delta
        self.node_edge_adjacency = node_edge_adjacency
        self.source_list      = source_list
        self.destination_list = destination_list
        self.true_edge_costs  = true_edge_costs
        self.edge_capacities  = edge_capacities

# function to get default simulation parameters
def get_simulation_params(network_type = 'multi-user', arrival_rate_scaling = None, noise_variance = None, is_regret_sim = True, visualize_network = True):
    if(arrival_rate_scaling == None): arrival_rate_scaling = 1 if (network_type == 'multi-user') else 2/3
    if(noise_variance == None): noise_variance = 0.1 if (network_type == 'multi-user') else 0.05

    # simulation lengths 
    # Use T_horizon = 10000 and N_runs = 500 to recreate the plots from the paper
    T_horizon = 100000
    T_horizon_list = np.linspace(5000, 100000, 20, dtype=int)
    N_runs = 1000 if is_regret_sim else 10000 # number of simulations

    # topology
    node_edge_adjacency, true_edge_costs, edge_capacities, \
        source_list, destination_list, arrival_rate_list = get_network_topology(network_type, arrival_rate_scaling, visualize_network)

    # noise and arrival rates
    def random_uniform(size = []):
        return 2*np.random.uniform(size = size)-1
    noise_distribution = random_uniform # np.random.standard_normal

    # algorithm parameters
    beta  = 4.5*noise_variance # exploration tuner (should theoretically be > 4 sigma^2)
    nu    = None # backlog-cost tradeoff tuner (should theoretically be T^{1/3}) (to be set later)
    delta = None # exploration tuner (should theoretically be T^{(-2 sigma^2)/(beta - 2 sigma^2)}) (to be set later)

    # pack parameters
    simulation_params = SimulationParameters(network_type, node_edge_adjacency,
                    true_edge_costs, edge_capacities, 
                    source_list, destination_list, 
                    noise_variance, noise_distribution,
                    arrival_rate_list, arrival_rate_scaling,
                    N_runs, T_horizon, T_horizon_list, 
                    beta, delta, nu)

    return simulation_params

def get_network_topology(network_type, arrival_rate_scaling, visualize_network):
    if(network_type == 'multi-user'):
        # topology
        edges_list = [[1,0],[2,1],[2,3],[0,4],[0,5],[6,1],
                    [3,6],[4,5],[5,6],[6,7],[8,4],[4,8],
                    [4,9],[9,5],[6,9],[6,10],[6,11],[11,7],
                    [9,8],[9,10],[10,9],[10,11]]
        N_nodes    = 12
        node_edge_adjacency = prepare_adjacency(edges_list, N_nodes)

        # source and destinations of each commodity, should follow the same order as arrival rates
        # Note: this assumes each commodity has a single source, maybe I can improve this in future
        source_list       = [0, 2, 3, 9]  
        destination_list  = [11, 8, 4, 7] 

        # arrival rates of each commodity, should follow same order as source_list
        # scale the default values as required using the arrival_rate_scaling argument
        arrival_rate_list = [arrival_rate_scaling*x for x in [2.5, 2.0, 0.5, 2.5]] # arrival rates for each commodity [60, 50, 10, 80]

        # edge properties
        edge_capacities = np.array([4,6,9,5,5,7,1,10,10,5,6,7,3,2,9,8,6,3,7,3,7,1])
        true_edge_costs = np.array([2,5,3,1,4,1,4,2,2,4,5,8,3,2,9,0,3,9,7,4,6,7])/25
    else:
        # topology
        edges_list       = [[0,1], [0,4], [0,2], [1,3], [1,4], [2,5], [3,6], [6,4], [4,6], [4,7], [5,4], [5,7], [6,8], [4,8], [7,8]]
        N_nodes          = 9
        node_edge_adjacency = prepare_adjacency(edges_list, N_nodes)

        # source and destinations of each commodity
        source_list       = [0]  
        destination_list  = [8] 

        # arrival rates of each commodity
        # scale the default values as required using the arrival_rate_scaling argument
        arrival_rate_list = [arrival_rate_scaling*x for x in [6]] 

        # edge properties
        edge_capacities = np.array([4,2,2,2,2,2,2,1,1,1,1,1,2,5,2]) # max-flow = 8
        true_edge_costs = np.array([2,5,1,1,2,1,1,1,1,1,1,3,3,1,1])/10

    # visualize topology
    if(visualize_network): plot_network(edges_list, N_nodes, source_list, destination_list, true_edge_costs, edge_capacities,)

    return node_edge_adjacency, true_edge_costs, edge_capacities, source_list, destination_list, arrival_rate_list

# function to modify simulation parameters
def set_simulation_params(simulation_params, T_horizon):
    # set T
    simulation_params.T_horizon = T_horizon

    # set nu
    simulation_params.nu = np.sqrt(T_horizon) 

    # set delta
    if(simulation_params.beta == 0): simulation_params.delta = 1
    else: simulation_params.delta = T_horizon**(-2*simulation_params.noise_variance/simulation_params.beta)

    return simulation_params

# prepares node-to-edge adjacency
def prepare_adjacency(edges, N_nodes):
    N_edges = len(edges)
    node_edge_adjacency = np.zeros([N_nodes, N_edges]) # node_edge_adjacency_(v,e) = {-1 if e = Out(v), 1 if e = In(v), 0 otherwise}
    for ll, vv in enumerate(edges):
        node_edge_adjacency[vv[0], ll] = -1
        node_edge_adjacency[vv[1], ll] = 1

    return node_edge_adjacency

# plots the network topology
def plot_network(edges_list, N_nodes, source_list, destination_list, true_edge_costs, edge_capacities,):
    # add edges, also create edge labels to mark edge capacities and costs
    G = nx.MultiDiGraph()
    edge_labels = {}
    for ii, edge in enumerate(edges_list):
        G.add_edge(edge[0], edge[1])
        edge_labels[(edge[0], edge[1])] =  '(' + str(edge_capacities[ii]) + ', ' + str(true_edge_costs[ii]) + ')'
        # networkx overwrites previous label if edge is bidirectional, so handle that separately 
        if((edge[1], edge[0]) in edge_labels): edge_labels[(edge[0], edge[1])] += '\n' + edge_labels[(edge[1], edge[0])]
        
    # fix node positions to match the diagram in paper
    if(N_nodes == 12): node_pos = {v : (v%4, 4-v//4) for v in range(N_nodes)}
    elif(N_nodes == 9): node_pos = {0: (0,2), 1: (0,1), 2: (1,2), 3: (0,0), 4: (1,1), 5: (2,2), 6: (1,0), 7: (2,1), 8: (2,0)}
    else: node_pos = nx.kamada_kawai_layout(G) 

    # mark sources and destinations
    color_map = []
    for v in list(G):
        if(v in source_list): color_map.append("red")
        elif(v in destination_list): color_map.append("green")
        else: color_map.append("blue")

    # Visualize the network
    nx.draw_networkx(G, pos = node_pos, node_color = color_map, with_labels = True, arrowsize = 15, alpha = 0.65)
    nx.draw_networkx_edge_labels(G, pos = node_pos, edge_labels = edge_labels)

    plt.title("Network Visualization")    
    plt.tight_layout()
    plt.show()