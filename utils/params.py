import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def get_settings(arrival_rate_scaling = 1, noise_variance = None, visualize_network = True):
    # simulation lengths 
    # Use T_horizon = 10000 and N_runs = 500 to recreate the plots from the paper
    T_horizon = 100
    T_horizon_list = np.linspace(1,1000,10, dtype=int)
    N_runs = 1000 # number of simulations

    # noise and arrival rates
    def random_uniform(size = []):
        return 2*np.random.uniform(size = size)-1
    noise_distribution = random_uniform # np.random.standard_normal
    if(noise_variance == None): noise_variance = 0.05 # sigma^2

    # algorithm parameters
    beta  = 4.5*noise_variance # exploration tuner (should theoretically be > 4 sigma^2)
    nu    = None # backlog-cost tradeoff tuner (should theoretically be T^{1/3}) (to be set later)
    delta = None # exploration tuner (should theoretically be T^{(-2 sigma^2)/(beta - 2 sigma^2)}) (to be set later)

    # topology
    edges_list = [[0,1], [1,2], [3,1], [1,4]]
    N_nodes    = 5
    node_edge_adjacency = prepare_adjacency(edges_list, N_nodes)

    # arrival rates of each commodity, scale the default values as required using the arrival_rate_scaling argument
    arrival_rate_list = [arrival_rate_scaling*x for x in [3, 1]] # arrival rates for each commodity
    
    # source and destinations of each commodity, should follow the same order as arrival rates
    # Note: this assumes each commodity has a single source, maybe I can improve this in future
    source_list       = [0, 3]  
    destination_list  = [2, 4] 

    # edge properties
    edge_capacities = np.array([4,6,2,3]) # max-flow = [4,2]
    true_edge_costs = np.array([2,5,1,1])/10

    # pack parameters
    simulation_params = SimulationParameters(node_edge_adjacency, 
                    true_edge_costs, edge_capacities, 
                    source_list, destination_list, 
                    noise_variance, noise_distribution,
                    arrival_rate_list, arrival_rate_scaling,
                    N_runs, T_horizon, T_horizon_list, 
                    beta, delta, nu)
    
    # visualize topology
    if(visualize_network): plot_network(edges_list, N_nodes)

    return simulation_params

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

# class to hold all simulaiton parameters
class SimulationParameters:
    def __init__(self, 
                 node_edge_adjacency, 
                 true_edge_costs, edge_capacities, 
                 source_list, destination_list, 
                 noise_variance, noise_distribution,
                 arrival_rate_list, arrival_rate_scaling,
                 N_runs, T_horizon, T_horizon_list,
                 beta, delta, nu):
        self.N_runs              = N_runs
        self.T_horizon           = T_horizon
        self.T_horizon_list      = T_horizon_list
        self.arrival_rate_list   = arrival_rate_list
        self.arrival_rate_scaling = arrival_rate_scaling
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


# plots the network topology
def plot_network(edges_list, N_nodes):
    # add edges
    G = nx.DiGraph()
    for edge_ii in range(len(edges_list)):
        G.add_edge(edges_list[edge_ii][0]+1, edges_list[edge_ii][1]+1)  
        
    # relabel nodes
    mapping = {v+1 : v for v in range(N_nodes)}
    G = nx.relabel_nodes(G, mapping)

    # Visualize the network
    pos = nx.spectral_layout(G) # pos = nx.spring_layout(G) 
    nx.draw_networkx(G, pos, with_labels=True)

    plt.title("Network Visualization")
    plt.show()