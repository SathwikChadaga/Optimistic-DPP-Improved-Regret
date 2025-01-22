import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import lsq_linear   

# class to hold all simulaiton parameters
class SimulationParameters:
    def __init__(self, 
                 node_edge_adjacency, 
                 true_edge_costs, edge_capacities, 
                 source_node, destination_node, 
                 noise_variance, noise_distribution,
                 arrival_rate, 
                 N_runs, T_horizon, 
                 beta, delta, nu):
        self.N_runs              = N_runs
        self.T_horizon           = T_horizon
        self.arrival_rate        = arrival_rate
        self.noise_variance      = noise_variance
        self.noise_distribution  = noise_distribution
        self.nu                  = nu
        self.beta                = beta         
        self.delta               = delta
        self.node_edge_adjacency = node_edge_adjacency
        self.source_node         = source_node
        self.destination_node    = destination_node
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
def prepare_cost_matrix(node_edge_adjacency, costs):
    if(len(costs.shape) > 1): 
        cost_matrix = np.expand_dims(node_edge_adjacency == -1, axis=0)*costs[:, np.newaxis, :] + 0.0
        cost_matrix[:, node_edge_adjacency != -1] = np.inf

    else: 
        cost_matrix = (node_edge_adjacency == -1)*costs + 0.0
        cost_matrix[node_edge_adjacency != -1] = np.inf
    return cost_matrix

# plots the network topology
def visualize_network(edges_list, N_nodes):
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

# fits regret curve into a theoretical curve
def fit_regret_curve(T_horizon_list, dpop_regret, start_index = 0):
    X = np.ones([T_horizon_list.shape[0], 2])
    X[:,1] =  (T_horizon_list**(1/2))*np.log(T_horizon_list) 

    regret_fit_dpop = lsq_linear(X[start_index:,:], dpop_regret[start_index:], bounds=([-np.inf,0], [np.inf,np.inf]))
    theoretical_dpop_regret = X@regret_fit_dpop.x
    print('Fit co-effs [1 sqrt(T)log(T)] = ' + np.array2string(regret_fit_dpop.x, precision=3, suppress_small=True))

    return theoretical_dpop_regret

