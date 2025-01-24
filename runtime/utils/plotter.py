import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# low pass filter to make plots smoother 
def windowed_average(input_array, window_size = 15):
    output_array = np.convolve(input_array, np.ones(window_size)/window_size, mode='valid')
    return np.append(input_array[0], output_array) # append used to match input and output size

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

# function to plot total queue backlogs
def plot_backlog_curve(ax, unknownT_backlog_at_tt, knownT_backlog_at_tt, oracle_backlog_at_tt, label_font_size):
    ax.plot(unknownT_backlog_at_tt, color = 'C0', label = 'DPOP (doubling)')
    ax.plot(knownT_backlog_at_tt, color = 'C1', label = 'DPOP (given T)')
    ax.plot(oracle_backlog_at_tt, color = 'C2', label = 'Oracle policy')

    # show values in scientific notation and show exponent near axes
    ax.set_xticks(ticks=1000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])
    ax.text(9100, -9, '$\\times 10^3$', fontdict=None, size=label_font_size)

    ax.set_xlim([-250,10000])
    ax.set_ylim([0,65])

    ax.set_xlabel('Time-slot')
    ax.set_ylabel('Total queue backlog')

    ax.legend(loc = 'lower right')
    ax.grid()

# function to plot total transmission costs
def plot_transmission_cost_curve(ax, unknownT_tran_cost_at_tt, knownT_tran_cost_at_tt, oracle_tran_cost_at_tt, label_font_size):
    ax.plot(windowed_average(oracle_tran_cost_at_tt), color = 'C2', label = 'Oracle policy')
    ax.plot(windowed_average(knownT_tran_cost_at_tt), color = 'C1', label = 'DPOP (given T)')
    ax.plot(windowed_average(unknownT_tran_cost_at_tt), color = 'C0', label = 'DPOP (doubling)')

    # show values in scientific notation and show exponent near axes
    ax.set_xticks(ticks=1000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])
    ax.text(9100, -0.7, '$\\times 10^3$', fontdict=None, size=label_font_size)

    ax.set_xlim([-250,10000])
    ax.set_ylim([0,5])

    ax.set_xlabel('Time-slot')
    ax.set_ylabel('Instantaneous transmission cost')

    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2]
    order.reverse()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'lower right')

    ax.grid()