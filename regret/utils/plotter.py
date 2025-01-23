import numpy as np
from scipy.optimize import lsq_linear  
import matplotlib.pyplot as plt
import networkx as nx

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

# fit curve to O(T^{1/2})
def fit_regret_curve(T_horizon_list, dpop_regret, start_index = 0):
    # create T^{1/2} for given values of T
    X = np.ones([T_horizon_list.shape[0], 2])
    X[:,1] = (T_horizon_list**(1/2))*np.log(T_horizon_list)
    bound_constraints = ([-np.inf,0], [np.inf,np.inf])

    # fit given regret to O(T^{1/2})
    regret_fit_dpop = lsq_linear(X[start_index:,:], dpop_regret[start_index:], bounds=bound_constraints)
    theoretical_dpop_regret = X@regret_fit_dpop.x

    # print co efficients and return
    print('Fit co-effs [1 sqrt(T)log(T)] = ' + np.array2string(regret_fit_dpop.x, precision=3, suppress_small=True))
    return theoretical_dpop_regret

# function to plot regret curve
def plot_regret_curve(ax, arrival_rate, noise_variance_list, show_theoretical, sweep_results_folder, plot_style, show_ylabel, label_font_size):
    # iterate for given values of noise variances
    for jj, noise_variance in enumerate(noise_variance_list):
        current_result = np.load(sweep_results_folder + '/regret-lambda-' + str(arrival_rate).replace('.','_') + '-var-' + str(noise_variance).replace('.','_') + '.npy')     
        ax.plot(current_result[0,:], current_result[1,:] - current_result[2,:], plot_style[jj], label = '$\sigma^2$ = ' + str(noise_variance), fillstyle = 'none', markeredgewidth=2, ms=8)  

        if(show_theoretical[jj]):
            theoretical_regret = fit_regret_curve(current_result[0,:], current_result[1,:] - current_result[2,:], start_index = 6)
            ax.plot(current_result[0,:], theoretical_regret, '--', label = r'$O(\sqrt{T}\log{T})$', linewidth=3)

    if(show_ylabel): ax.set_ylabel('Regret')
    ax.set_xlabel('Time horizon')
    
    # show values in scientific notation and show exponent near axes
    ax.set_xlim([5000,50000])
    ax.set_xticks(ticks=5000*np.arange(1,11), labels=['{:1.0f}'.format(s) for s in 5*np.arange(1,11)])
    ax.text(45500, -900, '$\\times 10^3$', fontdict=None, size=label_font_size)

    # ax.set_xlim([5000,25000])
    # ax.set_xticks(ticks=2500*np.arange(1,11), labels=['{:1.1f}'.format(s) for s in 2.5*np.arange(1,11)])
    # ax.text(20500, -900, '$\\times 10^3$', fontdict=None, size=label_font_size)
    
    # show values in scientific notation and show exponent near axes
    ax.set_ylim([0,5000])
    ax.set_yticks(ticks=np.linspace(0,5000,6), labels=['{:1.0f}'.format(s) for s in np.linspace(0,5,6)])
    ax.text(4500, 5075, '$\\times 10^3$', fontdict=None, size=label_font_size)
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2,3,4]
    order.reverse()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left')
    
    ax.grid()
