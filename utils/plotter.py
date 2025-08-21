import numpy as np
from scipy.optimize import lsq_linear  
import pickle as pkl
from matplotlib.ticker import FixedLocator
# import networkx as nx
# import matplotlib.pyplot as plt

# fit curve to O(T^{1/2})
def fit_regret_curve(T_horizon_list, dpop_regret, start_index = 3):
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
def plot_regret_curve(ax, network_type, backlog_cost_C_B, arrival_rate_scaling, noise_variance_list, show_theoretical, sweep_results_folder, plot_style, show_ylabel, label_font_size):
    # iterate for given values of noise variances
    for jj, noise_variance in enumerate(noise_variance_list):
        with open(sweep_results_folder + '/regret-lambda-' + ('%.3f'%(arrival_rate_scaling)).replace('.','_') + '-var-' + str(noise_variance).replace('.','_') + '.pkl', 'rb') as f: 
            current_result = pkl.load(f)   

        dpop_tran_costs =  current_result['dpop_tran_costs']
        dpop_backlogs = current_result['dpop_backlogs']
        stat_costs = current_result['stat_costs']
        T_horizon_list = current_result['T_horizon_list']

        ax.plot(T_horizon_list, dpop_tran_costs + dpop_backlogs*backlog_cost_C_B - stat_costs, plot_style[jj], label = '$\sigma^2$ = ' + str(noise_variance), fillstyle = 'none', markeredgewidth=2, ms=8)  

        if(show_theoretical[jj]):
            theoretical_regret = fit_regret_curve(T_horizon_list, dpop_tran_costs + dpop_backlogs*backlog_cost_C_B - stat_costs, start_index = 6)
            ax.plot(T_horizon_list, theoretical_regret, '--', label = r'$O(\sqrt{T}\log{T})$', linewidth=3)

    if(show_ylabel): ax.set_ylabel('Regret')
    ax.set_xlabel('Time horizon')

    # show values in scientific notation and show exponent near axes
    if(network_type == 'multi-user'):
        ax.set_ylim([0,3.7e4])
        y_exponent_label = '$\\times 10^4$'
        ax.set_yticks(ticks=np.linspace(0,3e4,4), labels=['{:1.0f}'.format(s) for s in np.linspace(0,3,4)])
    else:
        ax.set_ylim([0,5000])
        y_exponent_label = '$\\times 10^3$'
        ax.set_yticks(ticks=np.linspace(0,5000,6), labels=['{:1.0f}'.format(s) for s in np.linspace(0,5,6)])

    # show values in scientific notation and show exponent near axes
    ax.set_xlim([1e4,1e5])
    x_exponent_label = '$\\times 10^4$'
    ax.set_xticks(ticks=np.linspace(10000,100000,10), labels=['{:1.0f}'.format(s) for s in np.linspace(1,10,10)])

    # add texts on the sides to show the exponent value of the scientific notation
    y_lim_val = ax.get_ylim()
    x_lim_val = ax.get_xlim()
    ax.text(x_lim_val[0]+0.0005*(x_lim_val[1] - x_lim_val[0]), y_lim_val[0]+1.01*(y_lim_val[1] - y_lim_val[0]), y_exponent_label, fontdict=None, size=label_font_size)
    ax.text(x_lim_val[0]+0.901*(x_lim_val[1] - x_lim_val[0]), y_lim_val[0]-0.16*(y_lim_val[1] - y_lim_val[0]), x_exponent_label, fontdict=None, size=label_font_size)

    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2,3,4]
    order.reverse()
    location = 'best'
    if(round(arrival_rate_scaling, 3) == round(1.575/3, 3)): location = 'lower right'
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=location)
    
    ax.grid()

# function to plot total queue backlogs
def plot_backlog_curve(ax, network_type, unknownT_backlog_at_tt, knownT_backlog_at_tt, oracle_backlog_at_tt, label_font_size):
    ax.plot(unknownT_backlog_at_tt, color = 'C0', label = 'DPOP (doubling)', linewidth=2)
    ax.plot(knownT_backlog_at_tt, color = 'C1', label = 'DPOP (given T)', linewidth=2)
    ax.plot(oracle_backlog_at_tt, color = 'C2', label = 'Oracle policy', linewidth=2)

    ax.set_xlim([-2500,100000])

    if(network_type == 'multi-user'): 
        ax.set_ylim([0,3300])
        y_exponent_label = '$\\times 10^3$'
        tick_labels = ['{:1.1f}'.format(s) for s in np.linspace(0,3,7)]
        tick_labels[0] = '0'
        ax.set_yticks(ticks=np.linspace(0,3000,7), labels=tick_labels)
    # else: ax.set_ylim([0,900])

    # show values in scientific notation 
    ax.set_xticks(ticks=10000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])

    # show exponent information near axes
    y_lim_val = ax.get_ylim(); x_lim_val = ax.get_xlim()
    ax.text(x_lim_val[0]+0.9*(x_lim_val[1] - x_lim_val[0]), y_lim_val[0]-0.11*(y_lim_val[1] - y_lim_val[0]), \
            '$\\times 10^4$', fontdict=None, size=label_font_size)
    if(network_type == 'multi-user'): ax.text(x_lim_val[0]-0.12*(x_lim_val[1] - x_lim_val[0]), y_lim_val[0]+0.955*(y_lim_val[1] - y_lim_val[0]), \
                                              y_exponent_label, fontdict=None, size=label_font_size)

    ax.set_xlabel('Time-slot')
    ax.set_ylabel('Total queue backlog')

    ax.legend(loc = 'lower right')
    ax.grid()

# low pass filter to make plots smoother 
def windowed_average(input_array, window_size = 250):
    output_array = np.convolve(input_array, np.ones(window_size)/window_size, mode='valid')
    return np.append(input_array[0], output_array) # append used to match input and output size

# function to plot total transmission costs
def plot_transmission_cost_curve(ax, network_type, unknownT_tran_cost_at_tt, knownT_tran_cost_at_tt, oracle_tran_cost_at_tt, label_font_size):
    ax.plot(windowed_average(oracle_tran_cost_at_tt), color = 'C2', label = 'Oracle policy', linewidth=2)
    ax.plot(windowed_average(knownT_tran_cost_at_tt), color = 'C1', label = 'DPOP (given T)', linewidth=2)
    ax.plot(windowed_average(unknownT_tran_cost_at_tt), color = 'C0', label = 'DPOP (doubling)', linewidth=2)

    ax.set_xlim([-2500,100000])
    stat_cost = windowed_average(oracle_tran_cost_at_tt)[-1] 
    ax.set_ylim([0.8*stat_cost,1.2*stat_cost])

    # show values in scientific notation 
    ax.set_xticks(ticks=10000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])

    # show exponent information near axes
    y_lim_val = ax.get_ylim(); x_lim_val = ax.get_xlim()
    ax.text(x_lim_val[0]+0.9*(x_lim_val[1] - x_lim_val[0]), y_lim_val[0]-0.11*(y_lim_val[1] - y_lim_val[0]), \
            '$\\times 10^4$', fontdict=None, size=label_font_size)

    ax.set_xlabel('Time-slot')
    ax.set_ylabel('Instantaneous transmission cost')

    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2]
    order.reverse()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'best')

    ax.grid()

def get_average_rates(edge_rates, edge_capacities, until_T = -1):
    return np.mean(edge_rates[:until_T,:], axis = 0)/edge_capacities

def plot_edge_utilization(ax, until_T_list, until_T_labels, dpop_rates, oracle_rates, edge_capacities):

    sorted_indices = np.argsort(get_average_rates(oracle_rates, edge_capacities))[-1::-1]

    mult = len(until_T_list)+1
    width = 2
    offset = 2
    x = (mult*width + offset)*np.arange(oracle_rates.shape[1])

    for ii, until_T in enumerate(until_T_list):
        ax.bar(x + ii*width, get_average_rates(dpop_rates, edge_capacities, until_T)[sorted_indices], width, label='DPOP (until '+str(until_T_labels[ii])+')')
    ax.bar(x+(mult-1)*width, get_average_rates(oracle_rates, edge_capacities)[sorted_indices], width, label='Oracle')

    
    # ax.set_xlim([x[0]-2*offset, x[-1] + mult*width + offset])
    ax.grid()
    ax.legend()
    if(sorted_indices.shape[0]<20): 
        temp2 = np.char.mod('$e_%d$', np.arange(oracle_rates.shape[1])+1)
        ax.set_xticks(x+width*mult/2-width/2, temp2)
    else:
        temp1 = x+width*mult/2-width/2
        temp2 = np.char.mod('$e_%d$', np.arange(oracle_rates.shape[1])+1)
        ax.set_xticks(temp1[::], temp2[::], minor=False)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(FixedLocator(temp1[1::2]))
        ax.yaxis.set_tick_params(which='minor', bottom=False)
    ax.set_axisbelow(True)

    ax.set_xlabel('Edges (sorted by oracle\'s traffic)')
    ax.set_ylabel('Edge traffic (% usage)')

    # ax.set_ylim([0, 1.2])
    ax.set_xlim([x[0]-width, x[8]-width])
    

