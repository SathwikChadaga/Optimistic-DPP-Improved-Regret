import numpy as np
from scipy.optimize import lsq_linear  
import pickle as pkl

# low pass filter to make plots smoother 
def windowed_average(input_array, window_size = 15):
    output_array = np.convolve(input_array, np.ones(window_size)/window_size, mode='valid')
    return np.append(input_array[0], output_array) # append used to match input and output size



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
def plot_regret_curve(ax, arrival_rate_scaling, noise_variance_list, show_theoretical, sweep_results_folder, plot_style, show_ylabel, label_font_size):
    # iterate for given values of noise variances
    for jj, noise_variance in enumerate(noise_variance_list):
        with open(sweep_results_folder + '/regret-lambda-' + str(arrival_rate_scaling).replace('.','_') + '-var-' + str(noise_variance).replace('.','_') + '.pkl', 'rb') as f: 
            current_result = pkl.load(f)   

        dpop_costs =  current_result['dpop_costs']
        stat_costs = current_result['stat_costs']
        T_horizon_list = current_result['T_horizon_list']

        ax.plot(T_horizon_list, dpop_costs - stat_costs, plot_style[jj], label = '$\sigma^2$ = ' + str(noise_variance), fillstyle = 'none', markeredgewidth=2, ms=8)  

        if(show_theoretical[jj]):
            theoretical_regret = fit_regret_curve(T_horizon_list, dpop_costs - stat_costs, start_index = 6)
            ax.plot(T_horizon_list, theoretical_regret, '--', label = r'$O(\sqrt{T}\log{T})$', linewidth=3)

    if(show_ylabel): ax.set_ylabel('Regret')
    ax.set_xlabel('Time horizon')
    
    # show values in scientific notation and show exponent near axes
    # ax.set_xlim([10000,100000])
    # ax.set_xticks(ticks=np.linspace(10000,100000,10), labels=['{:1.0f}'.format(s) for s in np.linspace(1,10,10)])
    # ax.text(92000, -850, '$\\times 10^4$', fontdict=None, size=label_font_size)

    # ax.set_xlim([5000,25000])
    # ax.set_xticks(ticks=2500*np.arange(1,11), labels=['{:1.1f}'.format(s) for s in 2.5*np.arange(1,11)])
    # ax.text(20500, -900, '$\\times 10^3$', fontdict=None, size=label_font_size)
    
    # show values in scientific notation and show exponent near axes
    # ax.set_ylim([0,5000])
    # ax.set_yticks(ticks=np.linspace(0,5000,6), labels=['{:1.0f}'.format(s) for s in np.linspace(0,5,6)])
    # ax.text(9000, 5075, '$\\times 10^3$', fontdict=None, size=label_font_size)
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2,3,4]
    order.reverse()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left')
    
    ax.grid()


# function to plot total queue backlogs
def plot_backlog_curve(ax, unknownT_backlog_at_tt, knownT_backlog_at_tt, oracle_backlog_at_tt, label_font_size):
    ax.plot(unknownT_backlog_at_tt, color = 'C0', label = 'DPOP (doubling)')
    ax.plot(knownT_backlog_at_tt, color = 'C1', label = 'DPOP (given T)')
    ax.plot(oracle_backlog_at_tt, color = 'C2', label = 'Oracle policy')

    # show values in scientific notation and show exponent near axes
    # ax.set_xticks(ticks=1000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])
    # ax.text(9100, -9, '$\\times 10^3$', fontdict=None, size=label_font_size)

    # ax.set_xlim([-250,10000])
    # ax.set_ylim([0,65])

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
    # ax.set_xticks(ticks=1000*np.arange(0,11), labels=['{:1.0f}'.format(s) for s in np.arange(0,11)])
    # ax.text(9100, -0.7, '$\\times 10^3$', fontdict=None, size=label_font_size)

    # ax.set_xlim([-250,10000])
    # ax.set_ylim([0,5])

    ax.set_xlabel('Time-slot')
    ax.set_ylabel('Instantaneous transmission cost')

    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2]
    order.reverse()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'lower right')

    ax.grid()