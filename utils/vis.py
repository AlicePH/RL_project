import numpy as np
import matplotlib.pyplot as plt
import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.style as style
import os

# Function to calculate maximum drawdown
def calculate_max_drawdown(values):
    values = np.array(values)
    peak_index = np.argmax(np.maximum.accumulate(values) - values) # end of the period
    trough_index = np.argmax(values[:peak_index]) # start of period

    return values[trough_index] - values[peak_index]

# Function to apply common styles to plots
def apply_common_styles():
    plt.figure(figsize=(38.4 / 2, 21.6 / 2))

    plt.rcParams['font.size'] = 20
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['axes.titlepad'] = 24
    plt.rcParams['axes.labelpad'] = 12
    plt.rcParams['axes.facecolor'] = '#222222'
    plt.rcParams['axes.labelcolor'] = '#ffffff'
    plt.rcParams['savefig.facecolor'] = '#111111'
    plt.rcParams['xtick.color'] = '#ffffff'
    plt.rcParams['xtick.major.pad'] = 12
    plt.rcParams['xtick.minor.pad'] = 12
    plt.rcParams['ytick.color'] = '#ffffff'
    plt.rcParams['ytick.major.pad'] = 12
    plt.rcParams['ytick.minor.pad'] = 12
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False

    ax = plt.gca()
    ax.set_facecolor('#111111')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))

# Function to evaluate performance
def evaluate_performance(episode, env_eval, actor, config, save_path=None):
    # Initialize lists to store results
    end_weights = []
    end_portfolio_values = []
    min_portfolio_values = []
    max_portfolio_values = []
    mean_portfolio_values = []
    max_drawdowns = []

    # Reset environment
    state_eval, done_eval = env_eval.reset(config['w_init_test'], config['pf_init_test'], t=config['total_steps_train'])

    # Initialize portfolio and weight lists
    portfolio_values = [config['pf_init_test']]
    weights = [config['w_init_test']]

    # Iterate over steps
    for step in range(config['total_steps_train'], config['total_steps_train'] + config['total_steps_val'] - int(config['n'] / 2)):
        X_t = state_eval[0].reshape([-1]+ list(state_eval[0].shape))
        previous_weights = state_eval[1].reshape([-1]+ list(state_eval[1].shape))
        previous_portfolio_value = state_eval[2]

        action = actor.compute_W(X_t, previous_weights)
        state_eval, reward_eval, done_eval = env_eval.perform_action(action)

        X_next = state_eval[0]
        current_weights = state_eval[1]
        current_portfolio_value = state_eval[2]

        daily_return = X_next[-1, :, -1]

        portfolio_values.append(current_portfolio_value)
        weights.append(current_weights)

    # Append results to lists
    end_weights.append(weights[-1])
    end_portfolio_values.append(portfolio_values[-1])
    min_portfolio_values.append(np.min(portfolio_values))
    max_portfolio_values.append(np.max(portfolio_values))
    mean_portfolio_values.append(np.mean(portfolio_values))
    max_drawdowns.append(calculate_max_drawdown(portfolio_values))

    # Print results
    print('End of test PF value:', round(portfolio_values[-1]))
    print('Min of test PF value:', round(np.min(portfolio_values)))
    print('Max of test PF value:', round(np.max(portfolio_values)))
    print('Mean of test PF value:', round(np.mean(portfolio_values)))
    print('Max Draw Down of test PF value:', round(calculate_max_drawdown(portfolio_values)))
    print('End of test weights:', weights[-1])

    # Plot portfolio evolution
    apply_common_styles()
    plt.title('Portfolio evolution (validation set) episode {}'.format(episode))
    plt.plot(portfolio_values, label='Agent Portfolio Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    # Plot portfolio weights
    apply_common_styles()
    plt.title('Portfolio weights (end of validation set) episode {}'.format(episode))
    plt.bar(np.arange(+1), end_weights[-1])
    plt.xticks(np.arange(config['nb_stocks']+6), ['Money'] + config['list_stock'], rotation=45)

    # Save figure if path is provided
    if save_path:
        plt.savefig(os.path.join(save_path, 'portfolio_weights_episode_{}.png'.format(episode)))

    plt.show()



    names = ['Money'] + config['list_stock']
    w_list_eval = np.array(w_list_eval)

    for j in range(config['nb_stocks']+1):
        plt.plot(w_list_eval[:,j], label = 'Weight Stock {}'.format(names[j]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)

    if path_save:
        plt.savefig(
            os.path.join(path_save, 'assests_weights_episode_{}.png'.format(episode))
            )


    plt.show()
