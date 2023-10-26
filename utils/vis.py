import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.style as style
import seaborn as sns
import os

# Function to calculate maximum drawdown
def calculate_max_drawdown(values):
    values = np.array(values)
    peak_index = np.argmax(np.maximum.accumulate(values) - values) # end of the period
    trough_index = np.argmax(values[:peak_index]) # start of period

    return values[trough_index] - values[peak_index]

# # Function to apply common styles to plots
def apply_common_styles():
    plt.style.use('./utils/plot.mplstyle')


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

    print("\n")
    
    # Plot portfolio weights
    apply_common_styles()
    plt.title('Portfolio weights (end of validation set) episode {}'.format(episode))
    plt.bar(np.arange(+1), end_weights[-1])
    plt.xticks(np.arange(config['nb_stocks']+6), ['Money'] + config['list_stock'], rotation=45)

    # Save figure if path is provided
    if save_path:
        plt.savefig(os.path.join(save_path, 'portfolio_weights_episode_{}.png'.format(episode)))

    plt.show()


