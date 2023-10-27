import tensorflow as tf
import numpy as np
from collections import deque
import random
import pandas as pd
import ffn

import matplotlib
import matplotlib.pyplot as plt


from environment.environment import *
from actor.policy import Policy
from utils import *


from tqdm import tqdm 
import os
import argparse
import warnings
warnings.filterwarnings('ignore')




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-file', type=str, default= os.path.join(os.getcwd(), 'data', 'input.npy'),
                        help='The path to the dataset in folder data.')
    
    parser.add_argument('--vis-train', type=str, default=os.path.join(os.getcwd(), 'assets', 'train'),
                                   help='The path to save training vis.')
    
    parser.add_argument('--vis-analysis', type=bool, default=False,
                                   help='Make analysis')
    

    PATH_TRAIN_OUTPUT =  os.path.join(os.getcwd(), 'assets', 'train')
    

    parser.add_argument('--data-type', type=str, default='Crypto',
                                   help='Type of cryptocurrency')

    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    
    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)

    cfg['trading_period'] = np.load(args.data_file).shape[2]
    cfg['total_steps_val'] = int(cfg['val_ratio']*cfg['trading_period'])
    cfg['total_steps_val'] = int(cfg['val_ratio']*cfg['trading_period'])

    cfg['total_steps_test'] = cfg['trading_period'] - \
        cfg['total_steps_train'] - cfg['total_steps_val']



    return args, cfg




if __name__ == '__main__':
    args, config = parse_args()
    print('Create Environments')



    #policy network agent's environment for trading
    env = TradeEnv(data_path=args.data_file, 
                   window_size=config['n'],
                    initial_portfolio_value=config['p_init_train'], 
                    trading_cost=config['trading_cost'],
                    interest_rate=config['interest_rate'], 
                    train_size=config['train_ratio']
                    )


    #environment for unform weights
    env_eq = TradeEnv(data_path=args.data_file, 
                window_size=config['n'],
                initial_portfolio_value=config['p_init_train'], 
                trading_cost=config['trading_cost'],
                interest_rate=config['interest_rate'], 
                train_size=config['train_ratio']
                )

    #environment where agent keeps assets in cash
    env_s = TradeEnv(data_path=args.data_file, 
                window_size=config['n'],
                initial_portfolio_value=config['p_init_train'], 
                trading_cost=config['trading_cost'],
                interest_rate=config['interest_rate'], 
                train_size=config['train_ratio']
                )


    # every asset should have the same enfiironment
    env_fu = [TradeEnv(data_path=args.data_file, 
                window_size=config['n'],
                initial_portfolio_value=config['p_init_train'], 
                trading_cost=config['trading_cost'],
                interest_rate=config['interest_rate'], 
                train_size=config['train_ratio']
                ) for _ in range(5)] 
    

    action_fu = []
    for i in range(config['number_assets']):
        action = np.array([0]*(i+1) + [1] + [0]*(config['number_assets']-(i+1)))
        action_fu.append(action)


    w_eq = np.array(np.array([1 / (config['number_assets']+1)]*(config['number_assets']+1)))
    w_s = np.array(np.array([1] + [0.0]*config['number_assets']))






    ############# TRAINING #####################
    print('TRAINING')
    ###########################################
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    # session
    sess = tf.compat.v1.Session()
    optimizer =  tf.compat.v1.train.AdamOptimizer(config['learning'])



    # policy initialization
    actor = Policy(config['number_assets'], 
                   config['n'], 
                   sess, 
                   optimizer,
                   w_eq=w_eq,
                   trading_cost=config['trading_cost'],
                   interest_rate=config['interest_rate'])  

    sess.run(tf.compat.v1.global_variables_initializer())



    final_portfolio =  []
    final_portfolio_eq = []
    final_portfolio_s = []

    final_portfolio_fu = []

    state_fu = np.zeros(config['number_assets'], dtype=int).tolist()
    done_fu = np.zeros(config['number_assets'], dtype=int).tolist()

    portfolio_value_t_fu = np.zeros(config['number_assets'], dtype=int).tolist()

    for i in range(config['number_assets']):
        final_portfolio_fu.append(list())


    ###### Train #####
    for episode in range(config['n_episodes']):
        if episode==0:

            env_eval = TradeEnv(data_path=args.data_file, 
                                window_size=config['n'],
                                initial_portfolio_value=config['p_init_train'], 
                                trading_cost=config['trading_cost'],
                                interest_rate=config['interest_rate'], 
                                train_size=config['train_ratio']
                                )
            

            evaluate_performance('Before Training', env_eval, actor, config, save_path=args.vis_train)

        print('Episode: ', episode)


        #init memrybuffer
        memory = PVM(
                    config['sample_bias'], 
                    total_steps= config['total_steps_train'],
                    batch_size = config['batch_size'],
                    w_init = config['w_init_train'])

        for nb in range(config['n_batches']):
            i_start = memory.draw()

            state, done = env.reset(memory.get_W(i_start), config['p_init_train'], t=i_start )
            state_eq, done_eq = env_eq.reset(w_eq, config['p_init_train'], t=i_start )
            state_s, done_s = env_s.reset(w_s, config['p_init_train'], t=i_start )

            for i in range(config['number_assets']):
                state_fu[i], done_fu[i] = env_fu[i].reset(action_fu[i], config['p_init_train'], t=i_start )



            list_X_t, list_W_previous, list_portfolio_value_previous, list_dailyReturn_t = [], [], [], []
            list_portfolio_value_previous_eq, list_portfolio_value_previous_s = [],[]
            list_portfolio_value_previous_fu = []

            for i in range(config['number_assets']):
                list_portfolio_value_previous_fu.append([])


            for bs in range(config['batch_size']):

                #load the different inputs from the previous loaded state
                X_t = state[0].reshape([-1] + list(state[0].shape))
                W_previous = state[1].reshape([-1] + list(state[1].shape))
                portfolio_value_previous = state[2]


                if np.random.rand() < config['ratio_greedy']:
                    #computation of the action of the agent
                    action = actor.compute_W(X_t, W_previous)
                else:
                    action = get_random_action(config['number_assets'])

                # given the state and the action, call the environment to go one time step later
                state, reward, done = env.perform_action(action)
                state_eq, reward_eq, done_eq = env_eq.perform_action(w_eq)
                state_s, reward_s, done_s = env_s.perform_action(w_s)

                for i in range(config['number_assets']):
                    state_fu[i], _ , done_fu[i] = env_fu[i].perform_action(action_fu[i])



                #get the new state
                X_next = state[0]
                W_t = state[1]
                portfolio_value_t = state[2]

                portfolio_value_t_eq = state_eq[2]
                portfolio_value_t_s = state_s[2]

                for i in range(config['number_assets']):
                    portfolio_value_t_fu[i] = state_fu[i][2]


                #compute the returns
                dailyReturn_t = X_next[-1, :, -1]
                memory.update(i_start+bs, W_t)
                list_X_t.append(X_t.reshape(state[0].shape))
                list_W_previous.append(W_previous.reshape(state[1].shape))
                list_portfolio_value_previous.append([portfolio_value_previous])
                list_dailyReturn_t.append(dailyReturn_t)

                list_portfolio_value_previous_eq.append(portfolio_value_t_eq)
                list_portfolio_value_previous_s.append(portfolio_value_t_s)

                for i in range(config['number_assets']):
                    list_portfolio_value_previous_fu[i].append(portfolio_value_t_fu[i])


                if bs==config['batch_size']-1:
                    final_portfolio.append(portfolio_value_t)
                    final_portfolio_eq.append(portfolio_value_t_eq)
                    final_portfolio_s.append(portfolio_value_t_s)

                    for i in range(config['number_assets']):
                        final_portfolio_fu[i].append(portfolio_value_t_fu[i])


            list_X_t = np.array(list_X_t)
            list_W_previous = np.array(list_W_previous)
            list_portfolio_value_previous = np.array(list_portfolio_value_previous)
            list_dailyReturn_t = np.array(list_dailyReturn_t)


            #for each batch, train the network to maximize the reward
            actor.train(list_X_t, list_W_previous,
                        list_portfolio_value_previous, list_dailyReturn_t)
            
            
        env_eval = TradeEnv(data_path=args.data_file, 
                                window_size=config['n'],
                                initial_portfolio_value=config['p_init_train'], 
                                trading_cost=config['trading_cost'],
                                interest_rate=config['interest_rate'], 
                                train_size=config['train_ratio']
                                )

        evaluate_performance(episode, env_eval, actor, config, save_path=args.vis_train)


    print('End Training\n')

    print('Testing')


    #######TEST#######


    #initialization of the environment
    state, done = env.reset(config['w_init_test'], config['portfolio_init_test'], t = config['total_steps_train'])

    state_eq, done_eq = env_eq.reset(w_eq, config['portfolio_init_test'], t = config['total_steps_train'])
    state_s, done_s = env_s.reset(w_s, config['portfolio_init_test'], t = config['total_steps_train'])

    for i in range(config['number_assets']):
        state_fu[i],  done_fu[i] = env_fu[i].reset(action_fu[i], config['portfolio_init_test'], t = config['total_steps_train'])


    #first element of the weight and portfolio value
    p_list = [config['portfolio_init_test']]
    w_list = [config['w_init_test']]

    p_list_eq = [config['portfolio_init_test']]
    p_list_s = [config['portfolio_init_test']]


    p_list_fu = list()
    for i in range(config['number_assets']):
        p_list_fu.append([config['portfolio_init_test']])

    portfolio_value_t_fu = [0]*config['number_assets']

    prev_value = 10000


    for k in range(config['total_steps_train'] + config['total_steps_val']-\
                   int(config['n']/2), config['total_steps_train'] +config['total_steps_val'] + \
                    config['total_steps_test'] - config['n']):
        
        X_t = state[0].reshape([-1]+ list(state[0].shape))
        W_previous = state[1].reshape([-1]+ list(state[1].shape))
        portfolio_value_previous = state[2]
        #compute the action
        action = actor.compute_W(X_t, W_previous)
        #step forward environment
        state, reward, done = env.perform_action(action)
        state_eq, reward_eq, done_eq = env_eq.perform_action(w_eq)
        state_s, reward_s, done_s = env_s.perform_action(w_s)


        for i in range(config['number_assets']):
            state_fu[i], _ , done_fu[i] = env_fu[i].perform_action(action_fu[i])


        X_next = state[0]
        W_t = state[1]
        portfolio_value_t = state[2]

        portfolio_value_t_eq = state_eq[2]
        portfolio_value_t_s = state_s[2]

        for i in range(config['number_assets']):
            portfolio_value_t_fu[i] = state_fu[i][2]

        dailyReturn_t = X_next[-1, :, -1]


        difference = int(portfolio_value_previous)-prev_value
        diff = 'indrease' if difference>0 else 'decrease'

        if k%20 == 0:
            print('current portfolio value: ', int(portfolio_value_previous), f'{diff}: ', difference)
            print('weights\n', W_previous[0])


        p_list.append(portfolio_value_t)
        w_list.append(W_t)

        p_list_eq.append(portfolio_value_t_eq)
        p_list_s.append(portfolio_value_t_s)
        for i in range(config['number_assets']):
            p_list_fu[i].append(portfolio_value_t_fu[i])

        #here to breack the loop/not in original code
        if k== config['total_steps_train'] +config['total_steps_val']-int(config['n']/2) + 100:
            break






    if args.vis_analysis:
        print('\nAnalysis Done')

        path = os.path.join(os.getcwd(), "individual_stocks_5yr", "A_data.csv")
        times = pd.read_csv(path).date
        test_start_day =config['total_steps_train'] +config['total_steps_val']-int(config['n']/2)+10
        times = list(times[test_start_day:])


        plt.title('Portfolio Value (Test Set) {}: {}, {}, {}, {}, {}, {}, {}, {}'.format(args.data_type, 
                                                                                         config['batch_size'], 
                                                                                         config['learning'], 
                                                                                         config['ratio_greedy'], 
                                                                                         episode, 
                                                                                         config['n'], 
                                                                                         config['kernel_size'], 
                                                                                         config['n_batches'], 
                                                                                         config['ratio_regul']))
        plt.plot(p_list, label = 'Agent Portfolio Value')
        plt.plot(p_list_eq, label = 'Equi-weighted Portfolio Value')
        plt.plot(p_list_s, label = 'Secured Portfolio Value')
        for i in range(config['number_assets']):
            plt.plot(p_list_fu[i], label = 'Full Stock {} Portfolio Value'.format(config['list_stock'][i]))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


        plt.savefig(os.path.join(os.getcwd(), "assets", "analysis", 'portfolio_value.png'))
        plt.draw()




        names = ['Money'] + config['list_stock'].tolist()
        w_list = np.array(w_list)
        for j in range(config['number_assets']+1):
            plt.plot(w_list[:,j], label = 'Weight Stock {}'.format(names[j]))
            plt.title('Weight evolution during testing')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)



        plt.savefig(os.path.join(os.getcwd(), "assets", "analysis", 'Weight_evolution_during_testing.png'))
        plt.draw()

        plt.plot(np.array(p_list)-np.array(p_list_eq))

        plt.savefig(os.path.join(os.getcwd(), "assets", "analysis", 'something.png'))
        plt.draw()


        index1=0
        index2=-1

        plt.plot(final_portfolio[index1:index2], label = 'Agent Portfolio Value')
        plt.plot(final_portfolio_eq[index1:index2], label = 'Baseline Portfolio Value')
        plt.plot(final_portfolio_s[index1:index2], label = 'Secured Portfolio Value')

        plt.savefig(os.path.join(os.getcwd(), "assets", "analysis", 'results.png'))

        plt.legend()
        plt.draw()
