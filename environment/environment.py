import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.registration import register

class TradeEnv():
    """
    This class represents a trading environment that adjusts the weights of assets at each step.
    It simulates a trading scenario with a given portfolio value and allows for actions to be taken at each step.
    """

    def __init__(self, data_path, window_size=50, initial_portfolio_value=10000, 
                 trading_cost=0.25/100, interest_rate=0.02/250, train_size=0.7):
        """
        Initialize the trading environment with given parameters.
        """
        self.data_path = data_path
        self.asset_data = np.load(self.data_path)

        self.portfolio_value = initial_portfolio_value
        self.window_size = window_size
        self.transaction_cost = trading_cost
        self.interest_rate = interest_rate

        self.asset_count = self.asset_data.shape[1]
        self.feature_count = self.asset_data.shape[0]
        self.training_data_end = int((self.asset_data.shape[2]-self.window_size)*train_size)
        
        self.current_index = None
        self.current_state = None
        self.is_done = False
        self.initialize_random_seed()

    def get_portfolio_value(self):
        """
        Return the current portfolio value.
        """
        return self.portfolio_value
        
    def get_tensor_slice(self, tensor, time_index):
        """
        Return a slice of the tensor for a given time window.
        """
        return tensor[:, :, time_index-self.window_size:time_index]
    
    def get_update_vector(self, time_index):
        """
        Return the update vector for a given time index.
        """
        return np.array([1+self.daily_interest_rate]+self.asset_data[-1,:,time_index].tolist())

    def initialize_random_seed(self, seed=None):
        """
        Initialize the random seed for reproducibility.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, initial_weights, initial_portfolio, t=0):
        """
        Reset the environment to its initial state.
        """
        self.current_state = (self.get_tensor_slice(self.asset_data, self.window_size), initial_weights, initial_portfolio)
        self.current_index = self.window_size + t
        self.is_done = False
        
        return self.current_state, self.is_done

    def perform_action(self, action):
        """
        Perform an action in the environment and return the new state, reward, and done status.
        """
        time_index = self.current_index
        tensor_slice = self.get_tensor_slice(self.asset_data, time_index)
        done_status = self.is_done
        state = self.current_state
        previous_weights = state[1]
        previous_portfolio = state[2]
        
        update_vector = self.get_update_vector(time_index)

        allocation_weights = action
        portfolio_allocation = previous_portfolio
        
        transaction_cost = portfolio_allocation * np.linalg.norm((allocation_weights-previous_weights),ord = 1)* self.transaction_cost
        value_allocation = portfolio_allocation * allocation_weights
        portfolio_after_transaction = portfolio_allocation - transaction_cost
        value_after_transaction = value_allocation - np.array([transaction_cost]+ [0]*self.asset_count)
        
        value_evolution = value_after_transaction * update_vector
        portfolio_evolution = np.sum(value_evolution)
        weight_evolution = value_evolution / portfolio_evolution
        reward = (portfolio_evolution - previous_portfolio) / previous_portfolio
        time_index += 1
        
        state = (self.get_tensor_slice(self.asset_data, time_index), weight_evolution, portfolio_evolution)
        
        if time_index >= self.training_data_end:
            done_status = True
        
        self.current_state = state
        self.current_index = time_index
        self.is_done = done_status
        
        return state, reward, done_status
