import numpy as np

def get_random_action(nb_stocks):
    random_vec = np.random.rand(nb_stocks+1)
    return random_vec/np.sum(random_vec)