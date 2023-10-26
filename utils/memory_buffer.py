import numpy as np


class PVM(object):
   
    def __init__(self, sample_bias, total_steps=754, batch_size=50, w_init=np.array([1, 0, 0, 0, 0, 0])):
        
        """
        Initializes the Portfolio Vector Memory (PVM) class.

        Args:
        - sample_bias: The sample bias used in the draw function.
        - total_steps: The total number of steps in the memory.
        - batch_size: The batch size used in the draw function.
        - w_init: The initial portfolio vector.
        """

        self.memory = np.transpose(np.array([w_init]*total_steps))
        self.sample_bias = sample_bias
        self.total_steps = total_steps
        self.batch_size = batch_size

    def get_W(self, t):
        return self.memory[:, t]

    def update(self, t, w):
        self.memory[:, t] = w


    def draw(self, beta=5e-05):
        while True:
            z = np.random.geometric(p=beta)
            tb = self.total_steps - self.batch_size + 1 - z
            if tb >= 0:
                return tb

    def test(self):
        return self.memory