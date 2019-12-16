import numpy as np


class Relu(object):
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input):
        out = np.zeros_like(input)
        out[input < 0]


class RNNLayer(object):
    def __init__(self, input_width, state_width, activator, learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((state_width, 1)))
        self.U = np.random.uniform(-1e-4, 1e-4, (state_width, input_width))
        self.W = np.random.uniform(-1e-4, 1e-4, (state_width, state_width))
        self.bias = np.zeros((state_width,1))
        self.U_grad = np.zeros_like(self.U)
        self.W_grad = np.zeros_like(self.W)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, input_array):
        self.times += 1
        state = np.dot(self.U, input_array) + np.dot(
            self.W, self.state_list[self.times[-1]]) + self.bias
        state = activator.forward(state)
        self.state_list.append(state)

    def backward(self,sensitivity_array,activator):
        self.calc_delta(sensitivity_array,activator)
        self.calc_gradient()

    def update(self):
        self.W -= self.W_grad*self.learning_rate
        self.U -= self.U_grad*self.learning_rate
        self.bias -= self.bias_grad*learning_rate
    
    def calc_delta(self,sensitivity_array,activator):
        self.delta_list = []
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width,1)))
        self.delta_list.append(sensitivity_array)
        for k in range()

