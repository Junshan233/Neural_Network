import numpy as np

# 用RNN实现二进制加法，基于numpy


class Relu(object):
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input):
        out = np.zeros_like(input)
        out[input < 0]


class Sigmoid(object):
    def forward(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def gradient(self, output):
        return output * (1 - output)


class RNNLayer(object):
    def __init__(self, input_dim, hidden_dim, output_dim, input_length,
                 activator, learning_rate):
        self.input_dim = input_dim
        self.didden_dim = hidden_dim
        self.input_length = input_length  # 序列长度
        self.activator = activator
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.state_list = []  # 保存每一步状态
        self.state_list.append(np.zeros((hidden_dim, 1)))  # 初始状态为0
        self.ot = [] # 每一步输出
        self.prediction = np.zeros((1, input_length))  # 每一步的预测输出 0 或 1
        # 初始化模型参数
        # s(t) = sigmoid(U*x(t) + W*s(t-1))
        # output(t) = sigmoid(V*s(t))
        self.U = np.random.uniform(-1e-4, 1e-4, (hidden_dim, input_dim))
        self.W = np.random.uniform(-1e-4, 1e-4, (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-1e-4, 1e-4, (output_dim, hidden_dim))
        # 初始化梯度参数
        self.U_grad = np.zeros_like(self.U)
        self.W_grad = np.zeros_like(self.W)
        self.V_grad = np.zeros_like(self.V)

    def forward(self, input_matrix): # 低位在左，高位在右
        # input_array - shape(2,input_length)
        for i in range(self.input_length):
            xt = input_matrix[:, i]
            state = np.dot(self.U, xt) + np.dot(self.W, self.state_list[-1])
            state = self.activator.forward(state)
            ot = self.activator.forward(np.dot(self.V,state))
            self.ot.append(ot)
            self.state_list.append(state)
            self.prediction[1,i] = 1 if ot > 0.5 else 0


    def backward(self, sensitivity_array, activator):
        self.calc_delta(sensitivity_array, activator)
        self.calc_gradient()

    def update(self):
        self.W -= self.W_grad * self.learning_rate
        self.U -= self.U_grad * self.learning_rate

    def calc_delta(self, sensitivity_array, activator):
        self.delta_list = []
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.didden_dim, 1)))
        self.delta_list.append(sensitivity_array)
