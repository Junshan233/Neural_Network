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
        self.hidden_dim = hidden_dim
        self.input_length = input_length  # 序列长度
        self.activator = activator
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        # 保存每一步状态
        self.state_list = []
        self.z = []
        self.z2 = []
        self.xt_list = []

        self.state_list.append(np.zeros((hidden_dim, 1)))  # 初始状态为0
        self.ot = np.zeros((1,input_length)) # 每一步输出
        self.prediction = np.zeros((1, input_length))  # 每一步的预测输出 0 或 1
        # 初始化模型参数
        # s(t) = sigmoid(z) = sigmoid(U*x(t) + W*s(t-1))
        # output(t) = sigmoid(V*s(t))
        self.U = np.random.uniform(-1e-4, 1e-4, (hidden_dim, input_dim))
        self.W = np.random.uniform(-1e-4, 1e-4, (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-1e-4, 1e-4, (output_dim, hidden_dim))
        # 初始化梯度参数
        self.U_grad = np.zeros_like(self.U)
        self.W_grad = np.zeros_like(self.W)
        self.V_grad = np.zeros_like(self.V)

        self.loss = 0

    def forward(self, input_matrix):  # 低位在左，高位在右
        # input_array - shape(2,input_length)
        for i in range(self.input_length):
            xt = input_matrix[:, i]
            self.xt_list.append(xt)
            state = np.dot(self.U, xt) + np.dot(self.W, self.state_list[-1])
            self.z.append(state)
            state = self.activator.forward(state)
            z2 = np.dot(self.V, state)
            ot = self.activator.forward(z2)
            self.z2.append(z2)
            self.ot[1,i] = ot
            self.state_list.append(state)
            self.prediction[1, i] = 1 if ot > 0.5 else 0

    def calc_loss(self, y_label):
        # 交叉熵损失函数
        self.loss = np.squeeze(np.sum(np.abs(np.array(self.ot) - y_label)))
        self.loss = -1 / m * np.sum(y_label * np.log(self.ot) +
                                    (1 - y_label) * np.log(1 - self.ot))
        return self.loss

    def backward(self, sensitivity_array, activator, y_label):
        ot_grade = y_label / self.ot - (1 - y_label) / (1 - self.ot)
        next_delta = np.zeros((self.hidden_dim, 1))
        z_next = np.zeros((self.hidden_dim, 1))
        for i in range(self.input_length):
            ot_grade_now = ot_grade[1, self.input_length - 1 - i]
            z2_now = self.z2[self.input_length - 1 - i]
            ot_now = self.ot[1, self.input_length - 1 - i]
            st_now = self.state_list[self.input_length - i]
            xt_now = self.xt_list[self.input_length - 1 - i]
            z_now = self.z[self.input_length - 1 - i]
            delta = np.dot(
                np.dot(self.W.T, np.diag(activator.gradient(z_next))),
                next_delta) + np.dot(self.V.T,
                                     ot_grade_now * activator.gradient(z2_now))
            self.V_grad += np.dot(
                activator.gradient(ot_now) * ot_grade_now, st_now.T)
            self.U_grad += np.dot(
                np.dot(np.diag(activator.gradient(z_now)), delta), xt_now.T)
            self.W_grad += np.dot(
                np.dot(np.diag(activator.gradient(z_now)), delta),
                self.state_list[self.input_length - 1 - i])
            next_delta = delta
            z_next = z_now

    def update(self):
        self.W -= self.W_grad * self.learning_rate
        self.U -= self.U_grad * self.learning_rate
        self.V -= self.V_grad * self.learning_rate
        self.U_grad = np.zeros_like(self.U)
        self.W_grad = np.zeros_like(self.W)
        self.V_grad = np.zeros_like(self.V)

def gradient_check():
