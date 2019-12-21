import numpy as np
import matplotlib.pyplot as plt
# 用RNN实现二进制加法，基于numpy


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

        self.ot = np.zeros((1, input_length))  # 每一步输出
        self.prediction = np.zeros((1, input_length))  # 每一步的预测输出 0 或 1
        # 初始化模型参数
        # s(t) = sigmoid(z) = sigmoid(U*x(t) + W*s(t-1))
        # output(t) = sigmoid(V*s(t))
        self.U = np.random.randn(hidden_dim, input_dim)
        self.W = np.random.randn(hidden_dim, hidden_dim)
        self.V = np.random.randn(output_dim, hidden_dim)

        self.loss = 0

    def forward(self, input_matrix):  # 低位在左，高位在右
        # input_array - shape(2,input_length)
        # 保存每一步状态
        self.state_list = []
        self.xt_list = []
        self.state_list.append(np.zeros((self.hidden_dim, 1)))  # 初始状态为0
        for i in range(self.input_length):
            xt = input_matrix[:, i].reshape((2, 1))
            self.xt_list.append(xt)
            state = self.activator.forward(
                np.dot(self.U, xt) + np.dot(self.W, self.state_list[-1]))
            ot = self.activator.forward(np.dot(self.V, state))
            self.ot[0, i] = ot
            self.state_list.append(state)
            self.prediction[0, i] = 1 if ot > 0.5 else 0

    def calc_loss(self, y_label):
        # 交叉熵损失函数
        self.loss = np.squeeze(np.sum(np.abs(np.array(self.ot) - y_label)))
        self.loss = -1 / self.input_length * np.sum(y_label * np.log(self.ot) +
                                                    (1 - y_label) *
                                                    np.log(1 - self.ot))
        return self.loss

    def backward(self, y_label):
        # 初始化梯度参数
        self.U_grad = np.zeros_like(self.U)
        self.W_grad = np.zeros_like(self.W)
        self.V_grad = np.zeros_like(self.V)
        ot_grade = -1 / self.input_length * (y_label / self.ot -
                                             (1 - y_label) / (1 - self.ot))
        next_delta = np.zeros((self.hidden_dim, 1))
        st_next = np.zeros((self.hidden_dim, 1))
        for i in range(self.input_length):
            ot_grade_now = ot_grade[0, self.input_length - 1 - i]
            ot_now = self.ot[0, self.input_length - 1 - i]
            st_now = self.state_list[self.input_length - i]
            xt_now = self.xt_list[self.input_length - 1 - i]
            delta = np.dot(
                np.dot(self.W.T,
                       np.diag(np.squeeze(self.activator.gradient(st_next)))),
                next_delta) + np.dot(
                    self.V.T, ot_grade_now * self.activator.gradient(ot_now))
            self.V_grad += np.dot(
                self.activator.gradient(ot_now) * ot_grade_now, st_now.T)
            self.U_grad += np.dot(
                np.dot(np.diag(np.squeeze(self.activator.gradient(st_now))),
                       delta), xt_now.T)
            self.W_grad += np.dot(
                np.dot(np.diag(np.squeeze(self.activator.gradient(st_now))),
                       delta), self.state_list[self.input_length - 1 - i].T)
            next_delta = delta
            st_next = st_now

    def update(self):
        self.W -= self.W_grad * self.learning_rate
        self.U -= self.U_grad * self.learning_rate
        self.V -= self.V_grad * self.learning_rate

    def change_parameter(self, U, W, V):
        self.U = U
        self.W = W
        self.V = V


def int2binary(num, dim):
    '''
    input:
        - num: 范围为0 - 2^dim-1
        - dim: 整数，二进制位数
    return:
        - out: numpy.array,shape=(1,dim),转换后的二进制数组,低位在前，高位在后
    '''
    out = np.zeros((1, dim))
    i = 0
    while (num != 0):
        out[0, i] = num % 2
        num = int(num / 2)
        i += 1
    return out


def binary2int(array, dim):
    '''
    input:
        - array: numpy.array,shape=(1,dim),低位在前，高位在后
        - dim: array的位数
    return:
        - out: 整数
    '''
    temp = 1
    out = 0
    for i in range(dim):
        out += array[0, i] * temp
        temp *= 2
    return out


if __name__ == "__main__":
    EPOCH = 20000
    dim = 8
    largest_number = 2**dim - 1
    losses = []
    rnn = RNNLayer(input_dim=2,
                   hidden_dim=16,
                   output_dim=1,
                   input_length=dim,
                   activator=Sigmoid(),
                   learning_rate=0.1)
    for epoch in range(EPOCH):
        a_int = np.random.randint(largest_number / 2)
        b_int = np.random.randint(largest_number / 2)
        c_int = a_int + b_int
        a_binary = int2binary(a_int, dim)
        b_binary = int2binary(b_int, dim)
        c_binary = int2binary(c_int, dim)
        input_matrix = np.concatenate((a_binary, b_binary), axis=0).reshape(
            (2, dim))
        rnn.forward(input_matrix)
        rnn.backward(c_binary)
        rnn.update()
        if epoch % 100 == 0:
            loss = rnn.calc_loss(c_binary)
            losses.append(loss)

        if epoch % 1000 == 0:
            loss = rnn.calc_loss(c_binary)
            print("epoch:{}".format(epoch))
            print("Loss:{}".format(loss))
            print("Pred:" + str(rnn.prediction))
            print("True:" + str(c_binary))
            print("------------------------")
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
