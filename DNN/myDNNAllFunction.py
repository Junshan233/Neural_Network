import numpy as np


def init_parameters(layer_dims, method="None"):
    """
    argument:
    - layer_dims list
    - method "he" or "None"
    """
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        if method == "None":
            W = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        if method == "he":
            W = np.random.randn(layer_dims[i], layer_dims[i - 1]) / np.sqrt(
                layer_dims[i - 1] / 2.)
        b = np.zeros((layer_dims[i], 1))
        parameters["W" + str(i)] = W
        parameters["b" + str(i)] = b

    return parameters


def sigmoid(x):
    """
    argument: x - (n_x,m)
    return:   a - (n_x,m)
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def sigmoid_backword(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    return dA * s * (1 - s)


def relu_backward(dA, Z):
    dZ = np.ones_like(Z)
    dZ[Z <= 0] = 0
    return dA * dZ


def single_forward_propagation(A_prev, W, b, activation="relu"):
    Z = np.dot(W, A_prev) + b
    if activation == "relu":
        A = relu(Z)
    if activation == "sigmoid":
        A = sigmoid(Z)
    cache = (W, A_prev, b, Z, A)
    return cache


def L_forward_propagation(X, parameters):
    A = X
    L = len(parameters) // 2
    caches = []
    for l in range(1, L):
        A_prev = A
        cache = single_forward_propagation(A_prev,
                                           parameters["W" + str(l)],
                                           parameters["b" + str(l)],
                                           activation="relu")
        A = cache[4]
        caches.append(cache)
    cache = single_forward_propagation(A, parameters["W" + str(L)],
                                       parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    AL = cache[4]
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1. / m * np.sum(
        Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis=1, keepdims=True)
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    W, A_prev, _, _, _ = cache
    m = dZ.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    _, _, _, Z, _ = cache
    if activation == "relu":
        dZ = relu_backward(dA, Z)
    if activation == "sigmoid":
        dZ = sigmoid_backword(dA, Z)
    dA_prev, dW, db = linear_backward(dZ, cache)
    return dA_prev, dW, db


def L_backward_propagation(AL, Y, caches):
    grade = {}
    Y = Y.reshape(AL.shape)
    L = len(caches)
    dAL = -(Y / AL - (1 - Y) / (1 - AL))

    grade["dA" +
          str(L)], grade["dW" +
                         str(L)], grade["db" +
                                        str(L)] = linear_activation_backward(
                                            dAL, caches[L - 1], "sigmoid")

    for l in reversed(range(L - 1)):
        grade["dA" + str(l + 1)], grade["dW" + str(l + 1)], grade[
            "db" + str(l + 1)] = linear_activation_backward(
                grade["dA" + str(l + 2)], caches[l], "relu")

    return grade


def update_parameters(parameters, grade, learning_rate):

    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - grade["dW" + str(l)] * learning_rate
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - grade["db" + str(l)] * learning_rate

    return parameters
