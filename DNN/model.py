from myDNNAllFunction import *
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples
# The "-1" makes reshape flatten the remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.


def test(pred, lab):
    a = np.ones_like(pred)
    a[pred <= 0.5] = 0
    b = np.zeros_like(pred)
    b[a == lab] = 1
    return np.mean(b)


def getAccuracy(test_x, test_y, parameters):
    AL_test, _ = L_forward_propagation(test_x, parameters)
    accu = test(AL_test, test_y)
    return accu


def L_layers_model(X,
                   Y,
                   test_x,
                   test_y,
                   layers_dims,
                   learning_rate=0.0075,
                   num_iterations=1500,
                   print_cost=False,
                   method="None"):
    parameters = init_parameters(layers_dims, method)
    costs = []
    accus = []
    for i in range(num_iterations):
        AL, caches = L_forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grade = L_backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grade, learning_rate)
        if print_cost and i % 100 == 0:
            accus.append(getAccuracy(test_x, test_y, parameters))
            print("Cost after iteration %i:%f, Accuracy: %f" %
                  (i, cost, accus[-1]))
            costs.append(cost)
    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters, np.squeeze(costs), accus


layers_dims = [12288, 20, 7, 5, 1]
parameters1, cost1, accus1 = L_layers_model(train_x,
                                            train_y,
                                            test_x,
                                            test_y,
                                            layers_dims,
                                            num_iterations=2500,
                                            print_cost=True,
                                            method="he")
# parameters2, cost2,accus2 = L_layers_model(train_x,
#                                     train_y,
#                                     test_x,
#                                     test_y,
#                                     layers_dims,
#                                     num_iterations=2500,
#                                     print_cost=True,
#                                     method="None")
plt.subplot(2, 1, 1)
plt.plot(cost1, label="he")
# plt.plot(cost2, label="None")
plt.ylabel('cost')
plt.title("Learning rate = 0.0075")
plt.subplot(2, 1, 2)
plt.plot(accus1)
plt.ylabel("Accuracy")
plt.xlabel('iterations (per tens)')
plt.show()
