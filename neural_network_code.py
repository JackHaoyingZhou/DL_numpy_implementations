import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
from scipy.optimize import *
import matplotlib
import os
import sys
dynamic_path = os.path.abspath(__file__+"/../")
sys.path.append(dynamic_path)
matplotlib.use('Qt5Agg')

NUM_HIDDEN_LAYERS = 5
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 50 ]
# # NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 30 ]
# NUM_HIDDEN = [ 30, 40, 50 ]
NUM_OUTPUT = 10

############### Utility functions ####################


def load_data(set_name):
    f_name_image = f'fashion_mnist_{set_name}_images.npy'
    f_name_label = f'fashion_mnist_{set_name}_labels.npy'

    f_image_path = os.path.join(dynamic_path, f_name_image)
    f_label_path = os.path.join(dynamic_path, f_name_label)

    images = np.load(f_image_path).T / 255. - 0.5
    labels = np.load(f_label_path)

    # Convert labels vector to one-hot matrix (C x N).
    y = np.zeros((10, labels.shape[0]))
    y[labels, np.arange(labels.shape[0])] = 1

    return images, y


def weights_flatten(Ws, bs):
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])


def relu_activation(inputs):
    return np.maximum(0, inputs)


def grad_relu(inputs):
    return np.heaviside(inputs, 0)


# def softmax_old(inputs):
#     z = inputs - np.max(inputs, axis=-1, keepdims=True)
#     numerator = np.exp(z)
#     denominator = np.sum(numerator, axis=-1, keepdims=True)
#     softmax_value = numerator/denominator
#     return softmax_value

def softmax(inputs):
    z = inputs - np.max(inputs, axis=0)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=0)
    softmax_value = numerator/denominator
    return softmax_value


def forward_propagation(X, weights):
    Ws, bs = unpack(weights)
    activation_output = []
    raw_output = []
    output_a = X

    for i in range(len(Ws) - 1):
        z = Ws[i] @ output_a + bs[i].reshape(-1, 1)
        raw_output.append(z)
        output_a = relu_activation(z)
        activation_output.append(output_a)

    z_final = Ws[-1] @ activation_output[-1] + bs[-1].reshape(-1, 1)
    raw_output.append(z_final)

    yhat = softmax(z_final)

    return yhat, raw_output, activation_output


def loss_cross_entropy(yhat, y):
    return (-1 / y.shape[1]) * np.sum(y * np.log(yhat))


def backward_propagation(X, Y, weights):
    Ws, bs = unpack(weights)
    yhat, raw_output, activation_output = forward_propagation(X, weights)
    delta_w = [[] for _ in range(len(Ws))]
    delta_b = [[] for _ in range(len(bs))]
    error_o = yhat - Y
    for i in reversed(range(len(Ws) - 1)):
        error_i = Ws[i + 1].T @ error_o * grad_relu(activation_output[i])
        delta_w[i + 1] = error_o @ activation_output[i].T / Y.shape[1]
        delta_b[i + 1] = (np.sum(error_o, axis=1, keepdims=True) / Y.shape[1]).reshape((-1,))
        error_o = error_i
    delta_w[0] = error_o.dot(X.T)
    delta_b[0] = (np.sum(error_o, axis=1, keepdims=True) / Y.shape[1]).reshape((-1,))
    return delta_w, delta_b


def update_weights_bias(weights, delta_w, delta_b, lr):
    Ws, bs = unpack(weights)
    # print(self.layers[0].bias.shape)
    for i in range(len(Ws)):
        Ws[i] = Ws[i] - (lr * delta_w[i])
        bs[i] = bs[i] - (lr * delta_b[i])
    return weights_flatten(Ws, bs)


def get_accuracy_value(yhat, y):
    label_hat = np.argmax(yhat, axis=0)
    label_gt = np.argmax(y, axis=0)
    return (label_hat == label_gt).mean()


def predict(x, weights):
    yhat, _, _ = forward_propagation(x, weights)
    return yhat


############### Template given #######################


def unpack (weights):
    # Unpack arguments
    Ws = []
    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)
    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)
    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)
    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])
    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b)
    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)
    return Ws, bs


def fCE (X, Y, weights):
    # Ws, bs = unpack(weights)
    yhat, _, _ = forward_propagation(X, weights)
    ce = loss_cross_entropy(yhat, Y)
    # ...
    return ce


def gradCE (X, Y, weights):
    # Ws, bs = unpack(weights)
    delta_w, delta_b = backward_propagation(X, Y, weights)
    allGradientsAsVector = weights_flatten(delta_w, delta_b)
    # ...
    return allGradientsAsVector

# Creates an image representing the first layer of weights (W0).
def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()


def train(trainX, trainY, testX, testY, weights, lr = 5e-2, epochs=5, batch_size=128):
    # TODO: implement me
    loss_list = []
    for i in tqdm(range(epochs)):
        index_current = 0
        index_next = index_current + batch_size
        loss_current = loss_cross_entropy(predict(trainX, weights), trainY)
        loss_list.append(loss_current)
        print('loss: %s' % loss_current)
        # num_runs = int(trainX.shape[1]/batch_size)
        # pbar = tqdm(total = num_runs + 1)
        while (index_current < trainX.shape[1]):
            X_batch = trainX[:, index_current:index_next]
            Y_batch = trainY[:, index_current:index_next]
            yhat, _, _ = forward_propagation(X_batch, weights)
            delta_w, delta_b = backward_propagation(X_batch, Y_batch, weights)
            weights = update_weights_bias(weights, delta_w, delta_b, lr)
            index_current = index_next
            if index_current + batch_size > trainX.shape[1]:
                index_next = trainX.shape[1]
            else:
                index_next = index_current + batch_size
            # pbar.update(1)

            ### print accuracy
        yhat_train = predict(trainX, weights)
        acc_train = get_accuracy_value(yhat_train, trainY)
        yhat_test = predict(testX, weights)
        acc_test = get_accuracy_value(yhat_test, testY)
        print('train accuracy: %s' % (acc_train * 100))
        print('test accuracy: %s' % (acc_test * 100))
        # pbar.close()
    return weights, loss_list


def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i], NUM_HIDDEN[i+1]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs


if __name__ == "__main__":
    # Load training data.
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).
    trainX, trainY = load_data('train')
    testX, testY = load_data('test')

    Ws, bs = initWeightsAndBiases()
    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([W.flatten() for W in Ws ] + [ b.flatten() for b in bs])
    Ws, bs = unpack(weights)

    # print('test')
    # On just the first 5 training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).

    # time_start = time.time()
    # yhat = predict(trainX, weights)
    # loss = loss_cross_entropy(yhat, trainY)
    # print('runtime: %s' % (time.time() - time_start))
    # grad_validation = check_grad(lambda weights_: fCE(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]),
    #                                                   weights_),
    #                              lambda weights_: gradCE(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]),
    #                                                      weights_),
    #                              weights)
    # print('the average error of the gradient calculated is: %s' % (grad_validation/len(weights)))
    # fce_test = approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[:,0:5]),
    #                                                        np.atleast_2d(trainY[:,0:5]), weights_),
    #                          1e-6)
    # # # print(fce_test)
    # print('the forward estimation error is %s' % np.mean(fce_test))
    #
    weights, loss_list = train(trainX, trainY, testX, testY, weights, lr=1e-2, epochs=70, batch_size=128)

    time_start = time.time()
    yhat_train = predict(trainX, weights)
    acc_train = get_accuracy_value(yhat_train, trainY)
    yhat_test = predict(testX, weights)
    acc_test = get_accuracy_value(yhat_test, testY)
    print('train accuracy: %s' % (acc_train * 100))
    print('train accuracy: %s' % (acc_test * 100))
    show_W0(weights)
