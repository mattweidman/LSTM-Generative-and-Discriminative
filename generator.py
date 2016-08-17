import numpy as np
import random

from LSTM import LSTM_layer, LSTM

list_of_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
num_chars = len(list_of_chars)

def vector_to_char(vector):
    index = np.argmin(vector)
    return list_of_chars[index]

def matrix_to_string(mat):
    output_str = ''
    for v in mat:
        c = vector_to_char(v)
        output_str += c
    return output_str

def softmax(x):
    denominator = np.sum(np.exp(x), axis=1)[:,np.newaxis]
    return np.exp(x)/denominator

if __name__ == "__main__":
    # backprop for one layer
    input_size = 5
    output_size = input_size
    num_examples = 1000

    x = np.zeros((num_examples, input_size))
    for row in x:
        randind = random.randint(0, input_size-1)
        row[randind] = 1
    layer = LSTM_layer(input_size, output_size)

    def loss(h):
        n_ex = h.shape[0]
        return 1/(2*n_ex) * np.sum((x-h)**2)
    def dloss(h):
        n_ex = h.shape[0]
        return 1/n_ex * (h-x)

    layer_grad = layer.backprop(x, dloss)
    dLdtheta, dLdx, dLds_prev, dLdh_prev = layer_grad.to_tuple()

    def assert_same_shape(a1, a2):
        assert len(a1.shape) == len(a2.shape)
        for i in range(len(a1.shape)):
            assert a1.shape[i] == a2.shape[i]

    assert_same_shape(x, dLdx)
    assert_same_shape(layer.s0, dLds_prev)
    assert_same_shape(layer.h0, dLdh_prev)
    for i in range(len(dLdtheta)):
        assert_same_shape(layer.theta[i], dLdtheta[i])

    num_epochs = 10000
    learning_rate = .1
    for i in range(num_epochs):
        grad = layer.backprop(x, dloss)
        layer.update_theta_s0_h0(grad, learning_rate)
        outp = layer.forward_prop_once(x)
        print(loss(outp[1]))
    outp = layer.forward_prop_once(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]))
    print(outp[1])

    '''# construct the LSTM
    input_size = num_chars
    hidden_size = 30
    output_size = num_chars
    network = LSTM()
    network.add_layer(LSTM_layer(input_size, hidden_size))
    network.add_layer(LSTM_layer(hidden_size, output_size))

    # construct the input
    seq_length = 50
    num_examples = 10
    X_once = np.random.randn(num_examples, input_size)
    X_sequence = np.random.randn(num_examples, seq_length, input_size)

    # use the LSTM
    sequence_tensor = network.forward_prop_feedback(X_once, seq_length)
    for matx in sequence_tensor:
        outp = matrix_to_string(matx)
        print(outp)'''
