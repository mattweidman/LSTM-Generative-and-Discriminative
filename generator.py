import numpy as np
import random

from LSTM import LSTM_layer, LSTM

list_of_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
num_chars = len(list_of_chars)
char_dict = dict((c,i) for i,c in enumerate(list_of_chars))

def vector_to_char(vector):
    index = np.argmax(vector)
    return list_of_chars[index]

# matrix size: (seq_length, num_chars)
def matrix_to_string(mat):
    output_str = ''
    for v in mat:
        c = vector_to_char(v)
        output_str += c
    return output_str

# tensor size: (num_examples, seq_length, num_chars)
def tensor_to_strings(tensor):
    ans = ""
    for matx in tensor:
        outp = matrix_to_string(matx)
        ans += outp + "\n"
    return ans

def char_to_vec(c):
    ans = np.zeros((num_chars))
    ans[char_dict[c]] = 1
    return ans

def softmax(x):
    denominator = np.sum(np.exp(x), axis=1)[:,np.newaxis]
    return np.exp(x)/denominator

if __name__ == "__main__":

    # construct the LSTM
    input_size = num_chars
    hidden_size = 30
    output_size = num_chars
    network = LSTM()
    network.add_layer(LSTM_layer(input_size, hidden_size))
    network.add_layer(LSTM_layer(hidden_size, output_size))

    # construct the input
    seq_length = 5
    num_examples = 1000
    def gen_x_sequence():
        X_sequence = np.zeros((num_examples, input_size))
        for i in range(num_examples):
            randind = random.randint(0, num_chars-1)
            X_sequence[i,randind] = 1
        return X_sequence
    def normalize(v):
        return (v-v.mean())/v.std()

    # loss function and its gradient
    def get_n(h_):
        n = 1
        for dim in h_.shape[:-1]:
            n *= dim
        return n
    def loss(h_, x_):
        return 1/(2*get_n(h_)) * np.sum((h_-x_)**2)
    def dloss(h_, x_):
        return 1/get_n(h_) * (h_-x_)

    # train
    num_epochs = 1000
    learning_rate = 10
    for i in range(num_epochs):
        X_sequence = gen_x_sequence()
        dloss_i = lambda h_, i_: dloss(h_, X_sequence)
        grad = network.BPTT_feedback(normalize(X_sequence), seq_length, dloss_i)
        network.update_theta(grad, learning_rate)
        outp = network.forward_prop_feedback(normalize(X_sequence), seq_length)
        total_loss = loss(outp, X_sequence[:,np.newaxis,:].repeat(seq_length,
            axis=1))
        magnitude = sum([gl.magnitude_theta() for gl in grad])
        print("cost:%f\tgradient:%f" % (total_loss, magnitude))

    # use the LSTM
    def char_to_matx(c, length=seq_length):
        return [char_to_vec(c)] * length
    inp = np.array([char_to_matx(chr(c)) for c in range(97, 123)])
    print(inp.shape)
    outp = network.forward_prop_one2one(normalize(inp))
    for i in range(inp.shape[0]):
        print("%s\t%s" % (chr(i+97), matrix_to_string(outp[i])))

    '''# backprop for multiple layers
    input_size = 5
    hidden_size = 10
    output_size = 5
    num_examples = 1000

    x = np.zeros((num_examples, input_size))
    for row in x:
        randind = random.randint(0, input_size-1)
        row[randind] = 1

    def make_inner_matx(n_ex=num_examples, hid_size=hidden_size,
            out_size=output_size):
        return [np.zeros((n_ex, hid_size)),
            np.zeros((n_ex, out_size))]
    s_prev = make_inner_matx()
    h_prev = make_inner_matx()

    layer1 = LSTM_layer(input_size, hidden_size)
    layer2 = LSTM_layer(hidden_size, output_size)
    lstmnet = LSTM()
    lstmnet.add_layer(layer1)
    lstmnet.add_layer(layer2)

    def loss(h):
        n_ex = h.shape[0]
        return 1/(2*n_ex) * np.sum((x-h)**2)

    def dloss(h):
        n_ex = h.shape[0]
        return 1/n_ex * (h-x)

    def assert_same_shape(a1, a2):
        assert len(a1.shape) == len(a2.shape)
        for i in range(len(a1.shape)):
            assert a1.shape[i] == a2.shape[i]

    lstm_grad = lstmnet.backprop_once(x, dloss, s_prev, h_prev)

    grad1 = lstm_grad[0]
    assert_same_shape(x, grad1.dLdx)
    assert_same_shape(s_prev[0], grad1.dLds_prev)
    assert_same_shape(h_prev[0], grad1.dLdh_prev)
    for th, dth in zip(layer1.theta, grad1.dLdtheta):
        assert_same_shape(th, dth)

    grad2 = lstm_grad[1]
    assert_same_shape(np.zeros((num_examples, hidden_size)), grad2.dLdx)
    assert_same_shape(s_prev[1], grad2.dLds_prev)
    assert_same_shape(h_prev[1], grad2.dLdh_prev)
    for th, dth in zip(layer2.theta, grad2.dLdtheta):
        assert_same_shape(th, dth)

    num_epochs = 1000
    learning_rate = 3
    for i in range(num_epochs):
        grad = lstmnet.backprop_once(x, dloss, s_prev, h_prev)
        for layer, layer_grad in zip(lstmnet.layers, grad):
            layer.update_theta(layer_grad, learning_rate)
        outp = lstmnet.forward_prop_once(x, s_prev, h_prev)
        print(loss(outp[1][-1]))
    outp = lstmnet.forward_prop_once(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
        make_inner_matx(n_ex=5), make_inner_matx(n_ex=5))
    print(outp[1][-1])'''
