import numpy as np
import random

from LSTM_layer import LSTM_layer
from LSTM import LSTM

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
        X_sequence = np.zeros((num_examples, seq_length, input_size))
        for i in range(num_examples):
            randind = random.randint(0, num_chars-1)
            for j in range(seq_length):
                X_sequence[i,j,randind] = 1
        return X_sequence
    def gen_x_once():
        X = np.zeros((num_examples, input_size))
        for i in range(num_examples):
            randind = random.randint(0, num_chars-1)
            X[i,randind] = 1
        return X
    def normalize(v):
        return (v-v.mean())/v.std()

    # loss function and its gradient
    def get_n(h_):
        n = 1
        for dim in h_.shape[:-1]:
            n *= dim
        return n
    def loss(h_, y_):
        return 1/(2*get_n(h_)) * np.sum((h_-y_)**2)
    def dloss(h_, y_):
        return 1/get_n(h_) * (h_-y_)

    # train feedback
    X = gen_x_sequence()
    network.SGD(normalize(X[:,0,:]), X, loss, dloss, 1000, 10, # momentum=0,
        batch_size=200, seq_length=seq_length, print_progress=True)

    # use the LSTM, feedback
    def char_to_matx(c, length=seq_length):
        return [char_to_vec(c)] * length
    inp = np.array([char_to_vec(chr(c)) for c in range(97, 123)])
    print(inp.shape)
    outp = network.forward_prop(normalize(inp), seq_length=seq_length)
    for i in range(inp.shape[0]):
        print("%s\t%s" % (chr(i+97), matrix_to_string(outp[i])))

    '''# train one2one
    X = gen_x_sequence()
    network.SGD(normalize(X), X, loss, dloss, 1000, 10, momentum=0.9,
        batch_size=200, print_progress=True)

    # use the LSTM, one2one
    def char_to_matx(c, length=seq_length):
        return [char_to_vec(c)] * length
    inp = np.array([char_to_matx(chr(c)) for c in range(97, 123)])
    print(inp.shape)
    outp = network.forward_prop(normalize(inp))
    for i in range(inp.shape[0]):
        print("%s\t%s" % (chr(i+97), matrix_to_string(outp[i])))'''
