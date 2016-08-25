import numpy as np
import random

from generator import Generator
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

def test_SGD():
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

def test_abc():

    abc_list = ['a', 'b', 'c']
    abc_to_ind = dict((c,i) for i,c in enumerate(abc_list))

    def abc_to_vec(c):
        ind = abc_to_ind[c]
        ans = np.zeros(len(abc_list))
        ans[ind] = 1
        return ans

    def vec_to_abc(vec):
        return abc_list[np.argmax(vec)]

    # matrix size (sequence_length, len(abc_list))
    def matrix_to_string(matx):
        ans = ""
        for row in matx:
            ans += vec_to_abc(row)
        return ans

    def string_to_matrix(s):
        return [abc_to_vec(c) for c in s]

    def normalize(v):
        last_axis = len(v.shape)-1
        vmean = np.expand_dims(v.mean(axis=last_axis), axis=last_axis)
        vstd = np.expand_dims(v.std(axis=last_axis), axis=last_axis)
        return (v-vmean)/vstd

    X = normalize(np.array([abc_to_vec('a'), abc_to_vec('b'), abc_to_vec('c')]))
    Y = np.array([string_to_matrix('baabc'), string_to_matrix('abbbc'),
        string_to_matrix('cabac')])

    def get_n(tensor):
        n = 1
        for dim in tensor.shape[:-1]:
            n *= dim
        return n

    def loss(y_out, y_exp):
        return 1/(2*get_n(y_out)) * np.sum((y_out-y_exp)**2)

    def dloss(y_out, y_exp):
        return 1/get_n(y_out) * (y_out-y_exp)

    # construct the LSTM
    input_size = len(abc_list)
    hidden_size = 20
    output_size = len(abc_list)
    lstm = LSTM()
    lstm.add_layer(LSTM_layer(input_size, hidden_size))
    lstm.add_layer(LSTM_layer(hidden_size, output_size))

    # train the LSTM
    lstm.RMSprop(X, Y, loss, dloss, 500, 1, 0.1,
        seq_length=Y.shape[1], print_progress=True)

    # print the output of the LSTM
    outp = lstm.forward_prop(X, seq_length=Y.shape[1])
    for out_i, y_i in zip(outp, Y):
        print("%s\t%s" % (matrix_to_string(y_i), matrix_to_string(out_i)))

def test_generator():
    g = Generator(10, list_of_chars)
    seq_len = 150
    num_examples = 10
    chr_seqs = g.generate(seq_len, num_examples)
    for seq in chr_seqs:
        print(seq)

if __name__ == "__main__":
    test_generator()
