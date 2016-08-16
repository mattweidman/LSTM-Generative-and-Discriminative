import numpy as np

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

if __name__ == "__main__":
    # construct the LSTM
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
        print(outp)
