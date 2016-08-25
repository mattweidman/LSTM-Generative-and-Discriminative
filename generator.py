import numpy as np

from LSTM_layer import LSTM_layer
from LSTM import LSTM

class Generator:

    # input_size: size of the input layer
    # hidden_size: size of hidden layer
    # char_list: list of characters that can be output
    def __init__(self, hidden_size, char_list):
        self.input_size = len(char_list)
        self.hidden_size = hidden_size
        self.lstm = LSTM()
        self.lstm.add_layer(LSTM_layer(self.input_size, hidden_size))
        self.lstm.add_layer(LSTM_layer(hidden_size, len(char_list)))
        self.char_list = char_list
        self.char_to_ind = dict((c,i) for i,c in enumerate(char_list))

    # generates an output tensor Y of size (num_examples, sequence_length,
    # output_size)
    # X is the input, size (num_examples, input_size)
    # if X is None, a random value is chosen
    # default value for num_examples is 1, but it will be overriden if X is
    # not None
    def generate_tensor(self, sequence_length, num_examples=1, X=None):
        if X is None:
            X = np.random.randn(num_examples, self.input_size)
        return self.lstm.forward_prop(X, sequence_length)

    # converts a character embedding to a string
    # matrix Y is size (sequence_length, num_chars)
    def matrix_to_string(self, Y):
        ans = ""
        for c in Y:
            ind = np.argmax(c)
            ans = ans + self.char_list[ind]
        return ans

    # generates a list of sequences of characters
    # sequence_length: length of each character sequence
    # num_examples: number of sequences generated in the list
    # X is the input, size (num_examples, input_size)
    # if X is None, a random value is chosen
    # default value for num_examples is 1, but it will be overriden if X is
    # not None
    def generate(self, sequence_length, num_examples=1, X=None):
        tensor = self.generate_tensor(sequence_length, num_examples, X)
        gens = [self.matrix_to_string(matrx) for matrx in tensor]
        return gens
