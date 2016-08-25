import numpy as np

from LSTM import LSTM
from LSTM_layer import LSTM_layer

# h: sqeuence output from LSTM, size (num_examples, seq_length, 2)
# y: expected output from LSTM, same size as h
def loss(h, y):
    n = h.shape[0]
    return 1/(2*n) * ((h[:,-1,:]-y[:,-1,:])**2).sum()

# h: element output from LSTM, size (num_examples, 2)
# y: expected output from LSTM, same size as h
def dloss(h, y):
    if y.sum() == 0:
        return np.zeros(h.shape)
    n = h.shape[0]
    return 1/n * (h-y)

class Discriminator:

    # input_size: size of each input element
    # hidden_size: size of hidden layer
    # maxlen: length of longest sequence
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTM()
        self.lstm.add_layer(LSTM_layer(input_size, hidden_size))
        self.lstm.add_layer(LSTM_layer(hidden_size, 2))

    # learns to discriminate between datasets X1 and X2
    # X1 and X2 are size (num_examples, seq_length, input_size)
    # initial_lr and grad_multiplier are RMSprop parameters
    def train_RMS(self, X1, X2, num_epochs, initial_lr, grad_multiplier,
            batch_size, print_progress=False):
        # wrangle input and output data
        X = np.concatenate((X1, X2), axis=0)
        Y = np.zeros((X.shape[0], X.shape[1], 2))
        Y[:X1.shape[0], -1, 0] = 1
        Y[X1.shape[0]:, -1, 1] = 1
        # train
        self.lstm.RMSprop(X, Y, loss, dloss, num_epochs, initial_lr,
            grad_multiplier, batch_size, print_progress=print_progress)

    # X: input size (num_examples, seq_length, input_size)
    # returns a vector Y of length num_examples, where Y[i] is 0 if the LSTM
    # thinks X[i] is part of the first dataset, else Y[i] is 1
    def discriminate(self, X):
        Y_tensor = self.lstm.forward_prop(X)
        Y = np.zeros((Y_tensor.shape[0]))
        Y[Y_tensor[:,-1,0] < Y_tensor[:,-1,1]] = 1
        return Y

    # finds the accuracy of this discriminator
    # X1: data only from the first dataset
    # X2: data only from the second dataset
    # returns the proportion of correctly classified elements in both sets
    def accuracy(self, X1, X2):
        Y1 = self.discriminate(X1)
        Y2 = self.discriminate(X2)
        num_correct = (1-Y1).sum() + Y2.sum()
        total_examples = Y1.shape[0] + Y2.shape[0]
        return num_correct / total_examples
