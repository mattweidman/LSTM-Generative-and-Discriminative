import numpy as np

def random_matrix(height, width):
    return np.random.randn(height, width)

def phi(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(x))

class LSTM_layer:
    def __init__(self, input_size, output_size):
        # initialize all parameters
        self.input_size = input_size
        self.output_size = output_size
        self.s0 = random_matrix(1, output_size)
        self.h0 = random_matrix(1, output_size)
        self.Wgx = random_matrix(output_size, input_size)
        self.Wix = random_matrix(output_size, input_size)
        self.Wfx = random_matrix(output_size, input_size)
        self.Wox = random_matrix(output_size, input_size)
        self.Wgh = random_matrix(output_size, output_size)
        self.Wih = random_matrix(output_size, output_size)
        self.Wfh = random_matrix(output_size, output_size)
        self.Woh = random_matrix(output_size, output_size)
        self.bg = random_matrix(output_size, 1)
        self.bi = random_matrix(output_size, 1)
        self.bf = random_matrix(output_size, 1)
        self.bo = random_matrix(output_size, 1)

    # calculate the state and hidden layer vectors for the next time step
    # x: input matrix, size (num_examples, input_size)
    # s_prev: previous internal state size (num_examples, output_size)
    # h_prev: previous output from this hidden layer, same size as s_prev
    # returns (internal state, hidden layer) tuple
    def forward_prop_once(self, x, s_prev, h_prev):
        g = phi(self.Wgx.dot(x.T) + self.Wgh.dot(h_prev.T) + self.bg)
        i = sigmoid(self.Wix.dot(x.T) + self.Wih.dot(h_prev.T) + self.bi)
        f = sigmoid(self.Wfx.dot(x.T) + self.Wfh.dot(h_prev.T) + self.bf)
        o = sigmoid(self.Wox.dot(x.T) + self.Woh.dot(h_prev.T) + self.bo)
        s = g*i + s_prev.T*f
        h = phi(s)*o
        return s.T, h.T

class LSTM:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    # forward propagate through this entire LSTM network
    # s_prev and h_prev are lists of numpy matrices, where the ith element
    # of is input to the ith layer (x is input to only one layer)
    # elements of s_prev and h_prev are size (num_examples, layer_output_size)
    # returns (internal state, hidden layer) tuple (which are same
    # dimensions as s_prev and h_prev)
    def forward_prop_once(self, x, s_prev, h_prev):
        s = []
        h = []
        for i in range(len(self.layers)):
            si, hi = self.layers[i].forward_prop_once(x, s_prev[i], h_prev[i])
            s.append(si)
            h.append(hi)
            x = hi.copy()
        return s, h

    # using a sequence of inputs, creates a sequence of outputs
    # X is a tensor of size (num_examples, sequence_length, input_size)
    # the output is a tensor Y which is size (num_examples, sequence_length,
    # output_size)
    # there is exactly one output for every input
    def forward_prop_one2one(self, X):
        num_examples = X.shape[0]
        s = [layer.s0.repeat(num_examples, axis=0) for layer in self.layers]
        h = [layer.h0.repeat(num_examples, axis=0) for layer in self.layers]
        outp = np.zeros((num_examples, 0, self.layers[-1].output_size))
        for x in X.swapaxes(0,1):
            s, h = self.forward_prop_once(x, s, h)
            outp = np.concatenate((outp, h[-1][:,np.newaxis,:]), axis=1)
        return outp

    # using a single input, generate a sequence of outputs
    # x is a matrix of size (num_examples, input_size)
    # the output Y is size (num_examples, sequence_length, output_size)
    # the output at each timestep is calculated by using the previous output
    # as input
    # input_size and output_size must therefore be the same
    def forward_prop_feedback(self, x, sequence_length):
        num_examples = x.shape[0]
        s = [layer.s0.repeat(num_examples, axis=0) for layer in self.layers]
        h = [layer.h0.repeat(num_examples, axis=0) for layer in self.layers]
        outp = np.zeros((num_examples, 0, self.layers[-1].output_size))
        for i in range(sequence_length):
            s, h = self.forward_prop_once(x, s, h)
            outp = np.concatenate((outp, h[-1][:,np.newaxis,:]), axis=1)
            x = h[-1]
        return outp
