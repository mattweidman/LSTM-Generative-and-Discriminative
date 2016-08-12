import numpy as np

def random_vector(length):
    return np.random.randn(length)

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
        self.x0 = np.zeros((input_size)) # for the first layer, x is 0
                                         # this is ignored for later layers
        self.s0 = random_vector(output_size)
        self.h0 = random_vector(output_size)
        self.Wgx = random_matrix(output_size, input_size)
        self.Wix = random_matrix(output_size, input_size)
        self.Wfx = random_matrix(output_size, input_size)
        self.Wox = random_matrix(output_size, input_size)
        self.Wgh = random_matrix(output_size, output_size)
        self.Wih = random_matrix(output_size, output_size)
        self.Wfh = random_matrix(output_size, output_size)
        self.Woh = random_matrix(output_size, output_size)
        self.bg = random_vector(output_size)
        self.bi = random_vector(output_size)
        self.bf = random_vector(output_size)
        self.bo = random_vector(output_size)

    # calculate the state and hidden layer vectors for the next time step
    # x: input vector
    # s_prev: previous internal state
    # h_prev: previous output from this hidden layer
    # returns (internal state, hidden layer) tuple
    def forward_prop_once(self, x, s_prev, h_prev):
        g = phi(self.Wgx.dot(x) + self.Wgh.dot(h_prev) + self.bg)
        i = sigmoid(self.Wix.dot(x) + self.Wih.dot(h_prev) + self.bi)
        f = sigmoid(self.Wfx.dot(x) + self.Wfh.dot(h_prev) + self.bf)
        o = sigmoid(self.Wox.dot(x) + self.Woh.dot(h_prev) + self.bo)
        s = g*i + s_prev*f
        h = phi(s)*o
        return s, h

    # using the parameters, calculates a sequence of characters
    # of length sequence_length
    # returns a matrix of size output_size x sequence_length
    # assumes x is 0 every time step because you need other layers to
    # tell you what x is going to be
    def forward_prop_sequence(self, sequence_length):
        x = self.x0.copy()
        s = self.s0.copy()
        h = self.h0.copy()
        outp = np.zeros((self.output_size, 1))
        for j in range(sequence_length):
            s, h = self.forward_prop_once(x, s, h)
            outp = np.hstack((outp, h[:, np.newaxis]))
        return outp[:, 1:]

class LSTM:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    # forward propagate through this entire LSTM network
    # s_prev and h_prev are lists of numpy vectors, where the ith element
    # is input to the ith layer (x is input to only one layer)
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
    # the ith column of X is the input at the ith timestep
    # the output is a matrix Y where the ith column is the output
    # at the ith timestep
    # there is exactly one output for every input
    def forward_prop_sequence(self, X):
        s = [layer.s0 for layer in self.layers]
        h = [layer.h0 for layer in self.layers]
        outp = np.zeros((self.layers[-1].output_size, 0))
        for x in X.T:
            s, h = self.forward_prop_once(x, s, h)
            outp = np.hstack((outp, h[-1][:,np.newaxis]))
        return outp

list_of_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
num_chars = len(list_of_chars)

def vector_to_char(vector):
    index = np.argmin(vector)
    return list_of_chars[index]

def matrix_to_string(mat):
    output_str = ''
    for v in mat.T:
        c = vector_to_char(v)
        output_str += c
    return output_str

if __name__ == "__main__":

    # construct the LSTM
    hidden_size = 10
    network = LSTM()
    network.add_layer(LSTM_layer(num_chars, hidden_size))
    network.add_layer(LSTM_layer(hidden_size, num_chars))

    # construct the input
    seq_length = 150
    X = np.zeros((num_chars, seq_length))

    # use the LSTM
    sequence_matrix = network.forward_prop_sequence(X)
    outp = matrix_to_string(sequence_matrix)
    print(outp)
