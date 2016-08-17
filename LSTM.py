import numpy as np

def random_matrix(height, width):
    return np.random.randn(height, width)

# note to self: if I change this, I have to change backprop() as well
def phi(x):
    return np.tanh(x)

# note to self: if I change this, I have to change backprop() as well
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
        self.theta = [self.Wgx, self.Wix, self.Wfx, self.Wox,
            self.Wgh, self.Wih, self.Wfh, self.Woh,
            self.bg, self.bi, self.bf, self.bo]

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

    # finds the gradient of this LSTM layer by propagating forward and back
    # x, s_prev, and h_prev are as described in forward_prop_once
    # dloss is a function to compute the derivative of the loss with respect
    # to the output vector h. It should only be a function of h.
    # s_next_grad and h_next_grad are the gradients of s(t+1) and h(t+1)
    # returns tuple containing gradients of parameters theta, x, s_prev, and
    # h_prev, in that order
    # note that for all matrix arguments to this function, num_examples is
    # the size of the first dimension
    def backprop(self, x, s_prev, h_prev, dloss, s_next_grad, h_next_grad):
        # propagate forward
        g = phi(self.Wgx.dot(x.T) + self.Wgh.dot(h_prev.T) + self.bg)
        i = sigmoid(self.Wix.dot(x.T) + self.Wih.dot(h_prev.T) + self.bi)
        f = sigmoid(self.Wfx.dot(x.T) + self.Wfh.dot(h_prev.T) + self.bf)
        o = sigmoid(self.Wox.dot(x.T) + self.Woh.dot(h_prev.T) + self.bo)
        s = g*i + s_prev.T*f
        h = phi(s)*o

        # backprop to each gate
        dLdh = dloss(h.T).T + h_next_grad.T
        dLdo = dLdh * phi(s)
        dLds = dLdh * o * (1-phi(s)**2) + s_next_grad.T
        dLdg = dLds * i
        dLdi = dLds * g
        dLdf = dLds * s_prev.T
        dLds_prev = dLds * f

        # finds dL/dW?x, dL/dW?h, dL/db?, dL/dx_?, and dL/dh_?
        # grad_in is equal to dL/d? * sigmoid_prime or dL/d? * phi_prime
        # ? indicates the name of a gate - g, i, f, or o
        def backprop_gate(grad_in, W_x, W_h):
            dLdW_x = grad_in.dot(x)
            dLdW_h = grad_in.dot(h_prev)
            dLdb_ = grad_in.sum(axis=1)[:,np.newaxis]
            dLdx_ = W_x.T.dot(grad_in)
            dLdh_ = W_h.T.dot(grad_in)
            return dLdW_x, dLdW_h, dLdb_, dLdx_, dLdh_

        # backprop within each gate
        g_prime = dLdg * (1-g**2)
        dLdWgx, dLdWgh, dLdbg, dLdxg, dLdhg = backprop_gate(g_prime, self.Wgx,
            self.Wgh)
        i_prime = dLdi * i*(1-i)
        dLdWix, dLdWih, dLdbi, dLdxi, dLdhi = backprop_gate(i_prime, self.Wix,
            self.Wih)
        f_prime = dLdf * f*(1-f)
        dLdWfx, dLdWfh, dLdbf, dLdxf, dLdhf = backprop_gate(f_prime, self.Wfx,
            self.Wfh)
        o_prime = dLdo * o*(1-o)
        dLdWox, dLdWoh, dLdbo, dLdxo, dLdho = backprop_gate(o_prime, self.Wox,
            self.Woh)

        # combine everything into one data structure
        dLdtheta = [dLdWgx, dLdWix, dLdWfx, dLdWox, dLdWgh, dLdWih, dLdWfh,
            dLdWoh, dLdbg, dLdbi, dLdbf, dLdbo]
        dLdx = dLdxg + dLdxi + dLdxf + dLdxo
        dLdh_prev = dLdhg + dLdhi + dLdhf + dLdho
        return dLdtheta, dLdx.T, dLds_prev.T, dLdh_prev.T

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
