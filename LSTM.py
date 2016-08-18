import numpy as np

def random_matrix(height, width):
    return np.random.randn(height, width)

# note to self: if I change this, I have to change backprop() as well
def phi(x):
    return np.tanh(x)

# note to self: if I change this, I have to change backprop() as well
def sigmoid(x):
    return 1/(1+np.exp(-x))

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
    # if s_prev or h_prev is None, this function will use self.s0 and self.h0
    # returns (internal state, hidden layer) tuple
    def forward_prop_once(self, x, s_prev=None, h_prev=None):
        if s_prev is None:
            s_prev = self.s0
        if h_prev is None:
            h_prev = self.h0
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
    # default values for next_grad vectors are zero-vectors
    # returns an LSTM_layer_gradient object
    # note that for all matrix arguments to this function, num_examples is
    # the size of the first dimension
    def backprop(self, x, dloss, s_prev=None, h_prev=None,
            s_next_grad=None, h_next_grad=None):

        # default values for s_prev and h_prev
        if s_prev is None:
            s_prev = self.s0
        if h_prev is None:
            h_prev = self.h0

        # if s_prev or h_prev only has one row, repeat that row so that
        # there is one row for each example in x
        s_prev_first_dim = s_prev.shape[0]
        if s_prev_first_dim == 1:
            s_prev = s_prev.repeat(x.shape[0], axis=0)
        h_prev_first_dim = h_prev.shape[0]
        if h_prev_first_dim == 1:
            h_prev = h_prev.repeat(x.shape[0], axis=0)

        # default values for s_next_grad and h_next_grad
        if s_next_grad is None:
            s_next_grad = np.zeros(s_prev.shape)
        if h_next_grad is None:
            h_next_grad = np.zeros(h_prev.shape)

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

        # make sure the number of dimensions of dLds_prev and dLdh_prev are
        # the same as s_prev and h_prev
        if s_prev_first_dim == 1:
            dLds_prev = dLds_prev.sum(axis=1)[:,np.newaxis]
        if h_prev_first_dim == 1:
            dLdh_prev = dLdh_prev.sum(axis=1)[:,np.newaxis]

        return LSTM_layer_gradient(dLdtheta, dLdx.T, dLds_prev.T, dLdh_prev.T)

    # update the parameters of this LSTM layer by SGD
    # parameters <- parameters - gradient * learning rate
    # gradient: LSTM_layer_gradient object
    def update_theta(self, gradient, learning_rate):
        for param, grad in zip(self.theta, gradient.dLdtheta):
            param -= learning_rate * grad

    # update the parameters of this LSTM layer by SGD, including initial
    # s and h vectors
    # parameters <- parameters - gradient * learning rate
    # gradient: LSTM_layer_gradient object
    def update_theta_s0_h0(self, gradient, learning_rate):
        self.update_theta(gradient, learning_rate)
        self.s0 -= learning_rate * gradient.dLds_prev
        self.h0 -= learning_rate * gradient.dLdh_prev

class LSTM_layer_gradient():
    # dLdtheta: list of gradients with respect to LSTM_layer parameters
    # dLdx: gradient with prespect to x(t)
    # dLds_prev: gradient with respect to s(t-1)
    # dLdh_prev: gradient with respect to h(t-1)
    def __init__(self, dLdtheta, dLdx, dLds_prev, dLdh_prev):
        self.dLdtheta = dLdtheta
        self.dLdx = dLdx
        self.dLds_prev = dLds_prev
        self.dLdh_prev = dLdh_prev

    def to_tuple(self):
        return self.dLdtheta, self.dLdx, self.dLds_prev, self.dLdh_prev

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

    # perform backpropagation on one element in the sequence
    # x is the input, size (num_examples, input_size)
    # dloss is a function that computes the gradient of the loss function,
    # given the output
    # s_prev is a list of s(t-1) matrices for each layer
    # h_prev is a list of h(t-1) matrices for each layer
    # s_next_grad is a list of s(t+1) gradients for each layer
    # h_next_grad is a list of h(t+1) gradients for each layer
    # returns a list of LSTM_layer_gradient objects
    def backprop_once(self, x, dloss, s_prev=None, h_prev=None,
            s_next_grad=None, h_next_grad=None):
        # default values for s_prev and h_prev
        num_examples = x.shape[0]
        s_prev_is_none = False
        if s_prev is None:
            s_prev = [layer.s0.repeat(num_examples, axis=0) for layer in
                self.layers]
            s_prev_is_none = True
        h_prev_is_none = False
        if h_prev is None:
            h_prev = [layer.h0.repeat(num_examples, axis=0) for layer in
                self.layers]
            h_prev_is_none = True

        # default values for s_next_grad and h_next_grad
        s_next_is_none = False
        if s_next_grad is None:
            s_next_grad = [np.zeros((num_examples, layer.output_size))
                for layer in self.layers]
            s_next_is_none = True
        h_next_is_none = False
        if h_next_grad is None:
            h_next_grad = [np.zeros((num_examples, layer.output_size))
                for layer in self.layers]
            h_next_is_none = True

        # forward propagate
        s, h = self.forward_prop_once(x, s_prev, h_prev)

        # change s_prev, h_prev, s_next_grad, and h_next_grad so that backprop
        # uses default values
        if s_prev_is_none:
            s_prev = [None] * len(self.layers)
        if h_prev_is_none:
            h_prev = [None] * len(self.layers)
        if s_next_is_none:
            s_next_grad = [None] * len(self.layers)
        if h_next_is_none:
            h_next_grad = [None] * len(self.layers)

        # backprop each layer
        gradient = []
        for i in range(len(self.layers)-1, -1, -1):
            inp = x if i==0 else h[i-1]
            grad_i = self.layers[i].backprop(inp, dloss, s_prev[i], h_prev[i],
                s_next_grad[i], h_next_grad[i])
            dloss = lambda h_: grad_i.dLdx
            gradient.append(grad_i)

        return gradient[::-1]
