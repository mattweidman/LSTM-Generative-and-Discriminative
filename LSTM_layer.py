import math
import numpy as np

def random_matrix(height, width):
    return np.random.randn(height, width) / math.sqrt(width)

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

    # magnitude of weights and biases
    def magnitude_theta(self):
        return sum([(w**2).sum() for w in self.theta])

    # calculate the state and hidden layer vectors for the next time step
    # x: input matrix, size (num_examples, input_size)
    # s_prev: previous internal state size (num_examples, output_size)
    # h_prev: previous output from this hidden layer, same size as s_prev
    # returns (internal state, hidden layer) tuple
    # if return_gates is true, returns a (g, i, f, o, s, h) tuple as well, each
    # size (output_size, num_examples)
    def forward_prop_once(self, x, s_prev, h_prev, return_gates=False):
        g = phi(self.Wgx.dot(x.T) + self.Wgh.dot(h_prev.T) + self.bg)
        i = sigmoid(self.Wix.dot(x.T) + self.Wih.dot(h_prev.T) + self.bi)
        f = sigmoid(self.Wfx.dot(x.T) + self.Wfh.dot(h_prev.T) + self.bf)
        o = sigmoid(self.Wox.dot(x.T) + self.Woh.dot(h_prev.T) + self.bo)
        s = g*i + s_prev.T*f
        h = phi(s)*o
        if return_gates:
            return s.T, h.T, (g, i, f, o, s, h)
        else:
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
    # gate_values is a (g, i, f, o, s, h) tuple, each representing a gate, size
    # (output_size, num_examples)
    def backprop(self, x, dloss, s_prev, h_prev, s_next_grad=None,
            h_next_grad=None, gate_values=None):

        # default values for s_next_grad and h_next_grad
        if s_next_grad is None:
            s_next_grad = np.zeros(s_prev.shape)
        if h_next_grad is None:
            h_next_grad = np.zeros(h_prev.shape)

        # propagate forward
        if gate_values is None:
            g = phi(self.Wgx.dot(x.T) + self.Wgh.dot(h_prev.T) + self.bg)
            i = sigmoid(self.Wix.dot(x.T) + self.Wih.dot(h_prev.T) + self.bi)
            f = sigmoid(self.Wfx.dot(x.T) + self.Wfh.dot(h_prev.T) + self.bf)
            o = sigmoid(self.Wox.dot(x.T) + self.Woh.dot(h_prev.T) + self.bo)
            s = g*i + s_prev.T*f
            h = phi(s)*o
        else:
            g, i, f, o, s, h = gate_values

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

        return LSTM_layer_gradient(dLdtheta, dLdx.T, dLds_prev.T, dLdh_prev.T)

    # update the parameters of this LSTM layer by SGD
    # parameters <- parameters - gradient * learning rate
    # gradient: LSTM_layer_gradient object
    def update_theta(self, gradient, learning_rate):
        for param, grad in zip(self.theta, gradient.dLdtheta):
            param -= learning_rate * grad

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

    def add(self, other):
        dsumtheta = [sth+oth for sth, oth in zip(self.dLdtheta, other.dLdtheta)]
        return LSTM_layer_gradient(dsumtheta, self.dLdx+other.dLdx,
            self.dLds_prev+other.dLds_prev, self.dLdh_prev+other.dLdh_prev)

    def multiply(self, scalar):
        dprodtheta = [sth*scalar for sth in self.dLdtheta]
        return LSTM_layer_gradient(dprodtheta, self.dLdx*scalar,
            self.dLds_prev*scalar, self.dLdh_prev*scalar)

    def magnitude_theta(self):
        return sum([np.sum(p**2) for p in self.dLdtheta])
