import numpy as np

class LSTM:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    # quick way of choosing default values for None inputs
    def empty_or_same(self, num_examples, v):
        return [np.zeros((num_examples, l.output_size))
            for l in self.layers] if v is None else v

    # forward propagate through this entire LSTM network
    # s_prev and h_prev are lists of numpy matrices, where the ith element
    # of is input to the ith layer (x is input to only one layer)
    # elements of s_prev and h_prev are size (num_examples, layer_output_size)
    # returns (internal state, hidden layer) tuple (which are same
    # dimensions as s_prev and h_prev)
    # if return_gates is true, also returns list of (g,i,f,o,s,h) tuples
    def forward_prop_once(self, x, s_prev, h_prev, return_gates=False):
        s = []
        h = []
        gates = []
        for i in range(len(self.layers)):
            if return_gates:
                si, hi, gi = self.layers[i].forward_prop_once(x, s_prev[i],
                    h_prev[i], return_gates)
                gates.append(gi)
            else:
                si, hi = self.layers[i].forward_prop_once(x, s_prev[i],
                    h_prev[i])
            s.append(si)
            h.append(hi)
            x = hi.copy()
        if return_gates:
            return s, h, gates
        else:
            return s, h

    # forward propagate input through this LSTM
    # X is a matrix of size (num_examples, input_size) if feedback
    # X is size (num_examples, seq_length, input_size) if one2one
    # for feedback, the output at each timestep is calculated by using the
    # previous output as input, and seq_length MUST be provided
    # if seq_length is not provided, assume one2one
    # input_size and output_size must therefore be the same if feedback
    # returns (X, slist, hlist) tuple, where X is a
    # (num_examples, seq_length, input_size) tensor and slist and hlist are
    # lists of s and h values for each sequence element
    def forward_prop_lists(self, X, seq_length=None, s0=None, h0=None):

        # record whether this is feedback, sanity checks
        is_feedback = seq_length is not None
        if seq_length is None:
            assert len(X.shape) == 3
            seq_length = X.shape[1]
        else:
            assert len(X.shape) == 2

        # default values
        num_examples = X.shape[0]
        s0 = self.empty_or_same(num_examples, s0)
        h0 = self.empty_or_same(num_examples, h0)

        # forward prop through sequence
        s, h = s0, h0
        slist, hlist = [], []
        if is_feedback:
            x = X
            X = np.zeros((x.shape[0], 0, x.shape[1]))
        for i in range(seq_length):
            if is_feedback: X = np.concatenate((X, x[:,np.newaxis,:]), axis=1)
            else: x = X[:,i,:]
            s, h = self.forward_prop_once(x, s, h)
            slist.append(s)
            hlist.append(h)
            if is_feedback: x = h[-1]

        return X, slist, hlist

    # forward propagate input through this LSTM
    # X is a matrix of size (num_examples, input_size) if feedback
    # X is size (num_examples, seq_length, input_size) if one2one
    # for feedback, the output at each timestep is calculated by using the
    # previous output as input, and seq_length MUST be provided
    # if seq_length is not provided, assume one2one
    # input_size and output_size must therefore be the same if feedback
    # returns an output Y of size (num_examples, seq_length, output_size)
    def forward_prop(self, X, seq_length=None, s0=None, h0=None):
        X, slist, hlist = self.forward_prop_lists(X, seq_length, s0, h0)
        num_examples = X.shape[0]
        outp = np.zeros((num_examples, 0, self.layers[-1].output_size))
        for h in hlist:
            outp = np.concatenate((outp, h[-1][:,np.newaxis,:]), axis=1)
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
    def backprop_once(self, x, dloss, s_prev, h_prev, s_next_grad=None,
            h_next_grad=None):

        # default values for s_prev, h_prev, s_next_grad, and h_next_grad
        num_examples = x.shape[0]
        nonelist = lambda: [None] * len(self.layers)
        s_prev = self.empty_or_same(num_examples, s_prev)
        h_prev = self.empty_or_same(num_examples, h_prev)
        if s_next_grad is None: s_next_grad = nonelist()
        if h_next_grad is None: h_next_grad = nonelist()

        # forward propagate
        s, h, gates = self.forward_prop_once(x, s_prev, h_prev,
            return_gates=True)

        # backprop each layer
        gradient = []
        for i in range(len(self.layers)-1, -1, -1):
            inp = x if i==0 else h[i-1]
            grad_i = self.layers[i].backprop(inp, dloss, s_prev[i], h_prev[i],
                s_next_grad[i], h_next_grad[i], gates[i])
            dloss = lambda h_: grad_i.dLdx
            gradient.append(grad_i)

        return gradient[::-1]

    # perform backpropagation through time on input X
    # X is input size (num_examples, sequence_length, input_size) for one2one
    # X is size (num_examples, input_size) for feedback
    # seq_length MUST be provided for feedback BPTT, which is the length of
    # the sequence to be output
    # if seq_length is None, one2one will be assumed; else, feedback is assumed
    # s0 and h0 are initial internal state and hidden layer lists
    # sn_grad and hn_grad are gradients you can inject into the last element
    # of the sequence during backprop
    # dloss is the gradient of the loss function; function of h and i, where i
    # is the index of the current sequence element
    def BPTT(self, X, dloss, seq_length=None, s0=None, h0=None, sn_grad=None,
            hn_grad=None):

        num_examples = X.shape[0]

        # forward prop through sequence
        s, h = s0, h0
        X, slist, hlist = self.forward_prop_lists(X, seq_length=seq_length,
            s0=s0, h0=h0)

        is_feedback = seq_length is not None
        if seq_length is None: seq_length = X.shape[1]

        # backprop for every element in the sequence
        s_next_grad = sn_grad
        h_next_grad = hn_grad
        x_next_grad = np.zeros((num_examples, self.layers[0].input_size))
        gradients = []
        for i in range(seq_length-1, -1, -1):
            s_prev = s0 if i == 0 else slist[i-1]
            h_prev = h0 if i == 0 else hlist[i-1]
            grad = self.backprop_once(X[:,i,:],
                lambda h_: dloss(h_, i) + x_next_grad,
                s_prev, h_prev, s_next_grad, h_next_grad)
            s_next_grad = [gl.dLds_prev for gl in grad]
            h_next_grad = [gl.dLdh_prev for gl in grad]
            if is_feedback:
                x_next_grad = grad[0].dLdx
            gradients.append(grad)

        # sum the gradients
        gradsum = gradients[0]
        for i in range(1, len(gradients)):
            for sum_layer, grad_layer in zip(gradsum, gradients[i]):
                sum_layer = sum_layer.add(grad_layer)
        return gradsum

    # use the gradient to update parameters in theta
    def update_theta(self, gradient, learning_rate):
        for l, g in zip(self.layers, gradient):
            l.update_theta(g, learning_rate)
