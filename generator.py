import math
import numpy as np

from discriminator import dloss as discr_dloss
from discriminator import loss as discr_loss
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

    # trains this generator to maximize the probability the discriminator
    # returns 1 when its input is used
    # X: input size (num_examples, input_size)
    # seq_len: length of output sequences
    # discr: discriminator object
    # num_epochs: number of epochs to run training
    # data_index: 0 if discr treats the data from this generator as the first
    # dataset, 1 if discr treats it as the second dataset
    # initial_lr, grad_multiplier: RMSprop parameters
    def train_RMS(self, X, seq_len, discr, num_epochs, data_index, initial_lr,
            grad_multiplier, batch_size, print_progress=False):

        num_examples = X.shape[0]
        ms=0
        for i in range(num_epochs):

            # generate input and expected output to discriminator
            gen_output = self.generate_tensor(seq_len, num_examples, X)
            Y_exp = np.zeros((num_examples, seq_len, 2))
            Y_exp[:,-1,1-data_index] = 1

            # get gradient from discriminator
            discr_grads = discr.lstm.BPTT(gen_output, Y_exp, discr_dloss,
                return_list=True)
            discr_grads_x = np.array([grad[0].dLdx for grad in discr_grads])
            discr_grads_x = discr_grads_x.swapaxes(0,1)
            genr_dloss = lambda h, y: y

            # compute gradient for entire input
            if batch_size is None:
                grad = self.lstm.BPTT(X, discr_grads_x, genr_dloss, seq_len)

            # compute gradient for one batch
            else:
                batch_indices = np.random.choice(np.arange(0,num_examples),
                    batch_size)
                inpt = X[batch_indices]
                exp_outp = discr_grads_x[batch_indices]
                grad = self.lstm.BPTT(inpt, exp_outp, genr_dloss, seq_len)

            # choose new learning rate
            magnitude = sum([gl.magnitude_theta() for gl in grad])
            ms = (1-grad_multiplier) * ms + grad_multiplier * magnitude
            lr = initial_lr / math.sqrt(ms)

            # update parameters
            self.lstm.update_theta(grad, lr)

            # forward propagate and print the cost
            if print_progress:
                outp = self.lstm.forward_prop(X, seq_len)
                total_loss = discr_loss(outp, discr_grads_x)
                magnitude = sum(gl.magnitude_theta() for gl in grad)
                print("cost:%f\tgradient:%f" % (total_loss, magnitude))

        if print_progress:
            print("Training complete")
