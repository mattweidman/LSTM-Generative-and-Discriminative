import numpy as np

from discriminator import Discriminator
from generator import Generator

class GAN:

    # g_hidden_size: size of hidden layer in generator
    # d_hidden_size: size of hidden layer in discriminator
    # char_list: list of characters the generator can generate
    def __init__(self, g_hidden_size, d_hidden_size, char_list):
        self.char_list = char_list
        self.generator = Generator(g_hidden_size, char_list)
        self.discriminator = Discriminator(len(char_list), d_hidden_size)

    # X_actual: input data from dataset (not generated)
    # n_epochs: total epochs to train entire network
    # g_epochs: how long to train generator each epoch
    # d_epochs: how long to train disciminator each epoch
    # g_initial_lr, g_multiplier: generator RMSprop parameters
    # d_initial_lr, d_multiplier: discriminator RMSprop parameters
    # g_batch_size, d_batch_size: batch sizes for generator and discriminator
    def train(self, X_actual, seq_len, n_epochs, g_epochs, d_epochs,
            g_initial_lr, d_initial_lr, g_multiplier, d_multiplier,
            g_batch_size, d_batch_size, print_progress=False):

        num_examples = X_actual.shape[0]
        # TODO: make genr_input change every epoch
        genr_input = np.random.randn(num_examples, self.generator.input_size)
        for i in range(n_epochs):

            # generate text
            genr_output = self.generator.generate_tensor(seq_len, num_examples,
                genr_input)

            # train discriminator
            self.discriminator.train_RMS(X_actual, genr_output, d_epochs,
                d_initial_lr, d_multiplier, d_batch_size)

            # evaluate dicriminator
            if print_progress:
                genr_output = self.generator.generate_tensor(seq_len,
                    num_examples, genr_input)
                accuracy = self.discriminator.accuracy(X_actual, genr_output)
                print("accuracy before generator training: ", accuracy)

            # train generator
            self.generator.train_RMS(genr_input, seq_len, self.discriminator,
                g_epochs, 1, g_initial_lr, g_multiplier, g_batch_size)
            #print(sum(l.magnitude_theta() for l in self.generator.lstm.layers))

            # evaluate discriminator
            if print_progress:
                genr_output = self.generator.generate_tensor(seq_len,
                    num_examples, genr_input)
                accuracy = self.discriminator.accuracy(X_actual, genr_output)
                print("accuracy after generator training: ", accuracy)

            # display generator's output
            if print_progress:
                gen_text = self.generator.generate(seq_len, num_examples,
                    genr_input[:3,:])
                for line in gen_text:
                    print(line)
