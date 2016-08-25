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

    # X1, X2: first and second data sets
    # n_epochs: total epochs to train entire network
    # g_epochs: how long to train generator each epoch
    # d_epochs: how long to train disciminator each epoch
    # g_initial_lr, g_multiplier: generator RMSprop parameters
    # d_initial_lr, d_multiplier: discriminator RMSprop parameters
    # g_batch_size, d_batch_size: batch sizes for generator and discriminator
    def train(self, X1, X2, n_epochs, g_epochs, d_epochs, g_initial_lr,
            g_multiplier, g_batch_size, d_initial_lr, d_multiplier,
            d_batch_size, print_progress=False):
        pass
