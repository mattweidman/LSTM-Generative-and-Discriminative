# LSTM-Generative-and-Discriminative

Using LSTM neural network model for accurate text generation.

In this project, I create 2 LSTM networks: A generative network and a discriminative one. The generative network simply creates text. The discriminative network is trained to tell the difference between the generated text and human-written text. The hope is that the generative network can be trained to maximize the probability that the discriminative network thinks its generated text is human.

This is similar to the generative adversarial network (GAN) developed by Goodfellow et al, 2014.

Code for creating and training LSTM's is in LSTM.py and LSTM_layer.py. Testing code is in testing.py. Once the LSTM code is finalized, generator.py and discriminator.py will be written.

So far, this program only uses numpy, not theano or tensorflow.
