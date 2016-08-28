# LSTM-Generative-and-Discriminative

Using LSTM neural network model for accurate text generation.

In this project, I create 2 LSTM networks: A generative network and a discriminative one. The generative network simply creates text. The discriminative network is trained to tell the difference between the generated text and human-written text. The hope is that the generative network can be trained to maximize the probability that the discriminative network thinks its generated text is human.

This is similar to the generative adversarial network (GAN) developed by Goodfellow et al, 2014.

Code for creating and training LSTM's is in LSTM.py and LSTM_layer.py. Testing code is in testing.py. The generator and discriminator are located in generator.py and discriminator.py. The two are combined in gan.py.

I attempted to get the GAN to memorize (overfit) words from the animals.txt file.

Results: The discriminator was able to classify words with 100% accuracy almost every time it was trained. The generator would manage to reduce its accuracy, but when the discriminator was trained again, it went back to 100%. My hope was that the generator would learn to copy words from the dataset so that the discriminator would have no way to tell the difference. Although the generator sometimes displayed words that almost looked like words from the dataset, they never stayed for long. The generator's words changed too much and didn't seem to change in a consistent manner. For example, the generator almost spelled out the word "orangutan" (it spelled "oranguaa"), but the letters quickly descended into gibberish again after a few more epochs.

So far, this program only uses numpy, not theano or tensorflow.

If you want to run the GAN, run the file testing.py. Currently, it displays 3 things on the console each iteration: the accuracy of the discriminator before training the generator, the accuracy of the discriminator after the generator learns to trick it, and 3 example words the generator made.
