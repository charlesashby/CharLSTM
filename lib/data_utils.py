# Dataset needs to be shuffled before calling TextReader() instance;
# Use shuffle_dataset() to make a new CSV file, default is /datasets/train_set.csv

import random, csv
import numpy as np

PATH = '/home/ashbylepoc/PycharmProjects/tensorflow/'

# TODO: Add non-Ascii characters
emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '

DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}
ALPHABET_SIZE = len(emb_alphabet)

def shuffle_dataset(dataset, out_file='datasets/train_set.csv'):
    # Create a Shuffled Dataset

    with open(dataset, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)

    data = []
    for l in lines:
        split = l.split('","')
        data.append((split[0][1:], split[-1][:-2]))

    with open(out_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)

class TextReader(object):
    """ Util for Reading the Stanford Files """
    # TODO: Add support for larger files and Queues

    def __init__(self, file):
        # TextReader() takes a CSV file as input that it will read
        # through a buffer
        self.file = file

    def encode_one_hot(self, sentence):
        # Convert Sentence to np array of Shape (sent_length, DICT_SIZE)

        sent = []
        SENT_LENGTH = 0
        for char in sentence:
            try:
                encoding = DICT[char]
                one_hot = np.zeros(ALPHABET_SIZE)
                one_hot[encoding] = 1
                sent.append(one_hot)
                SENT_LENGTH += 1

            except Exception as e:
                pass

        return np.array(sent), SENT_LENGTH

    def make_minibatch(self, sentences):
        # Create a minibatch of sentences and convert sentiment
        # to a one-hot vector, also takes care of padding

        minibatch_x = []
        minibatch_y = []
        max_length = 0

        for sentence in sentences:
            minibatch_y.append(0 if sentence[:1] == '0' else 1)
            one_hot, length = self.encode_one_hot(sentence[2:-1])

            if length >= max_length:
                max_length = length
            minibatch_x.append(one_hot)


        # data is a np.array of shape ('b', 's', 'e') we want to
        # pad it with np.zeros of shape ('e',) to get ('b', 'SENTENCE_MAX_LENGTH', 'e')
        def numpy_fillna(data):
            # Get lengths of each row of data
            lens = np.array([len(i) for i in data])

            # Mask of valid places in each row
            mask = np.arange(lens.max()) < lens[:, None]

            # Setup output array and put elements from data into masked positions
            out = np.zeros(shape=(mask.shape + (ALPHABET_SIZE,)), dtype='float32')

            out[mask] = np.concatenate(data)
            return out

        # Padding...
        minibatch_x = numpy_fillna(minibatch_x)

        return minibatch_x, np.array(minibatch_y)

    def load_to_ram(self, batch_size):
        # Load n Rows from File f to Ram

        self.data = []
        n_rows = batch_size
        while n_rows > 0:
            self.data.append(next(self.file))
            n_rows -= 1

    def iterate_minibatch(self, batch_size):
        # Returns Next Batch and Catch Bound Errors

        n_batch = 1600000 // batch_size

        for i in range(n_batch):
            self.load_to_ram(batch_size)
            inputs, targets = self.make_minibatch(self.data)
            yield inputs, targets
