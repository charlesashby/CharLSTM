# coding=utf-8

# Dataset needs to be shuffled before calling TextReader() instance;
# Use shuffle_dataset() to make a new CSV file, default is /datasets/train_set.csv

import codecs
import random, csv
import numpy as np
from nltk.tokenize import word_tokenize
import sys
import string
import os
from lib.utils import count_lines


printable = string.printable

# PATH needs to be changed accordingly
TOP_LEVEL_CATEGORIES = ['Books', 'Movies & TV', 'Clothing, Shoes & Jewelry', 'Sports & Outdoors',
                        'Toys & Games', 'CDs & Vinyl', 'Musical Instruments', 'Tools & Home Improvement',
                        'Home & Kitchen', 'Health & Personal Care', 'Cell Phones & Accessories', 'Office Products',
                        'Electronics', 'Baby', 'Beauty', 'Automotive', 'Arts, Crafts & Sewing', 'Pet Supplies',
                        'Grocery & Gourmet Food', 'Industrial & Scientific', 'Patio, Lawn & Garden']

# PATH = '/home/ashbylepoc/PycharmProjects/CharLSTM/'
# TRAIN_SET = PATH + 'datasets/training.1600000.processed.noemoticon.csv'
# TEST_SET = PATH + 'datasets/testdata.manual.2009.06.14.csv'
# VALID_PERC = 0.05

# TODO: Add non-Ascii characters
emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '


DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}
ALPHABET_SIZE = len(emb_alphabet)


def reshape_lines(lines):
    data = []
    for l in lines:
        split = l.split('","')
        data.append((split[0][1:], split[-1][:-2]))
    return data


def save_csv(out_file, data):
    with open(out_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print('Data saved to file: %s' % out_file)


# TRAIN_SET = PATH + 'datasets/train_set.csv'
# TEST_SET = PATH + 'datasets/test_set.csv'
# VALID_SET = PATH + 'datasets/valid_set.csv'
TRAIN_SET = '/home/ashbylepoc/PycharmProjects/ml-marketvault/amazon_data/train_tlc.csv'
TEST_SET = '/home/ashbylepoc/PycharmProjects/ml-marketvault/amazon_data/test_tlc.csv'
VALID_SET = '/home/ashbylepoc/PycharmProjects/ml-marketvault/amazon_data/valid_tlc.csv'


class TextReader(object):
    """ Util for Reading the Stanford CSV Files """
    # TODO: Add support for larger files and Queues

    def __init__(self, file, max_word_length, file_pointer):
        # TextReader() takes a CSV file as input that it will read
        # through a buffer


        self.file = file
        self.file_pointer = file_pointer
        self.max_word_length = max_word_length

    def encode_one_hot(self, sentence):
        # Convert Sentences to np.array of Shape ('sent_length', 'word_length', 'emb_size')

        max_word_length = self.max_word_length
        sent = []
        SENT_LENGTH = 0
        # encoded_sentence = filter(lambda x: x in (printable), sentence)

        # print(encoded_sentence)
        # for word in word_tokenize(encoded_sentence.decode('utf-8', 'ignore').encode('utf-8')):
        for word in word_tokenize(sentence):
            word_encoding = np.zeros(shape=(max_word_length, ALPHABET_SIZE))

            for i, char in enumerate(word):

                try:
                    char_encoding = DICT[char]
                    one_hot = np.zeros(ALPHABET_SIZE)
                    one_hot[char_encoding] = 1
                    word_encoding[i] = one_hot

                except Exception as e:
                    pass

            sent.append(np.array(word_encoding))
            SENT_LENGTH += 1

        return np.array(sent), SENT_LENGTH

    def make_minibatch(self, sentences):
        # Create a minibatch of sentences and convert sentiment
        # to a one-hot vector, also takes care of padding

        max_word_length = self.max_word_length
        minibatch_x = []
        minibatch_y = []
        max_length = 0

        for sentence in sentences:
            # 0: Negative 1: Positive
            if len(sentence) > 2:
                title = ','.join(sentence[:-1])
                label = sentence[-1]
                print('weird sentence')
            else:
                title, label = sentence
            ohv = np.zeros(len(TOP_LEVEL_CATEGORIES))
            ohv[int(label)] = 1.
            # minibatch_y.append(np.array([0, 1]) if sentence[:1] == '0' else np.array([1, 0]))
            minibatch_y.append(ohv)
            one_hot, length = self.encode_one_hot(title)

            if length >= max_length:
                max_length = length
            minibatch_x.append(one_hot)


        # data is a np.array of shape ('b', 's', 'w', 'e') we want to
        # pad it with np.zeros of shape ('e',) to get ('b', 'SENTENCE_MAX_LENGTH', 'WORD_MAX_LENGTH', 'e')
        def numpy_fillna(data):
            # Get lengths of each row of data
            lens = np.array([len(i) for i in data])

            # Mask of valid places in each row
            mask = np.arange(lens.max()) < lens[:, None]

            # Setup output array and put elements from data into masked positions
            out = np.zeros(shape=(mask.shape + (max_word_length, ALPHABET_SIZE)),
                           dtype='float32')

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
        if n_rows == 0:
            return True
        else:
            return False

    def iterate_minibatch(self, batch_size):
        # Returns Next Batch and Catch Bound Errors

        n_samples = count_lines(self.file_pointer.name)
        n_batch = int(n_samples // batch_size)
        print(f'Found {n_samples} lines -> n_batch: {n_batch}')

        for i in range(n_batch):
            if self.load_to_ram(batch_size):
                inputs, targets = self.make_minibatch(self.data)
                yield inputs, targets


if __name__ == '__main__':
    file_pointer = open(TRAIN_SET)
    dataloader = TextReader(csv.reader(file_pointer), 50, file_pointer)

    for line in dataloader.iterate_minibatch(1024):
        print(line)