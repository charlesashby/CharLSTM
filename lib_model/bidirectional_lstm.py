from lib.data_utils import *
from lib.ops import *
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

PATH = '/home/ashbylepoc/PycharmProjects/tensorflow/'
TRAIN_SET = PATH + 'datasets/train_set.csv'
TEST_SET = PATH + 'datasets/test_set.csv'
VALID_SET = PATH + 'datasets/valid_set.csv'
SAVE_PATH = PATH + 'checkpoints/bidirectional_lstm'
LOGGING_PATH = PATH + 'checkpoints/log_bidirectional_lstm.txt'

class LSTM(object):
    """ Character-Level LSTM Implementation """

    def __init__(self):
        # X is of shape ('b', 'sentence_length', 'max_word_length', 'alphabet_size')
        self.hparams = self.get_hparams()
        max_word_length = self.hparams['max_word_length']
        self.X = tf.placeholder('float32', shape=[None, None, max_word_length, ALPHABET_SIZE], name='X')
        self.Y = tf.placeholder('float32', shape=[None, 2], name='Y')

    def build(self,
              kernels=[1, 2, 3, 4, 5, 6, 7],
              kernel_features=[25, 50, 75, 100, 125, 150, 175],
              rnn_size=650,
              dropout=0.0,
              size=700,
              train_samples=1600000 * 0.95,
              valid_samples=1600000 * 0.05):

        self.size = size
        self.hparams = self.get_hparams()
        self.max_word_length = self.hparams['max_word_length']
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        BATCH_SIZE = self.hparams['BATCH_SIZE']

        # Highway & TDNN Implementation are from https://github.com/mkroutikov/tf-lstm-char-cnn/blob/master/model.py
        def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
            """Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wy + b)
            z = t * g(Wy + b) + (1 - t) * y
            where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
            """

            with tf.variable_scope(scope):
                for idx in range(num_layers):
                    g = f(linear(input_, size, scope='highway_lin_%d' % idx))

                    t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

                    output = t * g + (1. - t) * input_
                    input_ = output

            return output

        def tdnn(input_, kernels, kernel_features, scope='TDNN'):
            ''' Time Delay Neural Network
            :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
            :kernels:         array of kernel sizes
            :kernel_features: array of kernel feature sizes (parallel to kernels)
            '''
            assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

            # input_ is a np.array of shape ('b', 'sentence_length', 'max_word_length', 'embed_size') we
            # need to convert it to shape ('b * sentence_length', 1, 'max_word_length', 'embed_size') to
            # use conv2D
            input_ = tf.reshape(input_, [-1, self.max_word_length, ALPHABET_SIZE])
            input_ = tf.expand_dims(input_, 1)

            layers = []
            with tf.variable_scope(scope):
                for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
                    reduced_length = self.max_word_length - kernel_size + 1

                    # [batch_size * sentence_length x max_word_length x embed_size x kernel_feature_size]
                    conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)

                    # [batch_size * sentence_length x 1 x 1 x kernel_feature_size]
                    pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

                    layers.append(tf.squeeze(pool, [1, 2]))

                if len(kernels) > 1:
                    output = tf.concat(layers, 1)
                else:
                    output = layers[0]

            return output

        cnn = tdnn(self.X, kernels, kernel_features)

        # tdnn() returns a tensor of shape [batch_size * sentence_length x kernel_features]
        # highway() returns a tensor of shape [batch_size * sentence_length x size] to use
        # tensorflow dynamic_rnn module we need to reshape it to [batch_size x sentence_length x size]
        cnn = highway(cnn, self.size)
        cnn = tf.reshape(cnn, [BATCH_SIZE, -1, self.size])

        with tf.variable_scope('LSTM'):

            def create_rnn_cell():
                lstm_fw_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0, reuse=False)
                lstm_bw_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0, reuse=False)

                if dropout > 0.0:
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=1. - dropout)
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=1. - dropout)

                return lstm_fw_cell, lstm_bw_cell

            fw_cell, bw_cell = create_rnn_cell()
            initial_fw_state = fw_cell.zero_state(BATCH_SIZE, dtype='float32')
            initial_bw_state = bw_cell.zero_state(BATCH_SIZE, dtype='float32')

            outputs, _  = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, cnn,
                                                                      initial_state_bw=initial_bw_state,
                                                                      initial_state_fw=initial_fw_state,
                                                                      dtype=tf.float32)
            output_fw, output_bw = outputs

            # In this implementation, we only care about the last outputs of the RNN
            # i.e. the output at the end of the sentence
            output_fw = tf.transpose(output_fw, [1, 0, 2])
            output_bw = tf.transpose(output_bw, [1, 0, 2])
            last_fw = output_fw[-1]
            last_bw = output_bw[-1]
            last = tf.concat([last_fw, last_bw], axis=1)

        self.prediction = softmax(last, 2)


    def train(self):
        BATCH_SIZE = self.hparams['BATCH_SIZE']
        EPOCHS = self.hparams['EPOCHS']
        max_word_length = self.hparams['max_word_length']
        learning_rate = self.hparams['learning_rate']

        pred = self.prediction

        cost = - tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

        predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

        acc = tf.reduce_mean(tf.cast(predictions, 'float32'))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        n_batch = self.train_samples // BATCH_SIZE

        # parameters for saving and early stopping
        saver = tf.train.Saver()
        patience = self.hparams['patience']

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            best_acc = 0.0
            DONE = False
            epoch = 1

            while epoch <= EPOCHS and not DONE:
                loss = 0.0
                batch = 1

                with open(TRAIN_SET, 'r') as f:
                    reader = TextReader(f, max_word_length)
                    for minibatch in reader.iterate_minibatch(BATCH_SIZE, dataset=TRAIN_SET):
                        batch_x, batch_y = minibatch

                        _, c, a = sess.run([optimizer, cost, acc], feed_dict={self.X: batch_x, self.Y: batch_y})

                        loss += c

                        if batch % 100 == 0:
                            # Compute Accuracy on the Training set and print some info
                            print('Epoch: %5d/%5d -- batch: %5d/%5d -- Loss: %.4f -- Train Accuracy: %.4f' %
                                  (epoch, EPOCHS, batch, n_batch, loss/batch, a))

                            # Write loss and accuracy to some file
                            log = open(LOGGING_PATH, 'a')
                            log.write('%s, %6d, %.5f, %.5f \n' % ('train', epoch * batch, loss/batch, a))
                            log.close()

                        # --------------
                        # EARLY STOPPING
                        # --------------

                        # Compute Accuracy on the Validation set, check if validation has improved, save model, etc
                        if batch % 500 == 0:
                            accuracy = []

                            # Validation set is very large, so accuracy is computed on testing set
                            # instead of valid set, change TEST_SET to VALID_SET to compute accuracy on valid set
                            with open(TEST_SET, 'r') as ff:
                                valid_reader = TextReader(ff, max_word_length)
                                for mb in valid_reader.iterate_minibatch(BATCH_SIZE, dataset=TEST_SET):
                                    valid_x, valid_y = mb
                                    a = sess.run([acc], feed_dict={self.X: valid_x, self.Y: valid_y})
                                    accuracy.append(a)
                                mean_acc = np.mean(accuracy)

                                # if accuracy has improved, save model and boost patience
                                if mean_acc > best_acc:
                                    best_acc = mean_acc
                                    save_path = saver.save(sess, SAVE_PATH)
                                    patience = self.hparams['patience']
                                    print('Model saved in file: %s' % save_path)

                                # else reduce patience and break loop if necessary
                                else:
                                    patience -= 500
                                    if patience <= 0:
                                        DONE = True
                                        break

                                print('Epoch: %5d/%5d -- batch: %5d/%5d -- Valid Accuracy: %.4f' %
                                     (epoch, EPOCHS, batch, n_batch, mean_acc))

                                # Write validation accuracy to log file
                                log = open(LOGGING_PATH, 'a')
                                log.write('%s, %6d, %.5f \n' % ('valid', epoch * batch, mean_acc))
                                log.close()

                        batch += 1

    def evaluate_test_set(self):
        '''
        Evaluate Test Set
        '''
        BATCH_SIZE = self.hparams['BATCH_SIZE']
        max_word_length = self.hparams['max_word_length']

        pred = self.prediction

        cost = - tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

        predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

        acc = tf.reduce_mean(tf.cast(predictions, 'float32'))

        # parameters for restoring variables
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Loading model %s...' % SAVE_PATH)
            saver.restore(sess, SAVE_PATH)
            print('Done!')
            loss = []
            accuracy = []

            with open(VALID_SET, 'r') as f:
                reader = TextReader(f, max_word_length)
                for minibatch in reader.iterate_minibatch(BATCH_SIZE, dataset=VALID_SET):
                    batch_x, batch_y = minibatch

                    c, a = sess.run([cost, acc], feed_dict={self.X: batch_x, self.Y: batch_y})
                    loss.append(c)
                    accuracy.append(a)

                loss = np.mean(loss)
                accuracy = np.mean(accuracy)
                print('Valid loss: %.5f -- Valid Accuracy: %.5f' % (loss, accuracy))
                return loss, accuracy


    def predict_sentences(self, sentences):
        '''
        Analyze Some Sentences

        :sentences: list of sentences
        e.g.: sentences = ['this is veeeryyy bad!!', 'I dont think he will be happy abt this',
                            'YOUre a such an incompetent fool!', 'Im sooo happY!!!']

        '''
        BATCH_SIZE = self.hparams['BATCH_SIZE']
        max_word_length = self.hparams['max_word_length']
        pred = self.prediction

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Loading model %s...' % SAVE_PATH)
            saver.restore(sess, SAVE_PATH)
            print('Done!')

            # Add placebo value '0,' at the beginning of the sentences to
            # use the make_minibatch() method
            sentences = ['0,' + s for s in sentences]

            with open(TEST_SET, 'r') as f:
                reader = TextReader(file=f, max_word_length=max_word_length)
                reader.load_to_ram(BATCH_SIZE)
                reader.data[:len(sentences)] = sentences
                batch_x, batch_y = reader.make_minibatch(reader.data)

                p = sess.run([pred], feed_dict={self.X: batch_x, self.Y: batch_y})
                for i, s in enumerate(sentences):
                    print('Sentence: %s , yielded results (pos/neg): %.5f/%.5f, prediction: %s' %
                          (s[2:], p[0][i][0], p[0][i][1], 'pos' if max(p[0][i]) == p[0][i][0] else 'neg'))
            return p

    def get_hparams(self):
        ''' Get Hyperparameters '''

        return {
            'BATCH_SIZE':       64,
            'EPOCHS':           500,
            'max_word_length':  16,
            'learning_rate':    0.0001,
            'patience':         10000,
        }

if __name__ == '__main__':
    network = LSTM()
    network.build()
    network.train()
