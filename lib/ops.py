# Modified From https://github.com/mkroutikov/tf-lstm-char-cnn/blob/master/model.py

import tensorflow as tf

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def softmax(input_, out_dim, scope=None):
    ''' SoftMax Output '''

    with tf.variable_scope(scope or 'softmax'):
        W = tf.get_variable('W', [input_.get_shape()[1], out_dim])
        b = tf.get_variable('b', [out_dim])

    return tf.nn.softmax(tf.matmul(input_, W) + b)

def MLP(input_, out_dim, size=128, scope=None):
    ''' MLP Implementation '''
    assert len(input_.get_shape) == 2, "MLP takes input of dimension 2 only"

    with tf.variable_scope(scope or "MLP"):
        W_h = tf.get_variable("W_hidden", [input_.get_shape()[1], size], dtype='float32')
        b_h = tf.get_variable("b_hidden", [size], dtype='float32')
        W_out = tf.get_variable("W_out", [size, out_dim], dtype='float32')
        b_out = tf.get_variable("b_out", [out_dim], dtype='float32')

    h = tf.nn.relu(tf.matmul(input_, W_h) + b_h)
    out = tf.matmul(h, W_out) + b_out
    return out

def ResBlock(input_, out_dim, size=128, scope=None):
    ''' Residual Block Implementation '''

    with tf.variable_scope(scope or "MLP"):
        W_h = tf.get_variable("W_hidden", [input_.get_shape()[1], size], dtype='float32')
        b_h = tf.get_variable("b_hidden", [size], dtype='float32')
        W_h_res = tf.get_variable("W_hidden_res", [input_.get_shape()[1], size], dtype='float32')
        b_h_res = tf.get_variable("b_hidden_res", [size], dtype='float32')

        W_out = tf.get_variable("W_out", [size, out_dim], dtype='float32')
        b_out = tf.get_variable("b_out", [out_dim], dtype='float32')

    h = tf.nn.relu(tf.matmul(input_, W_h) + b_h)
    h_res = tf.nn.relu(tf.matmul(input_, W_h_res) + b_h_res) + h
    out = tf.matmul(h_res, W_out) + b_out

    return out
