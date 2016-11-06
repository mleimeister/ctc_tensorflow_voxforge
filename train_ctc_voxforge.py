#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np
import pickle
import os

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

checkpoint_steps = 10

if not os.path.isdir('./checkpoints'):
    os.makedirs('./checkpoints')

if not os.path.isdir('./summaries/train'):
    os.makedirs('./summaries/train')

if not os.path.isdir('./summaries/test'):
    os.makedirs('./summaries/test')


# Number of input features
feature_dim = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 200
num_lstm_hidden = 128
batch_size = 4
learning_rate = 0.01
momentum = 0.9

# Loading the data
with open('train_data_batched.pkl') as f:
    batched_data = pickle.load(f)

# Load original text targets
with open('original_targets_batched.pkl') as f:
    original_targets = pickle.load(f)

num_valid_batches = 1
num_train_batches = len(batched_data) - num_valid_batches

valid_batches = batched_data[-num_valid_batches:]
valid_orig_targets = original_targets[-num_valid_batches:]
train_batches = batched_data[:num_train_batches]
train_orig_targets = original_targets[:num_train_batches]

del batched_data

graph = tf.Graph()
with graph.as_default():

    tf.set_random_seed(0)

    # The input has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, feature_dim])

    # Variables for the components of the sparse target tensor
    target_idx = tf.placeholder(tf.int64)
    target_vals = tf.placeholder(tf.int32)
    target_shape = tf.placeholder(tf.int64)

    # SparseTensor required by ctc_loss op.
    targets = tf.SparseTensor(target_idx, target_vals, target_shape)

    # Actual sequence length, 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Defining the LSTM cells
    fw_cell = tf.nn.rnn_cell.LSTMCell(num_lstm_hidden)
    bw_cell = tf.nn.rnn_cell.LSTMCell(num_lstm_hidden)

    # Use dynamic RNN to account for different sequence length. Second output is state which is not needed
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs,
                                                 sequence_length=seq_len, dtype=tf.float32)

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_lstm_hidden])

    # Weights for regression layer.
    W = tf.Variable(tf.truncated_normal([num_lstm_hidden, num_classes], stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name='b')

    # Apply linear transform
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Swap dimensions to time major for CTC loss.
    logits = tf.transpose(logits, (1, 0, 2))

    loss = ctc.ctc_loss(logits, targets, seq_len)
    cost = tf.reduce_mean(loss)

    # Record the loss
    tf.scalar_summary('loss', cost)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=momentum, use_nesterov=True).minimize(cost)

    decoded, log_prob = ctc.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len)

    # Label error rate using the edit distance between output and target
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

    # Record the label error rate
    tf.scalar_summary('label error rate', ler)

    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./summaries/train', graph)
    test_writer = tf.train.SummaryWriter('./summaries/test', graph)


def test_decoding(input_feed_dict, input_original):
    """
    Runs the classifier on a feed dictionary and prints the decoded predictions.
    """

    d = session.run(decoded, feed_dict=input_feed_dict)

    str_decoded = ''.join([chr(x) for x in np.asarray(d[0][1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

    print('Original: %s' % input_original)
    print('Decoded: %s' % str_decoded)
    print(' ')


with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.initialize_all_variables().run()

    for curr_epoch in xrange(num_epochs):
        train_cost = train_ler = 0

        for batch in xrange(num_train_batches):

            print('Batch {} / {}'.format(batch, num_train_batches))

            feed = {inputs: train_batches[batch][0],
                    target_idx: train_batches[batch][1][0],
                    target_vals: train_batches[batch][1][1],
                    target_shape: train_batches[batch][1][2],
                    seq_len: np.asarray(train_batches[batch][2])}

            batch_cost, _, summary = session.run([cost, optimizer, merged], feed)
            train_cost += batch_cost
            train_ler += session.run(ler, feed_dict=feed)
            train_writer.add_summary(summary, curr_epoch * num_train_batches + batch)

        train_cost /= num_train_batches
        train_ler /= num_train_batches

        valid_cost = valid_ler = 0

        for batch in xrange(num_valid_batches):

            val_feed = {inputs: valid_batches[batch][0],
                        target_idx: valid_batches[batch][1][0],
                        target_vals: valid_batches[batch][1][1],
                        target_shape: valid_batches[batch][1][2],
                        seq_len: valid_batches[batch][2]}

            val_cost, val_ler, summary = session.run([cost, ler, merged], feed_dict=val_feed)

            valid_cost += val_cost
            valid_ler += val_ler
            test_writer.add_summary(summary, curr_epoch * num_valid_batches + batch)

        valid_cost /= num_valid_batches
        valid_ler /= num_valid_batches

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, valid_cost = {:.3f}, valid_ler = {:.3f}"
        print(log.format(curr_epoch, num_epochs, train_cost, train_ler, valid_cost, valid_ler))

        if curr_epoch % checkpoint_steps == 0:
            saver.save(session, './checkpoints/model.ckpt')

            print('Train decoding: ')

            train_feed = {inputs: train_batches[0][0],
                    target_idx: train_batches[0][1][0],
                    target_vals: train_batches[0][1][1],
                    target_shape: train_batches[0][1][2],
                    seq_len: np.asarray(train_batches[0][2])}

            train_original = ' '.join(train_orig_targets[0])

            test_decoding(train_feed, train_original)

            print('Validation decoding: ')

            val_feed = {inputs: valid_batches[0][0],
                        target_idx: valid_batches[0][1][0],
                        target_vals: valid_batches[0][1][1],
                        target_shape: valid_batches[0][1][2],
                        seq_len: valid_batches[0][2]}

            valid_original = ' '.join(valid_orig_targets[0])

            test_decoding(val_feed, valid_original)


