import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

class RNNConfig():
    input_size = 1
    num_steps = 2
    lstm_size = 128
    num_layers = 1
    keep_prob = 0.8
    batch_size = 64
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 3  # 5
    max_epoch = 5  # 100 or 50
    scaler = MinMaxScaler()


config = RNNConfig()


def pre_process():

    num_steps = config.num_steps
    test_ratio = 0.2

    stock_data = pd.read_csv('AIG.csv')
    stock_data[['Close']] = config.scaler.fit_transform(stock_data[['Close']])

    seq = [price for tup in stock_data[['Close']].values for price in tup]

    seq = np.array(seq)
    print(seq)

    # split into number of input_size
    seq = [np.array(seq[i * config.input_size: (i + 1) * config.input_size])
           for i in range(len(seq) // config.input_size)]

    # split into groups of num_steps
    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])

    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

    train_size = int(len(X) * (1.0 - test_ratio))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    return train_X, train_y, test_X, test_y


def generate_batches(train_X, train_y, batch_size):
    num_batches = int(len(train_X)) // batch_size
    if batch_size * num_batches < len(train_X):
        num_batches += 1

    batch_indices = range(num_batches)
    # random.shuffle(batch_indices)
    for j in batch_indices:
        batch_X = train_X[j * batch_size: (j + 1) * batch_size]
        batch_y = train_y[j * batch_size: (j + 1) * batch_size]
        assert set(map(len, batch_X)) == {config.num_steps}
        yield batch_X, batch_y


def train_test():
    train_X, train_y, test_X, test_y = pre_process()

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="inputs")
        targets = tf.placeholder(tf.float32, [None,config.input_size], name="targets")
        keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        # cell = tf.contrib.rnn.MultiRNNCell(
        #     [_create_one_cell() for _ in range(config.num_layers)],
        #     state_is_tuple=True
        # ) if config.num_layers > 1 else _create_one_cell()

        lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * config.num_layers)

        val1, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        val = tf.transpose(val1, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

        weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.input_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[config.input_size]))

        prediction = tf.matmul(last, weight) + bias

        loss = tf.reduce_mean(tf.square(prediction - targets))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        minimize = optimizer.minimize(loss)

    # --------------------training------------------------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)

        tf.global_variables_initializer().run()

        iteration = 1

        learning_rates_to_use = [
            config.init_learning_rate * (
                    config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
            ) for i in range(config.max_epoch)]

        for epoch_step in range(config.max_epoch):

            current_lr = learning_rates_to_use[epoch_step]

            for batch_X, batch_y in generate_batches(train_X, train_y, config.batch_size):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: current_lr,
                    keep_prob: config.keep_prob
                }

                train_loss, _, value = sess.run([loss, minimize, val1], train_data_feed)

                if iteration % 5 == 0:
                    print("Epoch: {}/{}".format(epoch_step, config.max_epoch),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(train_loss))
                iteration += 1

        saver = tf.train.Saver()
        saver.save(sess, "checkpoints_stock/stock_pred.ckpt")

    # --------------------testing------------------------------------------------------

    # with tf.Session(graph=graph) as sess:
    #     merged_summary = tf.summary.merge_all()
    #     writer = tf.summary.FileWriter("log/stock_log.txt", sess.graph)
    #     writer.add_graph(sess.graph)
    #
    #     _summary = sess.run([merged_summary], test_data_feed)
    #     writer.add_summary(_summary, global_step=epoch_step)  # epoch_step in range(config.max_epoch)

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)

        saver.restore(sess, tf.train.latest_checkpoint('checkpoints_stock'))

        test_data_feed = {
            learning_rate: 0.0,
            keep_prob: 1.0,
            inputs: test_X,
            targets: test_y,
        }

        test_pred = sess.run(prediction, test_data_feed)

        # length = len(test_pred)
        # truth= config.scaler.inverse_transform(np.array(test_y).reshape(-1, 1))
        # pred = config.scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))[length:]

        days = range(len(test_y))

        plt.plot(days, test_y, label='truth close')
        plt.plot(days, test_pred, label='pred close')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("closing price")
        plt.ylim((min(test_y), max(test_y)))
        plt.grid(ls='--')
        plt.savefig("Stock price Prediction VS Truth uv.png", format='png', bbox_inches='tight', transparent=False)
        plt.close()

        meanSquaredError = mean_squared_error(test_y, test_pred)
        print("MSE: ", meanSquaredError)
        rootMeanSquaredError = sqrt(meanSquaredError)
        print("RMSE:", rootMeanSquaredError)


if __name__ == '__main__':
    train_test()
















