import os
from collections import Counter

import nltk
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib as mplt
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import f1_score, recall_score, precision_score
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def pre_process():

    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize,
                        strip_accents='unicode',
                        lowercase =True, analyzer='word', token_pattern=r'\w+',
                        use_idf=True, smooth_idf=True, sublinear_tf=False,
                        stop_words = 'english')

    vectorizer.fit(newsgroups_data.data)
    n_words = len(vectorizer.vocabulary_)

    features = vectorizer.transform(newsgroups_data.data)

    lb = LabelBinarizer()
    labels = np.reshape(newsgroups_data.target, [-1])
    labels = lb.fit_transform(labels)



    return features, labels, n_words

def get_batches(x, y, batch_size=1):

    for ii in range(0, len(y), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]

# def plot(prediction,actual):
#
#     font_size = 20
#     fig = plt.figure()
#     mplt.rcParams.update({'font.size': font_size})
#     plt.title("Prediction VS True values")
#
#     plt.ylabel('News Group',fontsize=font_size)
#     plt.xlabel('Data row',fontsize=font_size)
#
#     x_axiz_val = []
#
#     for i in range (1, len(actual)+1, 1):
#         x_axiz_val.append(i)
#
#
#     plt.plot(x_axiz_val, prediction, label='Prediction', color='blue', linewidth=1.8)
#     plt.scatter(x_axiz_val, actual, label='Truth', marker='.', color='green', s=60)
#
#     plt.legend(loc='upper right', fontsize=25)

    # fig.savefig('../res/figures/' + 'prediction vs actual' + '.jpg', format='jpg', bbox_inches='tight')

def plot_error(errorplot, datapoint, numberOfWrongPreds):
    errorplot.set_xdata(np.append(errorplot.get_xdata(), datapoint))
    errorplot.set_ydata(np.append(errorplot.get_ydata(), numberOfWrongPreds))
    errorplot.autoscale(enable=True, axis='both', tight=None)
    plt.draw()



def train_test():

    features, labels, n_words = pre_process()

    #Defining Hyperparameters

    epochs = 1
    lstm_layers = 1
    batch_size = 1
    lstm_size = 30
    learning_rate = 0.003

    print(lstm_size)
    print(batch_size)
    print(epochs)

    #--------------placeholders-------------------------------------

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        inputs_ = tf.placeholder(tf.float32, [1,None], name = "inputs")
        # labels_ = tf.placeholder(dtype= tf.int32)
        labels_ = tf.placeholder(tf.int32, [None,None], name = "labels")

        #getting dynamic batch size according to the input tensor size

        # dynamic_batch_size = tf.shape(inputs_)[0]

        #output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.

        keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

        # Size of the embedding vectors (number of units in the embedding layer)
        embed_size = 300

        # generating random values from a uniform distribution (minval included and maxval excluded)
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        #Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

        # inputs_withextradim = tf.expand_dims(inputs_, axis=2)
        # logit = tf.reshape(logit, [-1])

        # print((inputs_.shape))

        # inputShape = tf.shape(inputs_)
        #
        # inputs_withextradim = tf.reshape(inputs_, [1,inputShape[0],inputShape[1]])

        # inputs_withextradim = tf.split(inputs_, 1, 1)

        # print(inputs_withextradim.shape)

        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,  initial_state=initial_state)

        #hidden layer
        hidden = tf.layers.dense(outputs[:, -1], units=25, activation=tf.nn.relu)

        logit = tf.contrib.layers.fully_connected(hidden, 1, activation_fn=None)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels_))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()

    # ----------------------------online training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        state = sess.run(initial_state)
        wrongPred = 0
        errorplot, = plt.plot([], [])

        for ii, (x, y) in enumerate(get_batches(features, labels, batch_size), 1):

            feed = {inputs_: x.toarray(),
                    labels_: y,
                    keep_prob: 0.5,
                    initial_state: state}

            s = x.A

            print(s.shape)
            print(y)
            print(y)

            predictions = tf.round(tf.nn.softmax(logit)).eval(feed_dict=feed)

            print("----------------------------------------------------------")
            print("Iteration: {}".format(iteration))
            print("Prediction: ", predictions)
            print("Actual: ",y)

            pred = np.array(predictions)
            print(pred)
            print(y)

            if not ((pred==y).all()):
                wrongPred += 1

            if ii % 27 == 0:
                plot_error(errorplot,ii,wrongPred)

            loss, states, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            print("Train loss: {:.3f}".format(loss))
            iteration += 1

        saver.save(sess, "checkpoints/sentiment.ckpt")
        errorRate = wrongPred/len(labels)
        print("ERROR RATE: ", errorRate )

if __name__ == '__main__':
    train_test()
