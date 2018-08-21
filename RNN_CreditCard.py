import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score


hm_epochs = 8
n_classes = 1
rnn_size = 200
col_size = 29
batch_size = 35
fileName = "creditcard.csv"


data = pd.read_csv('creditcard.csv', skiprows=[0], header=None)

# total number of fraud records 492
data_1 = data.loc[data.iloc[:, -1] == 1]
# total number of non fraud records 284315
data_0 = data.loc[data.iloc[:, -1] == 0]

#sampling
data_0 = data_0.head(800)
totdata = data_1.append(data_0, ignore_index=True)
totdata = totdata.sample(frac=1)

#if you need to run with sampled data set use 'totdata' if you want to run the the original data set use 'data'

#currently running with sampled data
features = data.iloc[:, 1:30]
labels = data.iloc[:, -1]
X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, shuffle=False, random_state=42)


#shape of place holder when training (<batch size>, 30)
x = tf.placeholder('float',[None,col_size])
#shape of place holder when training (<batch size>,)
y = tf.placeholder('float')


def recurrent_neural_network_model(x):
    #giving the weights and biases random values
    layer ={ 'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
            'bias': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.split(x, col_size, 1)
    print(x)

    # x is a 2-dimensional Tensor and it is sliced along the dimension 1 (columns),
    # each slice is an element of the sequence given as input to the LSTM layer.

    # creates a LSTM layer and instantiates variables for all gates.
    #RNN cell means an RNN layer not just one unit
    #rnn_size is the size of your hidden state (both c and h in a LSTM).

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    #outputs is the output of the layer for each slice
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #outputs contains maxlen values, and they are the outputs of the layer
    # for each element (<tf.Tensor 'split:0' shape=(<batch size>, 1)) of the input sequence.
    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

    return output

def train_neural_network(x):
    logit = recurrent_neural_network_model(x)
    logit = tf.reshape(logit, [-1])

    # ratio = 492/284315
    # weight = tf.constant([[ratio]])
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            for i in range(int(len(X_train) / batch_size)):

                start = i
                end = i + batch_size

                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        pred = tf.round(tf.nn.sigmoid(logit)).eval({x: np.array(X_test), y: np.array(y_test)})
        f1 = f1_score(np.array(y_test), pred, average='macro')
        accuracy=accuracy_score(np.array(y_test), pred)
        recall = recall_score(y_true=np.array(y_test), y_pred= pred)
        print("F1 Score:", f1)
        print("Accuracy Score:",accuracy)
        print("Recall:", recall)


train_neural_network(x)











