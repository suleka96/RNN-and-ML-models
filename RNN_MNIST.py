import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("temp/data/", one_hot=True)
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


# cycles of feed forward and backprop
hm_epochs = 2
n_classes = 10
batch_size = 128
chunk_size=28
n_chunks=28
rnn_size = 128

#height by width dATA TO BE INPUTTED
x = tf.placeholder('float', [None, n_chunks, chunk_size])
#TARGET OR LABEL
y = tf.placeholder('float')

def recurrent_neural_network_model(x):
    # number of layers i
    layer ={ 'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
            'bias': tf.Variable(tf.random_normal([n_classes]))}
    print(x)
    x = tf.transpose(x,[1,0,2])
    print(x)
    x = tf.reshape(x, [-1,chunk_size])
    print(x)
    x = tf.split(x,n_chunks, 0)
    print(x)

    from tensorflow.contrib import rnn

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    print(prediction.shape)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss =0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x= epoch_x.reshape((batch_size,n_chunks,chunk_size ))
                #print(epoch_x)
                print(epoch_x.shape)
                print(epoch_y.shape)
                _, c = sess.run([optimizer,cost],feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'compleated out of', hm_epochs, 'loss:', epoch_loss)
            # print('F1 score: ', f1)

        correct =  tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)



