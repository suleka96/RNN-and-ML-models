import os
from collections import Counter
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from string import punctuation
import matplotlib as mplt
mplt.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def pre_process():
    direc = "enron/emails/"
    files = os.listdir(direc)
    emails = [direc+email for email in files]

    words = []
    temp_email_text = []
    labels = []
    hamcounter=0
    spamcounter =0

    for email in emails:
        if "ham" in email:
            labels.append(0)
            hamcounter +=1
        else:
            labels.append(1)
            spamcounter +=1
        f = open(email,encoding="utf8", errors='ignore')
        blob = f.read()
        all_text = ''.join([text for text in blob if text not in punctuation])
        all_text = all_text.split('\n')
        all_text = ''.join(all_text)
        temp_text = all_text.split(" ")

        for word in temp_text:
            if word.isalpha():
                temp_text[temp_text.index(word)] = word.lower()

        temp_text = list(filter(None, temp_text))
        temp_text = ' '.join([i for i in temp_text if not i.isdigit()])
        words += temp_text.split(" ")
        temp_email_text.append(temp_text)

    dictionary = Counter(words)
    #deleting spaces
    del dictionary[""]
    sorted_split_words = sorted(dictionary, key=dictionary.get, reverse=True)
    vocab_to_int = {c: i for i, c in enumerate(sorted_split_words, 1)}

    message_ints = []
    for message in temp_email_text:
        temp_message = message.split(" ")
        message_ints.append([vocab_to_int[i] for i in temp_message])

    #maximum message length = 3423

    message_lens = Counter([len(x) for x in message_ints])

    seq_length = 3425
    num_messages = len(temp_email_text)
    features = np.zeros([num_messages,seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    print(hamcounter)
    print(spamcounter)
    return features, np.array(labels), sorted_split_words

def get_batches(x, y, batch_size=100):
    for ii in range(0, len(y), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


def plot(noOfWrongPred, dataPoints):
    font_size = 14
    fig = plt.figure(dpi=100,figsize=(10, 6))
    mplt.rcParams.update({'font.size': font_size})
    plt.title("Distribution of wrong predictions", fontsize=font_size)
    plt.ylabel('Error rate', fontsize=font_size)
    plt.xlabel('Number of data points', fontsize=font_size)
    plt.plot(dataPoints, noOfWrongPred, label='Prediction', color='blue', linewidth=1.8)
    plt.savefig('distribution of wrong predictions ENRON.png')


def train_test():

    features, labels, sorted_split_words = pre_process()

    #Defining Hyperparameters

    lstm_layers = 1
    batch_size = 1
    lstm_size = 30
    n_words = len(sorted_split_words)
    learning_rate = 0.01

    print(n_words)
    print(lstm_size)
    print(batch_size)

    #--------------placeholders-------------------------------------

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        inputs_ = tf.placeholder(tf.int32, [None,None], name = "inputs")
        labels_ = tf.placeholder(tf.int32, [None,None], name = "labels")

        #output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.

        keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

        # Size of the embedding vectors (number of units in the embedding layer)
        embed_size = 300

        #generating random values from a uniform distribution (minval included and maxval excluded)
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)
        print(embedding.shape)
        print(embed.shape)

        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        #Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

        #hidden layer
        hidden = tf.layers.dense(outputs[:, -1], units=25, activation=tf.nn.relu)

        predictions = tf.contrib.layers.fully_connected(hidden, 1, activation_fn=tf.sigmoid)

        cost = tf.losses.mean_squared_error(labels_, predictions)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()

    # -----------------training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        state = sess.run(initial_state)

        wrongPred = 0
        noOfWrongPreds = []
        dataPoints = []

        for ii, (x, y) in enumerate(get_batches(np.array(features), np.array(labels), batch_size), 1):

            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}

            prediction = sess.run(predictions, feed_dict=feed)
            prediction = prediction.reshape([-1])
            prediction = np.round(prediction[0])
            prediction = prediction.astype(int)

            print(prediction)

            isequal = np.equal(prediction, y[0])

            if not (isequal):
                wrongPred += 1

            print("nummber of wrong preds: ", wrongPred)

            if iteration % 50 == 0:
                noOfWrongPreds.append(wrongPred / iteration)
                dataPoints.append(iteration)


            loss, states, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            print("Iteration: {}".format(iteration), "Train loss: {:.3f}".format(loss))
            iteration += 1


        saver.save(sess, "checkpoints/sentiment.ckpt")
        errorRate = wrongPred / len(labels)
        print("ERRORS: ", wrongPred)
        print("ERROR RATE: ", errorRate)
        plot(noOfWrongPreds, dataPoints)


if __name__ == '__main__':
    train_test()

