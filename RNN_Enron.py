import os
from collections import Counter
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from string import punctuation


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

    seq_length = 3425
    num_messages = len(temp_email_text)
    features = np.zeros([num_messages,seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    print(hamcounter)
    print(spamcounter)
    return features, np.array(labels), sorted_split_words

def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


def train_test():

    features, labels, sorted_split_words = pre_process()

    #splitting training, validation and testing sets

    split_frac1 = 0.8

    idx1 = int(len(features) * split_frac1)
    train_x, val_x = features[:idx1], features[idx1:]
    train_y, val_y = labels[:idx1], labels[idx1:]

    split_frac2 = 0.5

    idx2 = int(len(val_x) * split_frac2)
    val_x, test_x = val_x[:idx2], val_x[idx2:]
    val_y, test_y = val_y[:idx2], val_y[idx2:]

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    print("\t\t\Label Shapes:")
    print("Train set: \t\t{}".format(train_y.shape),
          "\nValidation set: \t{}".format(val_y.shape),
          "\nTest set: \t\t{}".format(test_y.shape))

    #Defining Hyperparameters

    epochs = 7
    lstm_layers = 1
    batch_size = 150
    lstm_size = 50
    n_words = len(sorted_split_words)+1
    learning_rate = 0.003

    print(n_words)
    print(lstm_size)
    print(batch_size)
    print(epochs)

    #--------------placeholders-------------------------------------

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        inputs_ = tf.placeholder(tf.int32, [None,None], name = "inputs")
        labels_ = tf.placeholder(tf.float32, [None,None], name = "labels")

        #getting dynamic batch size according to the input tensor size

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
        hidden = tf.layers.dense(outputs[:, -1], units=30, activation=tf.nn.relu)

        logit = tf.contrib.layers.fully_connected(hidden, num_outputs=1, activation_fn=None)

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=labels_))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        predictions = tf.round(tf.nn.sigmoid(logit))

        saver = tf.train.Saver()

    # -----------------training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range(epochs):
            state = sess.run(initial_state)
            for ii, (x, y) in enumerate(get_batches(np.array(train_x), np.array(train_y), batch_size), 1):

                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 0.5,
                        initial_state: state}
                loss, states, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

                if iteration % 5 == 0:
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(loss))
                iteration += 1

        saver.save(sess, "checkpoints/sentiment.ckpt")

    #-----------------testing validation set-----------------------------------------
    #
    print("starting validation set")
    prediction_vals = []
    y_vals = []
    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

        test_state = sess.run(cell.zero_state(batch_size, tf.float32))

        for ii, (x, y) in enumerate(get_batches(np.array(val_x), np.array(val_y), batch_size), 1):

            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}

            prediction = sess.run(predictions, feed_dict=feed)
            prediction = prediction.astype(int)

            for i in range(len(prediction)):
                prediction_vals.append(prediction[i][0])
                y_vals.append(y[i])

        accuracy = accuracy_score(y_vals, prediction_vals)
        f1 = f1_score(y_vals, prediction_vals, average='macro')
        recall = recall_score(y_true=y_vals, y_pred=prediction_vals, average='macro')
        precision = precision_score(y_vals, prediction_vals, average='macro')

        print("-----------------testing validation set-----------------------------------------")
        print("Test accuracy: {:.3f}".format(accuracy))
        print("F1 Score: {:.3f}".format(f1))
        print("Recall: {:.3f}".format(recall))
        print("Precision: {:.3f}".format(precision))

    # -----------------testing test set-----------------------------------------
    print("starting testing set")
    prediction_val = []
    y_val = []
    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))

        for ii, (x, y) in enumerate(get_batches(np.array(test_x), np.array(test_y), batch_size), 1):

            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}

            prediction = sess.run(predictions, feed_dict=feed)
            prediction = prediction.astype(int)

            for i in range(len(prediction)):
                prediction_val.append(prediction[i][0])
                y_val.append(y[i])

        accuracy = accuracy_score(y_val, prediction_val )
        f1 = f1_score(y_val, prediction_val, average='macro')
        recall = recall_score(y_true=y_val, y_pred=prediction_val, average='macro')
        precision = precision_score(y_val, prediction_val, average='macro')

        print("-----------------testing validation set-----------------------------------------")
        print("Test accuracy: {:.3f}".format(accuracy))
        print("F1 Score: {:.3f}".format(f1))
        print("Recall: {:.3f}".format(recall))
        print("Precision: {:.3f}".format(precision))


if __name__ == '__main__':
    train_test()

