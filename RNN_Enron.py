import os
from collections import Counter
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from string import punctuation
from sklearn.model_selection import train_test_split


def pre_process():
    direc = "enron/emails/"
    files = os.listdir(direc)
    emails = [direc+email for email in files]

    words = []
    temp_email_text = []
    labels = []

    for email in emails:
        if "ham" in email:
            labels.append(0)
        else:
            labels.append(1)
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

    # print(temp_email_text[0])
    # print(labels[:10])
    # print(message_ints[0])
    # print("\n")
    # print(len(temp_email_text[0]))
    # print(len(message_ints[0]))

    #maximum message length = 3423

    message_lens = Counter([len(x) for x in message_ints])
    # print("Zero-length messages: {}".format(message_lens[0]))
    # print("Maximum message length: {}".format(max(message_lens)))

    seq_length = 3425
    num_messages = len(temp_email_text)
    features = np.zeros([num_messages,seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    # print(len(features[0]))
    # print(len(features[1]))
    # blah = list(enumerate(message_ints))
    # print(blah[:2])

    return features, np.array(labels), sorted_split_words


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size

    batch_counter = 0
    ii = 0

    while(ii != len(x)):
        if(batch_counter == n_batches):
            yield x[ii:], y[ii:]
            ii = len(x)
        else:
            yield x[ii:ii + batch_size], y[ii:ii + batch_size]
            ii += batch_size

        batch_counter += 1

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
    batch_size = 32
    lstm_size = 30
    n_words = len(sorted_split_words)
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
        labels_ = tf.placeholder(tf.int32, [None,None], name = "labels")

        #getting dynamic batch size according to the input tensor size

        dynamic_batch_size = tf.shape(inputs_)[0]

        #output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.

        keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

        # Size of the embedding vectors (number of units in the embedding layer)
        embed_size = 300

        #generating random values from a uniform distribution (minval included and maxval excluded)
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        #Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(dynamic_batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

        #hidden layer
        hidden = tf.layers.dense(outputs[:, -1], units=23, activation=tf.nn.relu)

        predictions = tf.contrib.layers.fully_connected(hidden, 1, activation_fn=tf.sigmoid)

        cost = tf.losses.mean_squared_error(labels_, predictions)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()

    # -----------------training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range(epochs):
            for ii, (x, y) in enumerate(get_batches(np.array(train_x), np.array(train_y), batch_size), 1):

                tensor_x = tf.convert_to_tensor(x, np.int32)

                state = sess.run(cell.zero_state(tensor_x.get_shape()[0], tf.float32))

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
    test_acc = []
    f1 = []
    recall = []
    precision = []
    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

        for ii, (x, y) in enumerate(get_batches(np.array(val_x), np.array(val_y), batch_size), 1):

            tensor_x = tf.convert_to_tensor(x, np.int32)

            test_state = sess.run(cell.zero_state(tensor_x.get_shape()[0], tf.float32))

            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)

            prediction = sess.run([predictions], feed_dict=feed)
            prediction = np.array(prediction)
            prediction = prediction.reshape((x.get_shape()[0], 1))
            print(prediction.shape)

            batch_f1 = f1_score(np.array(y), prediction.round(), average='macro')
            batch_recall = recall_score(y_true=np.array(y), y_pred=prediction.round())
            batch_precision = precision_score(y, prediction.round(), average='macro')

            test_acc.append(batch_acc)
            f1.append(batch_f1)
            recall.append(batch_recall)
            precision.append(batch_precision)

        print("-----------------testing validation set-----------------------------------------")
        print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
        print("F1 Score: {:.3f}".format(np.mean(f1)))
        print("Recall: {:.3f}".format(np.mean(recall)))
        print("Precision: {:.3f}".format(np.mean(precision)))

        with open('results.txt', 'a') as f:
            f.write(str(np.mean(test_acc)))
            f.write(str(np.mean(f1)))
            f.write(str(np.mean(recall)))
            f.write(str(np.mean(precision)))

    # -----------------testing test set-----------------------------------------
    test_acc = []
    f1 = []
    recall = []
    precision = []
    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

        for ii, (x, y) in enumerate(get_batches(np.array(test_x), np.array(test_y), batch_size), 1):

            tensor_x = tf.convert_to_tensor(x, np.int32)

            test_state = sess.run(cell.zero_state(tensor_x.get_shape()[0], tf.float32))

            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)

            prediction = sess.run([predictions], feed_dict=feed)
            prediction = np.array(prediction)
            prediction = prediction.reshape((x.get_shape()[0], 1))
            print(prediction.shape)

            batch_f1 = f1_score(np.array(y), prediction.round(), average='macro')
            batch_recall = recall_score(y_true=np.array(y), y_pred=prediction.round())
            batch_precision = precision_score(y, prediction.round(), average='macro')

            test_acc.append(batch_acc)
            f1.append(batch_f1)
            recall.append(batch_recall)
            precision.append(batch_precision)

        print("-----------------testing test set-----------------------------------------")
        print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
        print("F1 Score: {:.3f}".format(np.mean(f1)))
        print("Recall: {:.3f}".format(np.mean(recall)))
        print("Precision: {:.3f}".format(np.mean(precision)))

        with open('results.txt', 'a') as f:
            f.write(str(np.mean(test_acc)))
            f.write(str(np.mean(f1)))
            f.write(str(np.mean(recall)))
            f.write(str(np.mean(precision)))

if __name__ == '__main__':
    train_test()


#8 and 6