from collections import Counter
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
import matplotlib as mplt
mplt.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')

#reduced number of classes to 4 and sequnce size to 1000 and changed state when training to let the previouse state to flow to the current state

def pre_process():


    # categories =['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
    #  'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
    #  'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    #  'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']


    categories_comp = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']

    comp = fetch_20newsgroups(subset='all', categories=categories_comp, remove=('headers', 'footers', 'quotes'))

    categories_rec = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    rec = fetch_20newsgroups(subset='all', categories=categories_rec, remove=('headers', 'footers', 'quotes'))

    categories_politics = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc']

    politics = fetch_20newsgroups(subset='all', categories=categories_politics, remove=('headers', 'footers', 'quotes'))

    categories_religion = ['talk.religion.misc', 'soc.religion.christian']

    religion = fetch_20newsgroups(subset='all', categories=categories_religion, remove=('headers', 'footers', 'quotes'))

    data_labels = []

    for post in comp.data:
        data_labels.append(1)

    for post in rec.data:
        data_labels.append(2)

    for post in politics.data:
        data_labels.append(3)

    for post in religion.data:
        data_labels.append(4)

    news_data = []

    for post in comp.data:
        news_data.append(post)

    for post in rec.data:
        news_data.append(post)

    for post in politics.data:
        news_data.append(post)

    for post in religion.data:
        news_data.append(post)

    newsgroups_data, newsgroups_labels = shuffle(news_data, data_labels, random_state=42)

    words = []
    temp_post_text = []
    print(len(newsgroups_data))

    for post in newsgroups_data:

        all_text = ''.join([text for text in post if text not in punctuation])
        all_text = all_text.split('\n')
        all_text = ''.join(all_text)
        temp_text = all_text.split(" ")

        for word in temp_text:
            if word.isalpha():
                temp_text[temp_text.index(word)] = word.lower()

        temp_text = [word for word in temp_text if word not in stopwords.words('english')]
        temp_text = list(filter(None, temp_text))
        temp_text = ' '.join([i for i in temp_text if not i.isdigit()])
        words += temp_text.split(" ")
        temp_post_text.append(temp_text)

    # temp_post_text = list(filter(None, temp_post_text))

    dictionary = Counter(words)
    # deleting spacesA
    # del dictionary[""]
    sorted_split_words = sorted(dictionary, key=dictionary.get, reverse=True)
    vocab_to_int = {c: i for i, c in enumerate(sorted_split_words,1)}

    message_ints = []
    for message in temp_post_text:
        temp_message = message.split(" ")
        message_ints.append([vocab_to_int[i] for i in temp_message])

    # # maximum message length = 4984
    message_lens = Counter([len(x) for x in message_ints])

    seq_length = 1000
    num_messages = len(temp_post_text)
    features = np.zeros([num_messages, seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        features[i, :len(row)] = np.array(row)[:seq_length]

    lb = LabelBinarizer()
    # lbl = newsgroups_data.target
    # labels = np.reshape(lbl, [-1])
    labels = lb.fit_transform(newsgroups_labels)

    # sequence_lengths = [len(msg) for msg in message_ints]

    sequence_lengths = []

    for msg in message_ints:
        lentemp = len(msg)
        if lentemp > 1000:
            lentemp = 1000
        sequence_lengths.append(lentemp)

    return features, labels, len(sorted_split_words)+1, sequence_lengths



def get_batches(x, y, sql, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size], sql[ii:ii+batch_size]


def train_test():
    features, labels, n_words, sequence_length = pre_process()

    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, shuffle=False, random_state=42)

    sequence_length_train = sequence_length[:len(train_y)]
    sequence_length_test= sequence_length[len(train_y):]


    # Defining Hyperparameters

    lstm_layers = 1
    batch_size = 50
    lstm_size = 50
    learning_rate = 0.01
    epoch = 25

    print("learning 32")

    # --------------placeholders-------------------------------------

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
        # labels_ = tf.placeholder(dtype= tf.int32)
        labels_ = tf.placeholder(tf.float32, [None, None], name="labels")
        sql_in = tf.placeholder(tf.int32, [None], name='sql_in')

        # Size of the embedding vectors (number of units in the embedding layer)
        embed_size = 50

        # generating random values from a uniform distribution (minval included and maxval excluded)
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size, forget_bias=1.0)

        # Getting an initial state of all zeros
        initial_state = lstm.zero_state(batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(lstm, embed, initial_state=initial_state, sequence_length=sql_in)

        out_batch_size = tf.shape(outputs)[0]
        out_max_length = tf.shape(outputs)[1]
        out_size = int(outputs.get_shape()[2])
        index = tf.range(0, out_batch_size) * out_max_length + (sql_in - 1)
        flat = tf.reshape(outputs, [-1, out_size])
        relevant = tf.gather(flat, index)

        # hidden layer
        hidden = tf.layers.dense(relevant, units=25, activation=tf.nn.relu)

        logit = tf.contrib.layers.fully_connected(hidden, num_outputs=4, activation_fn=None)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels_))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        saver = tf.train.Saver()

    # ----------------------------batch training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range (epoch):
            state = sess.run(initial_state)
            for ii, (x, y, sql) in enumerate(get_batches(np.array(train_x),  np.array(train_y), sequence_length_train, batch_size), 1):

                feed = {inputs_: x,
                        labels_: y,
                        sql_in: sql,
                        initial_state: state}

                loss, states, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)


                if iteration % 5 == 0:
                    print("Epoch: {}/{}".format(e, epoch),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(loss))
                iteration += 1
        saver.save(sess, "checkpoints/sentiment.ckpt")

     # -----------------testing test set-----------------------------------------
        print("starting testing set")
        argmax_pred_array = []
        argmax_label_array = []
        with tf.Session(graph=graph) as sess:
                tf.set_random_seed(1)
                saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
                test_state = sess.run(lstm.zero_state(batch_size, tf.float32))

                for ii, (x, y, sql) in enumerate(get_batches(np.array(test_x), np.array(test_y), sequence_length_test, batch_size), 1):
                    feed = {inputs_: x,
                            labels_: y,
                            sql_in: sql,
                            initial_state: test_state}

                    predictions = tf.nn.softmax(logit).eval(feed_dict=feed)

                    for i in range(len(predictions)):
                        argmax_pred_array.append(np.argmax(predictions[i], 0))
                        argmax_label_array.append(np.argmax(y[i], 0))

                    print(len(argmax_pred_array))
                    print(len(argmax_label_array))

                accuracy = accuracy_score(argmax_label_array, argmax_pred_array)

                batch_f1 = f1_score(argmax_label_array, argmax_pred_array, average="macro")

                batch_recall = recall_score(y_true=argmax_label_array, y_pred=argmax_pred_array, average='macro')

                batch_precision = precision_score(argmax_label_array, argmax_pred_array, average='macro')

                print("-----------------testing test set-----------------------------------------")
                print("Test accuracy: {:.3f}".format(accuracy))
                print("F1 Score: {:.3f}".format(batch_f1))
                print("Recall: {:.3f}".format(batch_recall))
                print("Precision: {:.3f}".format(batch_precision))


if __name__ == '__main__':
    train_test()

