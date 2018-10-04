from collections import Counter
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
import matplotlib as mplt
mplt.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.utils import shuffle
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')



def pre_process():
    categories_comp = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                       'comp.windows.x']

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
    # deleting spaces
    # del dictionary[""]
    sorted_split_words = sorted(dictionary, key=dictionary.get, reverse=True)
    vocab_to_int = {c: i for i, c in enumerate(sorted_split_words,1)}

    message_ints = []
    for message in temp_post_text:
        temp_message = message.split(" ")
        message_ints.append([vocab_to_int[i] for i in temp_message])


    # maximum message length = 6577

    # message_lens = Counter([len(x) for x in message_ints])AAA

    seq_length = 1000
    num_messages = len(temp_post_text)
    features = np.zeros([num_messages, seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        # print(features[i, -len(row):])
        # features[i, -len(row):] = np.array(row)[:seq_length]
        features[i, :len(row)] = np.array(row)[:seq_length]
        # print(features[i])

    lb = LabelBinarizer()
    # lbl = newsgroups_data.target
    # labels = np.reshape(lbl, [-1])
    labels = lb.fit_transform(newsgroups_labels)

    sequence_lengths = []

    for msg in message_ints:
        lentemp = len(msg)
        if lentemp > 1000:
            lentemp = 1000
        sequence_lengths.append(lentemp)

    return features, labels, len(sorted_split_words)+1, sequence_lengths


def get_batches(x, y, sql, batch_size=1):
    for ii in range(0, len(y), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size], sql[ii:ii+batch_size]


def plot(noOfWrongPred, dataPoints):
    font_size = 14
    fig = plt.figure(dpi=100,figsize=(10, 6))
    mplt.rcParams.update({'font.size': font_size})
    plt.title("Distribution of wrong predictions", fontsize=font_size)
    plt.ylabel('Error rate', fontsize=font_size)
    plt.xlabel('Number of data points', fontsize=font_size)

    plt.plot(dataPoints, noOfWrongPred, label='Prediction', color='blue', linewidth=1.8)
    # plt.legend(loc='upper right', fontsize=14)

    plt.savefig('distribution of wrong predictions 20 News LSTM.png')
    # plt.show()



def train_test():
    features, labels, n_words, sequence_length = pre_process()

    print(features.shape)
    print(labels.shape)

    # Defining Hyperparameters

    lstm_layers = 1
    batch_size = 1
    lstm_size = 50
    learning_rate = 0.05

    # --------------placeholders-------------------------------------

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
        # labels_ = tf.placeholder(dtype= tf.int32)
        labels_ = tf.placeholder(tf.float32, [None, None], name="labels")
        sql_in = tf.placeholder(tf.int32, [None], name= 'sql_in')

        # output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # Size of the embedding vectors (number of units in the embedding layer)
        embed_size = 50

        # generating random values from a uniform distribution (minval included and maxval excluded)
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1),trainable=True)
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        # print(embedding.shape)
        # print(embed.shape)
        # print(embed[0])

        # Your basic LSTM cell
        lstm =  tf.contrib.rnn.BasicLSTMCell(lstm_size)

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

        print(hidden.shape)

        logit = tf.contrib.layers.fully_connected(hidden, num_outputs=4, activation_fn=None)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels_))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        predictions = tf.nn.sigmoid(logit)

        saver = tf.train.Saver()

    # ----------------------------online training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        state = sess.run(initial_state)
        wrongPred = 0
        noOfWrongPreds = []
        dataPoints = []

        for ii, (x, y, sql) in enumerate(get_batches(features, labels, sequence_length, batch_size), 1):

            feed = {inputs_: x,
                    labels_: y,
                    sql_in : sql,
                    keep_prob: 0.5,
                    initial_state: state}

            prediction = sess.run(predictions, feed_dict=feed)

            print("----------------------------------------------------------")
            print("sez: ",sql)
            print("Iteration: {}".format(iteration))

            isequal = np.equal(np.argmax(prediction[0], 0), np.argmax(y[0], 0))

            print(np.argmax(prediction[0], 0))
            print(np.argmax(y[0], 0))

            if not (isequal):
                wrongPred += 1

            print("nummber of wrong preds: ",wrongPred)

            if iteration%50 == 0:
                noOfWrongPreds.append(wrongPred/iteration)
                dataPoints.append(iteration)

            loss, states, _ = sess.run([cost, outputs, optimizer], feed_dict=feed)

            print("Train loss: {:.3f}".format(loss))
            iteration += 1

        saver.save(sess, "checkpoints/sentiment.ckpt")
        errorRate = wrongPred / len(labels)
        print("ERRORS: ", wrongPred)
        print("ERROR RATE: ", errorRate)
        plot(noOfWrongPreds, dataPoints)


if __name__ == '__main__':
    train_test()

