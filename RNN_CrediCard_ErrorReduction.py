import tensorflow as tf
from tensorflow.contrib import rnn

# cycles of feed forward and backprop
hm_epochs = 30
n_classes = 1
rnn_size = 200
col_size = 30
batch_size = 24
try_epochs = 1
fileName = "creditcard.csv"

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31 = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30])
    return features, col31


def input_pipeline(fName, batch_size, num_epochs=None):
    # this refers to multiple files, not line items within files
    filename_queue = tf.train.string_input_producer([fName], shuffle=True, num_epochs=num_epochs)
    features, label = create_file_reader_ops(filename_queue)
    min_after_dequeue = 10000 # min of where to start loading into memory
    capacity = min_after_dequeue + 3 * batch_size # max of how much to load into memory
    # this packs the above lines into a batch of size you specify:
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch


creditCard_data, creditCard_label = input_pipeline(fileName, batch_size, try_epochs)

#(3, 30)
x = tf.placeholder('float',[None,col_size])
#(3,)
y = tf.placeholder('float')


def recurrent_neural_network_model(x):
    #giving the weights and biases random values
    layer ={ 'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
            'bias': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.split(x, 24, 0)
    print(x)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32 )
    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    print(prediction.shape)
    print(type(prediction))

    prediction = tf.reshape(prediction, [-1])
    print(prediction.shape)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for epoch in range(hm_epochs):
            epoch_loss = 0

            for counter in range(101):
                    feature_batch, label_batch = sess.run([creditCard_data, creditCard_label])
                    _, c = sess.run([optimizer, cost], feed_dict={x: feature_batch, y: label_batch})
                    epoch_loss += c
            print('Epoch', epoch, 'compleated out of', hm_epochs, 'loss:', epoch_loss)

train_neural_network(x)











