from collections import Counter
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
import matplotlib as mplt
mplt.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# import nltk
# nltk.download('stopwords')


class RNNConfig():
    input_size=1
    num_steps= 2
    lstm_size=128
    num_layers=1
    keep_prob=0.8
    batch_size = 3
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 3 #5
    max_epoch = 5 #100 or 50
    features = 2

config = RNNConfig()



def pre_process():
    input_size = config.input_size
    num_steps = config.num_steps
    test_ratio = 0.2

    stock_data = pd.read_csv('AIG.csv')
    # scaler = MinMaxScaler()
    # stock_data[['Volume', 'Close']] = scaler.fit_transform(stock_data[['Volume', 'Close']])

    seq = [price for tup in stock_data[['Volume', 'Close']].values for price in tup]

    seq = np.array(seq)
    print(seq)

    # split into items of features
    seq = [np.array(seq[i * config.features: (i + 1) * config.features])
           for i in range(len(seq) // config.features)]


    # seq = [seq[0] / seq[0][0] - 1.0] + [ curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]



    # split into groups of num_steps
    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])

    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

    # get only close value
    y = [y[i][1] for i in range(len(y))]

    train_size = int(len(X) * (1.0 - test_ratio))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    return train_X, train_y, test_X, test_y

def generate_batches(train_X,train_y,batch_size):
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

train_X, train_y, test_X, test_y = pre_process()


for batch_X, batch_y in generate_batches(train_X,train_y,config.batch_size):
    print(batch_X)
    print(batch_y)


