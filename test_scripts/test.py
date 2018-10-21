# # import os
# #
# #
# # direc = "enron/emails/"
# # files = os.listdir(direc)
# # emails = [direc + email for email in files]
# # print(len(emails))
# #
# # feature_set = []
# # labels = []
# # counter = 0
# # emailCounter = 0
# #
# # for email in emails:
# #     if "ham" in email:
# #         labels.append(0)
# #         counter += 1
# #         print("ham", counter)
# #     else:
# #         labels.append(1)
# #         counter += 1
# #         print("spam ", counter)
# #     print(email)
# #     print("email counter", emailCounter)
# #     emailCounter += 1
# #
# # print(len(labels))
# # print(labels)
#
# import tensorflow as tf
# from tensorflow.contrib import rnn
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn.metrics import f1_score, accuracy_score
#
#
#
# # hm_epochs = 30
# # n_classes = 1
# # rnn_size = 200
# # col_size = 30
# # batch_size = 24
# # fileName = "creditcard.csv"
# #
# #
# # data = pd.read_csv('creditcard.csv', skiprows=[0], header=None)
# # features = data.iloc[:, 1:30]
# # labels = data.iloc[:, -1]
# # print(features.head)
#
#
#
#
#
#
# # data_1 = data.loc[data.iloc[:, -1] == 1]
# # print(len(data_1))
# # data_0 = data.loc[data.iloc[:, -1] == 0]
# # print(len(data_0))
# # data_0 = data_0.head(400)
# # totdata = data_1.append(data_0, ignore_index=True)
# # totdata = totdata.sample(frac=1)
# #
# # batch_x = np.array(features)
# # batch_y = np.array(labels)
#
# # print(batch_x)
# # print(batch_y)
#
# # X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, random_state=4)
# # print (len(X_train))
# # print(len(X_test))
#
#
#
# # batch_x = np.array(X_train)
# # batch_y = np.array(y_train)
# # print(len(X_train))
# # print(len(X_test))
# # print(y_train.shape)
# #
# #
# # print("The factors of are:")
# # for i in range(1, (len(X_train)) + 1):
# #     if (len(X_train)) % i == 0:
# #            print(i)
# #
# # print("The factors of are:")
# # for i in range(1, (len(X_test)) + 1):
# #     if (len(X_test)) % i == 0:
# #            print(i)
# #
# #
# # batch_size= 24
# # i = 0
# # while i < len(features):
# #     start = i
# #     end = i + batch_size
# #
# #     batch_x = np.array(X_train[start:end])
# #     batch_y = np.array(y_train[start:end])
# #     i += batch_size
# # #
# #
# #
# #
# #
# #
# #
# #
# # define a function
# def print_factors(x):
#    # This function takes a number and prints the factors
#
#    print("The factors of",x,"are:")
#    for i in range(1, x + 1):
#        if x % i == 0:
#            print(i)
#
# # change this value for a different result.
# num = 320
#
# print_factors(18846)
#
# # 1
# # 3
# # 5
# # 9
# # 15
# # 27
# # 45
# # 135
# # 179
# # 537
# # 895
# # 1611
# # 2685
# # 4833
# # 8055
# # 24165
#
#24165
#
#
# #
# # def computeGCD(a, b):
# #     if a < b:
# #         smaller = a
# #     else:
# #         smaller = b
# #
# #     for i in range(1, smaller + 1):
# #         if (a % i == 0) & (b % i == 0):
# #             gcd = i
# #
# #     return gcd
# #
# #
# # print(computeGCD(24, 213605))
# # print(computeGCD(48, 71202))














# import os
# from collections import Counter
# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import f1_score, recall_score, precision_score
# from string import punctuation
# from sklearn.model_selection import train_test_split
#
#
# def pre_process():
#     direc = "enron/emails/"
#     files = os.listdir(direc)
#     emails = [direc+email for email in files]
#
#     words = []
#     temp_email_text = []
#     labels = []
#
#     for email in emails:
#         if "ham" in email:
#             labels.append(0)
#         else:
#             labels.append(1)
#         f = open(email,encoding="utf8", errors='ignore')
#         blob = f.read()
#         all_text = ''.join([text for text in blob if text not in punctuation])
#         all_text = all_text.split('\n')
#         all_text = ''.join(all_text)
#         temp_text = all_text.split(" ")
#
#         for word in temp_text:
#             if word.isalpha():
#                 temp_text[temp_text.index(word)] = word.lower()
#
#         temp_text = list(filter(None, temp_text))
#         temp_text = ' '.join([i for i in temp_text if not i.isdigit()])
#         words += temp_text.split(" ")
#         temp_email_text.append(temp_text)
#
#     dictionary = Counter(words)
#     #deleting spaces
#     del dictionary[""]
#     sorted_split_words = sorted(dictionary, key=dictionary.get, reverse=True)
#     vocab_to_int = {c: i for i, c in enumerate(sorted_split_words, 1)}
#
#     message_ints = []
#     for message in temp_email_text:
#         temp_message = message.split(" ")
#         message_ints.append([vocab_to_int[i] for i in temp_message])
#
#     # print(temp_email_text[0])
#     # print(labels[:10])
#     # print(message_ints[0])
#     # print("\n")
#     # print(len(temp_email_text[0]))
#     # print(len(message_ints[0]))
#
#     #maximum message length = 3423
#
#     message_lens = Counter([len(x) for x in message_ints])
#     # print("Zero-length messages: {}".format(message_lens[0]))
#     # print("Maximum message length: {}".format(max(message_lens)))
#
#     seq_length = 3425
#     num_messages = len(temp_email_text)
#     features = np.zeros([num_messages,seq_length], dtype=int)
#     for i, row in enumerate(message_ints):
#         features[i, -len(row):] = np.array(row)[:seq_length]
#
#     # print(len(features[0]))
#     # print(len(features[1]))
#     # blah = list(enumerate(message_ints))
#     # print(blah[:2])
#
#     return features, np.array(labels), sorted_split_words
#
#
# def get_batches(x, y, batch_size=100):
#     n_batches = len(x) // batch_size
#     # x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
#     print(n_batches)
#     batch_counter = 0
#     ii = 0
#     #start at 0 increment by catch_size and end at len(x)
#     # for ii in range(0, len(x), batch_size):
#
#     while(ii != len(x)):
#
#         if(batch_counter == n_batches):
#             yield x[ii:], y[ii:]
#             print("number of tuples left")
#             print(len(x)-ii)
#             ii = len(x)
#
#         else:
#             yield x[ii:ii + batch_size], y[ii:ii + batch_size]
#             ii += batch_size
#
#         batch_counter += 1
#         print(batch_counter)
#
#
# features, labels, sorted_split_words = pre_process()
#
# # splitting training, validation and testing sets
#
# split_frac1 = 0.8
#
# idx1 = int(len(features) * split_frac1)
# train_x, val_x = features[:idx1], features[idx1:]
# train_y, val_y = labels[:idx1], labels[idx1:]
#
# split_frac2 = 0.5
#
# idx2 = int(len(val_x) * split_frac2)
# val_x, test_x = val_x[:idx2], val_x[idx2:]
# val_y, test_y = val_y[:idx2], val_y[idx2:]
#
# print("\t\t\tFeature Shapes:")
# print("Train set: \t\t{}".format(train_x.shape),
#       "\nValidation set: \t{}".format(val_x.shape),
#       "\nTest set: \t\t{}".format(test_x.shape))
#
# print("\t\t\Label Shapes:")
#
# print("Train set: \t\t{}".format(train_y.shape),
#       "\nValidation set: \t{}".format(val_y.shape),
#       "\nTest set: \t\t{}".format(test_y.shape))
#
# for ii, (x, y) in enumerate(get_batches(np.array(train_x), np.array(train_y), 197), 1):
#     print(x.shape)
#     print(y.shape)
#     print()
#

# x_axiz_val = []
# y = [1,2,3,4,5,6,7,8,9,10]
#
# for i in range(1, len(y)+1, 1):
#     x_axiz_val.append(i)
#     print(i)

#
#
# #using MultinomialNB algorithm
#
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import metrics
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cross_validation import train_test_split
# from sklearn.cross_validation import cross_val_score
# import numpy as np
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import log_loss, accuracy_score
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.decomposition import TruncatedSVD
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# import re
# from nltk.corpus import stopwords
# import os
#
#
# newsgroups_train = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
#
# vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize,
#                         strip_accents='unicode',
#                         lowercase =True, analyzer='word',
#                         use_idf=True, smooth_idf=True, sublinear_tf=False,
#                         stop_words = 'english')
#
# print(newsgroups_train.data[0])
#
# vectorizer.fit(newsgroups_train.data)
# # summarize
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# # encode document
# vector = vectorizer.transform(newsgroups_train.data)
# # summarize encoded vector
# print(vector.shape)
# print(vector.toarray())


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


pre_process()
train_test()










