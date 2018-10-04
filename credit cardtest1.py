import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

data = pd.read_csv('creditcard.csv', skiprows=[0], header=None)

rows_list = []
index = 0

print(data.head(20))


dict1 = {}
# get input row in dictionary format
# key = col_name
dict1.update(data.iloc[[0]])
rows_list.append(dict1)

l = data.iloc[:, -1]

# for temp in l:
#     if temp == 1:
#         data_0 =
#         print(data_0)
#     index += 1
#
# print(index)
# print(Data)
# print(Data.shape)

# for row in l:
#     if row == 1:
#
#         dict1 = {}
#         # get input row in dictionary format
#         # key = col_name
#         dict1.update(data.iloc[[index]])
#         rows_list.append(dict1)
#     index += 1
#
# df = pd.DataFrame(rows_list)
# print(df)
# print(df.shape)
#
# features = df.iloc[:, 1:30]
# labels = df.iloc[:, -1]
#
# print(np.array(features.head(10)))
# print(np.array(labels.head(10)))

