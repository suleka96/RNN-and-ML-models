import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score

data = pd.read_csv('spam.csv', encoding='latin1')
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']

data['v1'] = data['v1'].replace(['ham','spam'], [0, 1])
y = data['v1'].as_matrix()
X_text = data['v2'].as_matrix()
print(X_text.shape)
print(y.shape)
print(X)