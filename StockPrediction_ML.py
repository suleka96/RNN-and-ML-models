import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.linear_model import LinearRegression


class RNNConfig():
    input_size = 1
    num_steps = 2
    lstm_size = 128
    num_layers = 1
    keep_prob = 0.8
    batch_size = 64
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 3  # 5
    max_epoch = 30  # 100 or 50
    features = 2
    test_ratio = 0.2
    fileName = 'AIG.csv'
    graph = tf.Graph()
    column1_min = 10
    column1_max = 2000
    column2_min = 0
    column2_max = 50000000
    column1 = 'Close'
    column2 = 'Volume'


config = RNNConfig()


def segmentation(data):

    seq = [price for tup in data[[config.column2, config.column1]].values for price in tup]

    seq = np.array(seq)

    # split into items of features
    seq = [np.array(seq[i * config.features: (i + 1) * config.features])
           for i in range(len(seq) // config.features)]

    # split into groups of num_steps
    temp_X = np.array([seq[i: i + config.num_steps] for i in range(len(seq) -  config.num_steps)])
    X = []

    for dataslice in temp_X:
        temp = dataslice.flatten()
        X.append(temp)

    X = np.asarray(X)

    y = np.array([seq[i +  config.num_steps] for i in range(len(seq) -  config.num_steps)])

    # get only close values
    y = [y[i][1] for i in range(len(y))]

    y = np.asarray(y)

    return X, y

def scale(data):

    data[config.column1] = (data[config.column1] - config.column1_min) / (config.column1_max - config.column1_min)

    data[config.column2] = (data[config.column2] - config.column2_min) / (config.column2_max - config.column2_min)

    return data


def pre_process():

    stock_data = pd.read_csv(config.fileName)
    stock_data = stock_data.reindex(index=stock_data.index[::-1])

    # ---for segmenting original data ---------------------------------
    original_data = pd.read_csv(config.fileName)
    original_data = original_data.reindex(index=original_data.index[::-1])

    train_size = int(len(stock_data) * (1.0 - config.test_ratio))

    train_data = stock_data[:train_size]
    test_data = stock_data[train_size:]
    original_data = original_data[train_size:]

    # -------------- processing train data---------------------------------------

    scaled_train_data = scale(train_data)
    train_X, train_y = segmentation(scaled_train_data)

    # -------------- processing test data---------------------------------------

    scaled_test_data = scale(test_data)
    test_X, test_y = segmentation(scaled_test_data)

    # ----segmenting original test data-----------------------------------------------

    nonescaled_X, nonescaled_y = segmentation(original_data)

    return train_X, train_y, test_X, test_y, nonescaled_y

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot(true_vals,pred_vals,name):

    days = range(len(true_vals))
    plt.plot(days, true_vals, label='truth close')
    plt.plot(days, pred_vals, label='pred close')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("closing price")
    plt.grid(ls='--')
    plt.savefig(name, format='png', bbox_inches='tight', transparent=False)
    plt.close()

def get_scores(name,pred_vals,nonescaled_y):

    print(name)
    meanSquaredError = mean_squared_error(nonescaled_y, pred_vals)
    rootMeanSquaredError = sqrt(meanSquaredError)
    print("RMSE:", rootMeanSquaredError)
    mae = mean_absolute_error(nonescaled_y, pred_vals)
    print("MAE:", mae)
    mape = mean_absolute_percentage_error(nonescaled_y, pred_vals)
    print("MAPE:", mape)

def Linear_Regression():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    reg = LinearRegression().fit(train_X, train_y)

    prediction = reg.predict(test_X)

    pred_vals, nonescaled_y = prediction, test_y
    # pred_vals = [(pred * (config.column1_max - config.column1_min)) + config.column1_min for pred in prediction]

    # pred_vals = np.asarray(pred_vals)

    get_scores("---------Linear Regression----------",pred_vals, nonescaled_y)

    plot(nonescaled_y,pred_vals,"Liner Regression Prediction Vs Truth.png")


def Random_Forest_Regressor():

    train_X, train_y, test_X, test_y, nonescaled_y = pre_process()

    rfr = RandomForestRegressor(random_state=42).fit(train_X, train_y)

    rfr_prediction = rfr.predict(test_X)

    pred_vals = [(pred * (config.column1_max - config.column1_min)) + config.column1_min for pred in rfr_prediction]

    pred_vals = np.asarray(pred_vals)

    get_scores("---------Random Forest Regressor----------",pred_vals,nonescaled_y)

    plot(nonescaled_y,pred_vals,"Random Forest Regressor Prediction Vs Truth.png")


if __name__ == '__main__':
    Linear_Regression()
    Random_Forest_Regressor()