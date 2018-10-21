from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


class RNNConfig():
    lag_order = 2
    degree_differencing = 1
    order_moving_avg = 0
    scaler = MinMaxScaler(feature_range=(0,1))
    test_ratio = 0.2
    fileName = 'AIG.csv'
    min = 10
    max = 2000
    column = 'Close'

config = RNNConfig()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def scale(data):

    data[config.column] = (data[config.column] - config.min) / (config.max - config.min)

    return data

def plot(original_test_list,pred_vals):
    days = range(len(original_test_list))
    pyplot.plot(days, original_test_list, color='blue', label='truth close')
    pyplot.plot(days, pred_vals, color='red', label='pred close')
    pyplot.legend(loc='upper left', frameon=False)
    pyplot.xlabel("day")
    pyplot.ylabel("closing price")
    pyplot.grid(ls='--')
    pyplot.savefig("ARIMA Prediction Vs Truth.png", format='png', bbox_inches='tight', transparent=False)

def preprocess():
    stock_data = pd.read_csv(config.fileName)
    stock_data = stock_data.reindex(index=stock_data.index[::-1])

    scaled_data = scale(stock_data)
    price = scaled_data[config.column]

    # ---for segmenting original data ---------------------------------
    original_data = pd.read_csv(config.fileName)
    original_data = original_data.reindex(index=original_data.index[::-1])
    nonescaled_price = original_data[config.column]

    size = int(len(stock_data) * (1.0 - config.test_ratio))
    train, test = price[:size], price[size:]
    original_train, original_test = nonescaled_price[0:size], nonescaled_price[size:len(price)]

    return train,test,original_train,original_test

def ARIMA_model():

    train, test, original_train, original_test = preprocess()

    predictions = list()
    history = [x for x in train]
    test_list= test.tolist()
    original_test_list = original_test.tolist()


    for t in range(len(test_list)):
        model = ARIMA(history, order=(config.lag_order, config.degree_differencing, config.order_moving_avg))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_list[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    pred_vals = [(pred[0] * (config.max - config.min)) + config.min for pred in predictions]

    meanSquaredError = mean_squared_error(original_test_list, pred_vals)
    rootMeanSquaredError = sqrt(meanSquaredError)
    print("RMSE:", rootMeanSquaredError)
    mae = mean_absolute_error(original_test_list, pred_vals)
    print("MAE:", mae)
    mape = mean_absolute_percentage_error(original_test_list, pred_vals)
    print("MAPE:", mape)

    plot(original_test_list,pred_vals)


if __name__ == '__main__':
    ARIMA_model()







