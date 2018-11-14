from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pylab as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


class RNNConfig():
    lag_order = 2
    degree_differencing = 1
    order_moving_avg = 0
    test_ratio = 0.2
    fileName = 'store285.csv'
    min = 0
    max = 50000
    column = 'Sales'
    store = 285

config = RNNConfig()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def scale(data):

    data[config.column] = (data[config.column] - config.min) / (config.max - config.min)

    return data

def plot(original_test_list,pred_vals):
    days = range(len(original_test_list))
    plt.plot(days, original_test_list, color='blue', label='truth close')
    plt.plot(days, pred_vals, color='red', label='pred close')
    plt.yscale('log')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("closing price")
    plt.grid(ls='--')
    plt.savefig("Sales ARIMA Prediction Vs Truth log.png", format='png', bbox_inches='tight', transparent=False)

def preprocess():

    stock_data = pd.read_csv(config.fileName)

    stock_data = stock_data.drop(stock_data[(stock_data.Open == 0) & (stock_data.Sales == 0)].index)

    stock_data = stock_data.drop(stock_data[(stock_data.Open != 0) & (stock_data.Sales == 0)].index)

    store_data_scale = stock_data
    store_data_orginal = stock_data.copy()

    scaled_data = scale(store_data_scale)
    sales = scaled_data[config.column]

    # ---for segmenting original data ---------------------------------
    nonescaled_sales= store_data_orginal[config.column]

    size = int(len(stock_data) * (1.0 - config.test_ratio))
    train, test = sales[:size], sales[size:]
    original_train, original_test = nonescaled_sales[:size], nonescaled_sales[size:]

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







