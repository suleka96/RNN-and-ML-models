import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import csv
from itertools import zip_longest

class RNNConfig():
    iterator = 0

config = RNNConfig()

def reverse():

    fields = ["Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"]
    with open('processed_train.csv', mode='a') as stock_file:
        writer = csv.writer(stock_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fields)

    for chunk in pd.read_csv("train.csv", chunksize=10):
        store_data = chunk.reindex(index=chunk.index[::-1])
        append_data_csv(store_data)

def append_data_csv(store_data):
    config.iterator += 1
    with open('processed_train.csv', mode='a') as store_file:
        writer = csv.writer(store_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if config.iterator == 1:
            for index, row in store_data.iterrows():
                writer.writerow([row['Store'],row['DayOfWeek'],row['Date'],row['Sales'],row['Customers'],row['Open'],row['Promo'],row['StateHoliday'],row['SchoolHoliday']])
        else:
            read_data = pd.read_csv('processed_train.csv')
            data = []
            for index, row in store_data.iterrows():
                data.append({'Store': row['Store'], 'DayOfWeek': row['DayOfWeek'],'Date' :row['Date'], 'Sales': row['Sales'],'Customers': row['Customers'],'Open': row['Open'],'Promo': row['Promo'],'StateHoliday': row['StateHoliday'],'SchoolHoliday': row['SchoolHoliday']})
            tempframe=pd.DataFrame(data)
            tempframe = tempframe.reindex_axis(read_data.columns, axis=1)
            f =pd.concat([tempframe, read_data],sort=True)
            print(f)


# , ignore_index=True,sort=True

def tryit():
    # stock_data = pd.read_csv("train.csv")
    # stock_data = stock_data.reindex(index=stock_data.index[::-1])
    # stock_data.to_csv('processed_train.csv', sep='\t', encoding='utf-8')

    # stock_data.index
    # print(stock_data.head())
    # for chunk in pd.read_csv("train.csv", chunksize=10):
    #     stock_data.index

    # stock_data = stock_data.drop( stock_data[(stock_data.Open == 0) & (stock_data.Sales == 0)].index)
    #
    # stock_data = stock_data.drop( stock_data[(stock_data.Open != 0) & (stock_data.Sales == 0)].index)

    ###################################################################################################

    # stock_data = pd.read_csv('processed_train.csv')
    #
    # datatof = stock_data[(stock_data.Store == 165)]
    #
    # datatof.to_csv('store285_test.csv', sep=',', encoding='utf-8')
    #
    # return stock_data

    ###################################################################################################

    #extracting date year and month from date column

    stock_data = pd.read_csv('store285_test.csv')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    stock_data['Year'] = stock_data['Date'].dt.year
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['Day'] = stock_data['Date'].dt.day

    with open(r'store285_test.csv', 'r') as f, open(r'store285_2.csv','w') as g:
        fr = csv.reader(f)
        gw = csv.writer(g)
        gw.writerow(next(fr))
        gw.writerows(a + [b] + [c] + [d] for a, b, c,d in zip_longest(fr,  stock_data['Year'], stock_data['Month'],  stock_data['Day'], fillvalue=[0]))








# reverse()

f = tryit()