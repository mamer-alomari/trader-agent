import os
import math
import logging

import pandas as pd
import numpy as np

import keras.backend as K


# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    # filter out the desired features
    
    # rename feature column names
    for i in df.columns: 
        if i == "Date": 
            df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            df = df.rename(columns={'Close': 'actual', 'Date': 'date' , 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'})
        elif i == "timestamp":
            df = df.rename(columns={'close': 'actual', 'timestamp': 'date'})
            df = df[['date', 'actual', 'high', 'low', 'open', 'volume']]

    df = df[:100]
    # df = df.rename(columns={'Close': 'actual', 'Date': 'date'})
    # convert dates from object to DateTime type
    df["combined"] = (np.log(df["actual"] + df["high"] + df["low"] + df["open"] / 4))

    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df['date'] = dates

    df.head()

    return list(df['combined']) ,


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
