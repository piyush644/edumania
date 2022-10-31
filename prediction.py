#modules
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def predict_results(symbol, day):
    #inputs
    crypto = symbol            # -> user input
    currency = 'USD'
    start = dt.datetime(2017,1,1)
    end = dt.datetime.now()

    #getting data
    info = pdr.DataReader(f'{crypto}-{currency}', 'yahoo', start, end)

    #new column for prediction
    info['Prediction'] = info[['Close']].shift(-day)

    #creation of independent numpy array
    in_A = np.array(info[['Close']])
    in_A = in_A[:-day]  #remove the last 'n' rows

    #creation of dependent numpy array
    de_A = info['Prediction'].values
    de_A = de_A[:-day]

    #division of data into training and testing data
    xtrain, xtest, ytrain, ytest = train_test_split(in_A, de_A, test_size = .1)

    #creation model
    lr = LinearRegression()
    #train model
    lr.fit(xtrain, ytrain)

    #testing
    lr_con = lr.score(xtest, ytest)

    A_pro = np.array(info[['Close']])[-day:]

    #print model's predictions for next 'n' days
    return lr.predict(A_pro)
