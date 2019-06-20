# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:30:01 2019

@author: baichen
"""
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from itertools import combinations,product

start = pd.to_datetime('2014-6-1')
end = pd.to_datetime('2019-5-25')
s = 'AMZN'

df = web.DataReader(s, 'yahoo', start=start, end=end)


# Kalman filter class
class KalmanFilter(object):
    def __init__(self, a, b, c, x, p, q, r):
        self.a = a
        self.b = b
        self.c = c
        self.current_state_estimate = x
        self.current_prob_estimate = p
        self.q = q  # process covariace
        self.r = r  # measurement covariance
        self.predicted_state_estimate = 0
        self.observation = 0

    def current_state(self):
        return self.current_state_estimate

    def predicted_state(self):
        return self.predicted_state_estimate

    def observe(self):
        return self.observation

    def process(self, control_input, measurement):
        # prediction
        self.predicted_state_estimate = self.a * self.current_state_estimate + self.b * control_input
        predicted_prob_estimate = (self.a * self.current_prob_estimate) * self.a + self.q

        # observation
        self.observation = measurement - self.c * self.predicted_state_estimate
        observation_cov = self.c * predicted_prob_estimate * self.c + self.r

        # update
        kalman_gain = predicted_prob_estimate * self.c / float(observation_cov)
        self.current_state_estimate = self.predicted_state_estimate + kalman_gain * self.observation

        self.current_prob_estimate = (1 - kalman_gain * self.c) * predicted_prob_estimate


X = df['Close'].values
size = int(len(X) * 0.6)
train, test = X[0:size], X[size:len(X)]

# acf
def acf():
    acf1 = acf(train)
    acf1 = pd.DataFrame([acf1]).T
    acf1.plot(kind='bar', figsize=(12, 10))
    plt.show();

# pacf
def pacf():
    pacf1 = pacf(train)
    pacf1 = pd.DataFrame([pacf1]).T
    pacf1.plot(kind='bar', figsize=(12, 10))
    plt.show();  # AR(1) is concerned

# test 1st differencing acf
def first_diff():
    global price_diff
    df1 = df.iloc[:size, :]
    df2 = df.iloc[:size, :].shift()
    price_diff = df1['Close'] - df2['Close']
    price_diff = price_diff.dropna()
    acf1_diff = acf(price_diff)
    acf1_diff = pd.DataFrame([acf1_diff]).T

    acf1_diff.plot(kind='bar', figsize=(12, 10))
    plt.show();

# test 1st differencing pacf
def first_diff_pacf():
    pacf1_diff = pacf(price_diff)
    pacf1_diff = pd.DataFrame([pacf1_diff]).T
    pacf1_diff.plot(kind='bar', figsize=(12, 10))
    plt.show();

# arima
def arima_summary():
    price = train
    model = ARIMA(price, order=(0, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    pred_arima = model_fit.predict(start=size + 1, end=size + 1, typ='levels')  # predict value at size+1

# ARIMA rolling forecast
def arima():
    global predicts
    history = [x for x in train]
    predicts = []
    for t in range(len(test)):
        model = ARIMA(history, order=(7, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predicts.append(yhat)
        obs = test[t]
        history.append(obs)  # update history data
        print('%.1f : ARIMA predicted=%f, expected=%f' % (t, yhat, obs))

    plt.figure(figsize=(12, 10))
    plt.plot(test)
    plt.plot(predicts, color='red')
    plt.show();
    print('predict', predicts)

# kalman filter part
def kalman_filter(data,a=1,b=0,c=1,q=0.01,r=1,x=200,p=1):   # try to control r
    filter = KalmanFilter(a, b, c, x, p, q, r)
    predictions = []
    estimate = []
    observe = []
    for d in data:
        filter.process(0, d)
        predictions.append(filter.current_state())
        estimate.append(filter.predicted_state())
        observe.append(filter.observe())

    predictions = [float(i) for i in predictions]
    estimate = [float(i) for i in estimate]
    observe = [float(i) for i in observe]

    # plt.figure(figsize=(12, 10))
    # plt.plot(data)
    # plt.plot(predictions)
    # plt.show();

    return predictions

def sim_kf():
    x = np.linspace(0, 10, num=200)
    y = -2 * x ** 2 + 10* np.random.standard_normal(len(x))
    filter = KalmanFilter(1, 0, 1, 100, 1, 1, 1)
    predictions = []
    for d in y:
        filter.process(0, d)
        predictions.append(filter.current_state())
    predictions = [float(i) for i in predictions]

    y=pd.Series(y)
    predictions=pd.Series(predictions)
    df=pd.concat([y,predictions],axis=1,keys=['Data','Kalman Filter'])
    df['Moving Average']=df['Data'].rolling(5).mean()

    df.plot(figsize=(12,10))
    plt.show();

def VWAP(df,window=20):

    train,test=df.iloc[:size,:],df.iloc[size:,:]


    train['price_vol']=train['Close']*train['Volume']
    train['VWAP']=train['price_vol'].rolling(window).sum()/train['Volume'].rolling(window).sum()

    predictions=kalman_filter(train['Close'].values)
    train['kalman']=np.array(predictions)


    # train[['VWAP','Close','kalman']].plot(figsize=(12,10))
    # plt.title('%s Training data'%s)
    # plt.show();
    return train[['VWAP','Close','kalman']]

def rsi(df,rsi_period=20):
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    chg=train['Close'].diff(1)
    gain=chg.mask(chg<0,0)
    loss=chg.mask(chg>0,0)

    avg_gain=gain.ewm(com=rsi_period-1,min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs=abs(avg_gain)/abs(avg_loss)
    rsi=100-100/(1+rs)
    train['RSI']=rsi

    fig=plt.figure(figsize=(12,10))

    ax1=fig.add_subplot(211)
    plt.plot(VWAP(df))


    ax2=fig.add_subplot(212,sharex = ax1)
    plt.plot(rsi)
    plt.axhline(y=40,color='r',linestyle='--')
    plt.axhline(y=70, color='r', linestyle='--')
    ax1.title.set_text('Price')
    ax2.title.set_text('RSI')
    plt.show();

def Volume_train():
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(211)
    plt.plot(train['Close'])

    ax2 = fig.add_subplot(212, sharex=ax1)
    plt.plot(train['Volume'])

    ax1.title.set_text('Price')
    ax2.title.set_text('Volume')
    plt.show();
def Cross_MA(df,window1,window2): # window1<window2
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    train['MA_%.f'%window1]=train['Close'].rolling(window1).mean()
    train['MA_%.f' % window2] = train['Close'].rolling(window2).mean()

    train[['Close','MA_%.f'%window1,'MA_%.f'%window2]].plot(figsize=(12,10))
    plt.show();

def SMA_train_performace(df,window1=15,window2=25):
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    train['Return']=train['Close'].pct_change()
    train['MA_%.f'%window1]=train['Close'].rolling(window1).mean()
    train['MA_%.f' % window2] = train['Close'].rolling(window2).mean()
    train['long']=np.where(train['MA_%.f'%window1]>train['MA_%.f' % window2],1,0)
    train['short']=np.where(train['MA_%.f'%window1]<train['MA_%.f' % window2],-1,0)
    train['positions']=train['long']+train['short']

    train['strategy_return']=train['positions'].shift(1)*train['Return']
    annual_return=train['strategy_return'].mean()*252
    annual_std=train['strategy_return'].std()*np.sqrt(252)
    sharpe_ratio=annual_return/annual_std
    print('SMA {} and SMA {} ~~ Annual return: {:.2f} , Annual std: {:.2f} , Sharpe Ratio: {:.2f}'.
          format(window1,window2,annual_return,annual_std,sharpe_ratio))
    #plot
    ax = train[['Close','MA_%.f'%window1,'MA_%.f' % window2, 'positions']].plot(figsize=(12, 10), secondary_y=['positions'])
    plt.title('{} training data \nSMA trading signal \nsharpe ratio: {:.2f}'.format(s, sharpe_ratio))
    plt.show();

def SMA_train_Optimize(windows=[10,15,20,25,30,40,50,60]):
    win1_win2=list(combinations(windows,r=2))
    for win1,win2 in win1_win2:
        SMA_train_performace(df,win1,win2)

def SMA_test(df,window1,window2):
    pass

def kalman_train_performance(r1=0.3,r2=3):
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    train['Return'] = train['Close'].pct_change()
    predictions1=kalman_filter(train['Close'].values,r=r1)
    predictions2 = kalman_filter(train['Close'].values, r=r2)
    train['kalman_r1']=np.array(predictions1)
    train['kalman_r2'] = np.array(predictions2)

    train['kf_long']=np.where(train['kalman_r1']>train['kalman_r2'],1,0)
    train['kf_short']=np.where(train['kalman_r1']<train['kalman_r2'],-1,0)
    train['kf_positions']=train['kf_long']+train['kf_short']
    train['kf_strategy_return'] = train['kf_positions'].shift(1) * train['Return']

    annual_return=train['kf_strategy_return'].mean()*252
    annual_std=train['kf_strategy_return'].std()*np.sqrt(252)
    sharpe_ratio=annual_return/annual_std
    print('r1:{:.1f} and r2:{:.1f} ~~ Annual return: {:.6f} , Annual std: {:.6f} , Sharpe Ratio: {:.2f}'.
          format(r1, r2, annual_return, annual_std, sharpe_ratio))
    ax=train[[ 'Close', 'kalman_r1','kalman_r2','kf_positions']].plot(figsize=(12, 10), secondary_y=['kf_positions'])
    plt.title('{} Kalman Filter training data \nr1= {:.1f}  r2= {:.1f} \nsharpe ratio: {:.2f}'.format(s,r1,r2, sharpe_ratio))
    plt.show();

def kalman_train_optimize(r1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],r2=[1,2,3,4,5,6,7,8,9,10]):
    r1_r2=list(product(r1,r2))
    for r1,r2 in r1_r2:
        kalman_train_performance(r1,r2)

def kalman_test(r1,r2):
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    test['Return'] = test['Close'].pct_change()
    predictions1 = kalman_filter(test['Close'].values, r=r1)
    predictions2 = kalman_filter(test['Close'].values, r=r2)
    test['kalman_r1'] = np.array(predictions1)
    test['kalman_r2'] = np.array(predictions2)

    test['kf_long'] = np.where(test['kalman_r1'] > test['kalman_r2'], 1, 0)
    test['kf_short'] = np.where(test['kalman_r1'] < test['kalman_r2'], -1, 0)
    test['kf_positions'] = test['kf_long'] + test['kf_short']
    test['kf_strategy_return'] = test['kf_positions'].shift(1) * test['Return']

    annual_return = test['kf_strategy_return'].mean() * 252
    annual_std = test['kf_strategy_return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std
    print('r1:{:.1f} and r2:{:.1f} ~~ Annual return: {:.6f} , Annual std: {:.6f} , Sharpe Ratio: {:.2f}'.
          format(r1, r2, annual_return, annual_std, sharpe_ratio))
    ax = test[['Close', 'kalman_r1', 'kalman_r2', 'kf_positions']].plot(figsize=(12, 10), secondary_y=['kf_positions'])
    plt.title('{} Kalman Filter test data \nr1= {:.1f}  r2= {:.1f} \nsharpe ratio: {:.2f}'.format(s, r1, r2,
                                                                                                      sharpe_ratio))
    plt.show();
def main():

    #VWAP(df)
    #rsi(df)
    #Cross_MA(df,15,25)
    #SMA_train_performace(df,20,50)
    #SMA_train_Optimize()
    #kalman_train_performance()
    #kalman_train_optimize()
    Volume_train()
    #kalman_train_optimize()
    #kalman_test(r1=0.3,r2=2)
    sim_kf()
main()