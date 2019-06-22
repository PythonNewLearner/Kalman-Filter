import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from itertools import combinations,product

start = pd.to_datetime('2018-1-1')
end = pd.to_datetime('2019-1-1')
s = 'SPY'

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


X = df['Adj Close'].values
size = int(len(X) * 0.6)
train, test = X[0:size], X[size:len(X)]

# kalman filter part
def kalman_filter(data,a=1,b=0,c=1,q=0.1,r=1,x=185,p=1):   # try to control r
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

def buy_hold():
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    train['Return'] = train['Close'].pct_change()
    annual_return=train['Return'].mean()*252
    annual_std=train['Return'].std()*np.sqrt(252)
    sharpe_ratio=annual_return/annual_std
    print("Buy& Hold annual return:",annual_return)
    print("Buy & Hold sharpe ratio:",sharpe_ratio)

def kalman_filter_pred_current(data,a=1,b=0,c=1,q=0.1,r=1,x=165,p=1):   # try to control r
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

    return predictions,estimate

def kf_pred_current_train(q):
    global train, test
    df = web.DataReader(s, 'yahoo', start=start, end=end)
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    train['Return'] = train['Adj Close'].pct_change()
    prediction,estimate=kalman_filter_pred_current(train['Adj Close'].values,q=q)
    train['estimate']=np.array(estimate)
    train['current_prediction']=np.array(prediction)


    train['kf_long'] = np.where(train['estimate'] < train['current_prediction'],1,0)
    train['kf_short'] = np.where(train['estimate'] > train['current_prediction'],-1, 0)
    train['kf_positions'] = train['kf_long'] + train['kf_short']
    train['kf_strategy_return'] = train['kf_positions'].shift(1) * train['Return']
    print(train[['kf_long','kf_short','kf_positions','estimate','current_prediction','kf_strategy_return','Adj Close']])

    annual_return = train['kf_strategy_return'].mean() * 252
    annual_std = train['kf_strategy_return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std
    print("annual return:", annual_return)
    print("sharpe ratio:", sharpe_ratio)

    print(' Annual return: {:.6f} , Annual std: {:.6f} , Sharpe Ratio: {:.2f}'.
          format(annual_return, annual_std, sharpe_ratio))

    return sharpe_ratio,annual_return,annual_std

def kf_pred_current_train_plot(q):
    sharpe_ratio, annual_return, annual_std=kf_pred_current_train(q=q)
    ax=train[['Adj Close','estimate','current_prediction','kf_positions']].plot(figsize=(12, 10), secondary_y=['kf_positions'])
    plt.title('{} Kalman Filter training data  \nsharpe ratio: {:.2f}'.format(s, sharpe_ratio,annual_return,annual_std))
    plt.show();
def kf_pred_current_train_optimize(): # to be continued
    Q=np.arange(0, 20, step=0.5)
    print(Q)
    SR=[]
    for q in Q:
        print("Process Covariance(Q): ",q)
        sr,annual_ret,annual_std=kf_pred_current_train(q=q)
        SR.append(sr)
    #SR=np.array(SR)

    # plt.figure(figsize=(12,10))
    # plt.plot(Q,SR)
    # plt.xlabel('Process Covariance (Q)')
    # plt.ylabel('Sharpe Ratio')
    # #plt.xticks(Q)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(Q, SR)

    ymax = max(SR)
    xpos = SR.index(ymax)
    xmax = Q[xpos]

    text = "Q={:.3f}, Sharpe Ratio={:.3f}".format(xmax, ymax)

    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax, ymax),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    plt.xlabel('Process Covariace(Q) ')
    plt.ylabel('Sharpe Ratio')
    ax.set_xlim(0, 20)
    plt.show()

def rankc(df, Filter):
    df1 = df[['Adj Close', 'Volume', 'Return']]
    df1.loc[:, 'MA5'] = df1.loc[:, 'Adj Close'].rolling(5).mean()
    df1.loc[:, 'MA10'] = df1.loc[:, 'Adj Close'].rolling(10).mean()
    df1.loc[:, 'MA20'] = df1.loc[:, 'Adj Close'].rolling(20).mean()
    df1.loc[:, 'MA30'] = df1.loc[:, 'Adj Close'].rolling(30).mean()
    # df1.loc[:,'MA60']=df1.loc[:,'Adj Close'].rolling(60).mean()

    # Calculate rank correlation
    from scipy.stats import spearmanr
    df1.loc[:, 'Rank Coefficient'] = np.nan
    rank = df1.iloc[:, 3:7].rank(axis=1).values
    for i in range(len(df1.index) - 30):
        i += 30
        data1 = [4, 3, 2, 1]
        coef, p = spearmanr(data1, rank[i])
        j = df1.index[i]
        df1.loc[j, 'Rank Coefficient'] = coef

    # position, daily_yield, cum_yield

    # Calculate filter
    df1['filter'] = abs(df1['Rank Coefficient'].rolling(30).mean())
    df1['trend position'] = np.nan
    df1.loc[df1['filter'] >= Filter, 'trend position'] = 1
    df1.loc[df1['filter'] < Filter, 'trend position'] = 0

    # Calculate return
    df1['position'] = np.nan
    df1['position'] = df1['trend position']
    df1['daily_yield'] = df1['position'].shift(1) * df1['Return']
    df1['cum_yield'] = (1 + df1['daily_yield']).cumprod()

    return (df1['trend position'], df1, Filter)  # return series and dataframe


def kalman2_MAs_correlation(df, r, Filter=0.4):
    train, test = df.iloc[:size, :], df.iloc[size:, :]
    test['Return'] = test['Adj Close'].pct_change()
    predictions, estimation = kalman_filter_pred_current(test['Close'].values, r)
    test['current predictions'] = np.array(predictions)
    test['estimation'] = np.array(estimation)


    test['correlation filter'] = rankc(test, Filter)[1]['filter']

    test = test.dropna()

    test['kf_long'] = np.where((test['estimation'] > test['current prediction']) & (abs(test['correlation filter']) > cor), 1,
                                0)
    test['kf_short'] = np.where((test['estimation'] < test['current prediction']) & (abs(test['correlation filter']) > cor),
                                 -1, 0)
    test['kf_positions'] = test['kf_long'] + test['kf_short']
    test['kf_strategy_return'] = test['kf_positions'].shift(1) * test['Return']

    # position, daily_yield, cum_yield
    test['daily_yield'] = test['kf_strategy_return']
    test['position'] =test['kf_positions']
    test['cum_yield'] = (1 + test['daily_yield']).cumprod()

    #print(test[['kf_positions', 'correlation filter']])
    annual_return = test['kf_strategy_return'].mean() * 252
    annual_std = test['kf_strategy_return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std
    print("annual return:", annual_return)
    print("sharpe ratio:", sharpe_ratio)

    print('Annual return: {:.6f} , Annual std: {:.6f} , Sharpe Ratio: {:.2f}'.
          format( annual_return, annual_std, sharpe_ratio))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    test[['Adj Close', 'current predictions', 'estimation', 'kf_positions']].plot(figsize=(12, 10), secondary_y=['kf_positions'],
                                                                        ax=ax1)

    test[['correlation filter']].plot(ax=ax2)
    ax2.axhline(y=cor, color='r', linestyle='--')
    plt.title('{} Kalman Filter training data \nsharpe ratio: {:.2f}'.format(s,sharpe_ratio))

    plt.show();
    return (test)



def main():
    #kf_pred_current_train(q=0.5)
    #kf_pred_current_train_plot(q=0.5)
    kf_pred_current_train_optimize()
    #buy_hold()

main()