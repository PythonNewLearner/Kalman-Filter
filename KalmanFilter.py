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
from statsmodels.tsa.stattools import acf,pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA


EURUSD=pd.read_csv('EURUSD60.csv',index_col=0,parse_dates=True)
train_EURUSD=EURUSD.iloc[:4000,:]['Close']
test_EURUSD=EURUSD.iloc[4001:,:]['Close']

start=pd.to_datetime('2014-6-1')
end=pd.to_datetime('2019-5-25')
s='AMZN'
df=pd.DataFrame()
df[s]=web.DataReader(s,'yahoo',start=start,end=end)['Close']

class KalmanFilter(object):
    def __init__(self,a,b,c,x,p,q,r):
        self.a=a
        self.b=b
        self.c=c
        self.current_state_estimate=x
        self.current_prob_estimate=p
        self.q=q   #process covariace
        self.r=r    #measurement covariance
        self.predicted_state_estimate=0
        self.observation=0



    def current_state(self):
        return self.current_state_estimate

    def predicted_state(self):
        return self.predicted_state_estimate

    def observe(self):
        return self.observation

    def process(self,control_input,measurement):
        #prediction
        self.predicted_state_estimate=self.a*self.current_state_estimate+self.b*control_input
        predicted_prob_estimate=(self.a*self.current_prob_estimate)*self.a+self.q

        #observation
        self.observation=measurement-self.c*self.predicted_state_estimate
        observation_cov=self.c*predicted_prob_estimate*self.c+self.r

        #update
        kalman_gain=predicted_prob_estimate*self.c/float(observation_cov)
        self.current_state_estimate=self.predicted_state_estimate+kalman_gain*self.observation

        self.current_prob_estimate=(1-kalman_gain*self.c)*predicted_prob_estimate

#acf
acf1=acf(train_EURUSD)
acf1=pd.DataFrame([acf1]).T
acf1.plot(kind='bar')
plt.show();

#pacf
pacf1=pacf(train_EURUSD)
pacf1=pd.DataFrame([pacf1]).T
pacf1.plot(kind='bar')
plt.show();   #AR(1) is concerned

#test 1st differencing acf
price_diff=train_EURUSD-train_EURUSD.shift()
price_diff=price_diff.dropna()
acf1_diff=acf(price_diff)
acf1_diff=pd.DataFrame([acf1_diff]).T
acf1_diff.plot(kind='bar')
plt.show();

#test 1st differencing pacf
pacf1_diff=pacf(price_diff)
pacf1_diff=pd.DataFrame([pacf1_diff]).T
pacf1_diff.plot(kind='bar')
plt.show();

#arima
price=train_EURUSD.values
model=ARIMA(price,order=(0,1,0))
model_fit=model.fit(disp=0)
print(model_fit.summary())
pred_arima=model_fit.predict(7000,typ='levels')

#ARIMA rolling forecast
X=EURUSD['Close'].values
size=4000
train, test = X[0:size], X[size:len(X)]
history=[x for x in train]
predicts=[]
for t in range(len(test)):

    #price=np.append(price,EURUSD['Close'].values[i])
    model=ARIMA(history,order=(5,2,0))
    model_fit=model.fit(disp=0)
    output=model_fit.forecast()
    yhat=output[0]
    predicts.append(yhat)
    obs=test[t]
    history.append(obs)    #update history data
    print('%.1f : ARIMA predicted=%f, expected=%f' % (t,yhat, obs))

plt.plot(test)
plt.plot(predicts,color='red')
plt.show();
print('predict',predicts)





#kalman filter part
a=1
b=0
c=1
q=0.00001
r=1
x=1.24
p=1

filter=KalmanFilter(a,b,c,x,p,q,r)
predictions=[]
estimate=[]
observe=[]
for data in train_EURUSD:
    filter.process(0,data)
    predictions.append(filter.current_state())
    estimate.append(filter.predicted_state())
    observe.append(filter.observe())

predictions=[float(i) for i in predictions]
estimate=[float(i) for i in estimate]
observe=[float(i) for i in observe]


plt.plot(train_EURUSD.values)
plt.plot(predictions)
#plt.plot(estimate)
#plt.plot(observe)
plt.show();




















