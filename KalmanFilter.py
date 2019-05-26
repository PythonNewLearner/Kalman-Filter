# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:30:01 2019

@author: baichen
"""

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

EURUSD=pd.read_csv('EURUSD30.csv',index_col=0,parse_dates=True)['Close']
print(EURUSD)
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

a=1
b=0
c=1
q=0.00001
r=1
x=1
p=1

filter=KalmanFilter(a,b,c,x,p,q,r)
predictions=[]
estimate=[]
observe=[]
for data in EURUSD:
    filter.process(0,data)
    predictions.append(filter.current_state())
    estimate.append(filter.predicted_state())
    observe.append(filter.observe())

predictions=[float(i) for i in predictions]
estimate=[float(i) for i in estimate]
observe=[float(i) for i in observe]
print(predictions)

plt.plot(EURUSD.values)
plt.plot(predictions)
#plt.plot(estimate)
#plt.plot(observe)
plt.show();




















