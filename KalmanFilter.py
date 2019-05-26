# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:36:01 2019

@author: baichen
"""

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

start=pd.to_datetime('2017-6-1')
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
        self.q=q
        self.r=r

    def current_state(self):
        return self.current_state_estimate

    def process(self,control_input,measurement):
        #prediction
        predicted_state_estimate=self.a*self.current_state_estimate+self.b*control_input
        predicted_prob_estimate=(self.a*self.current_prob_estimate)*self.a+self.q

        #obsservation
        observation=measurement-self.c*predicted_state_estimate
        observation_cov=self.c*predicted_prob_estimate*self.c+self.r

        #update
        kalman_gain=predicted_prob_estimate*self.c/float(observation_cov)
        self.current_state_estimate=predicted_state_estimate+kalman_gain*observation

        self.current_prob_estimate=(1-kalman_gain*self.c)*predicted_prob_estimate

a=1
b=0
c=1
q=0.005
r=1
x=1000
p=1

filter=KalmanFilter(a,b,c,x,p,q,r)
predictions=[]
for data in df.values:
    filter.process(0,data)
    predictions.append(filter.current_state())

predictions=[float(i) for i in predictions]
print(predictions)

plt.plot(df.values)
plt.plot(predictions)
plt.show();




















