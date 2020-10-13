#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:55:27 2020

@author: muguet
"""

#%%
# ## Get Walmart Stock Returns Data

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas_datareader.data import get_data_yahoo


enddate=datetime(2020,10,5)
startdate=datetime(2009,9,2)
stock_data = get_data_yahoo('WMT',startdate,enddate)
stock_data['Volume'].plot(legend=True,figsize=(10,4));
stock_data.head()
stock_data['Adj Close'].plot(legend=True,figsize=(10,4))

stock_data['returns']= stock_data['Adj Close'].pct_change(1) #returns

del client, enddate, query, startdate, stopwords

#%% Group tweets with same date and compute avarage sentiment

result['created_at'] = pd.to_datetime(result['created_at'])

y=result.groupby(pd.Grouper(key='created_at', freq='D')).mean()

stock_data["date"]=pd.to_datetime(stock_data.index)

stock_data2= stock_data.groupby(pd.Grouper(key='date', freq='D')).mean() #we already know that we have unique return for each day, I used this fonction to create a nan value for the missed dates (we have missed days because the stock market is not open every day )

#%% 
WalmartReturns= pd.DataFrame(stock_data2["returns"])    
WalmartVaderSentiment= pd.DataFrame(y["Vader Sentiment"])

#%% Create a data frame with stock return and vader sentiment

#Convert pandas timezone-aware DateTimeIndex to naive timestamp

WalmartVaderSentiment.index= pd.DatetimeIndex([i.replace(tzinfo=None) for i in WalmartVaderSentiment.index])

#merge dataframes

wmt = pd.concat([WalmartVaderSentiment, WalmartReturns],axis=1, join="inner")

wmt.head()



#%%Relationship between sentiments and Walmart return'

axw = wmt[['Vader Sentiment', 'returns']].plot(secondary_y = 'returns', title='Relationship between sentiments and Walmart return')
figw = axw.get_figure()
figw


#%%
from statsmodels.tsa.stattools import grangercausalitytests

#Drop rows with nan values
#Because we can not implement granger causalitytest on values contains NaN or inf 

wmt.dropna(subset = ['Vader Sentiment', 'returns'], inplace=True)

import statsmodels

statsmodels.tsa.stattools.adfuller(wmt.returns, regression = 'ct')
#non unit root

# According to the results, we believe there is no granger causality from sentiment to returns, ie all p-values are all above .05, accepting the null hypothesis, the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1.

grangercausalitytests(wmt[['returns', 'Vader Sentiment']], maxlag=4)

# ## Regress
# 
# $$
# r_{i, t}=\alpha+\beta_{1} \Delta s_{1, t}+\beta_{2} \Delta s_{i, t-1}+\epsilon_{t}
# $$
# 

wmt['sentiment_lag'] = wmt['Vader Sentiment'].shift(1)

wmt['L_s1'] = wmt['Vader Sentiment'].pct_change(1)
wmt['L_s2'] = wmt['sentiment_lag'].pct_change(1)
wmt.head()

import statsmodels.formula.api as smf

wmt= wmt.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

mod1 = smf.ols(formula='returns ~ L_s1 + L_s2', data=wmt).fit()
mod1.summary()
