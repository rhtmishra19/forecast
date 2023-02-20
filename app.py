# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:47:30 2023

@author: Anil Kumar
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime
import statsmodels.api as sm
import streamlit as st
from pickle import dump
from pickle import load

st.title('Apple Sales Price Forecasting')

data_close = load(open('data_close.sav', 'rb'))

periods = st.number_input('Number of Days',min_value=1)

datetime = pd.date_range('2020-01-01', periods=periods,freq='B')
date_df = pd.DataFrame(datetime,columns=['Date'])


model_sarima_final = sm.tsa.SARIMAX(data_close.Close,order=(0,1,2),seasonal_order=(1,1,0,57))
sarima_fit_final = model_sarima_final.fit()
forecast = sarima_fit_final.predict(len(data_close),len(data_close)+periods-1)
forecast_df = pd.DataFrame(forecast)
forecast_df.columns = ['Stock Price']

data_forecast = forecast_df.set_index(date_df.Date)
st.success('Forecasting stock price value for '+str(periods)+' days')
st.write(data_forecast)



fig,ax = plt.subplots(figsize=(16,8),dpi=100)
ax.plot(data_close, label='Actual')
ax.plot(data_forecast,label='Forecast')
ax.set_title('Apple Stock Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend(loc='upper left',fontsize=12)
ax.grid(True)
st.pyplot(fig)

