#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from prophet import Prophet
import warnings; 
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# In[2]:


df = pd.read_csv(r"C:\Users\ASUS\Downloads\hb_ac.csv")


# In[3]:


df = pd.read_csv(r"C:\Users\ASUS\Downloads\hb_ac.csv")
df[['Year', 'Month']] = df['Month/Year'].str.split('_', expand=True)

# Combine the Year and Month columns to create a datetime column
df['ds'] = pd.to_datetime(df['Year'] + '-' + df['Month'], format='%Y-%m')
df['ds'] = df['ds'].dt.strftime('%Y-%m')
df['ds'] = pd.to_datetime(df['ds'])
df['ds'] = pd.to_datetime(df['ds'])
df = df[['ds', 'Aircraft hours']]
df = df.rename(columns={'Aircraft hours': 'y'})


# In[ ]:





# In[4]:


df.head()


# In[4]:


test_result=adfuller(df['y'])
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(y):
    result=adfuller(y)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    


# In[6]:


adfuller_test(df['y'])


# In[7]:


plt.plot(df['y'], label= "")


# In[8]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF and PACF plots
plt.figure(figsize=(12, 6))

plt.subplot(211)
plot_acf(df['y'], lags=30, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')

plt.subplot(212)
plot_pacf(df['y'], lags=30, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()


# In[7]:


split_date = '2019-12-01'
train_data = df[df['ds'] <= split_date]  # Training data till 2020-01-31
test_data = df[df['ds'] > split_date] 


# In[10]:


result = seasonal_decompose(df['y'], model='additive', period=12)  # Try period=6 or 4 if needed
result.plot()
plt.show()


# In[8]:


model=sm.tsa.statespace.SARIMAX(train_data['y'],order=(1, 0, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[12]:


n_steps = len(test_data)  # Number of steps to forecast (same as length of test data)
forecast = results.get_forecast(steps=n_steps)

# Get the forecasted mean values
forecast_mean = forecast.predicted_mean

# Get confidence intervals (optional, for plotting uncertainty bounds)
conf_int = forecast.conf_int()

# Plot the training data, actual test data, and forecasted values
plt.figure(figsize=(10, 6))

# Plot the training data
plt.plot(train_data['ds'], train_data['y'], label='Training Data', color='black')

# Plot the actual test data
plt.plot(test_data['ds'], test_data['y'], label='Actual Test Data', color='black')

# Plot the forecasted values
plt.plot(test_data['ds'], forecast_mean, label='Forecasted Data', color='red')

# Optionally, plot the confidence intervals
plt.fill_between(test_data['ds'], 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='red', alpha=0.1)

# Add titles and labels
plt.title('hot air balloon aircraft hours')
plt.xlabel('Year')
plt.ylabel('Aircraft Hours')

# Show the plot
plt.show()



# In[14]:


mse = mean_squared_error(test_data['y'], forecast_mean)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(test_data['y'], forecast_mean)
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')


# In[13]:


n_steps = 36  # Number of steps to forecast (same as length of test data)
future_forecast = results.get_forecast(steps=n_steps)

# Get the forecasted mean values
forecast_mean = future_forecast.predicted_mean
last_date = df['ds'].max()
future_dates = pd.date_range(start=last_date, periods=n_steps + 1, freq='M')[1:]
future_forecast_df = pd.DataFrame({'ds': future_dates, 'Predicted': future_forecast})

# Plot the training data, actual test data, and forecasted values
plt.figure(figsize=(10, 6))

# Plot the training data
plt.plot(df['ds'], df['y'], label='Historical Data', color='green')

# Plot the forecasted values
plt.plot(future_forecast_df['ds'], forecast_mean, label='Forecasted Data', color='red')
plt.title('Historical and Forecasted Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

# Display the plot
plt


# In[7]:


n_steps = 36  # Example: forecasting for 12 months

# Generate future dates starting from the last date in your test_data
last_date = test_data['ds'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=n_steps, freq='MS')  # 'MS' for month start
# Forecast future values for n_steps into the future using the fitted model
forecast = results.get_forecast(steps=n_steps)

# Get the predicted mean values (the actual forecasted values)
forecast_mean = forecast.predicted_mean
# Create a DataFrame with future dates and forecasted values
future_forecast_df = pd.DataFrame({
    'ds': future_dates,
    'forecasted_values': forecast_mean
})

# Save the DataFrame to a CSV file
future_forecast_df.to_csv(r"C:\Users\ASUS\Downloads\hotairb_aircraft_hrs_forecast.csv", index=False)


