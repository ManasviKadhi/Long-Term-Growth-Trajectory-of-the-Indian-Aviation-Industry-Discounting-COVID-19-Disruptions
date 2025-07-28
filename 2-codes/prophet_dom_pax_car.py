#!/usr/bin/env python
# coding: utf-8

# In[16]:


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


# In[6]:


df = pd.read_csv(r"C:\Users\ASUS\Downloads\dom_pax_mov.csv")
df1= pd.read_csv(r"C:\Users\ASUS\Downloads\dom_pax_mov (1).csv")


# In[14]:


df1.head()


# In[14]:


df[['Month', 'Year']] = df['Month/Year'].str.split('_', expand=True)

# Convert the two-digit year to four digits (assuming it's from 2000s)
df['Year'] = '20' + df['Year']

# Combine the Year and Month columns to create a datetime column
df['ds'] = pd.to_datetime(df['Year'] + '-' + df['Month'], format='%Y-%m')
df['ds'] = df['ds'].dt.strftime('%Y-%m')
df['ds'] = pd.to_datetime(df['ds'])
# Now you have a proper datetime column 'ds'
print(df[['Month/Year', 'ds']])


# In[7]:


df1[['Month', 'Year']] = df1['Month/Year'].str.split('_', expand=True)

# Convert the two-digit year to four digits (assuming it's from 2000s)
df1['Year'] = '20' + df1['Year']

# Combine the Year and Month columns to create a datetime column
df1['ds'] = pd.to_datetime(df1['Year'] + '-' + df1['Month'], format='%Y-%m')
df1['ds'] = df1['ds'].dt.strftime('%Y-%m')
df1['ds'] = pd.to_datetime(df1['ds'])
# Now you have a proper datetime column 'ds'
print(df1[['Month/Year', 'ds']])


# In[15]:


df['ds'] = pd.to_datetime(df['ds'])
df = df[['ds','India']]
df = df.rename(columns={'India': 'y'})


# In[13]:


x = ["Mumbai", "Kolkata", "Bangalore", "Pune", "Chennai"]
df1['ds'] = pd.to_datetime(df1['ds'])
df1 = df1[['ds'] + x]  # Ensure you are selecting 'ds' and all columns in the list `x`


# In[8]:


df.dtypes


# In[33]:


import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import product

# Assuming df is your original DataFrame with 'ds' and 'y' columns
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Define a function to evaluate the model with given hyperparameters
def evaluate_model(changepoint_prior_scale, seasonality_prior_scale):
    model = Prophet(
        seasonality_mode='multiplicative',  
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_df), freq='M')
    forecast = model.predict(future)
    
    # Calculate RMSE
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].iloc[train_size:].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return rmse

# Define the range of hyperparameters to search
changepoint_prior_scales = [0.01, 0.1, 0.5, 1.0, 5.0]
seasonality_prior_scales = [1.0, 5.0, 10.0, 15.0]

# Generate combinations of hyperparameters
param_grid = product(changepoint_prior_scales, seasonality_prior_scales)

# Initialize a DataFrame to store results
results = []

# Loop through parameter combinations and evaluate each model
for cp, sp in param_grid:
    rmse = evaluate_model(cp, sp)
    results.append({'changepoint_prior_scale': cp, 'seasonality_prior_scale': sp, 'RMSE': rmse})

# Create a DataFrame to see the results
results_df = pd.DataFrame(results)

# Find the best hyperparameters
best_params = results_df.loc[results_df['RMSE'].idxmin()]
print("Best Hyperparameters:")
print(best_params)


# In[16]:


holidays =pd.DataFrame({
    'holiday': ['New Year', 'Maha Shivratri', 'Holi', 'Independence Day', 
                'Raksha Bandhan', 'Janmashtami', 'Gandhi Jayanti', 
                'Dussehra', 'Diwali', 'Guru Nanak Jayanti', 
                'Christmas', 'New Year\'s Eve'],
    'ds': pd.to_datetime([
        '2023-01-01',   # New Year's Day
        '2023-02-18',   # Maha Shivratri
        '2023-03-08',   # Holi
        '2023-08-15',   # Independence Day
        '2023-08-22',   # Raksha Bandhan
        '2023-09-06',   # Janmashtami
        '2023-10-02',   # Gandhi Jayanti
        '2023-10-24',   # Dussehra
        '2023-11-12',   # Diwali
        '2023-11-27',   # Guru Nanak Jayanti
        '2023-12-25',   # Christmas
        '2023-12-31',   # New Year's Eve
    ]),
    'lower_window': [0] * 12,  # Effect on the holiday itself
    'upper_window': [1] * 12,  # Effect on the day after the holiday
})


# In[23]:


import matplotlib.pyplot as plt

# Step 1: Prepare the training and test data
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Step 2: Initialize Prophet Model
model = Prophet(
    seasonality_mode='multiplicative',  # Additive seasonality like Holt-Winters
    yearly_seasonality=True,            # Allow yearly seasonality (for monthly data)
    daily_seasonality=False,            # No daily seasonality (for monthly data)
    weekly_seasonality=False,           # No weekly seasonality (for monthly data)
    holidays=holidays,
    seasonality_prior_scale=1,
    changepoint_prior_scale=1
)

# Step 3: Fit the model on the training data
model.fit(train_df)

# Step 4: Make a DataFrame for future predictions, based on the test period
future = model.make_future_dataframe(periods=len(test_df), freq='M')

# Step 5: Make predictions
forecast = model.predict(future)

# Step 6: Extract the predictions for the test period
predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[train_size:]

# Step 7: Plotting the results
plt.figure(figsize=(12, 6))

# Plot the training data
plt.plot(train_df['ds'], train_df['y'], color='black', label='Training Data')

# Plot the test data
plt.plot(test_df['ds'], test_df['y'], color='black', label='Test Data')

# Plot the forecasted values (yhat)
plt.plot(predictions['ds'], predictions['yhat'], color='red', label='Forecast')

# Plot the confidence intervals as shaded regions
plt.fill_between(predictions['ds'], predictions['yhat_lower'], predictions['yhat_upper'], color='red', alpha=0.3, label='Confidence Interval')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Passengers Carried')

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()


# In[18]:


# Assuming df1 is the DataFrame with your data
df1['ds'] = pd.to_datetime(df1['ds'])

# List of city columns to fit the model
city_columns = ['Mumbai', 'Kolkata', 'Bangalore', 'Pune', 'Chennai']

# Iterate over each city column and fit a Prophet model
for city in city_columns:
    # Step 1: Prepare the data for training and testing
    temp_df = df1[['ds', city]].rename(columns={city: 'y'})
    train_size = int(0.8 * len(temp_df))
    train_df = temp_df[:train_size]
    test_df = temp_df[train_size:]

    # Step 2: Initialize the Prophet Model
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False,
        seasonality_prior_scale=1,
        changepoint_prior_scale=1
    )

    # Step 3: Fit the model on the training data
    model.fit(train_df)

    # Step 4: Create a DataFrame for future predictions based on the test period
    future = model.make_future_dataframe(periods=len(test_df), freq='M')

    # Step 5: Make predictions
    forecast = model.predict(future)

    # Step 6: Extract the predictions for the test period
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[train_size:]

    # Step 7: Plotting the results
    plt.figure(figsize=(12, 6))
    plt.title(f'Forecast for {city}')

    # Plot the training data
    plt.plot(train_df['ds'], train_df['y'], color='black', label='Training Data')

    # Plot the test data
    plt.plot(test_df['ds'], test_df['y'], color='black', linestyle='dashed', label='Test Data')

    # Plot the forecasted values (yhat)
    plt.plot(predictions['ds'], predictions['yhat'], color='red', label='Forecast')

    # Plot the confidence intervals as shaded regions
    plt.fill_between(predictions['ds'], predictions['yhat_lower'], predictions['yhat_upper'], color='red', alpha=0.3, label='Confidence Interval')

    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Passengers Carried')
    plt.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# In[26]:


# Assuming df1 is the DataFrame with your data
df1['ds'] = pd.to_datetime(df1['ds'])

# List of city columns to fit the model
city_columns = ['Bangalore', 'Pune', 'Chennai']
colors = ['#bababa', '#7e7cc5', '#7aa6b2']

# Dictionary to store forecasts for plotting
forecasts = {}

# Iterate over selected city columns to fit a model and get forecasts
for i, city in enumerate(city_columns):
    # Step 1: Prepare the data for training and testing
    temp_df = df1[['ds', city]].rename(columns={city: 'y'})
    train_size = int(0.8 * len(temp_df))
    train_df = temp_df[:train_size]
    test_df = temp_df[train_size:]

    # Step 2: Initialize the Prophet Model
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False,
        seasonality_prior_scale=1,
        changepoint_prior_scale=1
    )

    # Step 3: Fit the model on the training data
    model.fit(train_df)

    # Step 4: Create a DataFrame for future predictions based on the test period
    future = model.make_future_dataframe(periods=len(test_df), freq='M')

    # Step 5: Make predictions
    forecast = model.predict(future)
    
    # Filter the forecast to only include the forecasted period
    forecast_filtered = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[train_size:].reset_index(drop=True)
    forecasts[city] = forecast_filtered

# Step 6: Plotting the results
plt.figure(figsize=(14, 8))
plt.title('Forecasts for Bangalore, Pune, and Chennai')

# Plot each forecast on the same graph with different colors
for i, city in enumerate(city_columns):
    plt.plot(forecasts[city]['ds'], forecasts[city]['yhat'], label=f'Forecast for {city}', linestyle='dotted', color=colors[i])
    plt.fill_between(
        forecasts[city]['ds'],
        forecasts[city]['yhat_lower'],
        forecasts[city]['yhat_upper'],
        color=colors[i],
        alpha=0.1,
        label=f'Confidence Interval for {city}'
    )

# Plot original data for visual comparison
for i, city in enumerate(city_columns):
    plt.plot(df1['ds'], df1[city], label=f'Actual Data for {city}', color=colors[i])

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Passengers Carried')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# In[20]:


plt.figure(figsize=(12, 6))
plt.plot(train_df['ds'], train_df['y'], color='black')
plt.plot(test_df['ds'], test_df['y'], color='black')
plt.plot(predictions['ds'], predictions['yhat'], color='red')
plt.xlabel('Year')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[271]:


# Step 1: Extract the true values (y) and predicted values (yhat) for the test period
y_true = test_df['y'].values
y_pred = predictions['yhat'].values

# Step 2: Calculate MAE, MSE, RMSE, and MAPE
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
mape = (abs((y_true - y_pred) / y_true).mean()) * 100

# Step 3: Print the results
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')


# In[ ]:


# Plot residuals
residuals = test_df['y'] - predictions['yhat']
plt.figure(figsize=(12, 6))
plt.plot(test_df['ds'], residuals, marker='o', color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


from prophet.diagnostics import cross_validation, performance_metrics

# Perform cross-validation
df_cv = cross_validation(model, initial='730 days', period='30 days', horizon='30 days')

# Evaluate cross-validation results
df_p = performance_metrics(df_cv)

# Print metrics for cross-validation
print(df_p.head())


# In[273]:


df1 = pd.read_csv(r"C:\Users\ASUS\Downloads\nph_ac_dept.csv")
df1[['Year', 'Month']] = df1['Month/Year'].str.split('_', expand=True)

# Convert the two-digit year to four digits (assuming it's from 2000s)
df1['Year'] = '20' + df1['Year']

# Combine the Year and Month columns to create a datetime column
df1['ds'] = pd.to_datetime(df1['Year'] + '-' + df1['Month'], format='%Y-%m')
df1['ds'] = df1['ds'].dt.strftime('%Y-%m')
df1['ds'] = pd.to_datetime(df1['ds'])
# Now you have a proper datetime column 'ds'
print(df1[['Month/Year', 'ds']])


# In[276]:


df1['ds'] = pd.to_datetime(df1['ds'])
df1 = df1[['ds', 'Aircraft Departures']]
df1 = df1.rename(columns={'Aircraft Departures': 'y'})


# In[277]:


df1.head()


# In[278]:


train_size = int(0.8 * len(df1))
train_df = df1[:train_size]
test_df = df1[train_size:]

# Define a function to evaluate the model with given hyperparameters
def evaluate_model(changepoint_prior_scale, seasonality_prior_scale):
    model = Prophet(
        seasonality_mode='multiplicative',  
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_df), freq='M')
    forecast = model.predict(future)
    
    # Calculate RMSE
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].iloc[train_size:].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return rmse

# Define the range of hyperparameters to search
changepoint_prior_scales = [0.01, 0.1, 0.5, 1.0, 5.0]
seasonality_prior_scales = [1.0, 5.0, 10.0, 15.0]

# Generate combinations of hyperparameters
param_grid = product(changepoint_prior_scales, seasonality_prior_scales)

# Initialize a DataFrame to store results
results = []

# Loop through parameter combinations and evaluate each model
for cp, sp in param_grid:
    rmse = evaluate_model(cp, sp)
    results.append({'changepoint_prior_scale': cp, 'seasonality_prior_scale': sp, 'RMSE': rmse})

# Create a DataFrame to see the results
results_df = pd.DataFrame(results)

# Find the best hyperparameters
best_params = results_df.loc[results_df['RMSE'].idxmin()]
print("Best Hyperparameters:")
print(best_params)


# In[279]:


train_size = int(0.8 * len(df))
train_df = df1[:train_size]
test_df = df1[train_size:]
# Step 2: Initialize Prophet Model
model1 = Prophet(
    seasonality_mode='multiplicative',  # Additive seasonality like Holt-Winters
    yearly_seasonality=True,      # Allow yearly seasonality (for monthly data)
    daily_seasonality=False,      # No daily seasonality (for monthly data)
    weekly_seasonality=False,      # No weekly seasonality (for monthly data)
    holidays=holidays,
    seasonality_prior_scale=1,        
)
# Step 3: Fit the model on the training data
model1.fit(train_df)

# Step 4: Make a DataFrame for future predictions, based on the test period
future = model1.make_future_dataframe(periods=len(test_df), freq='M')


# Step 5: Make predictions
forecast1 = model1.predict(future)

# Step 6: Extract the predictions for the test period
predictions1 = forecast1[['ds', 'yhat']].iloc[train_size:]


# In[280]:


plt.figure(figsize=(12, 6))
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue', marker='o')
plt.plot(test_df['ds'], test_df['y'], label='Test Data', color='orange', marker='o')
plt.plot(predictions1['ds'], predictions1['yhat'], label='Predictions', color='red', linestyle='--')
plt.title('Prophet Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[281]:


# Step 1: Extract the true values (y) and predicted values (yhat) for the test period
y_true = test_df['y'].values
y_pred = predictions['yhat'].values

# Step 2: Calculate MAE, MSE, RMSE, and MAPE
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
mape = (abs((y_true - y_pred) / y_true).mean()) * 100

# Step 3: Print the results
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')


# In[283]:


df2 = pd.read_csv(r"C:\Users\ASUS\Downloads\nph_pax_c.csv")
df2[['Year', 'Month']] = df2['Month/Year'].str.split('_', expand=True)

# Convert the two-digit year to four digits (assuming it's from 2000s)
df2['Year'] = '20' + df2['Year']

# Combine the Year and Month columns to create a datetime column
df2['ds'] = pd.to_datetime(df2['Year'] + '-' + df2['Month'], format='%Y-%m')
df2['ds'] = df2['ds'].dt.strftime('%Y-%m')
df2['ds'] = pd.to_datetime(df2['ds'])
# Now you have a proper datetime column 'ds'
print(df2[['Month/Year', 'ds']])
df2['ds'] = pd.to_datetime(df1['ds'])
df2 = df2[['ds', 'Passengers Carried']]
df2 = df2.rename(columns={'Passengers Carried': 'y'})


# In[284]:


train_size = int(0.8 * len(df2))
train_df = df2[:train_size]
test_df = df2[train_size:]

# Define a function to evaluate the model with given hyperparameters
def evaluate_model(changepoint_prior_scale, seasonality_prior_scale):
    model = Prophet(
        seasonality_mode='multiplicative',  
        yearly_seasonality=True,
        daily_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_df), freq='M')
    forecast = model.predict(future)
    
    # Calculate RMSE
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].iloc[train_size:].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return rmse

# Define the range of hyperparameters to search
changepoint_prior_scales = [0.01, 0.1, 0.5, 1.0, 5.0]
seasonality_prior_scales = [1.0, 5.0, 10.0, 15.0]

# Generate combinations of hyperparameters
param_grid = product(changepoint_prior_scales, seasonality_prior_scales)

# Initialize a DataFrame to store results
results = []

# Loop through parameter combinations and evaluate each model
for cp, sp in param_grid:
    rmse = evaluate_model(cp, sp)
    results.append({'changepoint_prior_scale': cp, 'seasonality_prior_scale': sp, 'RMSE': rmse})

# Create a DataFrame to see the results
results_df = pd.DataFrame(results)

# Find the best hyperparameters
best_params = results_df.loc[results_df['RMSE'].idxmin()]
print("Best Hyperparameters:")
print(best_params)


# In[285]:


train_size = int(0.8 * len(df))
train_df = df2[:train_size]
test_df = df2[train_size:]
# Step 2: Initialize Prophet Model
model1 = Prophet(
    seasonality_mode='multiplicative',  # Additive seasonality like Holt-Winters
    yearly_seasonality=True,      # Allow yearly seasonality (for monthly data)
    daily_seasonality=False,      # No daily seasonality (for monthly data)
    weekly_seasonality=False,      # No weekly seasonality (for monthly data)
    holidays=holidays,
    seasonality_prior_scale=15,        
)
# Step 3: Fit the model on the training data
model1.fit(train_df)

# Step 4: Make a DataFrame for future predictions, based on the test period
future = model1.make_future_dataframe(periods=len(test_df), freq='M')


# Step 5: Make predictions
forecast1 = model1.predict(future)

# Step 6: Extract the predictions for the test period
predictions1 = forecast1[['ds', 'yhat']].iloc[train_size:]


# In[286]:


plt.figure(figsize=(12, 6))
plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue', marker='o')
plt.plot(test_df['ds'], test_df['y'], label='Test Data', color='orange', marker='o')
plt.plot(predictions1['ds'], predictions1['yhat'], label='Predictions', color='red', linestyle='--')
plt.title('Prophet Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[287]:


# Step 1: Extract the true values (y) and predicted values (yhat) for the test period
y_true = test_df['y'].values
y_pred = predictions['yhat'].values

# Step 2: Calculate MAE, MSE, RMSE, and MAPE
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
mape = (abs((y_true - y_pred) / y_true).mean()) * 100

# Step 3: Print the results
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

