#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# # Loading Data

# In[2]:


df = pd.read_excel('online_retail_II.xlsx', engine = 'openpyxl')
#df.head()


# # Data Cleaning

# In[5]:


df = df.dropna(subset=['Customer ID'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[~df['Invoice'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['Quantity'] < 50000]


# # Selecting a product

# In[7]:


#top_product_id = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending = False).index[0]
#print({top_product_id})

top_product_id = '85123A'


# In[9]:


#Checking the data for this top product
heart_white = df[df['StockCode'] == '85123A']

print("--- Price Stats ---")
print(heart_white['Price'].describe())

print("\n--- Qty Stats ---")
print(heart_white['Quantity'].describe())

print("\n--- Top 5 Orders ---")
print(heart_white.sort_values(by='Quantity', ascending=False).head(5)[['InvoiceDate', 'Quantity', 'Price', 'Country']])

#Dataframe for just this product
product_df = df[df['StockCode'] == top_product_id].copy()


# In[11]:


#Prophet prep
weekly_sales = product_df.set_index('InvoiceDate')['Quantity'].resample('W').sum().reset_index()
weekly_sales.columns = ['ds', 'y']


# In[13]:


#Splitting data
cutoff_date = '2011-10-01'
train_data = weekly_sales[weekly_sales['ds'] < cutoff_date]
test_data = weekly_sales[weekly_sales['ds'] >= cutoff_date]

#Modeling
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode = 'multiplicative',
    changepoint_prior_scale = 0.03
)

m.add_country_holidays(country_name='UK')

m.fit(train_data)

future = m.make_future_dataframe(periods=26, freq='W')
forecast = m.predict(future)

forecast[['yhat', 'yhat_lower', 'yhat_upper']] = (
    forecast[['yhat', 'yhat_lower', 'yhat_upper']]
    .clip(lower=0)
)

forecast_test = forecast[forecast['ds'].isin(test_data['ds'])]

mape = mean_absolute_percentage_error(test_data['y'], forecast_test['yhat'])

rmse = np.sqrt(mean_squared_error(test_data['y'], forecast_test['yhat']))

print("MAPE:", mape)
print("RMSE:", rmse)


# # Plotting - Prophet

# In[15]:


fig = m.plot(forecast)

plt.axvline(
    pd.to_datetime(cutoff_date),
    color='gray',
    linestyle='--',
    label='Forecast Start'
)

plt.plot(test_data['ds'], test_data['y'], color='red', linestyle = '--', label='Actual Sales (Hidden Data)')

# Formatting
plt.title(f"Forecast vs Actuals for Product {top_product_id}")
plt.legend()
m.plot_components(forecast)
plt.show()


# # Safety Stock calculation - Prophet method

# In[17]:


z_score = 1.645 #service level of 95%
lead_time = 2 # 2 weeks

safety_stock = z_score * rmse * np.sqrt(lead_time)

print(f"--- INVENTORY DECISION ---")
print(f"To maintain a 95% Service Level:")
print(f"We must hold {int(safety_stock)} units of Safety Stock.")


# # SARIMA Model

# In[19]:


weekly_sales_sarima = (
    product_df
    .set_index('InvoiceDate')['Quantity']
    .resample('W')
    .sum()
)

train = weekly_sales_sarima[weekly_sales_sarima.index < cutoff_date]
test = weekly_sales_sarima[weekly_sales_sarima.index >= cutoff_date]

print("Training weeks:", len(train))
print("Testing weeks:", len(test))


# In[25]:


sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_results = sarima_model.fit(disp=False)
print(sarima_results.summary())


# In[27]:


sarima_forecast = sarima_results.get_forecast(steps=len(test))
sarima_pred = sarima_forecast.predicted_mean
sarima_pred = sarima_pred.clip(lower=0)


# In[29]:


sarima_mape = mean_absolute_percentage_error(test, sarima_pred)
sarima_rmse = np.sqrt(mean_squared_error(test, sarima_pred))

print("SARIMA MAPE:", sarima_mape)
print("SARIMA RMSE:", sarima_rmse)


# In[31]:


plt.figure(figsize=(12,6))

plt.plot(train.index, train, label='Training Data', color='black')
plt.plot(test.index, test, label='Actual Sales', color='red', linestyle='--')
plt.plot(test.index, sarima_pred, label='SARIMA Forecast', color='blue')

plt.axvline(
    pd.to_datetime(cutoff_date),
    color='gray',
    linestyle='--',
    label='Forecast Start'
)

plt.title(f"SARIMA Forecast vs Actuals for Product {top_product_id}")
plt.xlabel("Date")
plt.ylabel("Weekly Demand")
plt.legend()
plt.show()


# # Safety stock calculation - SARIMA Model

# In[37]:


safety_stock_2 = z_score * sarima_rmse * np.sqrt(lead_time)

print(f"--- INVENTORY DECISION ---")
print(f"To maintain a 95% Service Level:")
print(f"We must hold {int(safety_stock_2)} units of Safety Stock.")


# # Comparison of the two methods

# In[35]:


comparison_df = pd.DataFrame({
    'Model': ['Prophet', 'SARIMA'],
    'MAPE': [mape, sarima_mape],
    'RMSE': [rmse, sarima_rmse]
})

comparison_df


# In[ ]:




