#!/usr/bin/env python
# coding: utf-8

# # StockSavvy: Real-Time Stock Data Analysis

# In[1]:


##Import necessary libraries
import requests
import matplotlib.pyplot as plt


# In[2]:


##Define your API key and symbol
api_key = 'QFO8HG8RDQCGGHAL'
symbol = 'AAPL'


# In[3]:


##Define the base URL and parameters for the API request
base_url = 'https://www.alphavantage.co/query'
params = {
    'function': 'GLOBAL_QUOTE',
    'symbol': symbol,
    'apikey': api_key
}


# In[4]:


##Make API request and handle the response
response = requests.get(base_url, params=params)

if response.status_code == 200:
    data = response.json()
    if 'Global Quote' in data:
        global_quote = data['Global Quote']
        price = float(global_quote['05. price'])
        change = float(global_quote['09. change'])
        change_percent = float(global_quote['10. change percent'][:-1])  # Remove '%' sign
    else:
        print("No data found for the symbol.")
else:
    print("Failed to retrieve data. Check your API key and try again.")


# In[5]:


##Display the retrived data
print(f"Symbol: {symbol}")
print(f"Price: ${price:.2f}")
print(f"Change: ${change:.2f}")
print(f"Change Percent: {change_percent:.2f}%")


# In[6]:


import requests
import matplotlib.pyplot as plt

# Your Alpha Vantage API key
api_key = 'QFO8HG8RDQCGGHAL'

# Symbol for the stock we want to retrieve data for
symbol = 'AAPL'  #  Apple Inc.

# Base URL for the Alpha Vantage API
base_url = 'https://www.alphavantage.co/query'

# Parameters for the API request
params = {
    'function': 'GLOBAL_QUOTE',  
    'symbol': symbol,             
    'apikey': api_key             
}

try:
    # Make the API request
    response = requests.get(base_url, params=params)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        
        # Check if the response contains global quote data
        if 'Global Quote' in data:
            global_quote = data['Global Quote']
            price = float(global_quote['05. price'])
            change = float(global_quote['09. change'])
            change_percent = float(global_quote['10. change percent'].strip('%'))
            
            # Data for pie chart
            sizes = [abs(change), abs(change_percent)]
            labels = ['Change', 'Change Percent']
            colors = ['gold', 'lightcoral']
            explode = (0.1, 0)  # explode 1st slice
            
            # Plotting pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')
            plt.title(f'Stock Information for {symbol}')
            plt.show()
        else:
            print("No data found for the symbol.")
    else:
        print("Failed to retrieve data. Check your API key and try again.")
except Exception as e:
    print(f"An error occurred: {str(e)}")


# In[7]:


##This code retrieves real-time stock data for the given symbol (in this case, 'AAPL') using the Alpha Vantage API, prints the information, and optionally visualizes it using a pie chart


# In[8]:


pip install alpha-vantage


# In[9]:


from alpha_vantage.timeseries import TimeSeries

# Set your Alpha Vantage API key
api_key = 'QFO8HG8RDQCGGHAL'

# Initialize TimeSeries object with your API key
ts = TimeSeries(key=api_key, output_format='pandas')

# Specify the stock symbol and fetch daily stock data
symbol = 'AAPL'  # Example: Apple Inc.
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')

# Display the fetched data
print(data.head())


# In[10]:


#The provided data represents the daily stock prices of a particular security over a span of five days, from February 26th, 2024, to March 1st, 2024. Each row corresponds to a single trading day, with the following columns:

#Open: The opening price of the security at the beginning of the trading day.
#High: The highest price of the security reached during the trading day.
#Low: The lowest price of the security reached during the trading day.
#Close: The closing price of the security at the end of the trading day.
#Volume: The total number of shares traded for the security during the trading day.
#This dataset provides essential information for analyzing the price movements and trading activity of the security over the specified period. Analyzing these factors can help investors make informed decisions about buying, selling, or holding the security.


# In[11]:


#Prediction Analysis


# In[12]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = {
    'open': [179.55, 181.27, 182.51, 181.10, 182.24],
    'high': [180.53, 182.57, 183.12, 183.9225, 182.76],
    'low': [177.38, 179.53, 180.13, 179.56, 180.65],
    'close': [179.66, 180.75, 181.42, 182.63, 181.16],
    'volume': [73563082.0, 136682597.0, 48953939.0, 54318851.0, 40867421.0]
}
df = pd.DataFrame(data)

# Define features (X) and target variable (y)
X = df.drop('close', axis=1)
y = df['close']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[13]:


#The Mean Squared Error (MSE) is a measure of the average squared difference between the actual values (y_true) and the predicted values (y_pred) in a regression problem.

#In this context, a MSE of 0.5054 means that, on average, the squared difference between the actual close prices of the stock and the predicted close prices by the model is 0.5054.
# this is relativeley good! 

