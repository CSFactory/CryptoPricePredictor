import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import mplfinance as mpf
import pandas as pd
from tensorflow.python.ops.gen_math_ops import mod
import pandas_datareader as web
import datetime as dt
import requests

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential

crypto_currency = 'BTC'
against_currency = 'USD'


#Specify the time-frame for training

start = dt.datetime(2014,9,17) #2013
end = dt.datetime(2019,1,29) #end = dt.datetime(2021,1,29)

#Scrape the actual data
#data_wrx = requests.get('https://api.wazirx.com/api/v2/tickers') #testing wrx data
#data_dcx = requests.get('https://api.coindcx.com/exchange/ticker')
data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end) #e.g: BTC-USD, yahoo, 2012,1,1, NOW.
#data['Date']=pd.to_datetime(data['Date'])
data_df = pd.DataFrame(data)
#data_df = data_df.set_index('Close')
data_df.reset_index(inplace=True)
#data_df.set_index('Date').tail(100)
#Prepare data for Neural Network
    #Scale Down data b/w 0 and 1 so that NN can work better with that squeezed data

print(data_df.Date.head()) #web data read
print(data_df.Date.tail())
# We will concentrate only on "CLOSE" data column
# because we will predict "CLOSE" 
#print(data_df.Date.head()) 

#Start Scaling
scaler = MinMaxScaler(feature_range=(0,1)) #Squeeze all values b/w 0 and 1
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) #-1 to 1 is the format of data which scaler requires which is done by reshape function

#Choose a number for the PREDICTION_DAYS: Number of days on which our prediction is based

#IDEA: look at past 60 days close data and predict the close data for 61st day
prediction_days = 60

#EXPERIMENTAL (PREDICTING THE 30th day in future from current 60 days data)
future_day = 1
#future_day = 0
#Prepare the training data, we need to have xdata and ydata
 #we will make NN see past 60 days data and NN will predict 61st day data

x_train , y_train = [], [] #x and y train data set have empty lists

#fill x and y train data set with values
for x in range(prediction_days, len(scaled_data) - future_day): #length of scaled data is the length of whole dataset
    x_train.append(scaled_data[(x-prediction_days):x, 0]) 
    y_train.append(scaled_data[x + future_day, 0]) 

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Create Neural Network
model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))) #Input_Layer
#model.add(Dropout(0.20)) #Dropout to prevent Overfitting
model.add(LSTM(units=50, return_sequences=True))
#model.add(Dropout(0.20))
model.add(LSTM(units=50))
#model.add(Dropout(0.20))
model.add(Dense(units=1)) #Output Price Prediction Layer

#checkpoint = ModelCheckpoint(model, monitor='loss', verbose=1, save_best_only=True)
#callbacks_list = [checkpoint]

#Compile the Model
model.compile(optimizer = 'adam',loss = 'mean_squared_error')
#model.compile(optimizer = 'adam', loss='mean_squared_logarithmic_error')

#Fit the Model
model.fit(x_train, y_train, epochs = 50 , batch_size=16, verbose=1) #16

#Test the Model
#Y_pred comparison with actual Y to check the accuracy of the model

test_start = dt.datetime(2019,1,30) #Start date for the test data #2021 #2 #1
test_end = dt.datetime(2023,1,30)   #.now()
#test_predicted_date = dt.datetime(2022,2,1)

test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)

actual_prices = test_data['Close'].values

definite_prices = data['Close'].values #Time to test as compared to actual data with test data and predicted data

total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

for x in range(prediction_days, 350): #350
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

#pred_price_on_date = model.predict(test_predicted_date) #NOTE: TESTING
#pred_price_on_date = scaler.inverse_transform(pred_price_on_date)
#years = range(2014,2022,6)
plt.plot(actual_prices, color = "black", label = "Actual Prices")
plt.plot(prediction_prices, color = "orange", label = "Predicted Prices")
#plt.plot(definite_prices, color = "green", label = "Original Prices")
#plt.plot(pred_price_on_date, color = "orange", label = "On Date")
plt.title(f'{crypto_currency} price prediction')
#plt.yscale(dt.datetime(year,month,date))
#dates = [date for date in data]
#plt.xscale(data_df.Date)
#plt.xlabel('Year',size =14)
#plt.ticklabel_format(style='plain')
#plt.xticks(years)
plt.xlabel(f'{test_end}Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()



#mpf.plot(data_df.set_index('Date').tail(120), 
        #type='candle', style='charles', 
        #volume=True, 
        #title='ETH-USD Last 120 Days', 
        #mav=(10,20,30))


#PREDICT THE NEXT DAY PRICE

#real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs) + 1, 0]]
#real_data = np.array(real_data)
#real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

#prediction = model.predict(real_data)
#prediction = scaler.inverse_transform(prediction)
#print(prediction)
















