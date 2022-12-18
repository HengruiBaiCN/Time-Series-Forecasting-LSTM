import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# load data
company = 'FB'

start = dt.datetime(2009,1,1)
end = dt.datetime(2022,1,1)

data = web.DataReader(company, 'iex', start, end)

# prepare data
scaler = MinMaxScaler(feature_range=(0,1)) # transform the data range from (0-1000) to (0-1)
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) # only interest the closing price # reshape to (-1, 1) represent (row, column) - column as 1 but row unknown

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0]) # x = [[12345, 678910]], x[0, 1] = 678910
    y_train.append(scaled_data[x, 0])
    pass

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #  prediction of the next closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test the model accuracy on Existing data '''

# Load the data
test_start = dt.datetime(2022,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])
    pass

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
plt.plot(actual_prices, color='black', label=f'Actual price')
plt.plot([predicted_prices], color='red', label=f'Predicted price')
plt.title(f'{company} Share Price')
plt.xlabel('time')
plt.ylabel('share price')
plt.legend()
plt.show()
