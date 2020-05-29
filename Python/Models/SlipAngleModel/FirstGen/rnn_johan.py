# RNN - LSTM

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60(can be changed) timesteps and 1 output
lookback = 60 # Decide how far to look back
X_train = []
y_train = []

for i in range(lookback, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-lookback:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# adding first layer
regressor.add(LSTM(units = 50,
                   return_sequences = True,
                   input_shape = ( X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# adding second layer
regressor.add(LSTM(units = 50,
                   return_sequences = True))
regressor.add(Dropout(0.2))

# adding third layer
regressor.add(LSTM(units = 50,
                   return_sequences = True))
regressor.add(Dropout(0.2))

# adding fourth layer last layer so return = false
regressor.add(LSTM(units = 50,
                   return_sequences = False))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1))

# Compilling the RNN
regressor.compile(optimizer = 'adam',
                  loss = 'mean_squared_error')

# Fitting the RNN
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Importing the training set
dataset_test = pd.read_csv('Google_Stock_Price_test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
# Creating a data structure
X_test = []
for i in range(lookback, inputs.shape[0]):
    X_test.append(inputs[i-lookback:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Pred')
plt.title('RNN Prediction of Google Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


