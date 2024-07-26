import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



# Load data
currency = 'EURPLN=X'

print("1. Euro\n2. Dollar\n3. Funt")

while True:
    choice = int(input("Choose: "))

    if choice == 1:
        currency = 'EURPLN=X'
        currency_str = "EURO"
        break
    if choice == 2:
        currency = 'PLN=X'
        currency_str = "DOLLAR"
        break
    if choice == 3:
        currency = 'GBPPLN=X'
        currency_str = "FUNT"
        break

start = dt.datetime(2019, 1, 1)
end = dt.datetime(2023, 6, 11)

data = yf.download(currency, start, end)

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Prepare data for Random Forest and SVM
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
x_rf_svm = []
y_rf_svm = []

for x in range(prediction_days, len(scaled_data)):
    x_rf_svm.append(scaled_data[x - prediction_days:x, 0])
    y_rf_svm.append(scaled_data[x, 0])

x_rf_svm, y_rf_svm = np.array(x_rf_svm), np.array(y_rf_svm)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(x_rf_svm, y_rf_svm)

# Train SVM model
svm_model = SVR(kernel='rbf', C=1000, gamma=0.1)
svm_model.fit(x_rf_svm, y_rf_svm)

# Przygotowanie danych dla przewidywania kursu na kolejny dzień
real_data = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)

# Przewidywanie kursu na kolejny dzień dla wszystkich trzech modeli
prediction_lstm = model.predict(real_data)
prediction_lstm = scaler.inverse_transform(prediction_lstm)

prediction_svm = svm_model.predict(real_data.reshape((1, prediction_days)))
prediction_svm = prediction_svm.reshape(-1, 1)
prediction_svm = scaler.inverse_transform(prediction_svm)

prediction_rf = rf_model.predict(real_data.reshape((1, prediction_days)))
prediction_rf = prediction_rf.reshape(-1, 1)
prediction_rf = scaler.inverse_transform(prediction_rf)

# Wyświetlanie wyników przewidywania kursu na kolejny dzień
print(f"LSTM Prediction for Next Day: {prediction_lstm}")
print(f"SVM Prediction for Next Day: {prediction_svm}")
print(f"Random Forest Prediction for Next Day: {prediction_rf}")

