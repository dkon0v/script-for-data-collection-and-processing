import requests
import hashlib
import hmac
import time
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

#Устанавливаем значения для API ключа и секретного ключа

api_key = 'YOUR_API_KEY'
secret_key = 'YOUR_SECRET_KEY'

#Устанавливаем значения символа, интервала и лимита для получения данных с Binance

symbol = 'BTCUSDT'
interval = '1h'
limit = 1000

#Устанавливаем метку времени для запроса к API Binance

timestamp = str(int(time.time() * 1000))

#Устанавливаем параметры запроса к API Binance

params = {'symbol': symbol, 'interval': interval, 'limit': limit, 'timestamp': timestamp}

#Формируем строку запроса на основе параметров

query_string = '&'.join([f"{k}={v}" for k,v in params.items()])

#Создаем подпись запроса на основе секретного ключа

signature = hmac.new(secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()

#Формируем URL для запроса к API Binance на основе строки запроса и подписи

url = f"https://api.binance.com/api/v3/klines?{query_string}&signature={signature}"

#Устанавливаем заголовок для запроса к API Binance на основе API ключа

headers = {'X-MBX-APIKEY': api_key}

#Отправляем запрос к API Binance и получаем ответ в формате JSON

response = requests.get(url, headers=headers)

#Записываем полученные данные в файл csv

with open('binance_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    for row in response.json():
        writer.writerow(row)

# Чтение данных из файла в Pandas DataFrame
df = pd.read_csv('binance_data.csv')

# Удаление пропусков
df.dropna(inplace=True)

# Преобразование даты и времени
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')

# Выделение признаков и меток
features = df.drop('Close', axis=1).values
labels = df['Close'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Загружаем данные из файла data.csv
data = pd.read_csv('binance_data.csv', header=0, index_col=0)

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Разделение на обучающую и тестовую выборки
train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size
train_data = data_scaled[0:train_size,:]
test_data = data_scaled[train_size:len(data_scaled),:]

# Функция, создающая временные последовательности
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), :])
        y.append(data[(i+seq_length), 0])
    return np.array(X), np.array(y)

# Создание временных последовательностей для обучающей и тестовой выборок
seq_length = 30
train_X, train_y = create_sequences(train_data, seq_length)
test_X, test_y = create_sequences(test_data, seq_length)

# Создание LSTM-модели
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Компиляция модели
model.compile(optimizer='adam', loss='mse')

# Обучение модели
model.fit(train_X, train_y, epochs=100, batch_size=32, verbose=1)

# Оценка модели на тестовой выборке
test_loss = model.evaluate(test_X, test_y, verbose=0)

# Прогнозирование на будущее
future_seq_length = 30
future_data = data_scaled[-seq_length:,:].reshape(1,seq_length,-1)
future_predictions = []
for i in range(future_seq_length):
    prediction = model.predict(future_data)
    future_predictions.append(prediction[0,0])
    future_data = np.append(future_data[:,1:,:], [[prediction]], axis=1)

# Обратное масштабирование
future_predictions = np.array(future_predictions).reshape(-1,1)
future_predictions = scaler.inverse_transform(future_predictions)

# Вывод результатов
print('Test loss:', test_loss)
print('Future predictions:', future_predictions)

#Оценка качества модели 

test_loss = model.evaluate(test_X, test_y, verbose=0)
print('Test loss (MSE):', test_loss)

test_predictions = model.predict(test_X)
test_mae = np.mean(np.abs(test_predictions - test_y))
print('Test MAE:', test_mae)

# Прогнозирование на тестовой выборке
test_predictions = model.predict(test_X)

# Обратное масштабирование
test_predictions = scaler.inverse_transform(test_predictions)
test_y = scaler.inverse_transform(test_y.reshape(-1, 1))

# Вывод графика
plt.figure(figsize=(12, 6))
plt.plot(test_y, label='True Values')
plt.plot(test_predictions, label='Predictions')
plt.legend()
plt.show()

# Прогнозирование на час вперед от последних полученных данных
future_seq_length = 1
future_data = data_scaled[-seq_length:,:].reshape(1,seq_length,-1)
future_predictions = []
for i in range(future_seq_length):
    prediction = model.predict(future_data)
    future_predictions.append(prediction[0,0])
    future_data = np.append(future_data[:,1:,:], [[prediction]], axis=1)

# Обратное масштабирование
future_predictions = np.array(future_predictions).reshape(-1,1)
future_predictions = scaler.inverse_transform(future_predictions)

# Вывод прогноза на час вперед
print('Future prediction for next hour:', future_predictions[0, 0])

