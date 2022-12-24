import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import bittrex

# Подключение к Binance
client = bittrex.Bittrex(None, None)

# Получение данных из Binance
data = client.get_market_history('BTC-USDT')

# Преобразование данных в датафрейм
df = pd.DataFrame(data['result'])

# Преобразование даты в формат даты
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

# Сортировка датафрейма по времени
df = df.sort_values('TimeStamp')

# Преобразование данных в массив
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['TimeStamp', 'Close'])
for i in range(0,len(data)):
    new_data['TimeStamp'][i] = data['TimeStamp'][i]
    new_data['Close'][i] = data['Close'][i]

# Нормализация данных
dataset = new_data.values
train = dataset[0:len(dataset),:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)

# Создание датасета для обучения
x_train, y_train = [], []
for i in range(60,len(scaled_data)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Преобразование данных для входа в нейронную сеть
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Создание модели нейронной сети
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Предсказание цены
inputs = new_data[len(new_data) - len(data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# Торговля
trade_data = new_data.tail(60)
trade_data['Prediction'] = closing_price
trade_data['Action'] = np.where(trade_data['Prediction'] > trade_data['Close'], 1, 0)

# Вывод результатов
print(trade_data)
