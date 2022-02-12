import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler


class StockData:
    def __init__(self, stock, start_date, validation_date, project_folder):
        self._stock = stock
        self._sec = yf.Ticker(self._stock)
        self.start_date = start_date
        self.validation_date = validation_date
        self.project_folder = project_folder
        self._min_max = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
        print('Проверка данных')
        print('Средняя:', train.mean(axis=0))
        print('Максимум:', train.max())
        print('Минимум:', train.min())
        print('Стандартное отклонение:', train.std(axis=0))

    def get_short_name(self):
        return self._sec.info['shortName']

    def get_currency(self):
        return self._sec.info['currency']

    def get_ticker(self):
        return self._sec.info['symbol']

    def get_min_max(self):
        return self._min_max

    def download_to_numpy(self, time_steps):
        end_date = datetime.date.today()
        # print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download(tickers=[self._stock], start=self.start_date, end=end_date)[['Close']]
        # добавляем индекс для данных
        data = data.reset_index()
        # сохраняем данные в cvs
        data.to_csv(os.path.join(self.project_folder, self._stock + '_downloaded_data' + '.csv'))

        # разделяем данные на training & test переменной validation_date
        training_data = data[data['Date'] < self.validation_date].copy()
        test_data = data[data['Date'] >= self.validation_date].copy()
        # Set the data frame index using column Date
        training_data = training_data.set_index('Date')
        test_data = test_data.set_index('Date')
        # print(test_data)

        # нормализуем данные от 0 до 1
        train_scaled = self._min_max.fit_transform(training_data)
        # self.__data_verification(train_scaled)

        # Training Data Transformation (2D -> 3D)
        x_train = []
        y_train = []
        for i in range(time_steps, train_scaled.shape[0]):
            x_train.append(train_scaled[i - time_steps:i])
            y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]

        test_scaled = self._min_max.fit_transform(inputs)

        # Testing Data Transformation (2D -> 3D)
        x_test = []
        y_test = []
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return (x_train, y_train), (x_test, y_test), (training_data, test_data)
