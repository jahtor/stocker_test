import os
import yfinance as yf
import datetime
# from sklearn.preprocessing import MinMaxScaler

class StockData:
    def __init__(self, stock):
        self._stock = stock
        # self._sec = yf.Ticker(self._stock.get_ticker())
        self._sec = yf.Ticker(self._stock)
        # self._min_max = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
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

    def get_start_date(self):
        return self._sec.info['startDate']

    # def download_to_numpy(self, time_steps, project_folder):
    def download_to_numpy(self):
        end_date = datetime.date.today()
        # print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download(tickers=[self._stock], start=self.get_start_date(), end=end_date)[['Close']]
        data = data.reset_index()
        datafile = self._stock + '_downloaded_data' + '.csv'
        # data.to_csv(os.path.join('./data', datafile))
        # print(data)

        training_data = data[data['Date'] < '2020-01-01'].copy()
        test_data = data[data['Date'] >= '2020-01-01'].copy()
        # Set the data frame index using column Date
        training_data = training_data.set_index('Date')
        test_data = test_data.set_index('Date')
        # print(test_data)

        # train_scaled = self._min_max.fit_transform(training_data)
        # self.__data_verification(train_scaled)

        self.__data_verification(training_data)

        # Training Data Transformation
        x_train = []
        y_train = []
        # for i in range(time_steps, training_data.shape[0]):
        #     x_train.append(training_data[i - time_steps:i])
        #     y_train.append(training_data[i, 0])
