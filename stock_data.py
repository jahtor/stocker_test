import os
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler


class StockData:
    def __init__(self, stock):
        self.stock = stock
        # self._sec = yf.Ticker(self._stock.get_ticker())
        self._sec = yf.Ticker(stock)
        self._min_max = MinMaxScaler(feature_range=(0, 1))

    def download_to_numpy(self, time_steps, project_folder):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download([self.stock.get_ticker()], start=self.stock.get_start_date(), end=end_date)[['Close']]
        data = data.reset_index()
        data.to_csv(os.path.join(project_folder, 'downloaded_data_' + self.stock.get_ticker() + '.csv'))
        print(data)
