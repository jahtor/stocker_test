import os
import pandas as pd
import secrets
from datetime import datetime
from stock_data import StockData
import matplotlib.pyplot as plt


def train_LSTM_network(stock, start_date, validation_date, project_folder):
    data = StockData(stock, start_date, validation_date, project_folder)
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_to_numpy(TIME_STEPS)

    # print("plotting Data and Histogram")
    plt.figure(figsize=(12, 5))
    plt.plot(training_data.Close, color='green')
    plt.plot(test_data.Close, color='red')
    plt.ylabel('Price [' + data.get_currency() + ']')
    plt.xlabel("Date")
    plt.legend(["Training Data", "Validation Data >= " + validation_date.strftime("%Y-%m-%d")])
    plt.title(data.get_short_name())
    # plt.savefig(os.path.join(data.project_folder, data.get_short_name().strip().replace('.', '') + '_price.png'))
    # fig, ax = plt.subplots()
    # training_data.hist(ax=ax)
    # fig.savefig(os.path.join(data.project_folder, data.get_short_name().strip().replace('.', '') + '_hist.png'))
    plt.pause(0.001)
    plt.show()


if __name__ == '__main__':
    STOCK_TICKER = 'BTC-USD'
    STOCK_START_DATE = pd.to_datetime('2013-04-28')
    STOCK_VALIDATION_DATE = pd.to_datetime('2021-01-01')
    EPOCHS = 100
    BATCH_SIZE = 32
    TIME_STEPS = 3
    TODAY_RUN = datetime.today().strftime("%Y%m%d")
    # TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
    TOKEN = STOCK_TICKER + '_' + TODAY_RUN

    print('Ticker: ' + STOCK_TICKER)
    print('Start date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation date: ' + STOCK_VALIDATION_DATE.strftime("%Y-%m-%d"))
    print('Test Run Folder: ' + TOKEN)

    PROJECT_FOLDER = os.path.join(os.getcwd(), 'data', TOKEN)
    if not os.path.exists(PROJECT_FOLDER):
        os.makedirs(PROJECT_FOLDER)

    train_LSTM_network(STOCK_TICKER, STOCK_START_DATE, STOCK_VALIDATION_DATE, PROJECT_FOLDER)
