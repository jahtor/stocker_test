import os
import pandas as pd
from datetime import datetime
from stock_data import StockData
from stock_plotter import Plotter
from stock_lstm import LongShortTermMemory


def train_LSTM_network(stock, start_date, validation_date, project_folder):
    data = StockData(stock, start_date, validation_date, project_folder)
    # загружаем набор данных
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_to_numpy(TIME_STEPS)

    # рисуем график с разделением на данные
    plotter = Plotter(True, project_folder, data.get_short_name(), data.get_currency(), data.get_ticker())
    # plotter.data_split(training_data, test_data, validation_date)

    # создаем LSTM модель
    lstm = LongShortTermMemory(project_folder)
    model = lstm.create_model(x_train)
    # выводим сводку по модели
    model.summary()
    # приступаем к обучению модели
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
                        callbacks=[lstm.get_callback()])

    # сохраняем модель в файл
    # print('Сохраняем модель')
    model.save(project_folder + '/model_weights.h5')

    # plotter.plot_loss(history)
    # plotter.plot_mse(history)

    # оценка модели
    print("Содержание модели:")
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)

    test_predictions = model.predict(x_test)
    test_predictions = data.get_min_max().inverse_transform(test_predictions)
    test_predictions = pd.DataFrame(test_predictions)
    # test_predictions.to_csv(os.path.join(project_folder), 'predictions.cvs')
    test_predictions.rename(columns={0: STOCK_TICKER + '_predicted'}, inplace=True)
    test_predictions = test_predictions.round(decimals=0)
    test_predictions.index = test_data.index
    plotter.plot_predictions(test_predictions, test_data)


if __name__ == '__main__':
    STOCK_TICKER = 'BTC-USD'
    STOCK_START_DATE = pd.to_datetime('2013-04-28')
    STOCK_VALIDATION_DATE = pd.to_datetime('2021-01-01')
    EPOCHS = 25
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
