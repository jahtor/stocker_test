# библиотека для получения котировок
import yfinance as yf
# библиотека для анализа данных
import pandas as pd
# библиотека для визуализации данных
import matplotlib.pyplot as plt
import datetime

# tickers = ['AAPL', 'MSFT', 'FB', 'AMZN', 'GOOG']

# Import pandas
# data = pd.DataFrame(columns=tickers)

# получаем данные
# data = yf.download(tickers, '2021-01-01', datetime.date.today())['Close']


start = pd.to_datetime('2012-05-18')
ticker = ['FB']
data = yf.download(ticker, start, datetime.date.today())['Close']
# print(data)

# рисуем график цен закрытия
((data.pct_change()+1).cumprod()).plot(figsize=(10, 6))

# показываем "легенду"
plt.legend()

# Define the label for the title of the figure
plt.title("Close Price", fontsize=16)

# подписи к графику x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)

# рисуем сетку
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

plt.show()
