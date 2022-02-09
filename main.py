import json
import yfinance

from stock_data import StockData

# data = yfinance.Ticker("FB")
# print('Info')
# print(json.dumps(data.info, indent=4, sort_keys=True))

data = StockData('FB')

