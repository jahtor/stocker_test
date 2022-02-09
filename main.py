import json
from stock_data import StockData

StockData('FB').download_to_numpy()
# d = StockData('FB')._sec
# print('Info')
# print(json.dumps(d.info, indent=4, sort_keys=True))
