from binance.client import Client
import csv
import pandas as pd
import btalib
from talib import EMA, MACD
import pandas as pd
import talib

api_key = "yM3lvuf1BmfadulCoC90Xse8851q7F5ZOtT1n8OOQziJnuie3WCFFVwswsydUd92"
api_secret = "WtD0ZRRevPcNSEReMkGUFqQvc41oqzQkQzGizkC588pglRc9YBEwOkpuZ2uefoKY"
client = Client(api_key, api_secret)
# info = client.get_symbol_info('BTCUSDT')
# print(info)
timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1d')
bars = client.get_historical_klines('BTCUSDT', '1d', timestamp, limit=1000)
with open('btc_bars.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    for line in bars:
        wr.writerow(line)
for line in bars:
    del line[5:]
btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
btc_df.set_index('date', inplace=True)
btc_df.to_csv('btc_bars3.csv')
btc_df['RSI'] = talib.RSI(btc_df['close'], timeperiod=14)
print(btc_df.head())
btc_df.to_csv('btc_bars4.csv')

