from binance.client import Client
import csv
import pandas as pd
import btalib
from talib import EMA, MACD
import pandas as pd
import os
import talib

api_key = os.environ.get('binance_api')
api_secret = os.environ.get('binance_secret')
client = Client(api_key, api_secret)
# info = client.get_symbol_info('BTCUSDT')
# print(info)

def funcdatagather(filenameforfile,stringval):
    timestamp = client._get_earliest_valid_timestamp('BTCUSDT', stringval)
    bars = client.get_historical_klines('BTCUSDT', stringval, timestamp, limit=1000)

    with open(filenameforfile, 'w', newline='') as f:
        wr = csv.writer(f)
        for line in bars:
            wr.writerow(line)
    for line in bars:
        del line[5:]
    btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
    btc_df.set_index('date', inplace=True)
    btc_df.index = pd.to_datetime(btc_df.index, unit='ms')
    btc_df['sma20'] = btalib.sma(btc_df.close, period=20).df
    btc_df['sma50'] = btalib.sma(btc_df.close, period=50).df
    btc_df['sma100'] = btalib.sma(btc_df.close, period=100).df
    btc_df['sma200'] = btalib.sma(btc_df.close, period=200).df
    btc_df['ema8'] = EMA(btc_df.close, timeperiod=8)
    btc_df['ema12'] = EMA(btc_df.close, timeperiod=12)
    btc_df['ema21'] = EMA(btc_df.close, timeperiod=21)
    btc_df['ema55'] = EMA(btc_df.close, timeperiod=55)
    btc_df['ema200'] = EMA(btc_df.close, timeperiod=200)
    btc_df['RSI'] = talib.RSI(btc_df['close'], timeperiod=14)
    btc_df.loc[btc_df['open']>btc_df['close'],'newvalue'] = 'bear'
    btc_df.loc[btc_df['close']>btc_df['open'],'newvalue'] = 'bull'
    btc_df.loc[btc_df['close']==btc_df['open'],'newvalue'] = 'stagnant'
    btc_df.to_csv(filenameforfile)

def main():
    hourly = '1h'
    filenameforfile = 'btc_bars_1h.csv'
    funcdatagather(filenameforfile, hourly)
    hourly = '2h'
    filenameforfile = "btc_bars_2h.csv"
    funcdatagather(filenameforfile, hourly)
    hourly = '4h'
    filenameforfile = 'btc_bars_4h.csv'
    funcdatagather(filenameforfile, hourly)
    hourly = '6h'
    filenameforfile = 'btc_bars_6h.csv'
    funcdatagather(filenameforfile, hourly)
    hourly = '8h'
    filenameforfile = "btc_bars_8h.csv"
    funcdatagather(filenameforfile, hourly)
    hourly = '12h'
    filenameforfile = "btc_bars_12h.csv"
    funcdatagather(filenameforfile, hourly)
    hourly = '1d'
    filenameforfile = 'btc_bars_1d.csv'
    funcdatagather(filenameforfile, hourly)


if __name__ == '__main__':
    main()
