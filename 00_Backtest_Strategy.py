import sys
sys.path.append('../backtrader-master/')

import backtrader as bt
import backtrader.analyzers as btanalyzers
# from datetime import datetime
import datetime as dt
import pandas as pd

import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt

class SMA_Crossover_Strategy(bt.Strategy):

    params = (
        ('fast_length', 9),
        ('slow_length', 21)
    )
     
    def __init__(self):
        self.crossovers = []
        self.portfolio_values = []  # List to store portfolio values
         
        for d in self.datas: 
            # ma_fast = bt.ind.EMA(d, period = self.params.fast_length)
            # ma_slow = bt.ind.EMA(d, period = self.params.slow_length)
            ma_fast = bt.ind.SMA(d, period = self.params.fast_length)
            ma_slow = bt.ind.SMA(d, period = self.params.slow_length)
            self.crossovers.append(bt.ind.CrossOver(ma_fast, ma_slow))
 
    def next(self):
        for i, d in enumerate(self.datas):
            if not self.getposition(d).size:
                if self.crossovers[i] > 0: 
                    self.buy(data = d)
            elif self.crossovers[i] < 0: 
                self.close(data = d)

            print(f"{self.datas[i].datetime.date(0)} {cerebro.broker.getvalue()}: Portfolio value")

            # Append the current portfolio value to the list
            self.portfolio_values.append(cerebro.broker.getvalue())

cerebro = bt.Cerebro()
df_tics = pd.read_hdf("datasets/df_ohlcv_daily_sp500.h5", "df", mode = 'r')
tickers_list = df_tics['tic'].unique().tolist()

tickers_list = ['TSLA']
for tic in tickers_list: 
    df_tic = df_tics[df_tics['tic'] == tic]
    df_tic = df_tic.set_index('date')
    # df_tic['date'] = pd.to_datetime(df_tic['date'])
    # print(df_tic.head(1))

    data = bt.feeds.PandasData(dataname = df_tic,
                                            # datetime=None, 
                                            open =1,
                                            high=2,
                                            low=3,
                                            close=4,
                                            volume=6,
                                            openinterest=-1,
                                            timeframe = bt.TimeFrame.Days,
                                            fromdate=dt.datetime(2022, 10, 1),  # Specify the start date
                                            todate=dt.datetime(2023, 8, 24),   # Specify the end date
                                        )

    cerebro.adddata(data, name = tic)

cerebro.addstrategy(SMA_Crossover_Strategy)
cerebro.broker.setcash(30000.0)
cerebro.addsizer(bt.sizers.PercentSizer, percents = 100)

back = cerebro.run(stdstats=True)
# cerebro.broker.getvalue()

# Retrieve the vector of portfolio values
portfolio_values_vector = back[0].portfolio_values

print(portfolio_values_vector[-1])


