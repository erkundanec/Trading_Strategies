from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime as dt  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd

# Import the backtrader platform
import backtrader as bt

# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    tickers_list = ['AAPL']
    df_tic = pd.read_hdf("datasets/df_SnP_500_ohlcv.h5", "df", mode = 'r')
    df_tic = df_tic[df_tic['tic'].isin(tickers_list)]

    df_tic = df_tic.set_index('date')
    data = bt.feeds.PandasData(dataname = df_tic,
                                            datetime=None, 
                                            open =1,
                                            high=2,
                                            low=3,
                                            close=4,
                                            volume=6,
                                            openinterest=-1,
                                            timeframe = bt.TimeFrame.Days,
                                            fromdate=dt.datetime(2023, 1, 1),  # Specify the start date
                                            todate=dt.datetime(2023, 6, 30),   # Specify the end date
                                        )


    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())