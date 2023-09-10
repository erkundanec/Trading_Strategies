import datetime as dt
import backtrader as bt
import pandas as pd

class MovingAverageCrossStrategy(bt.Strategy):
    params = (
        ("short_period", 20),
        ("long_period", 50),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_period
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_period
        )

    def next(self):
        if self.short_ma > self.long_ma:
            # Buy signal
            self.buy()

        elif self.short_ma < self.long_ma:
            # Sell signal
            self.sell()

def main():
    tickers_list = ['AAPL']

    df_tic = pd.read_hdf("datasets/df_SnP_500_ohlcv.h5", "df", mode = 'r')
    df_tic = df_tic[df_tic['tic'].isin(tickers_list)]

    df_tic = df_tic.set_index('date')
    print(df_tic.head(5))

    # Backtrader parse data
    # tsla_df_daily_parsed = bt.feeds.PandasData(dataname = df_tic,datetime=None, open =1,high=2,low=3,close=4,volume=6,openinterest=-1)
    # bt.feeds.GenericCSVData

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

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MovingAverageCrossStrategy)

    cerebro.broker.set_cash(100000)  # Set your initial cash amount
    cerebro.broker.setcommission(commission=0.001)  # Set commission value

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("Ending Portfolio Value: %.2f" % cerebro.broker.getvalue())

    #Resample Data
    # cerebro = bt.Cerebro()

    # cerebro.run()

    cerebro.plot(iplot=False)

if __name__=="__main__":
    main()

