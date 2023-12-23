import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import datetime as dt

# Create a subclass of Strategy to define the indicators and logic
class SMA_CrossStrategy(bt.Strategy):
 
    def __init__(self):
        ma_fast = bt.ind.SMA(period = 9)
        ma_slow = bt.ind.SMA(period = 21)
         
        self.crossover = bt.ind.CrossOver(ma_fast, ma_slow)
 
    def next(self):
        if not self.position:
            if self.crossover > 0: 
                self.buy()
        elif self.crossover < 0: 
            self.close()

    # def next(self):
    #     if not self.position:
    #         if self.crossover > 0: 
    #             self.buy()
    #         elif self.crossover < 0:
    #             self.sell()
    #     elif self.crossover < 0: 
    #         self.close()
 

def main():
    cerebro = bt.Cerebro()

    tickers_list = ['AAPL']
    df_tic = pd.read_hdf("datasets/df_SnP_500_ohlcv.h5", "df", mode = 'r')
    df_tic = df_tic[df_tic['tic'].isin(tickers_list)]
    df_tic = df_tic.set_index('date')
    # df_tic['date'] = pd.to_datetime(df_tic['date'])
    print(df_tic.head(5))
    data = bt.feeds.PandasData(dataname = df_tic,
                                            # datetime=None, 
                                            open =1,
                                            high=2,
                                            low=3,
                                            close=4,
                                            volume=6,
                                            openinterest=-1,
                                            timeframe = bt.TimeFrame.Days,
                                            fromdate=dt.datetime(2023, 1, 1),  # Specify the start date
                                            todate=dt.datetime(2023, 8, 24),   # Specify the end date
                                        )
    
    
    # data = bt.feeds.YahooFinanceData(dataname = 'AAPL', fromdate = datetime(2010, 1, 1), todate = datetime(2020, 1, 1))
    cerebro.adddata(data)
    
    cerebro.addstrategy(SMA_CrossStrategy)
    
    cerebro.broker.setcash(10000.0)
    
    cerebro.addsizer(bt.sizers.PercentSizer, percents = 100)
    
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name = "sharpe")
    cerebro.addanalyzer(btanalyzers.Transactions, _name = "trans")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name = "trades")
    
    back = cerebro.run()
    
    cerebro.broker.getvalue()
    
    back[0].analyzers.sharpe.get_analysis()
    
    back[0].analyzers.trans.get_analysis()
    
    back[0].analyzers.trades.get_analysis()
    
    cerebro.plot(style='candlestick', barup='green', bardown='red')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotates the date labels for better visibility


if __name__=="__main__":
    main()