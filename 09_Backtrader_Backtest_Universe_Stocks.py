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
 
cerebro = bt.Cerebro()
df_tics = pd.read_hdf("datasets/df_SnP_500_ohlcv.h5", "df", mode = 'r')

tickers_list = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'V']
# tickers_list = ['MSFT']
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
                                            fromdate=dt.datetime(2021, 1, 1),  # Specify the start date
                                            todate=dt.datetime(2023, 8, 24),   # Specify the end date
                                        )

    # data = bt.feeds.YahooFinanceData(dataname = s, fromdate = datetime(2010, 1, 1), todate = datetime(2020, 1, 1))
    cerebro.adddata(data, name = tic)

cerebro.addstrategy(SMA_Crossover_Strategy)
cerebro.broker.setcash(30000.0)
cerebro.addsizer(bt.sizers.PercentSizer, percents = 20)
 
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name = "sharpe")
cerebro.addanalyzer(btanalyzers.Returns,     _name = "returns")
cerebro.addanalyzer(btanalyzers.Transactions, _name = "trans")

# # Disable the default symbol plot
# cerebro.plotinfo.plot = False
# # Enable the broker and trade plots
# cerebro.plotinfo.plotlog = True
# cerebro.plotinfo.plotting = "buy"  # This will plot buy signals on the chart

# plotinfo = dict(plot=True,
#                 subplot=False,
#                 plotname='',
#                 plotskip=False,
#                 plotabove=False,
#                 plotlinelabels=False,
#                 plotlinevalues=True,
#                 plotvaluetags=True,
#                 plotymargin=0.0,
#                 plotyhlines=[],
#                 plotyticks=[],
#                 plothlines=[],
#                 plotforce=False,
#                 plotmaster=None,
#                 plotylimited=True,
#            )




back = cerebro.run(stdstats=True)
# back = cerebro.run(stdstats=False)

cerebro.broker.getvalue()
print(back[0].analyzers.returns.get_analysis()['rnorm100'])
print(back[0].analyzers.sharpe.get_analysis())
# trans_dict = back[0].analyzers.trans.get_analysis()
# trans_df = pd.DataFrame(trans_dict)
# trans_df.to_excel('transactions.xlsx', index = False)
# print(back[0].analyzers.trans.get_analysis())

# cerebro.plot(style='candlestick', barup='green', bardown='red')
cerebro.plot(numfigs=1, volume=False, iplot=False, style='bar')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # Rotates the date labels for better visibility

# print(back)