import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime,timedelta
# import datetime as dt
import numpy as np
from matplotlib import pyplot
from alive_progress import alive_bar
import time
import xlwings as xw

import warnings
warnings.filterwarnings('ignore')

def myRSI(price, n=20):
    delta = price['close'].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(window=n).mean()
    RolDown = dDown.rolling(window=n).mean().abs()
    
    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    return rsi

def price_pivot_id(df1, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0
    
    pividlow=1
    pividhigh=1
    for i in range(l-n1, l+n2+1):
        if(df1.low[l]>df1.low[i]):
            pividlow=0
        if(df1.high[l]<df1.high[i]):
            pividhigh=0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0

def RSI_pivot_id(df1, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0

    pividlow=1
    pividhigh=1
    for i in range(l-n1, l+n2+1):
        if(df1.RSI[l]>df1.RSI[i]):
            pividlow=0
        if(df1.RSI[l]<df1.RSI[i]):
            pividhigh=0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0 
    
def price_point_pos(x):
    if x['price_pivot_id']==1:
        return x['low']-1e-3
    elif x['price_pivot_id']==2:
        return x['high']+1e-3
    else:
        return np.nan

def RSI_point_pos(x):
    if x['RSI_pivot_id']==1:
        return x['RSI']-1
    elif x['RSI_pivot_id']==2:
        return x['RSI']+1
    else:
        return np.nan

def add_technicals(df, technicals_list):
    for technical in technicals_list:
        if technical == 'RSI':
            df['RSI'] = df.ta.rsi(length=14)

    return df

def RSI_divergence(df, tic, right_window = 5, left_window = 5, backcandles = 40, plot_flag = False):
    technicals_list = ['RSI']
    df = add_technicals(df, technicals_list)
    candleid = len(df)-1   # last candle

    df = df.iloc[candleid-backcandles-left_window:candleid+1]
    df = df.reset_index(drop = True)

    candleid = len(df)-1   # last candle

    df['price_pivot_id'] = df.apply(lambda x: price_pivot_id(df, x.name, left_window, right_window), axis=1)
    df['RSI_pivot_id'] = df.apply(lambda x: RSI_pivot_id(df, x.name, left_window, right_window), axis=1)
    df['price_point_pos'] = df.apply(lambda row: price_point_pos(row), axis=1)
    df['RSI_point_pos'] = df.apply(lambda row: RSI_point_pos(row), axis=1)

    # maxim = np.array([])
    # minim = np.array([])
    # xxmin = np.array([])
    # xxmax = np.array([])

    # maximRSI = np.array([])
    # minimRSI = np.array([])
    # xxminRSI = np.array([])
    # xxmaxRSI = np.array([])

    # # for i in range(candleid-backcandles, candleid+1):
    # for i in range(len(df)):
    #     if df.iloc[i].price_pivot_id == 1:
    #         minim = np.append(minim, df.iloc[i].low)
    #         xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
    #     if df.iloc[i].price_pivot_id == 2:
    #         maxim = np.append(maxim, df.iloc[i].high)
    #         xxmax = np.append(xxmax, i) # df.iloc[i].name
    #     if df.iloc[i].RSI_pivot_id == 1:
    #         minimRSI = np.append(minimRSI, df.iloc[i].RSI)
    #         xxminRSI = np.append(xxminRSI, df.iloc[i].name)
    #     if df.iloc[i].RSI_pivot_id == 2:
    #         maximRSI = np.append(maximRSI, df.iloc[i].RSI)
    #         xxmaxRSI = np.append(xxmaxRSI, df.iloc[i].name)
            
    # slmin, intercmin = np.polyfit(xxmin, minim,1)
    # slmax, intercmax = np.polyfit(xxmax, maxim,1)
    # slminRSI, intercminRSI = np.polyfit(xxminRSI, minimRSI,1)
    # slmaxRSI, intercmaxRSI = np.polyfit(xxmaxRSI, maximRSI,1)

    # print(slmin, slmax, slminRSI, slmaxRSI)

    # start_idx = -100
    # end_idx = len(df)

    # signal = divsignal(df, backcandles, plot_flag = plot_flag)
    signal = divsignal2(df, tic, backcandles, plot_flag = plot_flag)
    return signal


# def when_to_re_enter(r, latest_exit_index):

#     for i in range(latest_exit_index + 1, len(r)):
#         n1 = 30
#         n2 = 10
#         last_30_avg = r[max(i - n1, 0) : i].mean()
#         last_30_std =  r[max(i - n1, 0) : i].std()

#         last_10_avg = r[max(i - n2, 0) : i].mean()
#         last_10_std = r[max(i - n2, 0) : i].std()

#         if last_30_avg < 0 and last_10_avg > 0 and last_10_std < last_30_std:
#             # if last_30_avg <= last_10_avg and last_10_std <= last_30_std:
#             return i

#     return len(r)-1

class Load_n_Preprocess:
    def __init__(self, tickers_list, start_date, end_date, path_data = None):
        self.tickers_list = tickers_list
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.path_data = path_data

    def load_data(self):
        file_ext = os.path.splitext(self.path_data)[-1]
        if file_ext == '.csv':
            df_tics = pd.read_csv(self.path_data)
        elif file_ext == '.h5':
            df_tics = pd.read_hdf(self.path_data, "df",mode = 'r') 
            df_tics = df_tics[df_tics['volume']!=0]                   # special care to volume

        if len(self.tickers_list) == 0:
            self.tickers_list = df_tics['tic'].unique().tolist()
        else:
            df_tics = df_tics[df_tics['tic'].isin(self.tickers_list)]
        
        df_tics['date'] = pd.to_datetime(df_tics['date'])               #,utc=True)              # convert date column to datetime
        df_tics = df_tics.sort_values(by=['date', 'tic'],ignore_index=True)
        df_tics.index = df_tics['date'].factorize()[0]

        return df_tics
    
    def clean_data(self, df_tics):
        # if len(df_tics['tics'].unique()) >1:
        uniqueDates = df_tics['date'].unique()
        # print("===================================================")
        # print(f'Number of Unique dates in between {self.start_date.date()} and {self.end_date.date()} is {len(uniqueDates)}')
        # print("===================================================")
        df_dates = pd.DataFrame(uniqueDates, columns=['date'])

        df_tics_list = []
        for tic in self.tickers_list:
            df_tic = df_tics[df_tics['tic'] == tic]
            df_tic = df_dates.merge(df_tic, on='date', how='left')
            df_tic['tic'] = tic
            # print("No. of missing values before imputation for %5s = %5d"%(tic,df_tic['close'].isna().sum()))
            df_tic = df_tic.fillna(method='ffill').fillna(method='bfill')
            df_tics_list.append(df_tic)

        df_tics = pd.concat(df_tics_list)
        df_tics = df_tics.sort_values(by=['date', 'tic'],ignore_index=True)
        df_tics = df_tics[(df_tics['date'] >= self.start_date) & (df_tics['date'] <= self.end_date)]

        df_tics.index = df_tics['date'].factorize()[0]
        print("Data cleaned!")
        return df_tics
    
def divsignal2(df, tic, nbackcandles, plot_flag = False):      # pivot point to find divergence 
    backcandles = nbackcandles 
    # candleid = int(x.name)

    closp = np.array([])
    xxclos = np.array([])
    
    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])

    maximRSI = np.array([])
    minimRSI = np.array([])
    xxminRSI = np.array([])
    xxmaxRSI = np.array([])

    for i in range(len(df)):
        closp = np.append(closp, df.iloc[i].close)
        xxclos = np.append(xxclos, i)
        if df.iloc[i].price_pivot_id == 1:
            minim = np.append(minim, df.iloc[i].low)
            xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
        if df.iloc[i].price_pivot_id == 2:
            maxim = np.append(maxim, df.iloc[i].high)
            xxmax = np.append(xxmax, i) # df.iloc[i].name
        if df.iloc[i].RSI_pivot_id == 1:
            minimRSI = np.append(minimRSI, df.iloc[i].RSI)
            xxminRSI = np.append(xxminRSI, df.iloc[i].name)
        if df.iloc[i].RSI_pivot_id == 2:
            maximRSI = np.append(maximRSI, df.iloc[i].RSI)
            xxmaxRSI = np.append(xxmaxRSI, df.iloc[i].name)

    if maxim.size<2 or minim.size<2 or maximRSI.size<2 or minimRSI.size<2:
        print("not sufficient points")
        return 0

    slclos, interclos = np.polyfit(xxclos, closp, 1)
    if slclos > 1e-4 and (maximRSI.size<2 or maxim.size<2):
        return 0
    if slclos < -1e-4 and (minimRSI.size<2 or minim.size<2):
        return 0
    # print(tic)
    if plot_flag == True:
        fig = make_subplots(rows=2, cols=1)
        fig.append_trace(go.Candlestick(x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close']), row=1, col=1)
        fig.add_scatter(x=df.index, y=df['price_point_pos'], mode="markers",
                        marker=dict(size=4, color="MediumPurple"),
                        name="pivot", row=1, col=1)
        fig.add_trace(go.Scatter(x=xxmin[-2:], y=minim[-2:], mode='lines', name='min slope'), row=1, col=1)
        fig.add_trace(go.Scatter(x=xxmax[-2:], y=maxim[-2:], mode='lines', name='max slope'), row=1, col=1)

        fig.append_trace(go.Scatter(x=df.index, y=df['RSI']), row=2, col=1)
        fig.add_scatter(x=df.index, y=df['RSI_point_pos'], mode="markers",
                        marker=dict(size=2, color="MediumPurple"),
                        name="pivot", row=2, col=1)
        fig.add_trace(go.Scatter(x=xxminRSI[-2:], y=minimRSI[-2:], mode='lines', name='min slope'), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxmaxRSI[-2:], y=maximRSI[-2:], mode='lines', name='max slope'), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    # print(f"slcos = {slclos}, maximRSI_1 = {maximRSI[-2]}, maximRSI_2 = {maximRSI[-1]}")
    # print(f"slcos = {slclos}, maxim_1 = {minim[-2]}, maxim_2 = {minim[-1]}")

    # signal decisions here !!!
    if slclos > 1e-4:
        if maxim[-1] > maxim[-2] and maximRSI[-1] > maximRSI[-2]:
            return 11                   # price increase rsi increase (bullish)
        if maxim[-1] > maxim[-2] and maximRSI[-1] < maximRSI[-2]:
            return 12                   # price increase and rsi decrease
        if maxim[-1] < maxim[-2] and maximRSI[-1] > maximRSI[-2]:
            return 13                   # price decrease and rsi increase
        if maxim[-1] < maxim[-2] and maximRSI[-1] < maximRSI[-2]:
            return 14                   # price decrease and rsi decrease
        
    elif slclos < - 1e-4:
        if minim[-1] > minim[-2] and minimRSI[-1] > minimRSI[-2]:
            return 21                   # price increase rsi increase (bullish)
        if minim[-1] > minim[-2] and minimRSI[-1] < minimRSI[-2]:
            return 22                   # price increase and rsi decrease
        if minim[-1] < minim[-2] and minimRSI[-1] > minimRSI[-2]:
            return 23                   # price decrease and rsi increase
        if minim[-1] < minim[-2] and minimRSI[-1] < minimRSI[-2]:
            return 24                   # price decrease and rsi decrease

    # signal decisions here !!!
    # if slclos > 1e-4:
    #     if maximRSI[-1]<maximRSI[-2] and maxim[-1] > maxim[-2]:
    #         return 1        # price increase rsi decrease (bearish divergence)
    #     if maximRSI[-1]<maximRSI[-2] and maxim[-1] < maxim[-2]:
    #         return 3        # price decrease and rsi also decrease
    # elif slclos < -1e-4:
    #     if minimRSI[-1]>minimRSI[-2] and minim[-1]<minim[-2]:
    #         return 2        # price decrease rsi increase (bullish divergence)
    #     if minimRSI[-1]<minimRSI[-2] and minim[-1]<minim[-2]:  # price decrease and rsi also decrease
    #         return 3        # price decrease and rsi also decrease
    # else:
    #     return 0
    
def divsignal(df, nbackcandles, plot_flag = False):          # slope to find divergence
    backcandles = nbackcandles 
    # candleid = int(x.name)

    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])

    maximRSI = np.array([])
    minimRSI = np.array([])
    xxminRSI = np.array([])
    xxmaxRSI = np.array([])

    for i in range(len(df)):
        if df.iloc[i].price_pivot_id == 1:
            minim = np.append(minim, df.iloc[i].low)
            xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
        if df.iloc[i].price_pivot_id == 2:
            maxim = np.append(maxim, df.iloc[i].high)
            xxmax = np.append(xxmax, i) # df.iloc[i].name
        if df.iloc[i].RSI_pivot_id == 1:
            minimRSI = np.append(minimRSI, df.iloc[i].RSI)
            xxminRSI = np.append(xxminRSI, df.iloc[i].name)
        if df.iloc[i].RSI_pivot_id == 2:
            maximRSI = np.append(maximRSI, df.iloc[i].RSI)
            xxmaxRSI = np.append(xxmaxRSI, df.iloc[i].name)

    if plot_flag == True:
        fig = make_subplots(rows=2, cols=1)
        fig.append_trace(go.Candlestick(x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close']), row=1, col=1)
        fig.add_scatter(x=df.index, y=df['price_point_pos'], mode="markers",
                        marker=dict(size=4, color="MediumPurple"),
                        name="pivot", row=1, col=1)
        fig.add_trace(go.Scatter(x=xxmin, y=slmin*xxmin + intercmin, mode='lines', name='min slope'), row=1, col=1)
        fig.add_trace(go.Scatter(x=xxmax, y=slmax*xxmax + intercmax, mode='lines', name='max slope'), row=1, col=1)

        fig.append_trace(go.Scatter(x=df.index, y=df['RSI']), row=2, col=1)
        fig.add_scatter(x=df.index, y=df['RSI_point_pos'], mode="markers",
                        marker=dict(size=2, color="MediumPurple"),
                        name="pivot", row=2, col=1)
        fig.add_trace(go.Scatter(x=xxminRSI, y=slminRSI*xxminRSI + intercminRSI, mode='lines', name='min slope'), row=2, col=1)
        fig.add_trace(go.Scatter(x=xxmaxRSI, y=slmaxRSI*xxmaxRSI + intercmaxRSI, mode='lines', name='max slope'), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    if maxim.size<2 or minim.size<2 or maximRSI.size<2 or minimRSI.size<2:
        print("not sufficient points")
        return 0
    
    slmin, intercmin = np.polyfit(xxmin, minim,1)
    slmax, intercmax = np.polyfit(xxmax, maxim,1)
    slminRSI, intercminRSI = np.polyfit(xxminRSI, minimRSI,1)
    slmaxRSI, intercmaxRSI = np.polyfit(xxmaxRSI, maximRSI,1)
    
    
    if slmin > 1e-4 and slmax > 1e-4 and slmaxRSI <-0.1:
        return 1             # bearish divergence
    elif slmin < -1e-4 and slmax < -1e-4 and slminRSI > 0.1:
        return 2             # bullish divergence
    else:
        return 0             # hidden divergence

def main(today,sheet):
    #%% parameter initialization
    right_window = 5
    left_window = 5
    backcandles = 60

    # tickers_list = []
    tickers_list = ['AES']
    # tickers_list = ['AES','ALLE','AMCR','AME']
    
    if len(tickers_list)==1:
        plot_flag = True
    else:
        plot_flag = False
    
    # start_date = pd.to_datetime('2023-01-01').date()
    # end_date = pd.to_datetime('2021-12-31')
    end_date =  today
    start_date = end_date - timedelta(days = 125)

    #%% Load and preprocess the data (data imputation)
    LP = Load_n_Preprocess(tickers_list = tickers_list,
                           start_date = start_date,
                           end_date = end_date,
                           path_data = "datasets/df_SnP_500_ohlcv.h5",)
    df_tics = LP.load_data()
    df_tics = LP.clean_data(df_tics)

    #%% Generate and Filter Signals
    signal_dict = {}
    tickers_list = df_tics['tic'].unique().tolist()
    with alive_bar(len(tickers_list), force_tty = True) as bar:
        for tic in tickers_list:
            time.sleep(0.005)
            bar()

            df_tic = df_tics[df_tics['tic'] == tic]
            df_tic =  df_tic.drop(columns = ['tic','adj_close'])

            signal = RSI_divergence(df_tic, tic, 
                                    right_window = right_window, 
                                    left_window = left_window, 
                                    backcandles = backcandles, 
                                    plot_flag = plot_flag,
                                    )
            
            signal_dict[tic] = signal

    df_signals = pd.DataFrame(signal_dict.items(),columns=['tic', 'signal'])
    df_signals = df_signals[(df_signals['signal'] == 23)]




    wb = xw.Book("results/df_signals.xlsx")
    ws = wb.sheets[sheet]
    ws["A1"].options(pd.DataFrame, header = True, index = False, expand='table').value = pd.DataFrame(np.nan, index=np.arange(550), columns=['A', 'B'])
    ws["A1"].options(pd.DataFrame, header = True, index = False, expand='table').value = df_signals
    # df_signals.to_excel("results/df_signals.xlsx", index = False)
    wb.save()

    return df_signals



if __name__ == "__main__":
    today =  pd.to_datetime('today').date()
    stocks_list_today = main(today,"RSI_Signals_Today")

    # yesterday =  pd.to_datetime('today').date()- timedelta(days = 5)
    # stocks_list_yesterday = main(yesterday,"RSI_Signals_Yesterday")

    # stocks_recommendation = list(set(stocks_list_today['tic'])- set(stocks_list_yesterday['tic']) )

    # wb = xw.Book("results/df_signals.xlsx")
    # ws = wb.sheets["Common"]
    # ws["A1"].options(pd.DataFrame, header = True, index = False, expand='table').value = pd.DataFrame(np.nan, index=np.arange(550), columns=['A', 'B'])
    # ws["A1"].options(pd.DataFrame, header = True, index = False, expand='table').value = pd.DataFrame(stocks_recommendation,columns=['tic'])
    # # df_signals.to_excel("results/df_signals.xlsx", index = False)
    # wb.save()
    
    # print(stocks_recommendation)


