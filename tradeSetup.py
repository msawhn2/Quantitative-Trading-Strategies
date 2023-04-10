#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import quandl
import functools
import seaborn as sns
# import plotnine as p9
pd.options.display.float_format = '{:,.4f}'.format
from itertools import permutations
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import wrds
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import datetime
from itertools import product
import pyfolio as pf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import norm
sns.set_theme()
import plotly.io as pio
pio.renderers.default = 'svg'


# In[8]:


key = 'qfe4Bb4GV7d_kaAkhQho'


# ## Functions

# In[9]:


@functools.lru_cache(maxsize=16)
def get_quandl_data(security, start, end):
    
    qdata = quandl.get_table('QUOTEMEDIA/PRICES',
                            ticker = security, api_key = key,
                            date = {'gte':start, 'lte':end})
    qdata = qdata[['date', 'adj_close', 'volume']]
    qdata.sort_values('date',inplace=True)
    qdata.set_index('date',inplace=True)

    return qdata


# In[10]:


@functools.lru_cache(maxsize=16)
def get_quandl_data_full(security, start, end):
    
    qdata = quandl.get_table('QUOTEMEDIA/PRICES',
                            ticker = security, api_key = key,
                            date = {'gte':start, 'lte':end})
    qdata.sort_values('date',inplace=True)
    qdata.set_index('date',inplace=True)

    return qdata


# In[11]:


@functools.lru_cache(maxsize=16)
def get_yield_curves(code, start_date, end_date):
    ycdata = quandl.get('YC/'+code, api_key = key, returns="pandas",                         start_date=start_date, end_date=end_date)
    return ycdata


# In[12]:


# Used from the Portfolio Theory class taught by Mark Hendricks
def performance_summary(return_data, ticker, annualization = 12):
    """ 
        Returns the Performance Stats for given set of returns
        Inputs: 
            return_data - DataFrame with Date index and Monthly Returns for different assets/strategies.
        Output:
            summary_stats - DataFrame with annualized mean return, vol, sharpe ratio. Skewness, Excess Kurtosis, \
                            Var (0.5) and CVaR (0.5) and drawdown based on monthly returns. 
    """
    summary_stats = return_data.mean().to_frame('Mean').apply(lambda x: x*annualization)
    summary_stats['Volatility'] = return_data.std().apply(lambda x: x*np.sqrt(annualization))
    summary_stats['Sharpe Ratio'] = summary_stats['Mean']/summary_stats['Volatility']

    summary_stats['Skewness'] = return_data.skew()
    summary_stats['Excess Kurtosis'] = return_data.kurtosis()
    summary_stats['VaR (0.05)'] = pd.to_numeric(return_data.iloc[:,0]).quantile(.05)
    summary_stats['CVaR (0.05)'] = return_data[return_data <= pd.to_numeric(return_data.iloc[:,0]).quantile(.05)]                                    .mean()
    
    wealth_index = 1000*(1+return_data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks

    summary_stats['Max Drawdown'] = drawdowns.min()
    summary_stats['Peak'] = [pd.to_numeric(previous_peaks[col][:pd.to_numeric(drawdowns[col]).idxmin()]).idxmax()                              for col in previous_peaks.columns]
    summary_stats['Bottom'] = pd.to_numeric(drawdowns.iloc[:,0]).idxmin()
    
    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][:pd.to_numeric(drawdowns[col]).idxmin()].max()
        recovery_wealth = pd.DataFrame([wealth_index[col][pd.to_numeric(drawdowns[col]).idxmin():]]).T
        recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
    summary_stats['Recovery'] = recovery_date
    
    summary_stats = summary_stats.rename(index={'Return on Capital': ticker})
    
    return summary_stats


# ## Data Extraction
# 
# ### ETF Data
# 
# Note: The data is extracted from 2009-01-01 to 2021-12-31.

# In[13]:


tickers = ['SPY', 'IWM', 'GLD', 'XLE', 'IYF', 'IYR', 'AGG', 'TLT', 'XLV', 'HYG', 'EEM', 'DJP']
dfs = {}

for ticker in tickers:
    df = get_quandl_data(ticker, '2009-01-01', '2021-12-31')
    df["ADTV"] = df["volume"].rolling(window=60, min_periods=60).median()
    dfs[ticker] = df


# ### Options Data

# In[14]:


conn = wrds.Connection(wrds_username='anvita29')


# In[15]:


def get_option_data(year, ticker, op_type, price, curr_date, min_exp_date):
    
    table = {2009: 'optionm_all.opprcd2009', 2010: 'optionm_all.opprcd2010', 2011: 'optionm_all.opprcd2011',              2012: 'optionm_all.opprcd2012', 2013: 'optionm_all.opprcd2013', 2014: 'optionm_all.opprcd2014',              2015: 'optionm_all.opprcd2015', 2016: 'optionm_all.opprcd2016', 2017: 'optionm_all.opprcd2017',              2018: 'optionm_all.opprcd2018', 2019: 'optionm_all.opprcd2019', 2020: 'optionm_all.opprcd2020',              2021: 'optionm_all.opprcd2021'}
    
    order = {'P': 'DESC', 'C': 'ASC'}
    sign = {'P': '<', 'C': '>'}

    sql_query = """SELECT date, symbol, exdate, cp_flag, strike_price, best_bid, best_offer
                FROM """ + table[year] + \
                """ WHERE symbol ~ """ + "'" + ticker + "'" + \
                """ AND cp_flag = """ + "'" + op_type + "'" + \
                """ AND strike_price """ + sign[op_type] + """ """ + str(price) + \
                """ AND date = """ + "'" + curr_date + "'" + \
                """ AND exdate >= """ + "'" + min_exp_date + "'" + \
                """ ORDER BY strike_price """ + order[op_type] + ", best_offer ASC" + \
                """ LIMIT 1"""
    
    result = conn.raw_sql(sql_query)
        
    return result


# ## Motivation

# In[16]:


def rsi_indicator():
    sample_tickers=['IWM']
    for ticker in sample_tickers:
        df_btc= dfs[ticker]
        change = df_btc["adj_close"].diff()
        change_up = change.copy()
        change_down = change.copy()
        change_up[change_up<0] = 0
        change_down[change_down>0] = 0
        avg_up = change_up.rolling(14).mean()
        avg_down = change_down.rolling(14).mean().abs()
        rsi = 100 * avg_up / (avg_up + avg_down)
        plt.rcParams['figure.figsize'] = (10, 5)
        ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
        ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
        ax1.plot(df_btc['adj_close'], linewidth=2)
        ax1.set_title(ticker+ ' Close Price')
        ax2.set_title('Relative Strength Index')
        ax2.plot(rsi, color='orange', linewidth=1)
        ax2.axhline(30, linestyle='--', linewidth=1.5, color='green')
        ax2.axhline(70, linestyle='--', linewidth=1.5, color='red')
        plt.show()


# In[17]:


def momentum_indicator():
    length = 20
    mult = 2
    length_KC = 20
    mult_KC = 1.5

    GLD_ = get_quandl_data_full('GLD', '2009-01-01', '2021-12-31')
    df= GLD_
    m_avg = df['adj_close'].rolling(window=length).mean()
    m_std = df['adj_close'].rolling(window=length).std(ddof=0)
    df['upper_BB'] = m_avg + mult * m_std
    df['lower_BB'] = m_avg - mult * m_std
    df['tr0'] = abs(df["high"] - df["low"])
    df['tr1'] = abs(df["high"] - df["close"].shift())
    df['tr2'] = abs(df["low"] - df["close"].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    range_ma = df['tr'].rolling(window=length_KC).mean()
    df['upper_KC'] = m_avg + range_ma * mult_KC
    df['lower_KC'] = m_avg - range_ma * mult_KC
    df['squeeze_on'] = (df['lower_BB'] > df['lower_KC']) & (df['upper_BB'] < df['upper_KC'])
    df['squeeze_off'] = (df['lower_BB'] < df['lower_KC']) & (df['upper_BB'] > df['upper_KC'])
    highest = df['high'].rolling(window = length_KC).max()
    lowest = df['low'].rolling(window = length_KC).min()
    m1 = (highest + lowest) / 2
    df['value'] = (df['close'] - (m1 + m_avg)/2)
    fit_y = np.array(range(0,length_KC))
    df['value'] = df['value'].rolling(window = length_KC).apply(lambda x : np.polyfit(fit_y, x, 1)[0] * (length_KC-1) +np.polyfit(fit_y, x, 1)[1], raw=True)
    long_cond1 = (df['squeeze_off'][-2] == False) & (df['squeeze_off'][-1] == True) 
    long_cond2 = df['value'][-1] > 0
    enter_long = long_cond1 and long_cond2
    short_cond1 = (df['squeeze_off'][-2] == False) & (df['squeeze_off'][-1] == True) 
    short_cond2 = df['value'][-1] < 0
    enter_short = short_cond1 and short_cond2

    df= df[3100:]
    ohcl = df[['open', 'high', 'close', 'low']]

    colors = []
    for ind, val in enumerate(df['value']):
        if val >= 0:
            color = 'green'
            if val > df['value'][ind-1]:
                color = 'lime'
        else:
            color = 'maroon'
            if val < df['value'][ind-1]:
                color='red'
        colors.append(color)

    apds = [mpf.make_addplot(df['value'], panel=1, type='bar', color=colors, alpha=0.8, secondary_y=False),
            mpf.make_addplot([0] * len(df), panel=1, type='scatter', marker='x', markersize=50, color=['gray' if s else 'black' for s in df['squeeze_off']], secondary_y=False)]

    fig, axes = mpf.plot(ohcl, 
                  volume_panel = 2,
                  figratio=(2,1),
                  figscale=1, 
                  type='candle',
                  title= 'GLD',
                  addplot=apds,
                  returnfig=True)


# In[18]:


def corr_between_assets():
    corr_df= pd.DataFrame()
    for ticker in tickers:
        corr_df[ticker]= dfs[ticker]['adj_close'].pct_change()
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(corr_df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


# In[19]:


def risk_hedging_with_options():
    a = get_option_data(2020, 'SPY', 'P', 296000, '2020-01-02', '2020-01-31')
    b = get_option_data(2020, 'SPY', 'C', 296000, '2020-01-02', '2020-01-31')
    price = np.arange(0,500,1)
    strike = a['strike_price'][0]/1000
    premium_call = b['best_offer'][0]
    premium_put = a['best_offer'][0]
    payoff_long_call = [max(-premium_call, i-strike-premium_call) for i in price]
    payoff_long_put = [max(-premium_put, strike-i-premium_put) for i in price]
    payoff = np.sum([payoff_long_call, payoff_long_put], axis=0)

    df = pd.DataFrame()
    df['Long Call'] = payoff_long_call
    df['Long Put']= payoff_long_put
    df['Long Straddle']= payoff

    fig = px.line(df, render_mode='svg')

    fig = fig.update_layout(
        width       = 650,
        height      = 350,
        title       = dict(text='Hedging with Options', font=dict(size=25), x=0.5),
        yaxis_title = dict(text='Payoff', font=dict(size=13)),
        xaxis_title = dict(text='Strike', font=dict(size=13)),
        margin      = dict(l=0, r=20, t=55, b=20),
        legend_title = 'Option',
        showlegend  = True
    )
    fig.show("notebook")


# ## Trade Setup

# In[20]:


idx_start = dfs['SPY'].reset_index().groupby(dfs['SPY'].index.to_period('M'))['date'].idxmin()
month_start = dfs['SPY'].iloc[idx_start].index

idx_end = dfs['SPY'].reset_index().groupby(dfs['SPY'].index.to_period('M'))['date'].idxmax()
month_end = dfs['SPY'].iloc[idx_end].index


# In[21]:


def momentum_based_signal(df_orig):
    df = df_orig.copy()
    returns = df[['adj_close']].dropna().resample('MS').bfill().pct_change()
    idx = df.reset_index().groupby(df.index.to_period('M'))['date'].idxmin()
    returns.index = df.iloc[idx].index
    df['Returns'] = returns
    df = df.dropna()
    
    raw_mom = df['Returns'].rolling(12).mean()
    mean = raw_mom.rolling(12, min_periods=1).mean()
    std = raw_mom.rolling(12, min_periods=1).std()
    
    df['Momentum Signal'] = (raw_mom - mean) / std
    
    return df


# ## Function to get Theta
# 
# $$ \text{B11 is defined as the correlation of asset 1's returns with asset 1's rolling momentum signal}$$
# $$ \text{B22 is defined as the correlation of asset 2's returns with asset 2's rolling momentum signal}$$
# $$ \text{B12 is defined as the correlation of asset 1's returns with asset 2's rolling momentum signal}$$
# $$ \text{B21 is defined as the correlation of asset 2's returns with asset 1's rolling momentum signal}$$
# $$ \text{Rho12 is defined as the correlation of asset 1's momentum signal and the correlation of asset 2's momentum signal}$$
# $$ \text { Theta: composite score given by the below forumula} $$
# $$ (B11 - B12 + B22 - B21) \div \sqrt{(1- Rho12)\div (\pi)} $$

# In[22]:


def composite_score(df1, df2, win):
    results = pd.DataFrame()
    
    results['B11'] = df1['Momentum Signal'].shift(1).rolling(win).corr(df1['Returns'])
    results['B22'] = df2['Momentum Signal'].shift(1).rolling(win).corr(df2['Returns'])
    
    results['Rho12'] = df1['Momentum Signal'].shift(1).rolling(win).corr(df2['Momentum Signal'].shift(1))
    
    results['B12'] = df2['Momentum Signal'].shift(1).rolling(win).corr(df1['Returns'])
    results['B21'] = df1['Momentum Signal'].shift(1).rolling(win).corr(df2['Returns'])
    
    results['Theta'] = ((results['B11'] - results['B12']) + (results['B22'] - results['B21']))                         * np.sqrt((1 - results['Rho12']) / np.pi)
    
    return results.dropna()


# ## Function to determine daily positions
# 
# - The window parameter is used to determine the composite score, θij, using a rolling window of trailing signals and returns of length **win** months
# - The number of pairs is used to determine the top **n** pairs for inclusion in the portfolio

# In[23]:


def daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p):
    
    combs = [(a, b) for a, b in combinations(tickers, 2)]
    
    thetas = pd.DataFrame()
    buy = pd.DataFrame(columns = ['Ticker', 'Price', 'Theta^p', 'volume', 'ADTV'])
    sell = pd.DataFrame(columns = ['Ticker', 'Price', 'Theta^p','volume', 'ADTV'])
    buy_data = []
    sell_data = []
    
    for comb in combs:
        asset1 = comb[0]
        asset2 = comb[1]
        
        thetas[f'{asset1}_{asset2}'] = composite_score(monthly_df[asset1], monthly_df[asset2], win).iloc[:, -1]
        
    top_n = pd.DataFrame(thetas.apply(lambda x:list(thetas.columns[np.array(x).argsort()[::-1][:n]]), axis=1)                        .to_list(), columns=list(range(1, n+1)), index = thetas.index)
    
    thetas_daily = pd.DataFrame(data=thetas, index=dfs['SPY'].index).ffill().dropna()
    top_n_daily = pd.DataFrame(data=top_n, index=dfs['SPY'].index).ffill().dropna()
    
    
    flags = {1: False, 2: False, 3: False, 4: False}

    for date in top_n_daily.index:
        pairs = top_n_daily.loc[date].apply(lambda x: pd.Series(str(x).split('_')))
        
    
        for index in pairs.index:
            asset1 = pairs.loc[index, 0]
            asset2 = pairs.loc[index, 1]

            if date in month_start:
                flags[index] = monthly_df[asset1].loc[date, 'Momentum Signal'] >                                monthly_df[asset2].loc[date, 'Momentum Signal']

            theta = thetas_daily.loc[date, f'{asset1}_{asset2}'] ** p
            asset1_close = daily_df[asset1].loc[date, 'adj_close']
            asset2_close = daily_df[asset2].loc[date, 'adj_close']
            asset1_volume = daily_df[asset1].loc[date, 'volume']
            asset2_volume = daily_df[asset2].loc[date, 'volume']
            asset1_adtv = daily_df[asset1].loc[date, 'ADTV']
            asset2_adtv = daily_df[asset2].loc[date, 'ADTV']

            if flags[index]:
                buy_data.append([date, asset1, asset1_close, theta, asset1_volume, asset1_adtv])
                sell_data.append([date, asset2, asset2_close, theta, asset2_volume, asset2_adtv])
            else:
                buy_data.append([date, asset2, asset2_close, theta, asset2_volume, asset2_adtv])
                sell_data.append([date, asset1, asset1_close, theta, asset1_volume, asset2_adtv])
    
    buy = pd.DataFrame(buy_data, columns=['Date', 'Ticker','Price','Theta^p','Volume','ADTV'])
    buy = buy.set_index('Date')
    sell = pd.DataFrame(sell_data, columns=['Date', 'Ticker','Price','Theta^p','Volume','ADTV'])
    sell = sell.set_index('Date')
    buy['Weights'] = buy['Theta^p'] / (buy.groupby(buy.index)['Theta^p'].sum() * 2)
    sell['Weights'] = sell['Theta^p'] / (sell.groupby(sell.index)['Theta^p'].sum() * 2)

    return thetas_daily, top_n_daily, buy, sell


# ## Fama French Risk Free Rate
# 
# We use the Fama-French Risk Free Rate to determine the funding rate that will be used in the strategies that we test.

# In[24]:


@functools.lru_cache(maxsize=16)
def get_ff_data(start, end):
    
    ff_data = pd.read_csv(r'F-F_Research_Data_Factors_daily.CSV', skiprows = [0, 1, 2, 3, 25384])
    ff_data.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
    ff_data['Date'] = pd.to_datetime(ff_data['Date'].astype(str), format='%Y-%m-%d')
    ff_data.set_index('Date',inplace=True)
    ff_data = ff_data[start:end]
    funding = ff_data[['RF']]
    
    return funding

