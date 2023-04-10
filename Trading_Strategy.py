#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tradeSetup

from tradeSetup import *


# # Strategy Implementations

# ## Strategy with both Put and Call Options
# 
# Here, we will be backtesting a strategy where we source put and call options on the most weighted long and short position, respecitvely.

# In[12]:


funding_rate = get_ff_data('2009-01-01','2021-12-31')


# In[11]:


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
                flags[index] = monthly_df[asset1].loc[date, 'Momentum Signal'] > monthly_df[asset2].loc[date, 'Momentum Signal']

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


# In[4]:


def momentum_pairs_strategy(tickers, monthly_df, daily_df, win, n, p, Capital, funding=False, transaction_cost=False):
    
    thetas_daily, top_n_daily, buy, sell = daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p)
    buy['AUM'] = (buy['ADTV']* 0.1) / buy['Weights']
    sell['AUM'] = ( sell['ADTV']* 0.1) / sell['Weights']

    results = pd.DataFrame(index=thetas_daily.index, columns=['Action', 'Long Position', 'Short Position',                                                               'Put', 'Put Position', 'Call', 'Call Position',
                                                              'Net Position', 'ETF PnL', 'Realized PnL', \
                                                        'Cumulative PnL', 'Cash','Return on Capital'])
    if funding:
        funding_cost = Capital*(funding_rate/365)   
        
    else:
        funding_cost = pd.DataFrame(0, index=np.arange(len(thetas_daily)), columns=['RF'])
        funding_cost.index = thetas_daily.index
    
    for curr_date in thetas_daily.index:
        
        if curr_date in month_start: # open positions
            curr_buy = buy[(buy.index.month == curr_date.month) & (buy.index.year == curr_date.year)]
            curr_sell = sell[(sell.index.month == curr_date.month) & (sell.index.year == curr_date.year)]
            Capacity = min(min(curr_buy.loc[curr_date, 'AUM']), min(curr_sell.loc[curr_date, 'AUM']))
            
            buy_quantity = list(Capacity* curr_buy.loc[curr_date, 'Weights'] / curr_buy.loc[curr_date, 'Price'])
            sell_quantity = list(Capacity * curr_sell.loc[curr_date, 'Weights'] / curr_sell.loc[curr_date, 'Price'])
            
            
            buy_orig = list(curr_buy.loc[curr_date, 'Price'])
            sell_orig = list(curr_sell.loc[curr_date, 'Price'])
            
            i = month_start.get_loc(curr_date)
            min_exp_date = month_end[i]
            put_price = curr_buy.loc[curr_date, 'Price'][0] * 1000 * 0.98
            call_price = curr_sell.loc[curr_date, 'Price'][0] * 1000 * 1.02
            
            put_option = get_option_data(curr_date.year, curr_buy.loc[curr_date, 'Ticker'][0], 'P',                                          put_price, curr_date.strftime('%Y-%m-%d'), min_exp_date.strftime('%Y-%m-%d'))
            
            call_option = get_option_data(curr_date.year, curr_sell.loc[curr_date, 'Ticker'][0], 'C',                                         call_price, curr_date.strftime('%Y-%m-%d'), min_exp_date.strftime('%Y-%m-%d'))
            
            results.loc[curr_date, 'Action'] = 'Open'
            results.loc[curr_date, 'Long Position'] = -1 *(buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            
            try:
                results.loc[curr_date, 'Put'] = put_option['symbol'][0]
                results.loc[curr_date, 'Put Position'] =  -1*(put_option['best_offer'][0] * buy_quantity[0])
            except:
                put_option = put_option.append(pd.Series(np.nan, index=put_option.columns), ignore_index=True)
                results.loc[curr_date, 'Put'] = ''
                results.loc[curr_date, 'Put Position'] = 0
            
            try:
                results.loc[curr_date, 'Call'] = call_option['symbol'][0]
                results.loc[curr_date, 'Call Position'] = -1 * (call_option['best_offer'][0] * sell_quantity[0])
            except:
                call_option = call_option.append(pd.Series(np.nan, index=call_option.columns), ignore_index=True)
                results.loc[curr_date, 'Call'] = ''
                results.loc[curr_date, 'Call Position'] = 0
            
            results.loc[curr_date, 'ETF PnL'] = 0
            results.loc[curr_date, 'Realized PnL'] = 0
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) + abs(results.loc[curr_date,                                                                                               'Short Position'])
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF']+                         results.loc[curr_date, 'Net Position'] +  results.loc[curr_date, 'Put Position'] +                         results.loc[curr_date, 'Call Position'] -gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
   
            put_exercised = False
            call_exercised = False
            
        elif curr_date in month_end: # close positions
            
            put_trigger = curr_buy.loc[curr_date, 'Price'][0] * 1000 < put_option['strike_price'][0]
            call_trigger = curr_sell.loc[curr_date, 'Price'][0] * 1000 > call_option['strike_price'][0]
            
            results.loc[curr_date, 'Action'] = 'Close'
            
            results.loc[curr_date, 'Put'] = ''
            results.loc[curr_date, 'Put Position'] = max(0, put_option['strike_price'][0] / 1000 -                                                         curr_buy.loc[curr_date, 'Price'][0]) * buy_quantity[0]
            
            results.loc[curr_date, 'Call'] = ''
            results.loc[curr_date, 'Call Position'] = max(0, curr_sell.loc[curr_date, 'Price'][0] -                                                          call_option['strike_price'][0] / 1000) * sell_quantity[0]
            
            
            if put_trigger and call_trigger and not put_exercised and not call_exercised:
                results.loc[curr_date, 'Action'] = 'Close and Both Options Exercised'
                buy_quantity[0] = 0
                sell_quantity[0] = 0
                
            elif put_trigger and not put_exercised:
                results.loc[curr_date, 'Action'] = 'Close and Put Exercised'
                buy_quantity[0] = 0
            
            elif call_trigger and not call_exercised:
                results.loc[curr_date, 'Action'] = 'Close and Call Exercised'
                sell_quantity[0] = 0
            
            results.loc[curr_date, 'Long Position'] = -1*(sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            results.loc[curr_date, 'ETF PnL']=-1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'ETF PnL'] +                         results.loc[curr_date, 'Put Position'] + results.loc[curr_date, 'Call Position']
            
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF']+                             results.loc[curr_date, 'Net Position'] +  results.loc[curr_date, 'Put Position'] +                             results.loc[curr_date, 'Call Position'] -gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
          
            
        else: # maintain positions
            
            put_trigger = curr_buy.loc[curr_date, 'Price'][0] * 1000 < put_option['strike_price'][0]
            call_trigger = curr_sell.loc[curr_date, 'Price'][0] * 1000 > call_option['strike_price'][0]
                           
            results.loc[curr_date, 'Action'] = 'Maintain'
            results.loc[curr_date, 'Long Position'] = 0
            results.loc[curr_date, 'Short Position'] = 0
            results.loc[curr_date, 'Net Position'] = 0
            
            results.loc[curr_date, 'Put'] = ''
            results.loc[curr_date, 'Put Position'] = max(0, put_option['strike_price'][0] / 1000 -                                                         curr_buy.loc[curr_date, 'Price'][0]) * buy_quantity[0]
            
            results.loc[curr_date, 'Call'] = ''
            results.loc[curr_date, 'Call Position'] = max(0, curr_sell.loc[curr_date, 'Price'][0] -                                                          call_option['strike_price'][0] / 1000) * sell_quantity[0]
            
            results.loc[curr_date, 'ETF PnL']= -1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            if put_trigger and call_trigger and not put_exercised and not call_exercised:
                results.loc[curr_date, 'Action'] = 'Both Options Exercised'
                results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'Put Position'] +                                                          results.loc[curr_date, 'Call Position']
                buy_quantity[0] = 0
                sell_quantity[0] = 0
                put_exercised = True
                call_exercised = True
                
            elif put_trigger and not put_exercised:
                results.loc[curr_date, 'Action'] = 'Put Exercised'
                results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'Put Position']
                buy_quantity[0] = 0
                put_exercised = True
            
            elif call_trigger and not call_exercised:
                results.loc[curr_date, 'Action'] = 'Call Exercised'
                results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'Call Position']
                sell_quantity[0] = 0
                call_exercised = True
                
            else: # maintain
                results.loc[curr_date, 'Realized PnL'] = 0
            
            results.loc[curr_date, 'Cash'] =  Capital -  funding_cost.loc[curr_date, 'RF'] +                          results.loc[curr_date, 'Put Position'] + results.loc[curr_date, 'Call Position'] 
            Capital = results.loc[curr_date, 'Cash']
            
    results['Cumulative PnL'] = results['Realized PnL'].cumsum()
    results['Return on Capital'] = results['Realized PnL'] * 100 / results['Cash']
    
    return results


# ## Strategy with only Put Options
# 
# Here, we will be backtesting a strategy where we source only put options on the most weighted long position.

# In[5]:


def momentum_pairs_strategy_put(tickers, monthly_df, daily_df, win, n, p, Capital, funding=False,                                 transaction_cost=False):
    
    thetas_daily, top_n_daily, buy, sell = daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p)
    buy['AUM'] = (buy['ADTV']* 0.1) / buy['Weights']
    sell['AUM'] = ( sell['ADTV']* 0.1) / sell['Weights']

    results = pd.DataFrame(index=thetas_daily.index, columns=['Action', 'Long Position', 'Short Position',                                                               'Put', 'Put Position',
                                                              'Net Position', 'ETF PnL', 'Realized PnL', \
                                                        'Cumulative PnL', 'Cash','Return on Capital'])
    if funding:
        funding_cost = Capital*(funding_rate/365)   
        
    else:
        funding_cost = pd.DataFrame(0, index=np.arange(len(thetas_daily)), columns=['RF'])
        funding_cost.index = thetas_daily.index
    
    
    
    for curr_date in thetas_daily.index:
        
        if curr_date in month_start: # open positions
            curr_buy = buy[(buy.index.month == curr_date.month) & (buy.index.year == curr_date.year)]
            curr_sell = sell[(sell.index.month == curr_date.month) & (sell.index.year == curr_date.year)]
            Capacity = min(min(curr_buy.loc[curr_date, 'AUM']), min(curr_sell.loc[curr_date, 'AUM']))
            
            buy_quantity = list(Capacity* curr_buy.loc[curr_date, 'Weights'] / curr_buy.loc[curr_date, 'Price'])
            sell_quantity = list(Capacity * curr_sell.loc[curr_date, 'Weights'] / curr_sell.loc[curr_date, 'Price'])
            
            
            buy_orig = list(curr_buy.loc[curr_date, 'Price'])
            sell_orig = list(curr_sell.loc[curr_date, 'Price'])
            
            i = month_start.get_loc(curr_date)
            min_exp_date = month_end[i]
            put_price = curr_buy.loc[curr_date, 'Price'][0] * 1000 * 0.98
            
            put_option = get_option_data(curr_date.year, curr_buy.loc[curr_date, 'Ticker'][0], 'P',                                          put_price, curr_date.strftime('%Y-%m-%d'), min_exp_date.strftime('%Y-%m-%d'))
            
         
            results.loc[curr_date, 'Action'] = 'Open'
            results.loc[curr_date, 'Long Position'] = -1 *(buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            
            try:
                results.loc[curr_date, 'Put'] = put_option['symbol'][0]
                results.loc[curr_date, 'Put Position'] =  -1*(put_option['best_offer'][0] * buy_quantity[0])
            except:
                put_option = put_option.append(pd.Series(np.nan, index=put_option.columns), ignore_index=True)
                results.loc[curr_date, 'Put'] = ''
                results.loc[curr_date, 'Put Position'] = 0
            
            
            results.loc[curr_date, 'ETF PnL'] = 0
            results.loc[curr_date, 'Realized PnL'] = 0
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
            
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF']+                             results.loc[curr_date, 'Net Position'] +  results.loc[curr_date, 'Put Position']                              - gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
   
            put_exercised = False
            
        elif curr_date in month_end: # close positions
            
            put_trigger = curr_buy.loc[curr_date, 'Price'][0] * 1000 < put_option['strike_price'][0]            
            results.loc[curr_date, 'Action'] = 'Close'
            
            results.loc[curr_date, 'Put'] = ''
            results.loc[curr_date, 'Put Position'] = max(0, put_option['strike_price'][0] / 1000 -                                                         curr_buy.loc[curr_date, 'Price'][0]) * buy_quantity[0]
                
            if put_trigger and not put_exercised:
                results.loc[curr_date, 'Action'] = 'Close and Put Exercised'
                buy_quantity[0] = 0

            
            results.loc[curr_date, 'Long Position'] = -1*(sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            results.loc[curr_date, 'ETF PnL']=-1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'ETF PnL'] +                                     results.loc[curr_date, 'Put Position'] 
            
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF']+                             results.loc[curr_date, 'Net Position'] +  results.loc[curr_date, 'Put Position']                             -gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
          
            
        else: # maintain positions
            
            put_trigger = curr_buy.loc[curr_date, 'Price'][0] * 1000 < put_option['strike_price'][0]
                           
            results.loc[curr_date, 'Action'] = 'Maintain'
            results.loc[curr_date, 'Long Position'] = 0
            results.loc[curr_date, 'Short Position'] = 0
            results.loc[curr_date, 'Net Position'] = 0
            
            results.loc[curr_date, 'Put'] = ''
            results.loc[curr_date, 'Put Position'] = max(0, put_option['strike_price'][0] / 1000 -                                                         curr_buy.loc[curr_date, 'Price'][0]) * buy_quantity[0]
                       
            results.loc[curr_date, 'ETF PnL']= -1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            if put_trigger and not put_exercised:
                results.loc[curr_date, 'Action'] = 'Put Exercised'
                results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'Put Position']
                buy_quantity[0] = 0
                put_exercised = True

            else: # maintain
                results.loc[curr_date, 'Realized PnL'] = 0
            
            results.loc[curr_date, 'Cash'] =  Capital -  funding_cost.loc[curr_date, 'RF'] +                                  results.loc[curr_date, 'Put Position'] 
            Capital = results.loc[curr_date, 'Cash']
            
    results['Cumulative PnL'] = results['Realized PnL'].cumsum()
    results['Return on Capital'] = results['Realized PnL'] * 100 / results['Cash']
    
    return results


# ## Strategy with only Call Options
# 
# Here, we will be backtesting a strategy where we source only call options on the most weighted short position.

# In[6]:


def momentum_pairs_strategy_call(tickers, monthly_df, daily_df, win, n, p, Capital, funding=False,                                  transaction_cost=False):
    
    thetas_daily, top_n_daily, buy, sell = daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p)
    
    buy['AUM'] = (buy['ADTV']* 0.1) / buy['Weights']
    sell['AUM'] = ( sell['ADTV']* 0.1) / sell['Weights']

    results = pd.DataFrame(index=thetas_daily.index, columns=['Action', 'Long Position', 'Short Position',                                                               'Call', 'Call Position',
                                                              'Net Position', 'ETF PnL', 'Realized PnL', \
                                                        'Cumulative PnL', 'Cash','Return on Capital'])
    if funding:
        funding_cost = Capital*(funding_rate/365)   
        
    else:
        funding_cost = pd.DataFrame(0, index=np.arange(len(thetas_daily)), columns=['RF'])
        funding_cost.index = thetas_daily.index
    
    
    
    for curr_date in thetas_daily.index:
        
        if curr_date in month_start: # open positions
            curr_buy = buy[(buy.index.month == curr_date.month) & (buy.index.year == curr_date.year)]
            curr_sell = sell[(sell.index.month == curr_date.month) & (sell.index.year == curr_date.year)]
            Capacity = min(min(curr_buy.loc[curr_date, 'AUM']), min(curr_sell.loc[curr_date, 'AUM']))
            
            buy_quantity = list(Capacity* curr_buy.loc[curr_date, 'Weights'] / curr_buy.loc[curr_date, 'Price'])
            sell_quantity = list(Capacity * curr_sell.loc[curr_date, 'Weights'] / curr_sell.loc[curr_date, 'Price'])
            
            
            buy_orig = list(curr_buy.loc[curr_date, 'Price'])
            sell_orig = list(curr_sell.loc[curr_date, 'Price'])
            
            i = month_start.get_loc(curr_date)
            min_exp_date = month_end[i]
            call_price = curr_sell.loc[curr_date, 'Price'][0] * 1000 * 1.02
            
            call_option = get_option_data(curr_date.year, curr_sell.loc[curr_date, 'Ticker'][0], 'C',                                         call_price, curr_date.strftime('%Y-%m-%d'), min_exp_date.strftime('%Y-%m-%d'))
            
            results.loc[curr_date, 'Action'] = 'Open'
            results.loc[curr_date, 'Long Position'] = -1 *(buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            
          
            try:
                results.loc[curr_date, 'Call'] = call_option['symbol'][0]
                results.loc[curr_date, 'Call Position'] = -1 * (call_option['best_offer'][0] * sell_quantity[0])
            except:
                call_option = call_option.append(pd.Series(np.nan, index=call_option.columns), ignore_index=True)
                results.loc[curr_date, 'Call'] = ''
                results.loc[curr_date, 'Call Position'] = 0
            
            results.loc[curr_date, 'ETF PnL'] = 0
            results.loc[curr_date, 'Realized PnL'] = 0
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF'] +                         results.loc[curr_date, 'Net Position']  + results.loc[curr_date, 'Call Position']                         - gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
   
            call_exercised = False
            
        elif curr_date in month_end: # close positions
            
            call_trigger = curr_sell.loc[curr_date, 'Price'][0] * 1000 > call_option['strike_price'][0]
            
            results.loc[curr_date, 'Action'] = 'Close'
            
            results.loc[curr_date, 'Call'] = ''
            results.loc[curr_date, 'Call Position'] = max(0, curr_sell.loc[curr_date, 'Price'][0] -                                                          call_option['strike_price'][0] / 1000) * sell_quantity[0]
            
            if call_trigger and not call_exercised:
                results.loc[curr_date, 'Action'] = 'Close and Call Exercised'
                sell_quantity[0] = 0
            
            results.loc[curr_date, 'Long Position'] = -1*(sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            results.loc[curr_date, 'ETF PnL']=-1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'ETF PnL']  +                                                          results.loc[curr_date, 'Call Position']
            
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF'] +                             results.loc[curr_date, 'Net Position'] + results.loc[curr_date, 'Call Position']                             - gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
          
            
        else: # maintain positions
            
            call_trigger = curr_sell.loc[curr_date, 'Price'][0] * 1000 > call_option['strike_price'][0]
                           
            results.loc[curr_date, 'Action'] = 'Maintain'
            results.loc[curr_date, 'Long Position'] = 0
            results.loc[curr_date, 'Short Position'] = 0
            results.loc[curr_date, 'Net Position'] = 0
            
          
            results.loc[curr_date, 'Call'] = ''
            results.loc[curr_date, 'Call Position'] = max(0, curr_sell.loc[curr_date, 'Price'][0] -                                                          call_option['strike_price'][0] / 1000) * sell_quantity[0]
            
            results.loc[curr_date, 'ETF PnL']= -1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            if call_trigger and not call_exercised:
                results.loc[curr_date, 'Action'] = 'Call Exercised'
                results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'Call Position']
                sell_quantity[0] = 0
                call_exercised = True
                
            else: # maintain
                results.loc[curr_date, 'Realized PnL'] = 0
            
            results.loc[curr_date, 'Cash'] =  Capital -  funding_cost.loc[curr_date, 'RF']  +                                     results.loc[curr_date, 'Call Position'] 
            Capital = results.loc[curr_date, 'Cash']
            
    results['Cumulative PnL'] = results['Realized PnL'].cumsum()
    results['Return on Capital'] = results['Realized PnL'] * 100 / results['Cash']
    
    return results


# ## Strategy with No Options
# 
# Here, we will be backtesting a strategy where we source neither put nor call options.

# In[7]:


def momentum_pairs_strategy_no_options(tickers, monthly_df, daily_df, win, n, p, Capital, funding=False,                                        transaction_cost=False):
    
    thetas_daily, top_n_daily, buy, sell = daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p)
    
    buy['AUM'] = (buy['ADTV']* 0.1) / buy['Weights']
    sell['AUM'] = ( sell['ADTV']* 0.1) / sell['Weights']

    results = pd.DataFrame(index=thetas_daily.index, columns=['Action', 'Long Position', 'Short Position',                                                               'Net Position', 'ETF PnL', 'Realized PnL',                                                         'Cumulative PnL', 'Cash','Return on Capital'])
    if funding:
        funding_cost = Capital*(funding_rate/365)   
        
    else:
        funding_cost = pd.DataFrame(0, index=np.arange(len(thetas_daily)), columns=['RF'])
        funding_cost.index = thetas_daily.index
    
    
    for curr_date in thetas_daily.index:
        
        if curr_date in month_start: # open positions
            curr_buy = buy[(buy.index.month == curr_date.month) & (buy.index.year == curr_date.year)]
            curr_sell = sell[(sell.index.month == curr_date.month) & (sell.index.year == curr_date.year)]
            Capacity = min(min(curr_buy.loc[curr_date, 'AUM']), min(curr_sell.loc[curr_date, 'AUM']))
            
            buy_quantity = list(Capacity* curr_buy.loc[curr_date, 'Weights'] / curr_buy.loc[curr_date, 'Price'])
            sell_quantity = list(Capacity * curr_sell.loc[curr_date, 'Weights'] / curr_sell.loc[curr_date, 'Price'])
            
            
            buy_orig = list(curr_buy.loc[curr_date, 'Price'])
            sell_orig = list(curr_sell.loc[curr_date, 'Price'])

            
            results.loc[curr_date, 'Action'] = 'Open'
            results.loc[curr_date, 'Long Position'] = -1 *(buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            
          
            results.loc[curr_date, 'ETF PnL'] = 0
            results.loc[curr_date, 'Realized PnL'] = 0
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF'] +                                 results.loc[curr_date, 'Net Position']   -gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']

            
        elif curr_date in month_end: # close positions
            
            results.loc[curr_date, 'Action'] = 'Close'
            
            results.loc[curr_date, 'Long Position'] = -1*(sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            results.loc[curr_date, 'ETF PnL']=-1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'ETF PnL'] 
                                       
            
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF'] +                                     results.loc[curr_date, 'Net Position'] -gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
          
            
        else: # maintain positions
            
           
            results.loc[curr_date, 'Action'] = 'Maintain'
            results.loc[curr_date, 'Long Position'] = 0
            results.loc[curr_date, 'Short Position'] = 0
            results.loc[curr_date, 'Net Position'] = 0
            
          
            results.loc[curr_date, 'ETF PnL']= -1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            results.loc[curr_date, 'Realized PnL'] = 0
            
            results.loc[curr_date, 'Cash'] =  Capital -  funding_cost.loc[curr_date, 'RF']  
            Capital = results.loc[curr_date, 'Cash']
            
    results['Cumulative PnL'] = results['Realized PnL'].cumsum()
    results['Return on Capital'] = results['Realized PnL'] * 100 / results['Cash']
    
    return results


# ## Benchmark Strategy
# 
# The benchmark strategy is the standard top-down rank-based momentum based strategy, wherein each asset is ranked each month by its raw 12-month trailing momentum signal, we go long the top ≈ 1/3 of assets (4/13), short the bottom ≈ 1/3 (4/13), and neutral the middle ≈ 1/3 (5/13).

# In[8]:


def daily_pos_benchmark(tickers, monthly_df, daily_df, win, n, p):
    
    
    momentum = pd.DataFrame()
    buy = pd.DataFrame(columns = ['Ticker', 'Price', 'volume', 'ADTV','Weights'])
    sell = pd.DataFrame(columns = ['Ticker', 'Price','volume', 'ADTV','Weights'])
    buy_data = []
    sell_data = []
    
    for ticker in tickers:
        momentum[ticker] = monthly_df[ticker]['Momentum Signal']
        momentum = momentum.dropna()
    
    top_n = pd.DataFrame(momentum.apply(lambda x:list(momentum.columns[np.array(x).argsort()[::-1][:n]]), axis=1)                        .to_list(), columns=list(range(1, n+1)), index = momentum.index)
    bottom_n = pd.DataFrame(momentum.apply(lambda x:list(momentum.columns[np.array(x).argsort()[::-1][-n:]]), axis=1)                        .to_list(), columns=list(range(1, n+1)), index = momentum.index)
    
    momentum_daily = pd.DataFrame(data=momentum, index=dfs['SPY'].index).ffill().dropna()
    top_n_daily = pd.DataFrame(data=top_n, index=dfs['SPY'].index).ffill().dropna()
    bottom_n_daily = pd.DataFrame(data=bottom_n, index=dfs['SPY'].index).ffill().dropna()
    flags = {1: False, 2: False, 3: False, 4: False}

    for date in top_n_daily.index:
        
    
        for i in range(1, n+1):
            asset1 = top_n_daily.loc[date, i]
            asset2 = bottom_n_daily.loc[date, i]

          
            asset1_close = daily_df[asset1].loc[date, 'adj_close']
            asset2_close = daily_df[asset2].loc[date, 'adj_close']
            asset1_volume = daily_df[asset1].loc[date, 'volume']
            asset2_volume = daily_df[asset2].loc[date, 'volume']
            asset1_adtv = daily_df[asset1].loc[date, 'ADTV']
            asset2_adtv = daily_df[asset2].loc[date, 'ADTV']
            

            buy_data.append([date, asset1, asset1_close,  asset1_volume, asset1_adtv, 0.125])
            sell_data.append([date, asset2, asset2_close, asset2_volume, asset2_adtv,  0.125])
    
    buy = pd.DataFrame(buy_data, columns=['Date', 'Ticker','Price','Volume','ADTV', 'Weights'])
    buy = buy.set_index('Date')
    sell = pd.DataFrame(sell_data, columns=['Date', 'Ticker','Price','Volume','ADTV', 'Weights'])
    sell = sell.set_index('Date')


    return  top_n_daily, buy, sell


# In[9]:


def momentum_pairs_strategy_benchmark(tickers, monthly_df, daily_df, win, n, p, Capital, funding=False,                                       transaction_cost=False):
    
    top_n_benchmark, buy_benchmark, sell_benchmark = daily_pos_benchmark(tickers, monthly_df, daily_df, 60, 4, 3)
    
    buy = buy_benchmark
    sell = sell_benchmark
    buy['AUM'] = (buy['ADTV']* 0.1) / buy['Weights']
    sell['AUM'] = ( sell['ADTV']* 0.1) / sell['Weights']

    results = pd.DataFrame(index=top_n_benchmark.index, columns=['Action', 'Long Position', 'Short Position',                                                               'Net Position', 'ETF PnL', 'Realized PnL',                                                         'Cumulative PnL', 'Cash','Return on Capital'])
    if funding:
        funding_cost = Capital*(funding_rate/365)   
        
    else:
        funding_cost = pd.DataFrame(0, index=np.arange(len(top_n_benchmark)), columns=['RF'])
        funding_cost.index = top_n_benchmark.index
    
    
    for curr_date in top_n_benchmark.index:
        
        if curr_date in month_start: # open positions
            curr_buy = buy[(buy.index.month == curr_date.month) & (buy.index.year == curr_date.year)]
            curr_sell = sell[(sell.index.month == curr_date.month) & (sell.index.year == curr_date.year)]
            Capacity = min(min(curr_buy.loc[curr_date, 'AUM']), min(curr_sell.loc[curr_date, 'AUM']))
            
            buy_quantity = list(Capacity* curr_buy.loc[curr_date, 'Weights'] / curr_buy.loc[curr_date, 'Price'])
            sell_quantity = list(Capacity * curr_sell.loc[curr_date, 'Weights'] / curr_sell.loc[curr_date, 'Price'])
            
            
            buy_orig = list(curr_buy.loc[curr_date, 'Price'])
            sell_orig = list(curr_sell.loc[curr_date, 'Price'])
            
            results.loc[curr_date, 'Action'] = 'Open'
            results.loc[curr_date, 'Long Position'] = -1 *(buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            results.loc[curr_date, 'ETF PnL'] = 0
            results.loc[curr_date, 'Realized PnL'] = 0
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF'] +                                     results.loc[curr_date, 'Net Position'] -gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
            
        elif curr_date in month_end: # close positions
            
            results.loc[curr_date, 'Action'] = 'Close'
            
            
            results.loc[curr_date, 'Long Position'] = -1*(sell_quantity * curr_sell.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Short Position'] =  (buy_quantity * curr_buy.loc[curr_date, 'Price']).sum()
            results.loc[curr_date, 'Net Position'] = results.loc[curr_date, 'Long Position'] +                                                         results.loc[curr_date, 'Short Position']
            results.loc[curr_date, 'ETF PnL']=-1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            results.loc[curr_date, 'Realized PnL'] = results.loc[curr_date, 'ETF PnL'] 
                                                     
            
            
            if transaction_cost:
                gross_amount = abs(results.loc[curr_date, 'Long Position']) +                                 abs(results.loc[curr_date, 'Short Position'])
              
            else:
                gross_amount = 0
            
            results.loc[curr_date, 'Cash'] = Capital - funding_cost.loc[curr_date, 'RF'] +                                 results.loc[curr_date, 'Net Position']  -gross_amount*0.001 
            Capital = results.loc[curr_date, 'Cash']
          
            
        else: # maintain positions
                           
            results.loc[curr_date, 'Action'] = 'Maintain'
            results.loc[curr_date, 'Long Position'] = 0
            results.loc[curr_date, 'Short Position'] = 0
            results.loc[curr_date, 'Net Position'] = 0
            
            results.loc[curr_date, 'ETF PnL']= -1*(sell_quantity*(curr_sell.loc[curr_date, 'Price']-sell_orig)).sum()                                            + (buy_quantity * (curr_buy.loc[curr_date, 'Price'] - buy_orig)).sum()
            
            results.loc[curr_date, 'Realized PnL'] = 0
            
            results.loc[curr_date, 'Cash'] =  Capital -  funding_cost.loc[curr_date, 'RF'] 
            Capital = results.loc[curr_date, 'Cash']
            
    results['Cumulative PnL'] = results['Realized PnL'].cumsum()
    results['Return on Capital'] = results['Realized PnL'] * 100 / results['Cash']
    
    return results


# ## Optimization of the Put+Call Strategy Parameters

# In[10]:


def optimize(strategy, tickers, monthly_df, daily_df, funding=False, transaction_cost=False):
    
    comparison = pd.DataFrame()
    
    win = [60, 90, 120]
    n = [2, 4, 6]
    
    params = list(product(win, n))
    
    for param in params:
        if strategy == 'Put+Call':
            result = momentum_pairs_strategy(tickers, monthly_df, daily_df, param[0], param[1], 3,                                              10e6, funding, transaction_cost)
        elif strategy == 'Put':
            result = momentum_pairs_strategy_put(tickers, monthly_df, daily_df, param[0], param[1], 3,                                              10e6, funding, transaction_cost)
        elif strategy == 'Call':
            result = momentum_pairs_strategy_call(tickers, monthly_df, daily_df, param[0], param[1], 3,                                              10e6, funding, transaction_cost)
        elif strategy == 'No Option':
            result = momentum_pairs_strategy_no_option(tickers, monthly_df, daily_df, param[0], param[1], 3,                                              10e6, funding, transaction_cost)
        elif strategy == 'Benchmark':
            result = momentum_pairs_strategy_benchmark(tickers, monthly_df, daily_df, param[0], param[1], 3,                                              10e6, funding, transaction_cost)
            
        index = len(comparison.index)
        print('Testing combination', index+1)

        comparison.loc[index, 'Window'] = param[0]
        comparison.loc[index, 'Number of Pairs'] = param[1]
        comparison.loc[index, 'Net Cash'] = result.iloc[-1]['Cash'] - result.iloc[0]['Cash']
        comparison.loc[index, 'Number of Trades'] = result[result['Action'] != 'Maintain']['Action'].count()
    
    return comparison.sort_values('Net Cash', ascending=False).reset_index(drop=True)

