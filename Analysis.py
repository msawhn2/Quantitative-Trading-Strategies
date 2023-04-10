#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import tradeSetup

from tradeSetup import *


# In[3]:


def all_stats(strat_returns, strat_returns_put, strat_returns_call, strat_returns_no_options, strat_returns_benchmark):
    
    all_stats = performance_summary(strat_returns[['Return on Capital']]/100,                                   'Put + Call Strategy', 252)
    all_stats = all_stats.append(performance_summary(strat_returns_put[['Return on Capital']]/100, 'Put Strategy', 252))
    all_stats = all_stats.append(performance_summary(strat_returns_call[['Return on Capital']]/100, 'Call Strategy', 252))
    all_stats = all_stats.append(performance_summary(strat_returns_no_options[['Return on Capital']]/100,                                                      'No Options Strategy', 252))
    all_stats = all_stats.append(performance_summary(strat_returns_benchmark.loc['2015-04-01':'2021-12-31',                                                                 ['Return on Capital']]/100, 'Benchmark Strategy', 252))
    
    return all_stats


# In[4]:


def covid_stats(strat_returns, strat_returns_put, strat_returns_call, strat_returns_no_options, strat_returns_benchmark):
    covid_stats = performance_summary(strat_returns.loc['2020-03-01':'2021-12-31', ['Return on Capital']]/100,                                   'Put + Call Strategy', 252)
    covid_stats = covid_stats.append(performance_summary(strat_returns_put.loc['2020-03-01':'2021-12-31',                                                                         ['Return on Capital']]/100, 'Put Strategy', 252))
    covid_stats = covid_stats.append(performance_summary(strat_returns_call.loc['2020-03-01':'2021-12-31',                                                                         ['Return on Capital']]/100, 'Call Strategy', 252))
    covid_stats = covid_stats.append(performance_summary(strat_returns_no_options.loc['2020-03-01':'2021-12-31',                                                                 ['Return on Capital']]/100, 'No Options Strategy', 252))
    covid_stats = covid_stats.append(performance_summary(strat_returns_benchmark.loc['2020-03-01':'2021-12-31',                                                                 ['Return on Capital']]/100, 'Benchmark Strategy', 252))

    return covid_stats


# In[5]:


def mean_returns_covid(df):
    df1 = df.copy()
    fig = go.Figure([go.Bar(x=df1['Mean'].index, y=df1['Mean'].loc[df1['Mean'].index])])
    
    fig = fig.update_layout(
        title='Average returns for each strategy in the COVID Era',
        xaxis_title='Strategy',
        yaxis_title='Average Returns'
    )
    
    for i, val in enumerate(df1['Mean']):
        if i ==1:
            yshift = -15
        elif i==3:
            yshift = -15
        else:
            yshift = -0.5
            
        fig =fig.add_annotation(
            x=df1['Mean'].index[i], 
            y=val, 
            text=str(round(val, 2)),
            font=dict(color='black'),  
            showarrow=False, 
            xanchor='center',  
            yanchor='bottom', 
            yshift=yshift  
        )
    
    
    fig.show("notebook")


# In[6]:


def plot_covid_weights(tickers, monthly_df, daily_df, win, n, p):
    thetas_daily, top_n_daily, buy, sell = daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p)
    buy_2020 = buy[buy.index.year == 2020]
    buy_2020['Month'] = buy_2020.index.month
    sell_2020 = sell[sell.index.year == 2020]
    sell_2020['Month'] = sell_2020.index.month

    sb1 = px.sunburst(buy_2020, path=['Month', 'Ticker'], values='Weights')
    sb2 = px.sunburst(sell_2020, path=['Month', 'Ticker'], values='Weights')

    fig = make_subplots(rows=1, cols=2, specs=[
        [{"type": "sunburst"}, {"type": "sunburst"}]
    ], subplot_titles=('Long Weights (2020)',  'Short Weights (2020)'))

    fig = fig.add_trace(sb1.data[0], row=1, col=1)
    fig = fig.add_trace(sb2.data[0], row=1, col=2)
    fig = fig.update_layout(title='Depiction of Weights by Tickers during COVID', title_x=0.5)
    fig.show("notebook")


# In[7]:


def performance_comparison_plots(df1_, df2_, df3_, df4_, df5_, value_column, returns = True):
    
    df1 = df1_.copy().reset_index()
    df1['year'] = df1['date'].dt.strftime('%Y')
    df2 = df2_.copy().reset_index()
    df2['year'] = df2['date'].dt.strftime('%Y')
    df3 = df3_.copy().reset_index()
    df3['year'] = df3['date'].dt.strftime('%Y')
    df4 = df4_.copy().reset_index()
    df4['year'] = df4['date'].dt.strftime('%Y')
    df5 = df5_.copy().reset_index()
    df5['year'] = df5['date'].dt.strftime('%Y')
    
    fig = make_subplots(rows=3, cols=2, subplot_titles=(value_column + ' with both options', value_column + ' with put options', value_column + ' with call options', value_column + ' without options', value_column + ' for Benchmark Portfolio'),
                    vertical_spacing=0.08)
    if returns:
        fig = fig.add_trace(
                go.Scatter(x=df1.sort_values('date')['date'], y=df1.sort_values('date')[value_column], mode='lines', name=value_column),
                row=1,
                col=1
            )

        fig = fig.add_trace(
                go.Scatter(x=df2.sort_values('date')['date'], y=df2.sort_values('date')[value_column], mode='lines', name=value_column),
                row=1,
                col=2
            )

        fig = fig.add_trace(
            go.Scatter(x=df3.sort_values('date')['date'], y=df3.sort_values('date')[value_column], mode='lines', name=value_column),
            row=2,
            col=1
        )

        fig = fig.add_trace(
            go.Scatter(x=df4.sort_values('date')['date'], y=df4.sort_values('date')[value_column], mode='lines', name=value_column),
            row=2,
            col=2
        )

        fig = fig.add_trace(
            go.Scatter(x=df5.sort_values('date')['date'], y=df5.sort_values('date')[value_column], mode='lines', name=value_column),
            row=3,
            col=1
        )
    else:
        fig = fig.add_trace(
                go.Box(x=df1.sort_values('year')['year'], y=df1.sort_values('year')[value_column],name=value_column),
                row=1,
                col=1
            )

        fig = fig.add_trace(
                go.Box(x=df2.sort_values('year')['year'], y=df2.sort_values('year')[value_column],name=value_column),
                row=1,
                col=2
            )

        fig = fig.add_trace(
            go.Box(x=df3.sort_values('year')['year'], y=df3.sort_values('year')[value_column],name=value_column),
            row=2,
            col=1
        )

        fig = fig.add_trace(
            go.Box(x=df4.sort_values('year')['year'], y=df4.sort_values('year')[value_column], name=value_column),
            row=2,
            col=2
        )

        fig = fig.add_trace(
            go.Box(x=df5.sort_values('year')['year'], y=df5.sort_values('year')[value_column], name=value_column),
            row=3,
            col=1
        )
        

    fig = fig.update_layout(title='Analysis of ' +  value_column + ' using momentum pairs strategy using different tail-risk hedging metrics',width=1100, height=1200)
    fig = fig.update_yaxes(title= value_column, row=1, col=1, title_standoff=0)
    fig = fig.update_yaxes(title=value_column, row=1, col=2, title_standoff=0)
    fig = fig.update_yaxes(title=value_column, row=2, col=1, title_standoff=0)
    fig = fig.update_yaxes(title=value_column, row=2, col=2, title_standoff=0)
    fig = fig.update_yaxes(title=value_column, row=3, col=1, title_standoff=0)
    fig = fig.update_xaxes(title="Date", row=1, col=1, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=1, col=2, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=2, col=1, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=2, col=2, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=3, col=1, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y", tickangle=45)
    fig.show("notebook")


# In[8]:


def ret_capital_trade(df1, df2, df3, df4, df5):
    
    check_strat = df1.copy()
    check_strat_2 = df2.copy()
    check_strat_3 = df3.copy()
    check_strat_4 = df4.copy()
    check_strat_5 = df5.copy()
    
    fig = make_subplots(rows=3, cols=2, subplot_titles=('With both options',' With put options',' With call options',  ' Without options',' Benchmark Portfolio'),
                    vertical_spacing=0.08)

    fig = fig.add_trace(go.Scatter(x=check_strat.index, y=check_strat['Return on Capital'], mode='lines', name='Return on Capital'),row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=check_strat[check_strat["Action"]=='Call Exercised'].index,
                             y=check_strat[check_strat["Action"]=='Call Exercised']['Return on Capital'],
                             mode='markers', name='Call Exercised', marker=dict(color='green', symbol='triangle-up', size=8)),row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=check_strat[check_strat["Action"]=='Put Exercised'].index,
                             y=check_strat[check_strat["Action"]=='Put Exercised']['Return on Capital'],
                             mode='markers', name='Put Exercised', marker=dict(color='red', symbol='triangle-down', size=8)),row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=check_strat[check_strat["Action"]=='Close and Put Exercised'].index,
                             y=check_strat[check_strat["Action"]=='Close and Put Exercised']['Return on Capital'],
                             mode='markers', name='Close and Put Exercised', marker=dict(color='orange', symbol='circle', size=8)),row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=check_strat[check_strat["Action"]=='Close and Call Exercised'].index,
                             y=check_strat[check_strat["Action"]=='Close and Call Exercised']['Return on Capital'],
                             mode='markers', name='Close and Call Exercised', marker=dict(color='purple', symbol='circle', size=8)),row=1, col=1)
    
    
    fig = fig.add_trace(go.Scatter(x=check_strat_2.index, y=check_strat_2['Return on Capital'], mode='lines', name='Return on Capital'),row=1, col=2)
    fig = fig.add_trace(go.Scatter(x=check_strat_2[check_strat_2["Action"]=='Put Exercised'].index,
                             y=check_strat_2[check_strat_2["Action"]=='Put Exercised']['Return on Capital'],
                             mode='markers', name='Put Exercised', marker=dict(color='red', symbol='triangle-down', size=8)),row=1, col=2)
    fig = fig.add_trace(go.Scatter(x=check_strat_2[check_strat_2["Action"]=='Close and Put Exercised'].index,
                             y=check_strat_2[check_strat_2["Action"]=='Close and Put Exercised']['Return on Capital'],
                             mode='markers', name='Close and Put Exercised', marker=dict(color='orange', symbol='circle', size=8)),row=1, col=2)
    
    fig = fig.add_trace(go.Scatter(x=check_strat_3.index, y=check_strat_3['Return on Capital'], mode='lines', name='Return on Capital'),row=2, col=1)
    fig = fig.add_trace(go.Scatter(x=check_strat_3[check_strat_3["Action"]=='Call Exercised'].index,
                             y=check_strat_3[check_strat_3["Action"]=='Call Exercised']['Return on Capital'],
                             mode='markers', name='Put Exercised', marker=dict(color='red', symbol='triangle-down', size=8)),row=2, col=1)
    fig = fig.add_trace(go.Scatter(x=check_strat_3[check_strat_3["Action"]=='Close and Call Exercised'].index,
                             y=check_strat_3[check_strat_3["Action"]=='Close and Call Exercised']['Return on Capital'],
                             mode='markers', name='Close and Put Exercised', marker=dict(color='orange', symbol='circle', size=8)),row=2, col=1)
    
    fig = fig.add_trace(go.Scatter(x=check_strat_4.index, y=check_strat_4['Return on Capital'], mode='lines', name='Return on Capital'),row=2, col=2)
    
    fig = fig.add_trace(go.Scatter(x=check_strat_5.index, y=check_strat_5['Return on Capital'], mode='lines', name='Return on Capital'),row=3, col=1)
    
    fig = fig.update_layout(title='Return on Capital by Action',
                      xaxis_title='Date',
                      yaxis_title='Return on Capital',
                     width=1300, height=1200)
    
    fig = fig.update_yaxes(title= 'Return on Capital', row=1, col=1, title_standoff=0)
    fig = fig.update_yaxes(title='Return on Capital', row=1, col=2, title_standoff=0)
    fig = fig.update_yaxes(title='Return on Capital', row=2, col=1, title_standoff=0)
    fig = fig.update_yaxes(title='Return on Capital', row=2, col=2, title_standoff=0)
    fig = fig.update_yaxes(title='Return on Capital', row=3, col=1, title_standoff=0)
    fig = fig.update_xaxes(title="Date", row=1, col=1, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=1, col=2, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=2, col=1, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=2, col=2, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y")
    fig = fig.update_xaxes(title="Date", row=3, col=1, title_standoff=10,dtick="M10", showgrid=True, ticklabelmode="period", tickformat="%b '%y" ,tickangle=45)
  

    fig.show("notebook")


# In[9]:


def violin_plot(df, col_names):
    fig = make_subplots(rows=3, cols=2, subplot_titles=col_names, vertical_spacing=0.1, horizontal_spacing=0.1)

    colors = px.colors.qualitative.Pastel
    for i, col_name in enumerate(col_names):
        fig = fig.add_trace(
            go.Violin(
                x=[col_name]*len(df[col_name]),
                y=df[col_name],
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.05,
                marker=dict(size=2, color=colors[i%len(colors)])
            ),
            row=(i//2)+1, col=(i%2)+1
        )
        fig = fig.update_xaxes(showticklabels=False, row=(i//2)+1, col=(i%2)+1)
        fig = fig.update_yaxes(title_text=col_name, row=(i//2)+1, col=(i%2)+1)
    
    fig = fig.update_layout(height=1200, width=1000, title_font_size=18, showlegend=False)
    fig.show("notebook")


# In[10]:


def plot_comp_spy(strat_returns, strat_returns_put, strat_returns_call, strat_returns_no_options, strat_returns_benchmark, daily_df):
    all_returns = pd.DataFrame()
    all_returns['Put+Call'] = strat_returns['Return on Capital']/100
    all_returns['Put'] = strat_returns_put['Return on Capital']/100
    all_returns['Call'] = strat_returns_call['Return on Capital']/100
    all_returns['No Options'] = strat_returns_no_options['Return on Capital']/100
    all_returns['Benchmark'] = strat_returns_benchmark['Return on Capital']/100
    all_returns['Market'] = daily_df['SPY']['adj_close'].pct_change()
    all_returns = all_returns.dropna()
    
    all_returns['id'] = all_returns.index

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    fig = fig.add_traces(go.Scatter(x=all_returns['id'], y = all_returns['Put+Call'], mode = 'lines',                               line=dict(color=colors[0]), name='Put+Call'))
    fig = fig.add_traces(go.Scatter(x=all_returns['id'], y = all_returns['Benchmark'], mode = 'lines',                               line=dict(color=colors[4]), name='Benchmark'))
    fig = fig.add_traces(go.Scatter(x=all_returns['id'], y = all_returns['Market'], mode = 'lines',                               line=dict(color=colors[5]), name='Market'))
    fig = fig.update_layout(title = dict(text='<b>Strategy Returns vs Market Returns</b>', font=dict(size=20), x=0.5))
    fig.show("notebook")


# In[11]:


def plot_corr_spy(strat_returns, strat_returns_put, strat_returns_call, strat_returns_no_options, strat_returns_benchmark, daily_df):
    all_returns = pd.DataFrame()
    all_returns['Put+Call'] = strat_returns['Return on Capital']/100
    all_returns['Put'] = strat_returns_put['Return on Capital']/100
    all_returns['Call'] = strat_returns_call['Return on Capital']/100
    all_returns['No Options'] = strat_returns_no_options['Return on Capital']/100
    all_returns['Benchmark'] = strat_returns_benchmark['Return on Capital']/100
    all_returns['Market'] = daily_df['SPY']['adj_close'].pct_change()
    all_returns = all_returns.dropna()
    
    all_returns['id'] = all_returns.index
    
    df_corr = all_returns.drop('id', axis=1).fillna(0).corr()
    fig = go.Figure()
    fig = fig.add_trace(
        go.Heatmap(
            x = df_corr.columns,
            y = df_corr.index,
            z = np.array(df_corr),
            text=df_corr.values,
            texttemplate='%{text:.2f}',
            colorscale='Viridis'
        )
    )
    fig = fig.update_layout(
        width       = 950,
        height      = 500,
        title       = dict(text='<b>Strategy Returns\' Correlation with Market Returns</b>', font=dict(size=20), x=0.5),
        margin      = dict(l=0, r=20, t=55, b=20),
        legend=dict(orientation="h", x=0, xanchor='left', y=-0.3, yanchor='bottom'))
    fig.show("notebook")


# In[12]:


palette_btc = {'orange': '#f7931a',
               'white' : '#ffffff',
               'gray'  : '#4d4d4d',
               'blue'  : '#0d579b',
               'green' : '#329239'
              }


# In[13]:


def weight_distribution_buy(tickers, monthly_df, daily_df, win, n, p):
    thetas_daily, top_n_daily, buy, sell = daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p)

    fig1 = px.box(buy.sort_values('Ticker'), x='Ticker', y='Weights',
              color_discrete_sequence=[palette_btc['green']])

    df_median = pd.DataFrame(buy.groupby('Ticker')['Weights'].median()).reset_index()
    fig2 = px.line(df_median, x='Ticker', y='Weights', markers=True,
                   color_discrete_sequence=[palette_btc['gray']])

    fig = go.Figure(data=fig1.data + fig2.data)

    fig = fig.update_layout(
        width       = 650,
        height      = 350,
        title       = dict(text='<b>Median Long Weights</b>', font=dict(size=20), x=0.5),
        yaxis_title = dict(text='Weight', font=dict(size=13)),
        xaxis       = dict(tickmode='linear'),
        xaxis_title = dict(text='ETF', font=dict(size=13)),
        margin      = dict(l=0, r=20, t=55, b=20)
    )

    fig.show("notebook")


# In[14]:


def weight_distribution_sell(tickers, monthly_df, daily_df, win, n, p):
    thetas_daily, top_n_daily, buy, sell = daily_pos_vectorized(tickers, monthly_df, daily_df, win, n, p)

    fig1 = px.box(sell.sort_values('Ticker'), x='Ticker', y='Weights',
                  color_discrete_sequence=[palette_btc['green']])

    df_median = pd.DataFrame(sell.groupby('Ticker')['Weights'].median()).reset_index()
    fig2 = px.line(df_median, x='Ticker', y='Weights', markers=True,
                   color_discrete_sequence=[palette_btc['gray']])

    fig = go.Figure(data=fig1.data + fig2.data)

    fig = fig.update_layout(
        width       = 650,
        height      = 350,
        title       = dict(text='<b>Median Short Weights</b>', font=dict(size=20), x=0.5),
        yaxis_title = dict(text='Weight', font=dict(size=13)),
        xaxis       = dict(tickmode='linear'),
        xaxis_title = dict(text='ETF', font=dict(size=13)),
        margin      = dict(l=0, r=20, t=55, b=20)
    )

    fig.show("notebook")


# In[15]:


def adf_test(Return):
    adf_results = []
    adf_output = sm.tsa.stattools.adfuller(Return.values)
    adf_statistic = adf_output[0]
    adf_pvalue = adf_output[1]
    adf_critical_values = adf_output[4]

    adf_results.append({

        'ADF Statistic': adf_statistic,
        'p-value': adf_pvalue,
        'Critical Value (1%)': adf_critical_values['1%'],
        'Critical Value (5%)': adf_critical_values['5%'],
        'Critical Value (10%)': adf_critical_values['10%']
    })

    adf_df = pd.DataFrame(adf_results)
    adf_df.sort_values(by='ADF Statistic', inplace=True)
    adf_df = adf_df.rename(index={0: 'Return'})
    return adf_df


# In[16]:


def kpss_test(Return):
    
    kpss_results = []
    res = Return
    kpss_stat, kpss_pval, kpss_lags, kpss_critval = sm.tsa.stattools.kpss(res)
    kpss_results.append([kpss_stat, kpss_pval, kpss_critval['1%'], kpss_critval['5%'], kpss_critval['10%']])
    kpss_df = pd.DataFrame(kpss_results, columns=['KPSS Statistic', 'p-value', '1% Crit. Val.',                                                   '5% Crit. Val.', '10% Crit. Val.'])
    kpss_df = kpss_df.rename(index={0: 'Return'})

    return kpss_df


# In[17]:


def ks_test(Return):
    ks_results = []
    res = pd.to_numeric(Return, errors='coerce').dropna()
    ks_stat, ks_pval = sm.stats.diagnostic.kstest_normal(res)
    ks_results.append([ks_stat, ks_pval])

    ks_df = pd.DataFrame(ks_results, columns=['KS Statistic', 'p-value'])
    ks_df = ks_df.rename(index={0: 'Return'})
    return ks_df


# In[18]:


def qq_plot(df, col_names):
    fig = make_subplots(rows=3, cols=2, subplot_titles=tuple([f'Q-Q Plot for {col_names[i]}' for i in range(len(col_names))]))
    fig = fig.update_layout(height=800, width=800, title_font_size=18)
    fig = fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig = fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig = fig.update_layout(
        margin=dict(l=20, r=20, t=80, b=20),
    )

    for i in range(len(col_names)):
        row = (i // 2) + 1
        col = (i % 2) + 1

        res = pd.to_numeric(df[col_names[i]], errors='coerce').dropna()
        sorted_res = np.sort(res)
        y = stats.norm.ppf((np.arange(len(sorted_res))+0.5)/len(sorted_res))
        trace = go.Scatter(x=y, y=sorted_res, mode='markers')
        fit = stats.linregress(y, sorted_res)
        y_fit = fit.slope * y + fit.intercept
        trendline = go.Scatter(x=y, y=y_fit, mode='lines', line=dict(color='red'))
        fig = fig.add_trace(trace, row=row, col=col)
        fig = fig.add_trace(trendline, row=row, col=col)

    fig = fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", showlegend=False)

    fig.show("notebook")


# In[19]:


def acf_plot(df, col_names):
    fig = make_subplots(rows=3, cols=2, subplot_titles=col_names, 
                        shared_xaxes=True, shared_yaxes=True, 
                        vertical_spacing=0.1, horizontal_spacing=0.1)
    fig = fig.update_layout(height=800, width=1000, title_font_size=18, 
                      xaxis_title_font_size=14, yaxis_title_font_size=14)
    
    for i, col in enumerate(col_names):
        row, col_num = divmod(i, 2)
        row += 1
        col_num += 1
        acf_vals, conf_int = sm.tsa.stattools.acf(df[col], nlags=50, fft=True, alpha=0.05)
        fig = fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF'), row=row, col=col_num)
        fig = fig.add_trace(go.Scatter(x=list(range(len(conf_int))), y=conf_int[:, 0], 
                                 mode='lines', name='Confidence Interval', 
                                 line=dict(dash='dash')), row=row, col=col_num)
        fig = fig.add_trace(go.Scatter(x=list(range(len(conf_int))), y=conf_int[:, 1], 
                                 mode='lines', showlegend=False, 
                                 line=dict(dash='dash')), row=row, col=col_num)
        
        fig = fig.update_xaxes(title_text='Lag', row=row, col=col_num)
        fig = fig.update_yaxes(title_text='Autocorrelation', row=row, col=col_num)

    fig.show("notebook")


# In[20]:


def pacf_plot(df, col_names):
    fig = make_subplots(rows=3, cols=2, subplot_titles=col_names, 
                        shared_xaxes=True, shared_yaxes=True, 
                        vertical_spacing=0.1, horizontal_spacing=0.1)
    fig = fig.update_layout(height=800, width=1000, title_font_size=18, 
                      xaxis_title_font_size=14, yaxis_title_font_size=14)
    
    for i, col in enumerate(col_names):
        row, col_num = divmod(i, 2)
        row += 1
        col_num += 1
        pacf_vals, conf_int = sm.tsa.stattools.pacf(df[col], nlags=50, alpha=0.05)
        fig = fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF'), row=row, col=col_num)
        fig = fig.add_trace(go.Scatter(x=list(range(len(conf_int))), y=conf_int[:, 0], 
                                 mode='lines', name='Confidence Interval', 
                                 line=dict(dash='dash')), row=row, col=col_num)
        fig = fig.add_trace(go.Scatter(x=list(range(len(conf_int))), y=conf_int[:, 1], 
                                 mode='lines', showlegend=False, 
                                 line=dict(dash='dash')), row=row, col=col_num)
        
        fig = fig.update_xaxes(title_text='Lag', row=row, col=col_num)
        fig = fig.update_yaxes(title_text='Partial Autocorrelation', row=row, col=col_num)

    fig.show("notebook")


# In[21]:


def hist_plot(df, col_names):
    fig = make_subplots(rows=3, cols=2, subplot_titles=col_names, vertical_spacing=0.1)

    for i in range(len(col_names)):
        res = pd.to_numeric(df[col_names[i]], errors='coerce').dropna()
        fig = fig.add_trace(go.Histogram(x=res, nbinsx=150, histnorm='probability density'), row=(i//2)+1, col=(i%2)+1)
        skewness = round(res.skew(), 2)
        kurtosis = round(res.kurtosis(), 2)
        fig = fig.update_xaxes(title_text=col_names[i], row=(i//2)+1, col=(i%2)+1)
        fig = fig.update_yaxes(title_text="Frequency", row=(i//2)+1, col=(i%2)+1)

    fig = fig.update_layout(height=1000, width=1000, title_text="Histograms of Stock Returns and Option Returns", font=dict(size=16), showlegend=False)
    fig.show("notebook")

