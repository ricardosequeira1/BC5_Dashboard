import pandas as pd
import numpy as np
from datetime import date as dt
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import plotly.graph_objects as go
import plotly.io as pio
from dateutil.relativedelta import relativedelta
import sklearn
from xgboost import XGBRegressor

pio.templates.default = "simple_white"

# --------1st try


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    dfs = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dfs.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dfs.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


df = yf.download('AAPL', start='2015-01-01', end=dt.today())
df['MA20'] = df.Close.rolling(20).mean()

information = yf.Ticker('AAPL').stats()

name = information.get('quoteType').get('shortName')
sector = information.get('summaryProfile').get('sector')
beta = information.get('defaultKeyStatistics').get('beta')
free_cash_flow = information.get('financialData').get('freeCashflow')
return_on_equity = information.get('financialData').get('returnOnEquity')

# load the dataset
series = df.Close
values = series.values
# transform the time series data into supervised learning
train = series_to_supervised(values, n_in=15)
# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]
# fit model
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(trainX, trainy)

predictions = []
predict_days = 5

dates = []
day = dt.today() - relativedelta(days=1)

for i in range(predict_days):
    # construct an input for a new prediction
    row = values[-15:].flatten()
    # make a one-step prediction
    yhat = model.predict(np.asarray([row]))
    # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))

    predictions.append(float(yhat))
    values = list(values)
    yhat = float(yhat)
    values.append(yhat)
    values = np.asarray(values)

    day = day + relativedelta(days=1)
    dates.append(str(day))

if len(predictions) == 1:
    value_future = float(predictions[0])
else:
    value_future = float(predictions[-1])


# graph style


# ---------- interactive components


asset_text = dcc.Textarea(id='asset_text',
                          placeholder='Select an asset:',
                          value='AAPL')

pick_year = dcc.DatePickerSingle(id='date-picker',
                                 min_date_allowed=dt(1900, 8, 5),
                                 max_date_allowed=dt.today(),
                                 initial_visible_month=dt(2015, 1, 1),
                                 date=dt(2015, 1, 1)
                                 )

indicator_radio = dcc.RadioItems(id='radio-indicators',
                                 options=[{'label': 'Moving Average: 20 days', 'value': 'MA20'},
                                          {'label': 'Moving Average: 50 days', 'value': 'MA50'},
                                          {'label': 'Moving Average: 200 days', 'value': 'MA200'},
                                          {'label': 'Exponential Moving Average: 20 days', 'value': 'EMA20'},
                                          {'label': 'Exponential Moving Average: 50 days', 'value': 'EMA50'},
                                          {'label': 'Exponential Moving Average: 200 days', 'value': 'EMA200'},
                                          {'label': 'MACD', 'value': 'MACD'},
                                          {'label': 'Boolinger Bands', 'value': 'BB'},
                                          {'label': 'Stochastic Oscillator', 'value': 'SO'}],
                                 value='MA20',
                                 labelStyle={'display': 'block'}
                                 )

predicted_days_radio = dcc.RadioItems(id='predictions_days',
                                      options=[{'label': '1 day', 'value': 1},
                                               {'label': '2 days', 'value': 2},
                                               {'label': '5 days', 'value': 5},
                                               {'label': '10 days', 'value': 10},
                                               {'label': '30 days', 'value': 30}],
                                      labelStyle={'display': 'block'},
                                      value=5
                                      )

# ---------- app

app = dash.Dash(__name__)

server = app.server
# ---------- app layout

app.layout = html.Div([
    html.Div([
        html.H1('Asset Analysis', style={'textAlign': 'center', 'background-color': '#f9f9f9'})
    ], className='pretty_container'),
    html.Div([
        html.Div([
            html.Div([
                html.H4('Choose an asset:', style={'textAlign': 'left', 'background-color': '#ffffff'}),
                asset_text,
                html.Br(),
                html.H4('Choose an initial date:', style={'textAlign': 'left', 'background-color': '#ffffff'}),
                pick_year,
                ], className='inside_container'),
            html.Div([
                html.Button('Submit', id='submit_button', className='Button'),
                ], className='inside_container'),
            html.Div([
                html.Br(),
                html.Br(),
                html.Img(src=app.get_asset_url('1stmenu.png'), style={'width': '100%',
                                                                      'position': 'bottom',
                                                                      'opacity': '80%'}),
            ])
        ], style={'width': '24%'}, className='pretty_container'),
        html.Div([
            html.Div([
                html.Div([
                    html.H4(id='title_1st', style={'background-color': '#ffffff'}),
                    html.H5(id='company_name', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4(id='title_2nd', style={'background-color': '#ffffff'}),
                    html.H5(id='sector_name', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4(id='title_3rd', style={'background-color': '#ffffff'}),
                    html.H5(id='beta_value', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4(id='title_4th', style={'background-color': '#ffffff'}),
                    html.H5(id='fcf_value', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4(id='title_5th', style={'background-color': '#ffffff'}),
                    html.H5(id='return_on_equity_value', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
            ], style={'height': '20%', 'display': 'flex'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='history_line_plot')
                ], style={'width': '76%'}, className='inside_container'),
                html.Div([
                    html.Div([
                        html.Label('1-Year Change:', style={'background-color': '#ffffff'}),
                        dcc.Graph(id='1year_change_value', style={'background-color': '#ffffff'})
                    ], style={'width': '85%', 'height': '18%'}, className='inside_container'),
                    html.Div([
                        html.Label('6-Month Change:', style={'background-color': '#ffffff'}),
                        dcc.Graph(id='6month_change_value', style={'background-color': '#ffffff'})
                    ], style={'width': '85%', 'height': '18%'}, className='inside_container'),
                    html.Div([
                        html.Label('1-Month Change:', style={'background-color': '#ffffff'}),
                        dcc.Graph(id='1month_change_value', style={'background-color': '#ffffff'})
                    ], style={'width': '85%', 'height': '18%'}, className='inside_container'),
                    html.Div([
                        html.Label('Today Change:', style={'background-color': '#ffffff'}),
                        dcc.Graph(id='today_change_value', style={'background-color': '#ffffff'})
                    ], style={'width': '85%', 'height': '17%'}, className='inside_container')
                ], style={'width': '19%'})
            ], style={'height': '80%', 'display': 'flex'})
        ], style={'width': '75%'}, className='pretty_container'),
    ], style={'display': 'flex'}),

    html.Div([
        html.Div([
            html.Div([
                html.H4('Choose an Indicator:', style={'textAlign': 'left', 'background-color': '#ffffff'}),
                indicator_radio,
            ], className='inside_container'),
            html.Div([
                html.H5(id='indicator_explanation', style={'background-color': '#ffffff'})
            ], className='inside_container'),
        ], style={'width': '24%'}, className='pretty_container'),
        html.Div([
            dcc.Graph(id='indicator_line_graph', className='inside_container')
        ], style={'width': '75%'}, className='pretty_container'),
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div([
                html.H4('Choose Prediction Days:', style={'textAlign': 'left', 'background-color': '#ffffff'}),
                predicted_days_radio
            ], className='inside_container'),
            html.Div([
                html.Img(src=app.get_asset_url('ToTheMoon.png'), style={'width': '100%',
                                                                        'position': 'bottom',
                                                                        'opacity': '80%'}),
            ]),
        ], style={'width': '27%'}, className='pretty_container'),
        html.Div([
            html.Div([
                dcc.Graph(id='predictions_line_graph', className='inside_container')
            ], style={'width': '80%'}),
            html.Div([
                html.Div([
                    html.Label('Prediction:', style={'background-color': '#ffffff'}),
                    html.H5(id='prediction_value', style={'background-color': '#ffffff'})
                ], style={'width': '150%', 'height': '35%'}, className='inside_container'),
                html.Div([
                    html.Label('Change to last Close Price:', style={'background-color': '#ffffff'}),
                    dcc.Graph(id='indicator_change', style={'background-color': '#ffffff'})
                ], style={'width': '150%', 'height': '45%'}, className='inside_container'),
            ], style={'width': '10%'})
        ], style={'width': '85%', 'display': 'flex'}, className='pretty_container'),
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.H2(id='title', style={'background-color': '#ffffff'})
        ], className='inside_container'),
        html.Div([
            html.Div([
                html.H3(id='title1_news', style={'background-color': '#ffffff'}),
                html.Br(),
                html.A('LINK', id='link1_news', target="_blank")
            ], style={'width': '32%'}, className='inside_container'),
            html.Div([
                html.H3(id='title2_news', style={'background-color': '#ffffff'}),
                html.Br(),
                html.A('LINK', id='link2_news', target="_blank")
            ], style={'width': '32%'}, className='inside_container'),
            html.Div([
                html.H3(id='title3_news', style={'background-color': '#ffffff'}),
                html.Br(),
                html.A('LINK', id='link3_news', target="_blank")
            ], style={'width': '32%'}, className='inside_container'),
        ], style={'display': 'flex'}),
    ], className='pretty_container'),
    html.Div([
        html.H3('Authors: Afonso Ramos, Beatriz Gonçalves, Helena Morais, Ricardo Sequeira ',
                style={'background-color': '#f9f9f9'})
    ], className='pretty_container'),
])

# ------------ app callback

##################################
# 1st graph #################
#################################


@app.callback(
    [
        Output("history_line_plot", "figure"),
    ],
    [
        Input("submit_button", "n_clicks")
    ],
    [
        State("asset_text", "value"),
        State("date-picker", "date")
    ]
)
def update_history(n_clicks, asset, date):

    df = yf.download(asset, start=date, end=dt.today())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.Close,
        name='Close',  # Style name/legend entry with html tags
        connectgaps=True  # override default to connect the gaps
    ))

    fig.update_layout(
        title={
            'text': f'History Closing Price of {asset} since {date}'},
        xaxis_title="Date",
        yaxis_title="Close Price")

    return [fig]

##################################
# 1st indicator #################
#################################


@app.callback(
    [
        Output("1year_change_value", "figure"),
        Output("6month_change_value", "figure"),
        Output("1month_change_value", "figure"),
        Output("today_change_value", "figure")
    ],
    [
        Input("submit_button", "n_clicks")
    ],
    [
        State("asset_text", "value"),
        State("date-picker", "date")
    ]
)
def update_indicator1(n_clicks, asset, date):

    df = yf.download(asset, start=date, end=dt.today())

    # 1year
    value_1year_ago = float(df.loc[df.index[-365]].Close)

    # 6months
    value_6month_ago = float(df.loc[df.index[-180]].Close)

    # 1month
    value_1month_ago = float(df.loc[df.index[-30]].Close)

    # 1day
    value_1day_ago = float(df.loc[df.index[-2]].Close)

    # Today
    value_today = float(df.loc[df.index[-1]].Close)

    # Figure 1

    fig1 = go.Figure(go.Indicator(
        mode="delta",
        value=df.loc[df.index[-1]].Close,
        delta={'reference': df.loc[df.index[-365]].Close, 'relative': True, 'valueformat': '.2%'}))

    fig1.update_traces(delta_font={'size': 18})
    fig1.update_layout(height=30, width=120)

    if value_today >= value_1year_ago:
        fig1.update_traces(delta_increasing_color='green')
    if value_today < value_1year_ago:
        fig1.update_traces(delta_decreasing_color='red')

    # Figure 2

    fig2 = go.Figure(go.Indicator(
        mode="delta",
        value=df.loc[df.index[-1]].Close,
        delta={'reference': df.loc[df.index[-180]].Close, 'relative': True, 'valueformat': '.2%'}))

    fig2.update_traces(delta_font={'size': 18})
    fig2.update_layout(height=30, width=120)

    if value_today >= value_6month_ago:
        fig2.update_traces(delta_increasing_color='green')
    if value_today < value_6month_ago:
        fig2.update_traces(delta_decreasing_color='red')

    # Figure 3

    fig3 = go.Figure(go.Indicator(
        mode="delta",
        value=df.loc[df.index[-1]].Close,
        delta={'reference': df.loc[df.index[-30]].Close, 'relative': True, 'valueformat': '.2%'}))

    fig3.update_traces(delta_font={'size': 18})
    fig3.update_layout(height=30, width=120)

    if value_today >= value_1month_ago:
        fig3.update_traces(delta_increasing_color='green')
    if value_today < value_1month_ago:
        fig3.update_traces(delta_decreasing_color='red')

    # Figure 4

    fig4 = go.Figure(go.Indicator(
        mode="delta",
        value=df.loc[df.index[-1]].Close,
        delta={'reference': df.loc[df.index[-2]].Close, 'relative': True, 'valueformat': '.2%'}))

    fig4.update_traces(delta_font={'size': 18})
    fig4.update_layout(height=30, width=120)

    if value_today >= value_1day_ago:
        fig4.update_traces(delta_increasing_color='green')
    if value_today < value_1day_ago:
        fig4.update_traces(delta_decreasing_color='red')

    return fig1, fig2, fig3, fig4


##################################
# 2nd graph #################
#################################


@app.callback(
    [
        Output("indicator_line_graph", "figure"),
        Output("indicator_explanation", "children")
    ],
    [
        Input("submit_button", "n_clicks"),
        Input("radio-indicators", "value")
    ],
    [
        State("asset_text", "value"),
        State("date-picker", "date")
    ]
)
def update_indicators(n_clicks, indicator, asset, date):

    df = yf.download(asset, start=date, end=dt.today())

    if indicator == 'MA20':
        df['MA20'] = df.Close.rolling(20).mean()
        text = 'Moving average is a simple, technical analysis tool. Moving averages are usually calculated to identify\
         the trend direction of a stock or to determine its support and resistance levels. It is a trend-following or \
         lagging—indicator because it is based on past prices.In this case the Moving Average is calculated\
         using the last 20 days.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.MA20,
            name='MA20',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Moving Average of the last 20 days of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'MA50':
        df['MA50'] = df.Close.rolling(50).mean()
        text = 'Moving average is a simple, technical analysis tool. Moving averages are usually calculated to identify\
         the trend direction of a stock or to determine its support and resistance levels. It is a trend-following or \
         lagging—indicator because it is based on past prices.In this case the Moving Average is calculated\
         using the last 50 days.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.MA50,
            name='MA50',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Moving Average of the last 50 days of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'MA200':
        df['MA200'] = df.Close.rolling(200).mean()
        text = 'Moving average is a simple, technical analysis tool. Moving averages are usually calculated to identify\
         the trend direction of a stock or to determine its support and resistance levels. It is a trend-following or \
         lagging—indicator because it is based on past prices.In this case the Moving Average is calculated\
         using the last 200 days.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.MA200,
            name='MA200',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Moving Average of the last 200 days of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'EMA20':
        df['EMA20'] = df.Close.ewm(span=20).mean()
        text = 'Exponential moving averages (EMA) is a weighted average that gives greater importance to the price of a\
         stock in more recent days, making it an indicator that is more responsive to new information. In this case the\
         Exponential Moving Average is calculated using the last 20 days.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.EMA20,
            name='EMA20',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Exponential Moving Average of the last 20 days of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'EMA50':
        df['EMA50'] = df.Close.ewm(span=50).mean()
        text = 'Exponential moving averages (EMA) is a weighted average that gives greater importance to the price of a\
        stock in more recent days, making it an indicator that is more responsive to new information. In this case the\
        Exponential Moving Average is calculated using the last 50 days.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.EMA50,
            name='EMA50',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Exponential Moving Average of the last 50 days of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'EMA200':
        df['EMA200'] = df.Close.ewm(span=200).mean()
        text = 'Exponential moving averages (EMA) is a weighted average that gives greater importance to the price of a\
        stock in more recent days, making it an indicator that is more responsive to new information. In this case the\
        Exponential Moving Average is calculated using the last 200 days.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.EMA200,
            name='EMA200',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Exponential Moving Average of the last 200 days of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'MACD':
        df['MACD'] = (df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean())
        df['signal'] = df['MACD'].ewm(span=9).mean()
        text = 'Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the \
        relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the \
        26-period exponential moving average (EMA) from the 12-period EMA.The result of that calculation is the \
        MACD line. A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line, \
        which can function as a trigger for buy and sell signals. Traders may buy the security when the MACD crosses \
        above its signal line and sell—or short—the security when the MACD crosses below the signal line. '

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.MACD,
            name='MACD',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.signal,
            name='signal',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))

        fig.update_layout(
            title={
                'text': f'Moving Average Convergence/Divergence of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'BB':
        df['stddev'] = df.Close.rolling(20).std()
        df['BB_Lower'] = df.Close.rolling(20).mean() - 2 * df['stddev']
        df['BB_Higher'] = df.Close.rolling(20).mean() + 2 * df['stddev']
        text = 'A Bollinger Band technical indicator has bands generally placed two standard deviations away from a \
        simple moving average. In general, a move toward the upper band suggests the asset is becoming overbought, \
        while a move close to the lower band suggests the asset is becoming oversold. Since standard deviation is used \
        as a statistical measure of volatility, this indicator adjusts itself to market conditions.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.BB_Lower,
            name='BB_Lower',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.BB_Higher,
            name='BB_Higher',  # Style name/legend entry with html tags
            connectgaps=True   # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Boolinger Bands of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    elif indicator == 'SO':
        df['14-low'] = df.High.rolling(14).min()
        df['14-high'] = df.Low.rolling(14).max()
        df['%K'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
        df['%D'] = df['%K'].rolling(3).mean()
        text = 'A stochastic oscillator is a momentum indicator comparing a particular closing price of a security to \
        a range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is \
        reducible by adjusting that time period or by taking a moving average of the result. It is used to generate\
        overbought and oversold trading signals, utilizing a 0–100 bounded range of values. Transaction signals are \
        created when the %K crosses through a three-period moving average, which is called the %D.'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['%D'],
            name='%D',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['%K'],
            name='%K',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.Close,
            name='Close',  # Style name/legend entry with html tags
            connectgaps=True  # override default to connect the gaps
        ))
        fig.update_layout(
            title={
                'text': f'Stochastic Oscillator of {asset} since {date}'},
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Indicator",
        )

    return fig, text


# -------informations

@app.callback(
    [
        Output("title_1st", "children"),
        Output("company_name", "children"),
        Output("title_2nd", "children"),
        Output("sector_name", "children"),
        Output("title_3rd", "children"),
        Output("beta_value", "children"),
        Output("title_4th", "children"),
        Output("fcf_value", "children"),
        Output("title_5th", "children"),
        Output("return_on_equity_value", "children")
    ],
    [
        Input("submit_button", "n_clicks")
    ],
    [
        State("asset_text", "value"),
        State("date-picker", "date")
    ]
)
def update_information(n_clicks, asset, date):

    ticker = yf.Ticker(asset)
    info = ticker.stats()

    if info.get('price').get('quoteType') == 'CRYPTOCURRENCY':

        title1 = 'Name:'
        name = info.get('summaryProfile').get('name')
        title2 = 'Sector:'
        sector = 'Cryptocurrency'
        title3 = 'Market Cap:'
        marketcap = info.get('price').get('marketCap')
        title4 = 'Volume 24h:'
        volume = info.get('price').get('volume24Hr')
        title5 = 'Start Date:'
        start = info.get('summaryProfile').get('startDate')

        return title1,\
               name,\
               title2,\
               sector, \
               title3,\
               marketcap, \
               title4,\
               volume, \
               title5,\
               start

    elif info.get('price').get('quoteType') == 'EQUITY':

        title1 = 'Company Name:'
        name = info.get('quoteType').get('shortName')
        title2 = 'Sector:'
        sector = info.get('summaryProfile').get('sector')
        title3 = 'Beta:'
        beta = info.get('defaultKeyStatistics').get('beta')
        title4 = 'Free Cash Flow:'
        free_cash_flow = info.get('financialData').get('freeCashflow')
        title5 = 'Return on Equity:'
        return_on_equity = info.get('financialData').get('returnOnEquity')

        if free_cash_flow is None:
            free_cash_flow = 'Unavailable'
        else:
            free_cash_flow = round(free_cash_flow / 1000000, 2)

        if return_on_equity is None:
            return_on_equity = 'Unavailable'
        else:
            return_on_equity = round(return_on_equity * 100, 2)

        return title1,\
               name, \
               title2, \
               sector, \
               title3, \
               beta, \
               title4, \
               f'{free_cash_flow}M $', \
               title5, \
               f'{return_on_equity}%'

    elif info.get('price').get('quoteType') == 'INDEX':

        title1 = 'Name:'
        name = info.get('quoteType').get('shortName')
        title2 = 'Sector:'
        sector = info.get('quoteType').get('quoteType')
        title3 = 'No More Information'
        beta = ''
        title4 = 'No More Information'
        free_cash_flow = ''
        title5 = 'No More Information'
        return_on_equity = ''

        return title1,\
               name, \
               title2, \
               sector, \
               title3, \
               beta, \
               title4, \
               free_cash_flow, \
               title5, \
               return_on_equity

    elif info.get('price').get('quoteType') == 'ETF':

        title1 = 'Name:'
        name = info.get('quoteType').get('shortName')
        title2 = 'Sector:'
        sector = info.get('quoteType').get('quoteType')
        title3 = 'Fund Family:'
        beta = info.get('defaultKeyStatistics').get('fundFamily')
        title4 = 'Category:'
        free_cash_flow = info.get('defaultKeyStatistics').get('category')
        title5 = 'Total Number of Assets:'
        return_on_equity = info.get('defaultKeyStatistics').get('totalAssets')

        return title1,\
               name, \
               title2, \
               sector, \
               title3, \
               beta, \
               title4, \
               free_cash_flow, \
               title5, \
               return_on_equity


# ---------predictions


@app.callback(
    [
        Output("predictions_line_graph", "figure"),
        Output("prediction_value", "children"),
        Output("indicator_change", "figure"),

    ],
    [
        Input("submit_button", "n_clicks"),
        Input("predictions_days", "value")

    ],
    [
        State("asset_text", "value"),
        State("date-picker", "date")
    ]
)
def update_predictions(n_clicks, predict_days, asset, date):
    # transform a time series dataset into a supervised learning dataset
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        dfs = pd.DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(dfs.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(dfs.shift(-i))
        # put it all together
        agg = pd.concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values

    df = yf.download(asset, start='2015-01-01', end=dt.today())
    # load the dataset
    series = df.Close
    values = series.values
    # transform the time series data into supervised learning
    train = series_to_supervised(values, n_in=15)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)

    predictions = []

    dates = []
    day = dt.today() - relativedelta(days=1)

    for i in range(predict_days):
        # construct an input for a new prediction
        row = values[-15:].flatten()
        # make a one-step prediction
        yhat = model.predict(np.asarray([row]))

        predictions.append(float(yhat))
        values = list(values)
        yhat = float(yhat)
        values.append(yhat)
        values = np.asarray(values)

        day = day + relativedelta(days=1)
        dates.append(str(day))

    new_df = pd.DataFrame(predictions, dates)
    new_df.columns = ['Predictions']

    rang = 7*predict_days

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[-rang:].index,
        y=df[-rang:].Close,
        name='Close',  # Style name/legend entry with html tags
        connectgaps=True  # override default to connect the gaps
    ))
    fig.add_trace(go.Scatter(
        x=new_df.index,
        y=new_df.Predictions,
        name='Predictions',
        connectgaps=True  # override default to connect the gaps
    ))

    fig.update_layout(
        title={
            'text': f'Predictions of asset {asset} for the next {predict_days} days.'},
        xaxis_title="Date",
        yaxis_title="Close Price"
    )

    # Figure

    if len(predictions) == 1:
        value_future = float(predictions[0])
    else:
        value_future = float(predictions[-1])

    value_last_day = float(df.Close[-1])

    fig1 = go.Figure(go.Indicator(
        mode="delta",
        value=value_future,
        delta={'reference': value_last_day, 'relative': True, 'valueformat': '.2%'}))

    fig1.update_traces(delta_font={'size': 18})
    fig1.update_layout(height=30, width=120)

    if value_future >= value_last_day:
        fig1.update_traces(delta_increasing_color='green')
    if value_future < value_last_day:
        fig1.update_traces(delta_decreasing_color='red')

    return fig, value_future, fig1

# ------------ news


@app.callback(
    [
        Output("title", "children"),
        Output("title1_news", "children"),
        Output("title2_news", "children"),
        Output("title3_news", "children"),
        Output("link1_news", "href"),
        Output("link2_news", "href"),
        Output("link3_news", "href"),

    ],
    [
        Input("submit_button", "n_clicks"),
    ],
    [
        State("asset_text", "value"),
    ]
)
def update_news(n_clicks,asset):

    news = yf.Ticker(asset).news

    if len(news) > 2:
        return f'Latest Financial News on {asset}',\
               news[0]['title'],\
               news[1]['title'],\
               news[2]['title'],\
               news[0]['link'],\
               news[1]['link'],\
               news[2]['link']

    elif len(news) == 2:
        return f'Latest Financial News on {asset}',\
               news[0]['title'],\
               news[1]['title'],\
               'Unavailable News',\
               news[0]['link'],\
               news[1]['link'],\
               ''

    elif len(news) == 1:
        return f'Latest Financial News on {asset}',\
               news[0]['title'],\
               'Unavailable News',\
               'Unavailable News',\
               news[0]['link'],\
               '',\
               ''

    else:
        return f'Latest Financial News on {asset}',\
               'Unavailable News',\
               'Unavailable News',\
               'Unavailable News',\
               '',\
               '',\
               ''


if __name__ == '__main__':
    app.run_server(debug=True, port=3002)
