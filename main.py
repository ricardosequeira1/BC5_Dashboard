import pandas as pd
import numpy as np
from datetime import date as dt
import yfinance as yf
from urllib.request import urlopen
import dash  # pip install dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dateutil.relativedelta import relativedelta
import calendar
import sklearn
from xgboost import XGBRegressor

pio.templates.default = "simple_white"


#############

symbols = []
s = urlopen("https://raw.githubusercontent.com/ricardosequeira1/BC5_Dashboard/main/data/sp500-symbol-list.txt?token=GHSAT0AAAAAABRUUZGFY2EE7JPV6X2XJNCQYUFBYYQ")

for line in s:
    symbols.append(line)

for i in range(len(symbols)):
    symbols[i] = str(symbols[i])

symb = []

for i in symbols:
    symb.append(i[2:-3])

c = urlopen("https://raw.githubusercontent.com/ricardosequeira1/BC5_Dashboard/main/data/crypto.txt")

cryptocurrencies = []

for line in c:
    cryptocurrencies.append(line)

for i in range(len(cryptocurrencies)):
    cryptocurrencies[i] = str(cryptocurrencies[i])

crypto = []

for i in cryptocurrencies:
    crypto.append(i[2:-3])

project = ['ADA-USD', 'ATOM-USD', 'AVAX-USD', 'AXS-USD', 'BTC-USD', 'ETH-USD', 'LINK-USD', 'LUNA1-USD', 'MATIC-USD',
           'SOL-USD']

for i in project:
    if i not in crypto:
        crypto.append(i)

for i in crypto:
    symb.append(i)

# graph style


# ---------- interactive components


asset_text = dcc.Textarea(id='asset_text',
                          placeholder='Select an asset:',
                          value='AAPL')

asset_dropdown = dcc.Dropdown(id='dropdown_stock',
                              options=symb,
                              searchable=True,
                              placeholder="Select an asset:",
                              value='AAPL')

pick_year = dcc.DatePickerSingle(id='date-picker',
                                 min_date_allowed=dt(1995, 8, 5),
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
                                      value=1
                                      )

# ---------- app

app = dash.Dash(__name__)
#                , external_stylesheets=[dbc.themes.BOOTSTRAP],
#                meta_tags=[{'name': 'viewport',
#                            'content': 'width=device-width, initial-scale=1.0'}]
server = app.server
# ---------- app layout

app.layout = html.Div([
    html.Div([
        html.H1('Asset Analysis', style={'textAlign': 'center'})
    ], className='pretty_container'),
    html.Div([
        html.Div([
            html.Div([
                html.H4('Choose an asset:', style={'textAlign': 'left','background-color': '#ffffff'}),
                asset_text,
                html.Br(),
                html.H4('Choose an initial date:', style={'textAlign': 'left','background-color': '#ffffff'}),
                pick_year,
                ], className='inside_container'),
            html.Div([
                html.Button('Submit', id='submit_button', className='Button'),
                ], className='inside_container'),
            html.Div([
                html.Img(src=app.get_asset_url('Finance.jpg'), style={'width': '100%', 'position': 'bottom', 'opacity': '80%'}),
            ], className='photo_container')
        ], style={'width': '24%'}, className='pretty_container'),
        html.Div([
            html.Div([
                html.Div([
                    html.H4('Name:', style={'background-color': '#ffffff'}),
                    html.H5(id='company_name', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4('Sector:', style={'background-color': '#ffffff'}),
                    html.H5(id='sector_name', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4('Beta', style={'background-color': '#ffffff'}),
                    html.H5(id='beta_value', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4('Free Cash Flow:', style={'background-color': '#ffffff'}),
                    html.H5(id='fcf_value', style={'background-color': '#ffffff'})
                ], style={'width': '20%'}, className='inside_container'),
                html.Div([
                    html.H4('Return on Equity:', style={'background-color': '#ffffff'}),
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
                html.H4('Choose an Indicator:', style={'textAlign': 'left','background-color': '#ffffff'}),
                indicator_radio,
            ], className='inside_container')
        ], style={'width': '24%'}, className='pretty_container'),
        html.Div([
            dcc.Graph(id='indicator_line_graph', className='inside_container')
        ], style={'width': '75%'}, className='pretty_container'),
    ], style={'display': 'flex'}),

    html.Div([

    ]),

    html.Div([
        html.Div([
            html.Div([
                html.H4('Choose Prediction Days:', style={'textAlign': 'left','background-color': '#ffffff'}),
                predicted_days_radio
            ], className='inside_container')
        ], style={'width': '27%'}, className='pretty_container'),
        html.Div([
            html.Div([
                dcc.Graph(id='predictions_line_graph', className='inside_container')
            ], style={'width': '81%'}),
            html.Div([
                html.Div([
                    html.Label('MAE:', style={'background-color': '#ffffff'}),
                    html.H5(id='mae_value', style={'background-color': '#ffffff'})
                ], style={'width': '150%', 'height': '25.5%'}, className='inside_container'),
                html.Div([
                    html.Label('Prediction:', style={'background-color': '#ffffff'}),
                    html.H5(id='prediction_value', style={'background-color': '#ffffff'})
                ], style={'width': '150%', 'height': '26%'}, className='inside_container'),
                html.Div([
                    html.Label('Change to last Close Price:', style={'background-color': '#ffffff'}),
                    dcc.Graph(id='indicator_change', style={'background-color': '#ffffff'})
                ], style={'width': '150%', 'height': '26%'}, className='inside_container'),
            ], style={'width': '10%'})
        ],style={'width': '85%', 'display': 'flex'}, className='pretty_container'),
    ],style={'display': 'flex'})
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
    df.reset_index(inplace=True)

    return [px.line(df, x='Date', y=df.Close, title="History Data")]

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
def update_indicator1(n_clicks, asset,date):

    df = yf.download(asset, start=date, end=dt.today())

    ## 1year
    value_1year_ago = float(df.loc[df.index[-365]].Close)

    ## 6months
    value_6month_ago = float(df.loc[df.index[-180]].Close)

    ## 1month
    value_1month_ago = float(df.loc[df.index[-30]].Close)

    ## 1day
    value_1day_ago = float(df.loc[df.index[-2]].Close)

    ## Today
    value_today = float(df.loc[df.index[-1]].Close)

    ## Figure 1

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

    ## Figure 2

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

    ## Figure 3

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

    ## Figure 4

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
    df.reset_index(inplace=True)

    if indicator == 'MA20':
        df[indicator] = df.Close.rolling(20).mean()
    elif indicator == 'MA50':
        df[indicator] = df.Close.rolling(50).mean()
    elif indicator == 'MA200':
        df[indicator] = df.Close.rolling(200).mean()
    elif indicator == 'EMA20':
        df[indicator] = df.Close.ewm(span=20).mean()
    elif indicator == 'EMA50':
        df[indicator] = df.Close.ewm(span=50).mean()
    elif indicator == 'EMA200':
        df[indicator] = df.Close.ewm(span=200).mean()
    elif indicator == 'MACD':
        df['MACD'] = (df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean())
        df['signal'] = df['MACD'].ewm(span=9).mean()

        return [px.line(df, x='Date', y=['MACD', 'signal', df.Close], title="Indicators")]

    elif indicator == 'BB':
        df['stddev'] = df.Close.rolling(20).std()
        df['BB_Lower'] = df.Close.rolling(20).mean() - 2 * df['stddev']
        df['BB_Higher'] = df.Close.rolling(20).mean() + 2 * df['stddev']

        return [px.line(df, x='Date', y=['BB_Lower', 'BB_Higher', df.Close], title="Indicators")]

    elif indicator == 'SO':
        df['14-low'] = df.High.rolling(14).min()
        df['14-high'] = df.Low.rolling(14).max()
        df['%K'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
        df[indicator] = df['%K'].rolling(3).mean()

    return [px.line(df, x='Date', y=[indicator, df.Close], title="Indicators")]


# -------informations

@app.callback(
    [
        Output("company_name", "children"),
        Output("sector_name", "children"),
        Output("beta_value", "children"),
        Output("fcf_value", "children"),
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

    if asset in crypto:

        cryptot = yf.Ticker(asset)
        info = cryptot.stats()

        name = info.get('summaryProfile').get('name')
        sector = 'Cryptocurrency'
        beta = 'Unavailable'
        free_cash_flow = 'Unavailable'
        return_on_equity = 'Unavailable'

        return name, sector, beta, free_cash_flow, return_on_equity

    else:

        company = yf.Ticker(asset)
        information = company.stats()

        name = information.get('quoteType').get('shortName')
        sector = information.get('summaryProfile').get('sector')
        beta = information.get('defaultKeyStatistics').get('beta')
        free_cash_flow = information.get('financialData').get('freeCashflow')
        return_on_equity = information.get('financialData').get('returnOnEquity')

        if free_cash_flow is None:
            free_cash_flow = 'Unavailable'
        else:
            free_cash_flow = round(free_cash_flow / 1000000, 2)

        if return_on_equity is None:
            return_on_equity = 'Unavailable'
        else:
            return_on_equity = round(return_on_equity * 100, 2)

    return name, sector, beta, f'{free_cash_flow}M $', f'{return_on_equity}%'



#---------predictions

@app.callback(
    [
        Output("predictions_line_graph", "figure"),
        Output("mae_value", "children"),
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
def update_predictions(n_clicks,predict_days,asset,date):
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
        #print('Input: %s, Predicted: %.3f' % (row, yhat[0]))

        predictions.append(float(yhat))
        values = list(values)
        yhat = float(yhat)
        values.append(yhat)
        values = np.asarray(values)

        day = day + relativedelta(days=1)
        dates.append(str(day))

    mae = 1

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

    ## Figure

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

    return fig, mae, predictions[-1], fig1

#[px.line(df[-predict_days:], x='Date', y=[df.Close[-predict_days:], new_df.Predictions], title="Predctions")]




if __name__ == '__main__':
    app.run_server(debug=True, port=3002)
