import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
from dash import  Dash, dash_table, dcc, html, ctx, Input, Output
import plotly.io as pio
import pandas as pd
from datetime import date
# import dash
my_key = "Y13BIJBJHU50T2YV"

def get_data(ticker):
    """
    Extracts daily stock price data for a given ticker from the AlphaVantage API and returns a pandas dataframe.

    Args:
    ticker (str): The stock symbol for the desired company, e.g. 'AAPL' for Apple Inc.

    Returns:
    pandas.DataFrame: A dataframe containing the daily closing prices for the given stock, with dates as the index.
    """
    source = "https://www.alphavantage.co/query?"
    func = "function=TIME_SERIES_DAILY_ADJUSTED"+'&'
    symbol = "symbol="+ticker + '&'
    datatype="datatype=json" + '&'
    outputsize = 'outputsize=full' + '&'
    apikey="apikey=" + my_key
    url = source + func + symbol + outputsize + datatype + apikey
    r = requests.get(url)
    data = r.json()
    if (len(data)>1):
        df = pd.DataFrame(data['Time Series (Daily)']).T["4. close"]
        return df.iloc[::-1]
    else:
        return 0

def arima_forecast(ticker, df2):
    """
    Performs an ARIMA forecast on a given DataFrame of stock prices.

    Args:
        ticker (str): The stock ticker symbol.
        df2 (pandas.DataFrame): The DataFrame containing historical stock prices.
    
    Returns:
        pandas.DataFrame: A DataFrame containing historical and forecasted stock prices, with columns for 'Ticker', 'Date', 'Close', 'Forecasted Price', 'Lower 95', 'Upper 95', 'Lower 75', 'Upper 75', 'Lower 50', and 'Upper 50'.
    """
    forecast_df = pd.DataFrame(columns=['Ticker','Date', 'Close', 'Forecasted Price'])
     # Convert the date column to datetime format
    df2=df2.reset_index()
    df2.columns = ['Date','Close']
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2['Close'] = pd.to_numeric(df2['Close'],errors='coerce')
    # Perform ARIMA forecast
    ar1_model = ARIMA(df2['Close'], order=(3, 2, 0))
    ar1_fit = ar1_model.fit()
    forecast = ar1_fit.forecast(steps=8)


    forecast_95 = ar1_fit.get_forecast(8)
    yhat_95 = forecast_95.conf_int(alpha=0.05)

    forecast_75=ar1_fit.get_forecast(8)
    yhat_75 = forecast_95.conf_int(alpha=0.25)

    forecast_50=ar1_fit.get_forecast(8)
    yhat_50 = forecast_95.conf_int(alpha=0.5)


    # Create a new dataframe with the forecasted values
    forecast_df = pd.DataFrame({'Ticker': ticker,'Date': pd.date_range(start=df2['Date'].iloc[-1]+pd.DateOffset(1), periods=8),
                                        'Forecasted Price': forecast,'Lower 95':yhat_95.iloc[:,0],'Upper 95':yhat_95.iloc[:,1],'Lower 75':yhat_75.iloc[:,0],'Upper 75':yhat_75.iloc[:,1],'Lower 50':yhat_50.iloc[:,0],'Upper 50':yhat_50.iloc[:,1]},
                                       index=None)
    merged_df = pd.concat([df2, forecast_df])#.reset_index()
    return merged_df

def search_by_ticker(ticker):
    '''
    This function takes a stock ticker symbol as input and returns a pandas DataFrame containing the closing price and
    forecasted prices for the next 8 days using the ARIMA model.

    Parameters:
    ticker (str): A stock ticker symbol for the company whose stock price is to be forecasted.

    Returns:
    pandas.DataFrame: A DataFrame with columns for the date, closing price, and forecasted prices for the next 8 days,
    along with confidence intervals at 50%, 75%, and 95% levels.
    '''
    
    closing = get_data(ticker)
    if type(closing) == int:
        return 0
    else:
        output_df = arima_forecast(ticker, closing)
        return output_df



marksd = {}
for i in  range(2000, 2024, 2):
    marksd[i] = str(i)


app = Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1(children='Fortune Teller', style={'text-align': 'center', 'margin-bottom': '10px'}),
    html.H2(children='Get a glimpse into the future of your favorite stocks', style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.H3(children='*Due to API limits, we currently support only 5 queries per minute.', style={'text-align': 'center', 'margin-bottom': '30px'}),
    html.Div([
        "Write a ticker (e.g. AAPL, AMZN, NFLX, SHEL etc.) and press Enter : ",
        dcc.Input(id='my-input', value='AAPL', type='text', debounce=True)
    ], style={'width': '80%', 'margin-left': 90}), 
    html.Div([
        dcc.Graph(id='my-subplot', style={'width': '95%'}),
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'margin-top': '20px'}),
    html.Div([
        html.P('Select a time interval using the slider:'),
        dcc.RangeSlider(
            min=2001,
            max=2023,
            step=None,
            marks={i: str(i) for i in range(2001, 2024, 2)},
            value=[2001, 2023],
            id='year-slider'
        ),
    ], style={'width': '80%', 'margin-left': 90}),
    html.Div([
        html.P('Select a time interval using the calander: '),
        dcc.DatePickerRange(
            id='my-date-picker-range',
            #min_date_allowed=date(2000, 8, 5),
            #max_date_allowed=date.today(),
            #initial_visible_month=date(2023, 1, 1)
            min_date_allowed=date(2001, 1, 1),
        max_date_allowed=date.today(),
        initial_visible_month=date(2023, 1, 1),
        )]
        ,style={'width': '80%','display': 'flex', 'flex-direction': 'column', 'align-items': 'flex-start', 'margin-left': 90}),


    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Graph(id='my-second-subplot', style={'width': '95%'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'center'}),
    html.Div([
        html.P('Select a confidence interval:'),
        dcc.RadioItems(
            id='ci-selector',
            options=[
                {'label': '50%', 'value': 0.5},
                {'label': '75%', 'value': 0.75},
                {'label': '95%', 'value': 0.95}
            ],
            value=0.95,
            labelStyle={'width': '100%','display': 'flex', 'flex-direction': 'row', 'margin':'5px'}
        )
    ], style={'width': '80%', 'margin': 'auto'}),
], style={'background-color': '#1a1a1a', 'color': 'white', 'font-family': 'Arial'})

@app.callback(
    Output('my-subplot', 'figure'),
     Output('my-second-subplot', 'figure'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('my-input', 'value'),
     Input('year-slider', 'value'),
     Input('ci-selector', 'value'))

def update_graph(start_date, end_date,input_data, year_value, confidence_interval):
    """
    Update the graphs in the app based on the selected inputs.

    Args:
    - start_date (str): The start date of the selected date range as a string in the format 'YYYY-MM-DD'.
    - end_date (str): The end date of the selected date range as a string in the format 'YYYY-MM-DD'.
    - input_data (str): The ticker symbol of the selected stock.
    - year_value (Tuple[int, int]): The range of years selected using the year slider.
    - confidence_interval (float): The confidence interval selected using the radio buttons.

    Returns:
    - Tuple[go.Figure, go.Figure]: A tuple of two Plotly figures. The first one displays the historical and forecasted 
    prices of the stock and the second one shows the distribution of forecasted prices and the confidence intervals 
    around them.
    """
    pio.templates.default = "plotly_dark"
    trig_id = ctx.triggered_id if not None else 'No clicks yet'
    #dfticker = df.loc[df['Ticker'] == input_data]
    dfticker = search_by_ticker(input_data)
    if type(dfticker)== int:
        raise dash.exceptions.PreventUpdate
    dftickerpast = dfticker.loc[dfticker['Close'].notnull()]
    dftickerfuture = dfticker.loc[dfticker['Forecasted Price'].notnull()]

    # Filter dftickerpast to get only the data for the 7 days before the start date of dftickerfuture
    start_date_f = dftickerfuture['Date'].min() - pd.Timedelta(days=7)
    end_date_f = dftickerfuture['Date'].min()
    dftickerpast_filtered = dftickerpast.loc[(dftickerpast['Date'] >= start_date_f) & (dftickerpast['Date'] <= end_date_f)]

    fig1 = go.Figure()
    fig2 = go.Figure()

    fig1.add_trace(go.Scatter(x=dftickerpast['Date'], y=dftickerpast['Close'], name='Close'))
    if trig_id == 'my-date-picker-range':
        start_date_object = date.fromisoformat(start_date)
        end_date_object = date.fromisoformat(end_date)
        fig1.update_xaxes(
            title='Date',
            range=(pd.Timestamp(year=start_date_object.year, month=start_date_object.month, day=start_date_object.day, hour=0),
                   pd.Timestamp(year=end_date_object.year, month=end_date_object.month, day = end_date_object.day, hour=0)),
            constrain='domain'
        )
    else: 
        fig1.update_xaxes(
            title='Date',
            range=(pd.Timestamp(year=year_value[0], month=1, day=1, hour=0),
                   pd.Timestamp(year=year_value[1], month=date.today().month, day = date.today().day, hour=0)),
            constrain='domain'
        )

    
    fig1.update_layout(title=f'Closing Price of {input_data} over the years')

    fig2.add_trace(go.Scatter(x=dftickerpast_filtered['Date'], y=dftickerpast_filtered['Close'], name='Close'))
    fig2.add_trace(go.Scatter(x=dftickerfuture['Date'], y=dftickerfuture['Forecasted Price'], name='Forecasted Price'))
    if confidence_interval == 0.5:
        lower_col = 'Lower 50'
        upper_col = 'Upper 50'
    elif confidence_interval == 0.75:
        lower_col = 'Lower 75'
        upper_col = 'Upper 75'
    elif confidence_interval == 0.95:
        lower_col = 'Lower 95'
        upper_col = 'Upper 95'
    else:
        lower_col = None
        upper_col = None

    if lower_col is not None and upper_col is not None:
        fig2.add_trace(go.Scatter(x=dftickerfuture['Date'], y=dftickerfuture[lower_col], name='', line=dict(color='gray', width=0)))
        fig2.add_trace(go.Scatter(x=dftickerfuture['Date'], y=dftickerfuture[upper_col], name=f'{int(confidence_interval*100)}% Confidence Interval', fill='tonexty', line=dict(color='gray', width=0)))

    fig2.update_xaxes(
        title='Date',
        range=(dftickerpast_filtered['Date'].min(), dftickerfuture['Date'].max()),
        constrain='domain'
    )

    fig1.update_yaxes(title='Price in Dollars')
    fig2.update_yaxes(title='Price in Dollars')


    fig2.update_layout(title=f"Forecasted Price of {input_data} with {int(confidence_interval*100)}% Confidence Interval")
    return fig1, fig2

if __name__ == '__main__':
    app.run_server(mode='inline')
    
