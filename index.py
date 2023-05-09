import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
from dash import  Dash,dash_table, dcc, html, ctx
from dash.dependencies import Input, Output
from datetime import date
my_key = "Y13BIJBJHU50T2YV"

def get_data(ticker):
    '''This function extracts data from the alphavantage API'''
    try:
        source = "https://www.alphavantage.co/query?"
        func = "function=TIME_SERIES_DAILY_ADJUSTED"+'&'
        symbol = "symbol="+ticker + '&'
        datatype="datatype=json" + '&'
        outputsize = 'outputsize=full' + '&'
        apikey="apikey=" + my_key
        url = source + func + symbol + outputsize + datatype + apikey
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['Time Series (Daily)']).T["4. close"]
        return df.iloc[::-1]
    except:
        return None


def arima_forecast(ticker, df2):
    '''This function creates a forecast'''
    if df2 is not None:
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
    else:
        return None

def search_by_ticker(ticker):
    '''This function takes ticker as input and returns a dataframe containing closing price and forecast as output'''
    closing = get_data(ticker)
    output_df = arima_forecast(ticker, closing)
    return output_df

# marksd = {}
# for i in  range(2000, 2024, 2):
#     marksd[i] = str(i)


app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.Div([
        "Enter a ticker: ",
        dcc.Input(id='my-input', value='AAPL', type='text')
    ]), 
    html.Div([
        dcc.Graph(id='my-subplot', style={'width': '100%'}),
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'center'}),
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
    ], style={'width': '80%', 'margin': 'auto'}),
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
        ,style={'width': '80%','display': 'flex', 'flex-direction': 'column', 'align-items': 'flex-start', 'margin-left': 100}),


    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Graph(id='my-second-subplot', style={'width': '100%'})
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
            labelStyle={'width': '100%','display': 'flex', 'flex-direction': 'row', 'margin':'0px'}
        )
    ], style={'width': '80%', 'margin': 'auto'}),
])

@app.callback(
    Output('my-subplot', 'figure'),
     Output('my-second-subplot', 'figure'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('my-input', 'value'),
     Input('year-slider', 'value'),
     Input('ci-selector', 'value'))

def update_graph(start_date, end_date,input_data, year_value, confidence_interval):
    try:
        trig_id = ctx.triggered_id if not None else 'No clicks yet'
        #dfticker = df.loc[df['Ticker'] == input_data]
        dfticker = search_by_ticker(input_data)
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
                title='',
                range=(pd.Timestamp(year=start_date_object.year, month=start_date_object.month, day=start_date_object.day, hour=0),
                    pd.Timestamp(year=end_date_object.year, month=end_date_object.month, day = end_date_object.day, hour=0)),
                constrain='domain'
            )
        else: 
            fig1.update_xaxes(
                title='',
                range=(pd.Timestamp(year=year_value[0], month=1, day=1, hour=0),
                    pd.Timestamp(year=year_value[1], month=date.today().month, day = date.today().day, hour=0)),
                constrain='domain'
            )

        
        fig1.update_layout(title='Closing Price')

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
            title='',
            range=(dftickerpast_filtered['Date'].min(), dftickerfuture['Date'].max()),
            constrain='domain'
        )
        fig2.update_layout(title='Forecasted Price with Confidence Interval')

        return fig1, fig2
    except:
        pass

if __name__ == '__main__':
    app.run_server(mode='inline')
    