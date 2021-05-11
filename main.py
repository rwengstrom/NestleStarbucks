import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from datetime import date
import makeclouds as wc
import DataFetching as fetch
from dash.dependencies import Input, Output, State
import base64

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

df='twi_march_12col.csv'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# App Layout

mask = 'twitter-logo-4.png'
image_filename = 'twitter cloud 1.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(
    [html.Div(className='title', children=[html.H2('Nestle Starbucks Keyword Analysis', style={'color': "#CECECE"}), ],
              style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
     html.Div(className='heading', children=[html.H5('Keyword Search:', style={'color': '#FFFFFF'}),
                                             html.H5('Date Range:', style={'color': '#FFFFFF'})],
              style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
     html.Div(className='Search', children=[
         html.Div(id='inputs',
                  children=dcc.Input(id='input1', type='text', style={'color': '#FF206E'})),
         html.Div(id='dates', children=dcc.DatePickerRange(id='date',
                                                           min_date_allowed=date(
                                                               1995, 8, 5),
                                                           max_date_allowed=date(
                                                               2021, 5, 1),
                                                           initial_visible_month=date(
                                                               2021, 4, 1),
                                                           end_date=date(2021, 4, 1)))
     ], style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
     html.Div(className='competitors', children=[
         html.H5('Baseline Keyword:', style={'color': '#FFFFFF'}),
         dcc.Input(id='baseline1', type='text', style={'color': '#FF206E'})
     ], style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000}),
     html.Div(id='searchbutton', children=[
         html.H5(id='tweets', style={'color': '#FFFFFF'}),
         html.H5(id='weeks', style={'color': '#FFFFFF'})
     ], style={'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
     html.Div(id='slider', children=[
         html.Div(dcc.Slider(
             id='tweet_slider',
             min=10,
             max=5000,
             step=10,
             marks={10: '10',
                    5000: '5000'
                    }
         )),
         html.Div(dcc.Slider(
             id='week_slider',
             min=1,
             max=52,
             step=1,
             marks={1: '1',
                    13: '13',
                    26: '26',
                    39: '39',
                    52: '52'
                    }),style={'margin-left': '95px'}
        )
     ], style={'width': '1200px', 'columnCount': 2}),
     html.Br(),
     html.Div(className='Buttons',children=[
         html.Button('Search', id='search', style={'margin-left': 10,'color': '#FFFFFF'}),
         html.Button('Load', id='load', style={'margin-left': 10,'color': '#FFFFFF'}),
         html.Button('Save', id='save', style={'margin-left': 10,'color': '#FFFFFF'})
     ]),
     html.Div(className='WordClouds', children=[
         html.Div(id='Heading1', children=[html.H4('Word Cloud 1', style={'color': '#FFFFFF'}),
                                           html.Img(id='wc1', src='data:image/png;base64,{}'.format(encoded_image.decode()), height=400)]),
         html.Div(id='heading2', children=[html.H4('Most Common Words', style={'color': '#FFFFFF'}),
                                           dcc.Graph(id='histogram')])
     ], style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2})
     ]
    , style={'backgroundColor': '#0C0F0A', 'margin-top': '-10px', 'height': '2000px'},
)


@app.callback(
    Output('wc1', 'src'),
    [Input('search', 'n_clicks')],
    state=[State('input1', 'value'),
           State('tweet_slider','slider')
             ])
def update_output(n_clicks,value,slider):
    df = fetch.scrape_by_keyword(value)
    source = wc.create_wordcloud(df, mask)
    return source

@app.callback(
    dash.dependencies.Output('tweets', 'children'),
    [dash.dependencies.Input('tweet_slider', 'value')])
def tweets_slider(value):
    return 'Number of Tweets: {}'.format(value)

@app.callback(
    dash.dependencies.Output('weeks', 'children'),
    [dash.dependencies.Input('week_slider', 'value')])
def tweets_slider(value):
    return 'Time Step: {} Week(s)'.format(value)

@app.callback(
    dash.dependencies.Output('histogram', 'children'),
    [dash.dependencies.Input('input1', 'value')])
def tweets_slider(value):
    df = fetch.scrape_by_keyword('value'),
    return wc.sentiment_graph(df)



# Run App
if __name__ == '__main__':
    app.run_server(debug=False)