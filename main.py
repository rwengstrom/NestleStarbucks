import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from datetime import date
import makeclouds as wc
import DataFetching as fetch
from dash.dependencies import Input, Output, State
import base64
import dash_bootstrap_components as dbc
import plotly.express as px

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

df=''

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# App Layout

mask = 'twitter-logo-4.png'
image_filename = 'twitter cloud 1.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
dictionary = {}

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.Br(),
    html.Div(className='title', children=[html.H1('Nestle Starbucks Sentiment Analysis',
                                                  style={'margin-left': 375, 'margin-top': 200,'color': '#FFFFFF'})]),
    html.Div(className='links', children=[
        html.A(html.Button('Twitter Analysis', style={'color': '#FFFFFF'}), href='/page-1', style={'margin-left': 520}),
        html.Br(),
        html.A(html.Button('Topic Modeling', style={'color': '#FFFFFF'}), href='/page-2', style={'margin-left': 50})
    ], style={'columnCount': 2})
], style={'backgroundColor': '#0C0F0A', 'margin-top': '-10px', 'height': '2000px'})

page_1_layout = html.Div(
    [html.Div(className='title', children=[dbc.Card(dbc.CardBody(
        [dbc.CardImg(id='logo', src="/assets/logo.png")])),
        html.H2('Nestle Starbucks Keyword Analysis', style={'color': "", 'width': 950, 'fontSize': 45, 'margin-right': 400})],
              style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 3}),
    html.Br(),
     html.Div(className='heading', children=[html.H5('Keyword Search:', style={'color': '', 'fontSize': 17}),
                                              dcc.Input(id='input1', type='text', style={'color': ''}),
                                              html.H5('Baseline', style={'color': '', 'fontSize': 17}),
                                              dcc.Input(id='baseline', type='text', style={'color': '', }),
                                              html.H5('Date Range:', style={'color': '', 'fontSize': 17}),
                                              html.Div(id='dates', children=dcc.DatePickerRange(id='date',
                                                                                                min_date_allowed=date(
                                                                                                    1995, 8, 5),
                                                                                                max_date_allowed=date(
                                                                                                    2021, 5, 1),
                                                                                                initial_visible_month=date(
                                                                                                    2021, 4, 1),
                                                                                                end_date=date(2021, 4,
                                                                                                              1),
                                                                                                with_portal=True)),
                                             html.Div(id='searchbutton', children=[
                                                 html.H5(id='tweets', style={'color': '','fontSize': 17, 'margin-left': 20})]),
                                              html.Div(id='slider', children=[
                                                  html.Div(dcc.Slider(id='tweet_slider',
                                                                      min=100,
                                                                      max=50000,
                                                                      step=100,
                                                                      marks={100: '100',
                                                                             50000: '50000'
                                                                             }
                                                                      ))]),
                                              ],
              style={'width': '90%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 4}),
     html.Br(),
     html.Div(className='Buttons', children=[
                                            html.Button('Search', id='search',
                                                style={'margin-left': 10, 'color': ''}),
                                            html.Button('Load', id='load',
                                                style={'margin-left': 10, 'color': ''}),
                                            html.Button('Save', id='save',
                                                style={'margin-left': 10, 'color': ''}),
                                            html.A(html.Button('Topics', style={'color': ''}),
                                                   href='/page-2', style={'margin-left': 50})
     ],
              style={'margin-left': 525}),

     html.Br(),
     html.Br(),
     html.Div(className='WordClouds', children=[
         html.Div(id='Heading1', children=[html.H4('Word Cloud', style={'color': '', 'fontSize':20}),
                                           html.Img(id='wc1',
                                                    src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                                    height=400)]),
         html.Div(id='heading2', children=[html.H4('Sentiment Graph', style={'color': '','fontSize':20}),
                                           dcc.Graph(id='bargraph', )])
     ], style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
     html.Div(className='WeekBreakdown', children=[dcc.Dropdown(id='dropdown',
                                                                options=[
                                                                    {'label': 'January', 'value': 'January'},
                                                                    {'label': 'February', 'value': 'February'},
                                                                    {'label': 'March', 'value': 'March'},
                                                                    {'label': 'April', 'value': 'April'},
                                                                    {'label': 'May', 'value': 'May'},
                                                                    {'label': 'June', 'value': 'June'},
                                                                    {'label': 'July', 'value': 'July'},
                                                                    {'label': 'August', 'value': 'August'},
                                                                    {'label': 'September', 'value': 'September'},
                                                                    {'label': 'October', 'value': 'October'},
                                                                    {'label': 'November', 'value': 'November'},
                                                                    {'label': 'December', 'value': 'December'}
                                                                ], value='January', style={'width': 200}),
         html.H4(id='weekheading', style={'color': '', 'fontSize':20})

     ])
     ]
    , style={'backgroundColor': '', 'margin-top': '-10px', 'height': '2000px'},
)

page_2_layout = html.Div([
    html.Div(className='title', children=[html.H2('Topic Modeling', style={'color': "#CECECE"})],
             style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
    html.Div(className='search', children=dcc.Input(id='input2', type='text', style={'color': '#FF206E'})),
    html.A(html.Button('Back to Main Page', style={'color': ''}), href='/page-1', style={'margin-left': 50}),
    html.Div(className='graphs', children=dcc.Graph(id='wordgraph'))],
    style={'backgroundColor': '', 'margin-top': '-10px', 'height': '2000px'}
)

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page

# Update the Main Word Cloud
@app.callback(
    Output('wc1', 'src'),
    [Input('search', 'n_clicks')],
    state=[State('input1', 'value'),
           State('date','start_date'),
           State('date','end_date')]
    )
def update_output(n_clicks,value,start_date,end_date):
    df = fetch.fetch_once(keyword=value, since=start_date, until=end_date)
    processed_df = wc.preprocess(df)
    source = wc.create_wordcloud(processed_df, mask)
    return source

# Update the number of tweets
@app.callback(
    dash.dependencies.Output('tweets', 'children'),
    [dash.dependencies.Input('tweet_slider', 'value')])
def tweets_slider(value):
    return 'Number of Tweets: {}'.format(value)

# Update the time step
@app.callback(
    dash.dependencies.Output('weeks', 'children'),
    [dash.dependencies.Input('week_slider', 'value')])
def tweets_slider(value):
    return 'Time Step: {} Week(s)'.format(value)

# Update the Dropdown
@app.callback(
    dash.dependencies.Output('weekheading', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def tweets_slider(value):
    return '{} Word Cloud '.format(value)

# @app.callback(
#     Output('bargraph', 'figure'),
#     [Input('input1', 'value')]
#     )
# def tweets_slider(value):
#     df = fetch.fetch_once(keyword=value)
#     processed_df = wc.preprocess(df)
#     return wc.sentiment_graph(processed_df)

# Update the
@app.callback(
    Output('wordgraph', 'figure'),
    [Input('input2', 'value')]
    )
def tweets_slider(value):
    df = fetch.fetch_once(keyword=value)
    processed_df = wc.preprocess(df)
    return wc.topics(processed_df)

# Run App
if __name__ == '__main__':
    app.run_server(debug=False)