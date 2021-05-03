import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import date
import makeclouds as wc
import DataFetching as fetch
import base64

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

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
                  children=dcc.Input(id='keyword1', value='Keyword', type='text', style={'color': '#FF206E'})),
         html.Div(id='dates', children=[html.Div(id='date-range', children=dcc.DatePickerRange(id='date',
                                                                                               min_date_allowed=date(
                                                                                                   1995, 8, 5),
                                                                                               max_date_allowed=date(
                                                                                                   2021, 5, 1),
                                                                                               initial_visible_month=date(
                                                                                                   2021, 4, 1),
                                                                                               end_date=date(2021, 4,
                                                                                                             1)))
                                        ])
     ], style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
     html.Div(className='competitors', children=[
         html.H5('Competitor Search:', style={'color': '#FFFFFF'}),
         dcc.Input(id='competitor1', value='Competitor', type='text', style={'color': '#FF206E'})
     ],style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000}),
     html.Div(id='searchbutton', children=[
         html.H5('Number of Tweets:', style={'color': '#FFFFFF'}),
         html.Div(className='button', children=[html.Button('Search', id='submit-val',style={'color': '#FFFFFF'})])
     ], style={'margin-left': 10, 'margin-right': 10, 'max-width': 50000, 'columnCount': 2}),
     html.Div(id='slider', children=[
         dcc.Slider(
             min=100,
             max=50000,
             step=100,
             marks={100: '100',
                    50000: '50000'
                    }, value=2
         )
     ], style={'width': '500px'}),
     html.Div(className='WordClouds', children=[
         html.Div(id='Heading1', children=[html.H4('Word Cloud 1', style={'color': '#FFFFFF'}),
                                           html.Img(id='wc1', style={'height': '10%'})]),
         html.Div(id='heading2', children=[html.H4('Most Common Words', style={'color': '#FFFFFF'})])
     ], style={'width': '98%', 'margin-left': 10, 'margin-right': 10, 'max-width': 50000,'columnCount': 2})
     ]
    , style={'backgroundColor': '#0C0F0A', 'margin-top': '-10px', 'height': '2000px'},
)


@app.callback(dash.dependencies.Output('wc1', 'src'),
              [dash.dependencies.Input('keyword1', 'value')])
def make_image(button,value):
    df = fetch.scrape_by_keyword(value)
    source = wc.create_wordcloud(df, mask, value)
    return source


# Run App
if __name__ == '__main__':
    app.run_server(debug=False)
