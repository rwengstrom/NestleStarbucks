import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from datetime import date
from dash.dependencies import Input


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# App Layout

dropdown = dcc.Dropdown(
        options=[
            {'label': 'Nestle', 'value': 'Nestle'},
            {'label': 'Starbucks', 'value': 'Starbucks'},
            {'label': 'Coffee', 'value': 'Coffee'}
        ],
        value='MTL'
    )


daterange = dcc.DatePickerRange(
        end_date=date(2021,4,22),
        display_format='MM/DD/YY',
        start_date_placeholder_text='MM/DD/YY'
    )

app.layout = html.Div([

    html.H1("Nestle Starbucks Social Media Data", style={'text-align': 'center'}),
    dbc.Row([dbc.Col(dropdown, width=12)]),
    html.Br(),
    dbc.Row([dbc.Col(daterange, width = 12)]),
    html.Br(),
    dcc.Input(id="input1", type="text", placeholder="Input Text Here"),
    html.Br(),
    html.Img(src='/assets/wordcloud/twitter cloud 1.png', style={'height':'20%', 'width':'20%'}),
    html.Br(),
    html.Img(src='/assets/wordcloud/twitter cloud 2.png', style={'height':'20%', 'width':'20%'})
    
])

@app.callback(
    Input("input1", "value"),
)
def update_output(input1):
    return u'Input 1 {}'.format(input1)

# Run App
if __name__ == '__main__':
    app.run_server(debug = False)
