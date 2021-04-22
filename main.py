import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px


app = dash.Dash(__name__)

# App Layout

dropdown = dcc.Dropdown(
        options=[
            {'label': 'Nestle', 'value': 'Nestle'},
            {'label': 'Starbucks', 'value': 'Starbucks'},
            {'label': 'Coffee', 'value': 'Coffee'}
        ],
        value='MTL'
    )

app.layout = html.Div([

    html.H1("Nestle Starbucks Social Media Data", style={'text-align': 'center'}),
    dbc.Row([dbc.Col(dropdown, width=12)]), 
    html.Br(),
    html.Img(src='/assets/wordcloud/twitter cloud 1.png', style={'height':'20%', 'width':'20%'}),
    html.Br(),
    html.Img(src='/assets/wordcloud/twitter cloud 2.png', style={'height':'20%', 'width':'20%'})
    
])


# Run App
if __name__ == '__main__':
    app.run_server(debug = False)
