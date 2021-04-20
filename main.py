import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

# App Layout

app.layout = html.Div([

    html.H1("Nestle Starbucks Social Media Data", style={'text-align': 'center'}),

    dcc.Dropdown(
        options=[
            {'label': 'Nestle', 'value': 'Nestle'},
            {'label': 'Starbucks', 'value': 'Starbucks'},
            {'label': 'Coffee', 'value': 'Coffee'}
        ],
        value='MTL'
    )
])

# Run App
if __name__ == '__main__':
    app.run_server(debug = True)