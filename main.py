import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from datetime import date
import makeclouds as wc
import DataFetching as fetch

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# App Layout

app.layout = html.Div([
    html.H1('Nestle Starbucks Word Cloud'),
    html.Br(),
    dcc.DatePickerRange(
        id='date-range',
        min_date_allowed=date(1995, 8, 5),
        max_date_allowed=date(2017, 9, 19),
        initial_visible_month=date(2017, 8, 5),
        end_date=date(2017, 8, 25)
    ),
    dcc.Input(id="input1", type="text", placeholder=""),
    html.Img(id='image')
])

print()

@app.callback(
    dash.dependencies.Output('image', 'src'),
     [dash.dependencies.Input('input1', 'value')])
def update_output(value):
    df = fetch.scrape_by_keyword(value)
    return wc.create_wordcloud(df)


# Run App
if __name__ == '__main__':
    app.run_server(debug=False)
