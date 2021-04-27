import dash
import dash_core_components as dcc
import dash_html_components as html
import twint
import pandas as pd
from datetime import date
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import string
from string import punctuation
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def scrape_by_keyword(keywords, since, until):
    # initiate data frame
    df = pd.DataFrame()

    # loop through all keywords and append result
    for word in keywords:
        # setup twint
        c = twint.Config()
        # configurations
        c.Limit = 500  # increments by 20, so 100 is 2,000 tweets
        c.Since = since
        c.Until = until
        c.Search = word
        c.Pandas = True
        c.Stats = True
        c.Lower_case = True
        c.Hide_output = True
        # run twint
        twint.run.Search(c)

        # store data frame
        Tweets_df = twint.storage.panda.Tweets_df
        # append to result data frame
        df = df.append(Tweets_df)

    return df

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
nltk.download('wordnet')
def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove english stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove contextual stop words
    more_stopwords = ["starbucks","https://t.co" ,"nestle", "nestlé", "starbuck", "starbucks", "https", "cc", "co", "ht", "tps", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the","Mr", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    text = [x for x in text if x not in more_stopwords]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only two letters
    text = [t for t in text if len(t) > 2]
    # join all
    text = " ".join(text)
    return(text)

# App Layout

dropdown = dcc.Dropdown(
    options=[
        {'label': 'Nestle', 'value': 'Nestle'},
        {'label': 'Starbucks', 'value': 'Starbucks'},
        {'label': 'Coffee', 'value': 'Coffee'}
    ],
    value='MTL'
)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-range',
        min_date_allowed=date(1995, 8, 5),
        max_date_allowed=date(2017, 9, 19),
        initial_visible_month=date(2017, 8, 5),
        end_date=date(2017, 8, 25)
    ),
    dcc.Input(id="input1", type="text", placeholder=""),
    html.Div(id='output-container-date-picker-range')
])

print()

@app.callback(
    dash.dependencies.Output('output-container-date-picker-range', 'children'),
    [dash.dependencies.Input('date-range', 'start_date'),
     dash.dependencies.Input('date-range', 'end_date'),
     dash.dependencies.Input('input1', 'value')])
def update_output(start_date, end_date,value):
    since = start_date
    until = end_date
    df = scrape_by_keyword(value,since,until)
    df.tweet = df.tweet.astype('str')
    df["tweet_clean"] = df["tweet"].apply(lambda x: clean_text(x))
    # Only the english reviews
    df = df[df.language == 'en'].reset_index(drop=True)
    # removing other characters
    df["tweet_clean"] = df["tweet_clean"].replace('_', '')
    df["tweet_clean"] = df["tweet_clean"].replace('?', '')
    df["tweet_clean"] = df["tweet_clean"].replace('•', '')
    df["tweet_clean"] = df["tweet_clean"].replace("@", '')
    df["tweet_clean"] = df["tweet_clean"].replace('▯', '')
    df["tweet_clean"] = df["tweet_clean"].replace("'", '')
    df["tweet_clean"] = df["tweet_clean"].replace(",", "")
    df["tweet_clean"] = df["tweet_clean"].replace(":", "")
    df["tweet_clean"] = df["tweet_clean"].replace("/", "")

# Run App
if __name__ == '__main__':
    app.run_server(debug=False)
