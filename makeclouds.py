def create_wordcloud(csv_name, mask_png, baseline=None):
    # Generates and plots a wordcloud for a dataset. Optionally rescales word weights to show their relative frequency to a baseline dataset.

    # Parameters:
    # csv_name: a csv file containing the tweets to be analyzed
    # mask_png: a png of the logo to use as the coloring scheme
    # baseline(optional): a csv file of data (competitors or overall category) to compare starbucks word frequencies to

    # imports
    from nltk.corpus import words
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    from spacy.lang.en import English
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk import download
    from nltk.corpus import wordnet
    #download('wordnet')
    from nltk.corpus import stopwords
    from wordcloud import WordCloud, ImageColorGenerator
    from nltk.stem import WordNetLemmatizer
    import base64

    # read and inspect data (optional)
    df1 = csv_name
    if baseline is not None:
        df2 = pd.read_csv(baseline)

    # if you want to test with a small subset to save time
    # df1 = df1.head(num_rows)

    # stopwords - NLTK
    #download('stopwords')
    # additional stopwords based on previous wordcloud results
    more_stopwords = ["starbucks", "want", "coffee", "like", "say", "put", "nestl", "nestle", "nestlé", "starbuck",
                        "starbucks", "https", "cc", "co", "ht", "tps", "i", "me", "my", "myself", "we", "our", "ours",
                        "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
                        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
                        "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
                        "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
                        "did", "doing", "a", "an", "the", "Mr", "and", "but", "if", "or", "because", "as", "until",
                        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                        "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
                        "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
                        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
                        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
                        "will", "just", "don", "should", "now"]
    stop = stopwords.words('english') + more_stopwords

    def clean_and_return_TFIDF_weights(df):
        # replace non-alphabetical characters with space using Regex
        df['tweet'] = df['tweet'].map(lambda x: re.sub(r'[^a-zA-Z] ', ' ', str(x)))
        # eliminate rows with empty values for tweet by dropping na's
        df = df.dropna(subset=['tweet'])

        # remove encoding of word- strip off any unwanted formatting/http://
        def remove_encoding_word(word):
            word = str(word)
            word = word.encode('ASCII', 'ignore').decode('ASCII')
            return word

        # apply remove_encoding_word to each word in text
        def remove_encoding_text(text):
            text = str(text)
            text = ' '.join(remove_encoding_word(word) for word in text.split() if word not in stop)
            return text

        # apply remove_encoding_word and create lemmatizer
        df['tweet'] = df['tweet'].apply(remove_encoding_text)
        text = ' '.join(words for words in df['tweet'])
        lemma = WordNetLemmatizer().lemmatize

        # apply lemmatizer (as opposed to stemming, lemmatizing breaks words down into similar dictionary definitions), filter short words and nonalphabetical characters, and fit TF-IDF model
        def tokenize(document):
            tokens = [lemma(w) for w in document.split() if len(w) > 3 and w.isalpha()]
            return tokens

        vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=((1, 1)),
                                     stop_words=stop, strip_accents='unicode')

        # fit vectorizer and transform tweets column (safe to ignore warning!)
        tdm = vectorizer.fit_transform(df['tweet'])
        # view words
        vectorizer.vocabulary_.items()
        # calculate TF-IDF weights - fast
        n = 1000
        items = list(vectorizer.vocabulary_.items())
        y = [dict(items[x:x + n + 1]) for x in range(0, len(vectorizer.vocabulary_), n + 1)]
        tfidf_weights = []
        counter = 0
        for d in y:
            counter += 1
            tfidf_weights.extend([(word, tdm.getcol(idx).sum()) for word, idx in d.items()])
            print("Processing subdictionary:", counter, "of", len(y))
        return tfidf_weights

    # get TF-IDF weights
    tfidf_weights_primary = clean_and_return_TFIDF_weights(df1)
    if baseline is not None:
        tfidf_weights_baseline = clean_and_return_TFIDF_weights(df2)

    # calculate TF-IDF weights - slow
    '''tfidf_weights = [(word, tdm.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
    len(tfidf_weights)
    tfidf_weights[0:10]'''

    # rescale TF-IDF weights
    if baseline is None:
        tfidf_weights_rescaled = tfidf_weights_primary
    else:
        tfidf_weights_rescaled = []
        tfidf_weights_primary_dict = dict(tfidf_weights_primary)
        tfidf_weights_baseline_dict = dict(tfidf_weights_baseline)
        for weight, word in enumerate(tfidf_weights_primary_dict):
            if word in tfidf_weights_baseline_dict.keys():
                rescaled_weight = (weight + 0.00000000000000000000000000001) / (
                            tfidf_weights_baseline_dict[word] + 0.00000000000000000000000000001)
                tfidf_weights_rescaled.append((word, rescaled_weight))
            else:
                tfidf_weights_rescaled.append((word, weight))

    # Create Word Cloud
    # a) including link to .png file in create_wordcloud command will turn the provided image into a mask for the wordcloud
    twitter_mask2 = np.array(Image.open(mask_png))
    image_colors = ImageColorGenerator(twitter_mask2)
    w = WordCloud(width=1500, height=1200, mask=twitter_mask2, background_color='white',
                  max_words=2000).fit_words(dict(tfidf_weights_rescaled))
    #plot = plt.figure(figsize=(20, 15))
    plt.imshow(w.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis('off')
    plt.savefig('tweets_wordcloud.png')

    # encode the image into source codes that dash can read
    image_filename = 'tweets_wordcloud.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    source = 'data:image/png;base64,{}'.format(encoded_image.decode())

    return source

def sentiment_graph(csv_name, english_only = True):
  ''''
  Takes a dataframe with tweets and plots a bar chart of the counts of postitive vs negative tweets.
  Parameters:
    -df: a dataframe with a 'tweets' column
    -english_only = filters non-english tweets if True, defaults to True
  '''
  import plotly.express as px
  from textblob import TextBlob
  import pandas as pd
  import plotly
  from  plotly.offline import plot
  from plotly.offline import iplot
  import plotly.graph_objects as go
  # add seniment column for each tweet
  def fetch_sentiment_using_textblob(text):
        sentiment = []
        for i in text:
          analysis = TextBlob(i)
          # set sentiment
          if analysis.sentiment.polarity >= 0:
              sentiment.append('positive')
          else:
              sentiment.append('negative')
        return sentiment
  df1 = csv_name
  df1 = preprocess(df1)
  tweet = df1['tweet']
  df1['sentiment']= fetch_sentiment_using_textblob(tweet)
  df1 = df1.groupby(["sentiment"]).count().reset_index()
  fig = px.bar(df1,
              y=df1['id'],
              x="sentiment",
              color='sentiment')
  # if you want to save a static image of the graph
  # fig.write_image("sentiments.png")
  fig.show()

def preprocess(df, english_only = True):
  '''
  Preprocesses the tweet column in a data frame and returns the dataframe with a cleaned tweets column.
  Parameters:
    -df: a dataframe with a 'tweets' column
    -english_only = filters non-english tweets if True, defaults to True
  '''
  #!pip install swifter
  import swifter
  #!pip install langid
  import langid
  from nltk.corpus import stopwords
  from nltk import download
  download('wordnet')
  from nltk.stem import WordNetLemmatizer
  import re
  import pandas as pd

  #df = dataframe.copy()
  #replace non-alphabetical characters with space using Regex
  df['tweet'] = df['tweet'].map(lambda x: re.sub(r'[^a-zA-Z] ', ' ', str(x)))
  #eliminate rows with empty values for tweet by dropping na's
  df = df.dropna(subset=['tweet'])
  #stopwords - NLTK
  download('stopwords')
  #additional stopwords based on previous wordcloud results
  more_stopwords = ["starbucks","want", "coffee", "like", "say", "put" , "nestl", "nestle", "nestlé", "starbuck", "starbucks", "https", "cc", "co", "ht", "tps", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the","Mr", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
  stop = stopwords.words('english') + more_stopwords

  # remove encoding of word- strip off any unwanted formatting/http://
  def remove_encoding_word(word):
      word = str(word)
      word = word.encode('ASCII', 'ignore').decode('ASCII')
      return word
  # apply remove_encoding_word to each word in text
  def remove_encoding_text(text):
      text = str(text)
      text = ' '.join(remove_encoding_word(word) for word in text.split() if word not in stop)
      return text
  # apply remove_encoding_word, lemmatize words, make lowercase
  df['tweet'] = df['tweet'].swifter.apply(remove_encoding_text)
  text = ' '.join(words for words in df['tweet'])
  lemma = WordNetLemmatizer().lemmatize
  def cleanup(document):
    tokens = [lemma(w).lower() for w in document.split() if len(w) > 3 and w.isalpha()]
    text = ' '.join(tokens)
    return text
  df['tweet'] = df['tweet'].swifter.apply(cleanup)
  # filter only English tweets
  df = df[df['tweet'] != '']
  if english_only == True:
    df['Language'] = df['tweet'].swifter.apply(lambda x: langid.classify(x))
    df = df.loc[df['language'] == 'en']
  return df