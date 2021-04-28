def create_wordcloud(csv_name, mask_png=None):

# calling this function, by default, will generate a wordcloud for the top 1000 terms in a provided dataset
# if you wish to create a custom mask, simply provide a link to the file location when you call the function ex: create_wordcloud(filepath_data,filepath_image)
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

    download('wordnet')
    from nltk.corpus import stopwords
    from wordcloud import WordCloud, ImageColorGenerator
    from nltk.stem import WordNetLemmatizer
    # read and inspect data (optional)
    df1 = pd.read_csv(csv_name)
    # if you want to test with a small subset to save time
    # df1 = df1.head(100)

    # clean data

    # replace non-alphabetical characters with space using Regex
    df1['tweet'] = df1['tweet'].map(lambda x: re.sub(r'[^a-zA-Z] ', ' ', str(x)))
    # eliminate rows with empty values for tweet by dropping na's
    df1 = df1.dropna(subset=['tweet'])
    # stopwords - NLTK
    download('stopwords')
    # additional stopwords based on previous wordcloud results
    more_stopwords = ["starbucks", "want", "coffee", "like", "say", "put", "nestl", "nestle", "nestlé", "starbuck",
                  "starbucks", "https", "cc", "co", "ht", "tps", "i", "me", "my", "myself", "we", "our", "ours",
                  "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
                  "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                  "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were",
                  "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
                  "Mr", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
                  "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
                  "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
                  "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
                  "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
                  "t", "can", "will", "just", "don", "should", "now"]
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


    # apply remove_encoding_word and create lemmatizer
    df1['tweet'] = df1['tweet'].apply(remove_encoding_text)
    text = ' '.join(words for words in df1['tweet'])
    lemma = WordNetLemmatizer().lemmatize


    # apply lemmatizer (as opposed to stemming, lemmatizing breaks words down into similar dictionary definitions), filter short words and nonalphabetical characters, and fit TF-IDF model
    def tokenize(document):
        tokens = [lemma(w) for w in document.split() if len(w) > 3 and w.isalpha()]
        return tokens


    vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=((1, 2)),
                                 stop_words=stop, strip_accents='unicode')

    # fit vectorizer and transform tweets column (safe to ignore warning!)
    tdm = vectorizer.fit_transform(df1['tweet'])
    # view words
    vectorizer.vocabulary_.items()
    # calculate TF-IDF weights - fast
    n = 1000
    items = list(vectorizer.vocabulary_.items())
    y = [dict(items[x:x + n + 1]) for x in range(0, len(vectorizer.vocabulary_), n + 1)]
    tfidf_weights2 = []
    counter = 0
    for d in y:
        counter += 1
        tfidf_weights2.extend([(word, tdm.getcol(idx).sum()) for word, idx in d.items()])
        print("Processing subdictionary:", counter, "of", len(y))
    tfidf_weights2[0:10]

    # calculate TF-IDF weights - slow
    '''tfidf_weights = [(word, tdm.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
    len(tfidf_weights)
    tfidf_weights[0:10]'''


    # Create Word Cloud
    # a) including link to .png file in create_wordcloud command will turn the provided image into a mask for the wordcloud
    if mask_png is not None:
        twitter_mask2 = np.array(Image.open(mask_png))
        image_colors = ImageColorGenerator(twitter_mask2)
        w = WordCloud(width=1500, height=1200, mask=twitter_mask2, background_color='white',
                    max_words=2000).fit_words(dict(tfidf_weights2))
        plt.figure(figsize=(20, 15))
        plt.imshow(w.recolor(color_func=image_colors), interpolation="bilinear")
        plt.axis('off')
        plt.savefig('tweets_wordcloud.png')
    # b) not including a .png will still generate a wordcloud, just without a mask
    else:
        w = WordCloud(width=1500, height=1200, background_color='white',
                        max_words=2000).fit_words(dict(tfidf_weights2))
        plt.figure(figsize=(20, 15))
        plt.imshow(w)
        plt.axis('off')
        plt.savefig('tweets_wordcloud.png')

create_wordcloud('twi_march_12col.csv')