import twint
import pandas as pd

# Data fetching main function
def scrape_by_keyword(keywords, since='2021-03-01', until='2021-04-01', limit=100, city=''):
    # initiate data frame
    df = pd.DataFrame()

    # loop through all keywords and append result
    for word in keywords:
      # setup twint
      c = twint.Config()
      # configurations
      c.Limit = limit   # increments of 20
      c.Search = word
      c.Since = since
      c.Until = until
      c.Near = city

      c.Pandas = True
      c.Stats = True
      c.Lower_case = True
      c.Hide_output = False
      # run twint
      twint.run.Search(c)

      # store data frame
      Tweets_df = twint.storage.panda.Tweets_df
      # append to result data frame
      df = df.append(Tweets_df)

    return df

