### Run these three line below if you're using for the first time:
import pip
import nest_asyncio
import pandas as pd
import twint
nest_asyncio.apply()
import numpy as np

from datetime import timedelta, date, datetime
import math


# Data fetching main function
# The input can either be a string or a list of strings
# If it's the latter, the data frame will have 3* limit number of rows
def fetch_once(keyword, since='2021-04-01', until='2021-04-02', limit=100, city=''):
    # initiate data frame
    df = pd.DataFrame()

    lst = []
    if type(keyword) == str:
        lst.append(keyword)
    else:
        lst = keyword

    for word in lst:
        # setup twint
        c = twint.Config()
        # configurations
        c.Limit = limit  # increments of 20
        c.Search = word
        c.Since = since
        c.Until = until
        c.Near = city

        c.Pandas = True
        c.Stats = False
        c.Lower_case = True
        c.Hide_output = True

        # run twint
        twint.run.Search(c)

        # store data frame
        Tweets_df = twint.storage.panda.Tweets_df
        # append to result data frame
        df = df.append(Tweets_df)

    return df


def fetch_for_range(keyword, start_year, start_month, start_date, end_year, end_month, end_date, interval=7, limit=100):
    print('fetching twitter data...')

    d0 = date(start_year, start_month, start_date)
    d1 = date(end_year, end_month, end_date)
    delta = d1 - d0

    df = pd.DataFrame()
    # fetch tweet data for each day
    for d in range(delta.days):
        since = (d0 + timedelta(days=d)).strftime("%Y-%m-%d")
        until = (d0 + timedelta(days=d + 2)).strftime("%Y-%m-%d")
        # fetch 1 day of data
        Tweets_df = fetch_once(keyword, since, until, limit)
        # Assign time group
        Tweets_df['group'] = math.ceil((d + 1) / interval)
        # Assign year
        Tweets_df['year'] = Tweets_df.date.apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d').year)
        # Assign month
        Tweets_df['month'] = Tweets_df.date.apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d').month)
        # Assign day
        Tweets_df['day'] = Tweets_df.date.apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d').day)
        df = df.append(Tweets_df)

    print('fetching complete, return', len(df), 'rows')

    return df


def fetch_by_year(keyword, year):
    start_year = year
    start_month = 1
    start_date = 1
    end_year = year + 1

    today = date.today()
    # check if it's the current year
    if year == today.year:
        end_month = today.month
        end_date = today.day
    else:
        end_month = 1
        end_date = 1

    # fetch data for the selected year
    # This might take a while....
    df = fetch_for_range(keyword, start_year, start_month, start_date, end_year, end_month, end_date, interval=1,
                         limit=10)
    return df

