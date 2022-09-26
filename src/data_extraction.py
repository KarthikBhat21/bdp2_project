import numpy as np
import pandas as pd
import os
import tweepy as tw
import yaml
import lxml

params = yaml.safe_load(open("params.yaml"))["data_extraction"]


# Test data set split ratio
num_of_tweets = params["num_tweets"]

# Twitter API Crendentials

consumer_key = "Ganmnp3P1N5vFHZSiSwQf53IN"
consumer_secret = "4U0YLsgsY1paRnwJYHum4WISEN01qN3zr4iS7kw3RvY6ye4Dij"
access_token = "1519783902604365824-7Yuy108u7v0BrNuIVFRlvoEiAQSW8h"
access_token_secret = "LhXMx038NNKdSxjkQPxNlFUVEDDRctraS8ZdIj25z8x8r"

# Authenticate
auth = tw.OAuthHandler(consumer_key, consumer_secret)
# Set Tokens
auth.set_access_token(access_token, access_token_secret)
# Instantiate API
api = tw.API(auth, wait_on_rate_limit=True)

hashtag = "#covid19-vaccine"
query = tw.Cursor(api.search_tweets, lang="en", q=hashtag).items(num_of_tweets)
posts = [{'Tweet':tweet.text} for tweet in query]

# Create a dataframe with a column called tweets

df = pd.DataFrame( [tweet['Tweet'] for tweet in posts] , columns=['Tweets'])


data_path = os.path.join("data", "extracted")
os.makedirs(data_path, exist_ok=True)

# data_extracted_path = os.path.join(data_path, "data.xml")

# df.to_xml(data_extracted_path)

data_extracted_path = os.path.join(data_path, "extracted_data.csv")

df.to_csv(data_extracted_path)