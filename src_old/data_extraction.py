import numpy as np
import pandas as pd
import os
import tweepy as tw
from textblob import TextBlob
import yaml


params = yaml.safe_load(open("params.yaml"))["extract"]

# if len(sys.argv) != 2:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython prepare.py data-file\n")
#     sys.exit(1)

# Test data set split ratio
num_of_tweets = params["num_tweets"]

# Twitter API Crendentials

consumer_key = "rOzqncGU7hTOEd68goZtqNWLD"
consumer_secret = "fNHpcWHnTuof3E6EH8HQc4cTSMui8eL0fzXQrKYGR8EJx2tVlw"
access_token = "1519783902604365824-dulfg3eq87bEm7wJiWeyzO3SLa7MjD"
access_token_secret = "UzvDBmZUq2GaeCqtlAr4FwEh8IS5F2UAc1R87VVAIME1n"

# Authenticate
auth = tw.OAuthHandler(consumer_key, consumer_secret)
# Set Tokens
auth.set_access_token(access_token, access_token_secret)
# Instantiate API
api = tw.API(auth, wait_on_rate_limit=True)

#hashtag = "#germany-weather"
hashtag = "#ukraine"
#query = tw.Cursor(api.search_tweets, q=hashtag).items(1000)
query = tw.Cursor(api.search_tweets, lang="en", q=hashtag).items(num_of_tweets)
posts = [{'Tweet':tweet.text} for tweet in query]
print(posts)

# Create a dataframe with a column called tweets

df = pd.DataFrame( [tweet['Tweet'] for tweet in posts] , columns=['Tweets'])

# parent_dir ="D:/BigData2/GitRepo/BigData2_project"
# # directory = "Data/extracted"
# data_path = os.path.join(parent_dir,directory)
# os.makedirs(data_path, exist_ok=True)

os.makedirs(os.path.join("data", "extracted"), exist_ok=True)

data_extracted_path = os.path.join("data", "extracted", "Sentiment_Analysis_Log_reg.csv")

df.to_csv(data_extracted_path)