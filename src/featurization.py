import os
import pickle
import sys
from textblob import TextBlob
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import random
from sklearn.model_selection import train_test_split

input = sys.argv[1]

data_input = os.path.join(input,"transformeddata.csv")


data_input = pd.read_csv(data_input)

df = pd.DataFrame(data_input)

text = df['Tweets']
# Create a function to get the subjectivity

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create two new columns
df['Subjectivity'] = df['text'].apply(getSubjectivity)
df['Polarity'] = df['text'].apply(getPolarity)


# Function to compute the negative, neutral and positive analysis

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    # elif score == 0:
    #     return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df['Polarity'].apply(getAnalysis)

df = df.replace('Positive', 1)
df = df.replace('Negative', -1)
# df = df.replace('Neutral', 0)


data_path = os.path.join("data", "features")

os.makedirs(data_path, exist_ok=True)

df.to_csv(os.path.join(data_path, "features.csv"), index=None)

