import numpy as np
import pandas as pd
import os
import tweepy as tw
from textblob import TextBlob

import preprocessor as p
import math
import json
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.parsing.preprocessing import remove_stopwords
import yaml
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error

params = yaml.safe_load(open("params.yaml"))["prepare"]

split = params["split"]

n_classes = 3

metrics_output = os.path.join("data", "prepared", "metrics.json")

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

data_input = os.path.join(sys.argv[1], "Sentiment_Analysis_Log_reg.csv")

#input = pd.read_csv(r'C:/BDBA/BDP2/Git_Repo/bdp2_project/data/UkraineData.csv')

data_input = pd.read_csv(data_input)

df = pd.DataFrame(data_input)

df = df.dropna()
df = df.drop_duplicates()

def preprocess_tweet(row):
    text = row['Tweets']
    text = p.clean(text)
    return text

df['text'] = df.apply(preprocess_tweet, axis = 1)

def stopword_removal(row):
    text = row['text']
    text = remove_stopwords(text)
    return text

df['text'] = df.apply(stopword_removal, axis = 1)

df['text'] = df['text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+',' ')


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
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df['Polarity'].apply(getAnalysis)


label_counts = df['sentiment'].value_counts()
#print(label_counts)

df = df.replace('Positive', 1)
df = df.replace('Negative', -1)
df = df.replace('Neutral', 0)

label_counts = df['sentiment'].value_counts()
#print(label_counts)

# Perform count vectorization 
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['text'])

y = df['sentiment']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

model = LogisticRegression(max_iter=10000,  dual=False)
model.fit(X_train, y_train)
pred_y = model.predict(X_test)

#pickle.dump(model, open(str(os.pa)))

acc = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')

print("Logistic Regression Sentiment Analysis Accuracy:\t")
accuracy = acc.mean()
print(accuracy)

# Calculating R-squared error:

r_squared = model.score(X_test, y_test)

# Calculating Root mean squared error:

rmse = math.sqrt(mean_squared_error(y_test, pred_y))

with open(metrics_output, "w") as outfile:
    json.dump(dict(accuracy = accuracy, r_square_error = r_squared, root_mean_squared_error = rmse), outfile)





