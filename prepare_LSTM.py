import io
import os
import random
import re
import sys
import xml.etree.ElementTree
import keras_preprocessing.text
import tensorflow as tf

import numpy as np
import pandas as pd
import os
import tweepy as tw
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense,SpatialDropout1D,LSTM
from keras.utils.np_utils import to_categorical
import re
from sklearn.model_selection import train_test_split

import preprocessor as p

from gensim.parsing.preprocessing import remove_stopwords

import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]

# if len(sys.argv) != 2:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython prepare.py data-file\n")
#     sys.exit(1)

# Test data set split ratio
split = params["split"]
#random.seed(params["seed"])

#input = sys.argv[1]

input = pd.read_csv(r'C:/BDBA/BDP2/Git_Repo/bdp2_project/data/Ukraine_Data.csv')

# output_train = os.path.join("data", "prepared", "train.csv")
# output_test = os.path.join("data", "prepared", "test.csv")


# Preprocess the data:

#####################################################################################################

df = pd.DataFrame(input)

df = df['Tweets']

df = df.dropna()
df = df.drop_duplicates()

def preprocess_tweet(row):
    #text = row['Tweets']
    text = row[0]
    text = p.clean(text)
    return text

df['text'] = df.apply(preprocess_tweet)

def stopword_removal(row):
    # text = row['text']
    text = row[0]
    text = remove_stopwords(text)
    return text

df['text'] = df.apply(stopword_removal)


df['text'] = df['text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+',' ')


#####################################################################################################

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


######################################### Sentiment Analysis #########################################

df = df[["text","sentiment"]]

df = df[df.sentiment != "Neutral"]
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',x)))

print(df[ df['sentiment'] == 'Positive'].size)
print(df[ df['sentiment'] == 'Negative'].size)

for idx,row in df.iterrows():
    row[0] = row[0].replace('rt',' ')

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
x = tokenizer.texts_to_sequences(df['text'].values)
x = pad_sequences(x)

#Build LSTM Model

embed_dim = 175
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = x.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

y = pd.get_dummies(df['sentiment']).values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)


# Train Model

batch_size = 32
model.fit(x_train, y_train, epochs = 7, batch_size = batch_size, verbose = 'auto')

model.evaluate(x_test,y_test)




























# Split the dataset using the parameters and save as csv files:

# df_train, df_test = train_test_split(final_input, test_size = split, random_state=random.seed(params["seed"]))

# df_train.to_csv(os.path.join("data", "prepared", "train.csv"))

# df_test.to_csv(os.path.join("data", "prepared", "test.csv"))

# df_train.to_csv(os.path.join("data/prepared",'/train.csv'), encoding="utf8")

# df_test.to_csv(os.path.join("data/prepared",'/train.csv'), encoding="utf8")


# df_train.to_csv('C:\\BDBA\\train.csv')

# df_test.to_csv('C:\\BDBA\\test.csv')










# def process_posts(fd_in, fd_out_train, fd_out_test, target_tag):
#     num = 1
#     for line in fd_in:
#         try:
#             fd_out = fd_out_train if random.random() > split else fd_out_test
#             attr = xml.etree.ElementTree.fromstring(line).attrib

#             pid = attr.get("Id", "")
#             label = 1 if target_tag in attr.get("Tags", "") else 0
#             title = re.sub(r"\s+", " ", attr.get("Title", "")).strip()
#             body = re.sub(r"\s+", " ", attr.get("Body", "")).strip()
#             text = title + " " + body

#             fd_out.write("{}\t{}\t{}\n".format(pid, label, text))

#             num += 1
#         except Exception as ex:
#             sys.stderr.write(f"Skipping the broken line {num}: {ex}\n")


# os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

# with io.open(final_input, encoding="utf8") as fd_in:
#     with io.open(output_train, "w", encoding="utf8") as fd_out_train:
#         with io.open(output_test, "w", encoding="utf8") as fd_out_test:
#             process_posts(fd_in, fd_out_train, fd_out_test, "<r>")
