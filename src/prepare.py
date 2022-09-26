import io
import os
import random
import re
import sys
import xml.etree.ElementTree
import preprocessor as p
from gensim.parsing.preprocessing import remove_stopwords
import yaml
from lxml import etree
import pandas as pd

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

input = sys.argv[1]

data_input = os.path.join(input)


data_input = pd.read_csv(data_input)

df = pd.DataFrame(data_input)

df = df.dropna()
df = df.drop_duplicates()

# def preprocess_tweet(row):
#     text = row['Tweets']
#     text = p.clean(text)
#     return text

# df['text'] = df.apply(preprocess_tweet, axis = 1)

def stopword_removal(row):
    text = row['Tweets']
    text = remove_stopwords(text)
    return text

df['text'] = df.apply(stopword_removal, axis = 1)

df['text'] = df['text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+',' ')

data_path = os.path.join("data", "prepared")

os.makedirs(data_path, exist_ok=True)

data_extracted_path = os.path.join(data_path, "transformeddata.csv")

df.to_csv(data_extracted_path)




