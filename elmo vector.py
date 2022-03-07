import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import pickle
import re
pd.set_option('display.max_colwidth', 200)

df = pd.read_csv("/Users/lidiiamelnyk/Documents/trans_train.csv",encoding='utf-8', sep = ';')
print(df.shape)
df_train = df.sample(frac = 0.85)
df_test = df.drop(df_train.index)

print(df_train.head(10))

def clean_df(data):
  lines = []
  for i, row in data.iterrows():
    for line in str(row['Text']).split('/n'):
      new_line = re.sub(r'http\S+', '', line)
      #lines.append(new_line)
      data.loc[i, 'clean_text'] = line
  return data

df_train = clean_df(df_train)
df_test = clean_df(df_test)
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

import spacy
for i, row in df_train.iterrows():
  for line in str(row['clean_text']).split('/n'):
    line = ''.join(ch for ch in line if ch not in set(punctuation))
    line = line.lower()
    line = line.replace("[0-9]", " ")

for i, row in df_test.iterrows():
  for line in str(row['clean_text']).split('/n'):
    line = ''.join(ch for ch in line if ch not in set(punctuation))
    line = line.lower()
    line = line.replace("[0-9]", " ")

import tensorflow as tf