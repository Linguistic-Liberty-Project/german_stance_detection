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


import de_core_news_sm
import de_core_news_sm
nlp = de_core_news_sm.load(disable=['parser', 'ner'] )

def lemmatization(texts):
  output = []
  for i in texts:
    s = [token.lemma_ for token in nlp(i)]
    output.append(' '.join(s))
  return output

df_train['text'] = lemmatization(df_train['clean_text'])
df_test['text'] = lemmatization(df_test['clean_text'])

from pathlib import Path
filepath_train = Path('/Users/lidiiamelnyk/Documents/german_stance_detection/df_train.csv')
filepath_train.parent.mkdir(parents=True, exist_ok=True)
df_train.to_csv(filepath_train)
filepath_test = Path('/Users/lidiiamelnyk/Documents/german_stance_detection/df_test.csv')
df_test.to_csv(filepath_test)
from flair.data import Corpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

#create embedding
document_embeddings = TransformerDocumentEmbeddings('bert-base-german-cased')

from flair.data import Corpus, Sentence
from flair.data_fetcher import NLPTaskDataFetcher


corpus: Corpus = NLPTaskDataFetcher.load_classification_corpus(Path('/Users/lidiiamelnyk/Documents/german_stance_detection/'),
                                                                    train_file ='df_train.csv',
                                                                    test_file = 'df_test.csv',
                                                                    label_type = 'Label',
                                                            )
label_type = 'Label'

label_dict = corpus.make_label_dictionary(label_type=label_type)

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

trainer = ModelTrainer(classifier, corpus)

trainer.fine_tune('resources/taggers/stance-classification-with-transformer',
                  learning_rate=5.0e-5,
                  mini_batch_size=4,
                  max_epochs=10,
                  )

classifier = TextClassifier.load('resources/taggers/stance-classification-with-transformer/final-model.pt')


# create example sentence
sentence = Sentence('Die LGBT Sache unterstütze ich gar nicht')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)



