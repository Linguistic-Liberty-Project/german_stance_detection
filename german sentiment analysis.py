import pandas as pd

xls = pd.ExcelFile('/Users/lidiiamelnyk/Documents/Non_binary_small.xlsx')

from germansentiment import SentimentModel

model = SentimentModel()
sheet_to_df_map = {}
i = 0
for sheet_name in xls.sheet_names:
    sheet_to_df_map[sheet_name] = xls.parse(sheet_name)

for key in sheet_to_df_map.keys():
  for i,row in sheet_to_df_map[key].iterrows():
    for m in str(row['Reply']).split('\n'):
      if m == 'nan':
        pass
      elif m != 'nan':
        sheet_to_df_map[key].loc[i, 'Comment'] = m

df = pd.DataFrame()

for key in sheet_to_df_map.keys():
  df = df.append(sheet_to_df_map[key])

list_text = [ ]
for i, row in df.iterrows():
    list_row = []
    df['Comment'] = df['Comment'].astype(str)
    for k in str(row['Comment']).split('/n'):
        for l in k.split('.'):
            list_row.append(l)
        list_text.append(list_row)

result_list = []
for l in list_text:
    result = model.predict_sentiment(l)
    print(result)
    result_list.append(result)

def list_restruction(list):
  for i, x in enumerate(list):
    for j, a in enumerate(x):
        if 'negative' in a:
            list[i][j] = a.replace('negative', '-1')
        if 'positive' in a:
            list[i][j] = a.replace('positive', '1')
        elif 'neutral' in a:
            list[i][j] = a.replace('neutral', '0')
  return list

new_list = list_restruction(result_list)

import numpy as np
def sentiment_calculation(list):
    for i, x in enumerate(list):
      for j in enumerate(x):
        total_sum = sum(int(j))
        total = total_sum/len(x)
        print(total)

total_sentiment = sentiment_calculation(new_list)