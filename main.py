import nltk
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')

import string
import re
import numpy as np
from nltk.corpus import words
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import cross_validate as cross_validation, ShuffleSplit, cross_val_score
import nltk
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

def labelStance(labelDict, data):
	for key, val in labelDict.items():
		data.loc[data["Stance"] == val, "Stance"] = int(key)
	return data

def readGlobalVecData(glove_word_vec_file):
	file = open(glove_word_vec_file, encoding="utf8")
	rawData = file.readlines()
	glove_word_vec_dict = {}
	for line in rawData:
		line = line.strip().split()
		tag = line[0]
		vec = line[1:]
		glove_word_vec_dict[tag] = np.array(vec, dtype=float)
	return glove_word_vec_dict

gloveFile = "/Users/lidiiamelnyk/Downloads/vectors.txt"

print("\nLoading Glove data in progress...")
glove_word_vec_dict = readGlobalVecData(gloveFile)
print("\nLoading Glove data is done...")

classifiers = ['Support Vector Machine', 'Random Forest Classifier', 'Gradient Boosting Classifier',  'K Neighbors Classifier', 'Decision Tree Classifier']

training = "/Users/lidiiamelnyk/Documents/trans_train.csv"

Comments = pd.read_csv(training,encoding='utf-8')

#For converting all the stances into numerical values in both training and test data
labelDict = {0:"Against", 1:"Favor", 2:"Neutral"}
Comments = labelStance(labelDict, Comments)
print('Shape of label tensor:', Comments.shape)