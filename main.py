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


#function to get the labels
def labelStance(labelDict, data):
	for key, val in labelDict.items():
		data.loc[data["Label"] == val, "Label"] = int(key)
	return data

#create a dictionary of glove vectors from the file
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
#read glove vectors with the function
glove_word_vec_dict = readGlobalVecData(gloveFile)
print("\nLoading Glove data is done...")

#load classifiers
classifiers = ['Support Vector Machine', 'Random Forest Classifier', 'Gradient Boosting Classifier',  'K Neighbors Classifier', 'Decision Tree Classifier']

training = "/Users/lidiiamelnyk/Documents/trans_train.csv"

Comments = pd.read_csv(training,encoding='utf-8', sep = ';')
Comments = Comments.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'])
#For converting all the stances into numerical values in both training and test data
labelDict = {0:"__label__AGAINST", 1:"__label__FAVOR", 2:"__label__NEUTRAL"}
Comments = labelStance(labelDict, Comments)
print('Shape of label tensor:', Comments.shape)

#get glove vectors if word in list
def getWordVector(word, glove_word_vec_dict):
	if word in glove_word_vec_dict:
		return glove_word_vec_dict[word]
	return np.zeros_like(glove_word_vec_dict['dummy'])

def sumVectors(finalList, glove_word_vec_dict):
	numNonZero = 0
	#get dummies
	vector = np.zeros_like(glove_word_vec_dict['dummy'])
	for word in finalList:
		vect = getWordVector(word, glove_word_vec_dict)
		if vect.sum() != 0:
			vector += vect
			numNonZero += 1

	if numNonZero:
		vector = vector/numNonZero
	return vector

def sumVectorsCNN(finalList, glove_word_vec_dict):
	numNonZero = 0
	vector = []
	for word in finalList:
		vector.append(getWordVector((word, glove_word_vec_dict)))
	return vector

def simplify(word):
	dump = ''
	temp = []
	listOfWords = list(filter(None, re.split('([A-Z][^A-Z]*)', word)))
	if len(listOfWords) == len(word):
		return word.lower()
	for i in range(len(listOfWords)):
		listOfWords[i] = listOfWords[i].lower()
		if len(listOfWords[i]) == 1:
			dump = dump + listOfWords[i]
			if dump in words.words() and len(dump) > 2:
				temp.append(dump)
				dump = ''
		else:
			temp.append(listOfWords[i])

	return temp

def glove(glove_word_vec_dict, trainComments):
	def createTokens(data, glove_word_vec_dict):
		listofComments = []
		listofStances = []
		CommentVector = []
		for ind, row in data.iterrows():
			#create a sentence using target and comment. it will be used to form wordvector
			example_sentence = 'Transsexual' + " " + str(row['Text'])
			#remove punctuation
			final_sentence = example_sentence.translate(string.punctuation)
			wordList = word_tokenize(final_sentence)
			finalList = []
			s = ' '.join([i for i in wordList if i.isalpha()])
			#create tokens from the string and stem them
			wordList = word_tokenize(s)
			wordList = [w.lower() for w in wordList]
			stop_words = set(stopwords.words('german'))
			wordList = [w for w in wordList if w not in stop_words]
			for word in wordList:
				#to break any combined word into its simplified components (e.g. hashtags)
				finalList += simplify(word)
			final_sentence = ' '.join(finalList)
			listofComments.append(final_sentence)
			listofStances.append(row['Label'])
			CommentVector.append(sumVectors(finalList, glove_word_vec_dict))
		return listofComments, listofStances, CommentVector
		#remove punctuation from and tokenize the comments

	listofComments, listofStances, trainCommentVector = createTokens(trainComments, glove_word_vec_dict)


	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(trainCommentVector, listofStances, test_size= 0.1, random_state = 42)

	Xtrain = np.asarray(X_train)
	Xtest = np.asarray(X_test)
	ytrain = np.asarray(y_train)
	ytest = np.asarray(y_test)

	return Xtrain, ytrain, Xtest, ytest

totalAcc = []

for classifier in classifiers:
	print( 'The machine learning model used for classification: ' + classifier )
	temp = []
	#['Support Vector Machine', 'Random Forest Classifier', 'Gradient Boosting Classifier',  'K Neighbors Classifier', 'Decision Tree Classifier']
	Xtrain, ytrain, Xtest, ytest = glove(glove_word_vec_dict, Comments)

	if classifier == 'Support Vector Machine':
		clf = SVC(kernel = 'rbf').fit(Xtrain, ytrain)

	elif classifier == 'Random Forest Classifier':
		clf = RandomForestClassifier(n_estimators= 90).fit(Xtrain, ytrain)

	elif classifier == 'Gradient Boosting Classifier':
		clf = GradientBoostingClassifier().fit(Xtrain, ytrain)

	elif classifier == 'K Neighbors Classifier':
		clf = GaussianNB().fit(Xtrain, ytrain)

	elif classifier == 'Decision Tree Classifier':
		clf = tree.DecisionTreeClassifier().fit(Xtrain, ytrain)

	acc = clf.score(Xtest, ytest)

	print("Total Test Accuracy is " + str(round(acc * 100, 2)) + "%")
	totalAcc.append(acc)


import matplotlib.pyplot as plt

x  = ['SVM', 'RFC', 'GBC', 'KNN', 'DT']
y  = totalAcc

plt.plot(x, y)
plt.plot()

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Comparison of Total Test Accuracy of different Baseline Models")
plt.show()