import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from sklearn.cross_validation import train_test_split
import math
import csv
from sklearn.preprocessing.data import PolynomialFeatures
from numpy.core.umath import power

# MDSKND6TTDTE01 | NTN2S36S972S01 | NFJ9HX6K50Y601 | MLQ53H07SXKX01 | MCB9K10YHQ0X01 | MOP7UU07SXKX01
Corpus=[] 
stopwordlist = []

# Load Stop Words
with open('StopWords.txt', 'rb') as fstop:
    stopwordlist = pickle.load(fstop)
fstop.close()

# Load 6 articles (Corpus)
# Load article text
#corpus = []
print "opening article document (pickle)\n"
with open('pickle_6_articles.txt', 'rb') as fassets:
    corpus = pickle.load(fassets)  
	
# check number or articles, should be 6	
print "Corpus length: ", len(corpus)

# Load KeyWords
# Main Keywords
businesswords=np.genfromtxt('Main Keywords.csv',delimiter=',',dtype=None)
print "Main Words from csv",businesswords.shape

# Poly Keywords
polywords=np.genfromtxt('Poly_Keywords.csv',delimiter=',',dtype=None)
print "Poly Words from csv",polywords.shape

# businesswords as numpy
businesswords=np.genfromtxt('Main Keywords.csv',delimiter=',',dtype=None)
print "Main Words from csv",businesswords.shape

# polywords as numpy
polywords=np.genfromtxt('Poly_Keywords.csv',delimiter=',',dtype=None)
print "Poly Words from csv",polywords.shape

wordslist={}
for word in businesswords:
    wordslist[word]=0
	
for word in polywords:
    wordslist[word]=0
	
print "After duplicate removal, Keywords Wessam:",len(wordslist)

vectorizer=CountVectorizer(analyzer="word",vocabulary=wordslist.keys(),ngram_range=(1,4),stop_words=stopwordlist)
X_main =  vectorizer.fit_transform(corpus)
print "Main Words Shape",X_main.shape

vectPoly=TfidfVectorizer(analyzer="word",vocabulary=polywords, ngram_range=(1,4),use_idf=True,stop_words=stopwordlist)
poly =  vectPoly.fit_transform(corpus)
print "Poly Words Shape",poly.shape
 
polyFeatures=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
X_poly=polyFeatures.fit_transform(poly.todense())
print "Poly Words into Poly Features Shape",X_poly.shape
 
X=np.concatenate((X_main.toarray(),X_poly),axis=1)
#X=np.concatenate((X_main.toarray(),power(poly.toarray(),3)),axis=1)
print "Matrix Shape",X.shape

'''
outfile = open('OUTPUT_6_articles_TFIDF.txt', 'w')
for item in X:
	temp = str(item)
	outfile.write(temp)

outfile.close()
'''

# to dump results as a pickle file
outfile = 'Output_6Articles_Tfidf_pickle.csv'
pickle.dump(X, open(outfile, "w"))
itemlist = pickle.load(open(outfile, 'rb'))
print itemlist
