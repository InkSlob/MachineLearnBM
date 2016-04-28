import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

Corpus=[] 
#fcorpus = open('/home/master/Documents/Blmbrg/Ln_Rg_Model/Corpus_15k_Stemmed.txt', 'rb') 
fcorpus = open('Corpus_15k_Stemmed.txt', 'rb') 
Corpus = pickle.load(fcorpus)

#check if pickle loaded properly
if(len(Corpus) == 15000):
    print "Article pickle file has loaded with proper size of 15,000.  Size = ", len(Corpus), "\n"
print "__________________________________________________________________________________________________________\n"
         
vectorizer = TfidfVectorizer(min_df=100)
X=vectorizer.fit_transform(Corpus)
print "The shape of Tfidf Vectorization is: ", X.shape
#Output appears as: (0, number of words) normalized rate ___   (0, 1015)	0.0205270786855
print "__________________________________________________________________________________________________________\n"

ShelfLife=[]
#fShelfLife = open('/home/master/Documents/Blmbrg/Ln_Rg_Model/ShelfLife_15k_Stemmed.txt', 'rb') 
fShelfLife = open('ShelfLife_15k_Stemmed.txt', 'rb') 
ShelfLife = pickle.load(fShelfLife)
#check if pickle loaded properly
if(len(ShelfLife) == 15000):
    print "ShelfLife pickle file has loaded with proper size of 15,000.  Size = ", len(ShelfLife), "\n"
print "__________________________________________________________________________________________________________\n"

print "linear regression model initialized\n"
#Split data into train & test 80 / 20
corpus_x_train,corpus_x_test,shelf_y_train,shelf_y_test = train_test_split(X,ShelfLife,test_size=0.20)

#Create linear regression object
regr = linear_model.LinearRegression()

#Train the model using training set
lr = regr.fit(corpus_x_train, shelf_y_train)

print "Cross-validation: \n"
score_cv = []
score_cv = cross_validation.cross_val_score(lr, corpus_x_test, shelf_y_test, cv=5)
print "The cross validation score is: ", score_cv, "\n"
print "__________________________________________________________________________________________________________\n"


#The Coefficients
coeff = regr.coef_
print "Coefficients: ", coeff, "\n"

#The root mean square error
pred = regr.predict(corpus_x_test)
baselineAVG = np.mean(shelf_y_train)
print "Shelf Life Training Data Baseline Average: ", baselineAVG, "\n"
RMSE_Base = (np.mean(baselineAVG - shelf_y_test)**2)**0.5
print "The RMSE Baseline is: ", RMSE_Base, "\n"

RMSE = (np.mean(pred - shelf_y_test)**2)**0.5
print("Residual sum of squares: %.2f"% RMSE)

#Explain variance score: 1 is perfect prediction
var = regr.score(corpus_x_test, shelf_y_test)
print('Variance score: %.2f'% var)
print "__________________________________________________________________________________________________________\n"

#plot
# pred shape is (3000, )
# shelf_y_test shape is (3000, )

#plt.scatter(pred, shelf_y_test, color='blue')
#plt.ylabel('shelflife test data')
#plt.xlabel('linear regression predicted data')
#plt.title('Scatter Plot')
#plt.axis([-1250,1000,-100,3550])
#plt.show()


