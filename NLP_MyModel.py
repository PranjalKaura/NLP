# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:08:51 2019

@author: Pranjal
"""
import glob
import os
import numpy as np
import pickle
dataset = []
file_list = glob.glob(os.path.join(os.getcwd(), "TestCaseBig/train/posTest", "*.txt"))
for file_path in file_list:
    with open(file_path, encoding="utf8") as f_input:
        text = f_input.read()
        dataset.append([text, 1])
   

file_list = glob.glob(os.path.join(os.getcwd(), "TestCaseBig/train/negTest", "*.txt"))
for file_path in file_list:
    with open(file_path, encoding="utf8") as f_input:
        text = f_input.read()
        dataset.append([text, 0])
 
  
      
print(len(dataset)) 

import re
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#import nltk
#nltk.download('stopwords') #List of all the words (Articles, Prepositions etc) that have to be removed during the txet cleaning phase

#Creating your own coprus file
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]'," ",  dataset[i][0])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english")) or word == "nor" or word == "no" or word == "not"] # Set method added for faster check as converted to set
    review = " ".join(review)
    corpus.append(review)


#Saving the Corpus File
#np.save("CorpusFile", corpus)
 
#Loading the Corpus FIle
#corpus = np.load("CorpusFile.npy") 
   
print("Stemmed and saved")

from sklearn.feature_extraction.text import CountVectorizer

#creating your own model
#CV = CountVectorizer(max_features = 3000)
#X = CV.fit_transform(corpus).toarray()

#saving the CV object for future use    
#CV = pickle.dump(CV.vocabulary_,open("CV_vocabulary.pkl","wb")) 

#Load the CV    
CV = CountVectorizer(vocabulary=pickle.load(open("CV_vocabulary.pkl", "rb"))) 
X = CV.fit_transform(corpus).toarray() 

#just a check to ensure correct library is being imported
#shared_items = {k: CV.vocabulary_[k] for k in CV.vocabulary_ if k in CV2.vocabulary and CV.vocabulary_[k] == CV2.vocabulary[k]}
#print (len(shared_items) )  



Y = []
for i in range(0, len(corpus)):
    Y.append(dataset[i][1])
    

splitRatio = 0.2
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = splitRatio,random_state = 0)

#Bayes classification

# Fitting the NaiveBayes Model to the dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

filename = 'finalized_model_Bayes.sav'

#SavingtheModel
pickle.dump(classifier, open(filename, 'wb'))

#loadingtheModel
classifier = pickle.load(open(filename, 'rb'))

# Predicting test Set results
Y_pred = classifier.predict(X_test)

# Analysing the results. Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(" ")
print("Bayes")
Acc = (cm[0][0] + cm[1][1])/(len(corpus)*splitRatio)
print("Accuracy ", Acc)
Prec = cm[1][1]/(cm[1][1] + cm[0][1])
print("Prec ", Prec )
Rec = cm[1][1]/(cm[1][1] +  cm[1][0])
print("Recall ", Rec )
print("F1 Score ", 2*Prec*Rec/(Prec + Rec ))





# Random Forest Classification

# Fitting the RandomForest Model to the dataset

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, n_jobs = -1)
classifier.fit(X_train, Y_train)


filename = 'finalized_model_RandomForest2.sav'
pickle.dump(classifier, open(filename, 'wb'))
#classifier = pickle.load(open(filename, 'rb'))

# Predicting test Set results
Y_pred = classifier.predict(X_test)

# Analysing the results. Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(" ")
print("Random Forest")
Acc = (cm[0][0] + cm[1][1])/(len(corpus)*splitRatio)
print("Accuracy ", Acc)
Prec = cm[1][1]/(cm[1][1] + cm[0][1])
print("Prec ", Prec )
Rec = cm[1][1]/(cm[1][1] +  cm[1][0])
print("Recall ", Rec )
print("F1 Score ", 2*Prec*Rec/(Prec + Rec ))




 

    
    
    