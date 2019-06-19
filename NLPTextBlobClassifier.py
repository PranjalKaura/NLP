# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:04:18 2019

@author: Pranjal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('2000_Data.csv')
X = dataset.iloc[:, 0:2].values #Esnurinig X is a matrix
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting the RandomForest Model to the dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting test Set results
Y_pred = classifier.predict(X_test)

# Analysing the results. Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

#print(cm)
print("Randomforest")
Acc = (cm[0][0] + cm[1][1])/6250
print("Accuracy ", Acc)
Prec = cm[1][1]/(cm[1][1] + cm[0][1])
print("Prec ", Prec )
Rec = cm[1][1]/(cm[1][1] +  cm[1][0])
print("Recall ", Rec )
print("F1 Score ", 2*Prec*Rec/(Prec + Rec ))

# Graphic analysis. Visualizing the test set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, Y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.04), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.04))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('RandomForest (Training set)')
#plt.xlabel('Polarity')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test,Y_pred)
#print(cm)
print(" ")
print("Bayes")
Acc = (cm[0][0] + cm[1][1])/6250
print("Accuracy ", Acc)
Prec = cm[1][1]/(cm[1][1] + cm[0][1])
print("Prec ", Prec )
Rec = cm[1][1]/(cm[1][1] +  cm[1][0])
print("Recall ", Rec )
print("F1 Score ", 2*Prec*Rec/(Prec + Rec ))




from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test,Y_pred)
#print(cm)
print(" ")
print("Kernel SVM")
Acc = (cm[0][0] + cm[1][1])/6250
print("Accuracy ", Acc)
Prec = cm[1][1]/(cm[1][1] + cm[0][1])
print("Prec ", Prec )
Rec = cm[1][1]/(cm[1][1] +  cm[1][0])
print("Recall ", Rec )
print("F1 Score ", 2*Prec*Rec/(Prec + Rec ))