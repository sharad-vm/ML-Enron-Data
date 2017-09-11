#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##linear
#########################################################
from sklearn import svm
from sklearn.metrics import accuracy_score

clflin = svm.SVC(kernel='linear')

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clflin.fit(features_train, labels_train)
print ("training time with SVM's linear kernel", time()-t0)

t1 = time()
predlin = clflin.predict(features_test)
print ("prediction time with SVM's linear kernel", time()-t1)

print(accuracy_score(labels_test, predlin))

##rbf - radial basis function
#########################################################
clfrbf = svm.SVC(kernel='rbf', C=10000)

t0 = time()
clfrbf.fit(features_train, labels_train)
print ("training time with SVM's rbf kernel", time()-t0)

t1 = time()
predrbf = clfrbf.predict(features_test)
print ("prediction time with SVM's rbf kernel", time()-t1)

print(accuracy_score(labels_test, predrbf))

print(len(predrbf[predrbf == 1]))


