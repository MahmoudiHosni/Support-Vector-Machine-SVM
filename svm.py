#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:30:48 2018

@author: hosni
"""
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
from sklearn.model_selection import train_test_split

data1=[]
with open("/home/hosni/Desktop/Classe1.csv") as file:
    fichier1=csv.reader(file,delimiter=",")
    #next(fichier1)
    for i in fichier1:
        data1.append(i)
 
print(data1)
data2=[]
with open("/home/hosni/Desktop/Classe2.csv") as file:
    fichier2=csv.reader(file,delimiter=",")
    #next(fichier2)
    for i in fichier2:
        data2.append(i)
print(data2)
data1_train,data1_test=train_test_split(data1,test_size=0.33)
data2_train,data2_test=train_test_split(data2,test_size=0.33)
print("data1_train",len(data1_train))
print("data1_test",len(data1_test))
print("data2_train",len(data2_train))
print("data2_test",len(data2_test))
label1=np.ones((335,1))
label2=np.zeros((335,1))
label=np.concatenate((label1,label2),axis=0)
print(label)
train=np.concatenate((data1_train,data2_train),axis=0)
print(train)
test=np.concatenate((data1_test,data2_test),axis=0)
print(test)
clf = svm.SVC(kernel='linear', C=1)
#clf = svm.SVC(kernel='rbf', C=1.1, gamma=1)
#clf = svm.SVC(kernel='poly', coef0=5, degree=2, gamma=1)
#clf = svm.SVC(kernel='sigmoid', coef0=1, gamma=1)
#T1=np.ravel(train)
#T2=np.ravel(label)
clf.fit(train,np.ravel(label))
predicted=clf.predict(test)
print("Predicted =",predicted)
labeltest1=np.ones((165,1))
labeltest2=np.zeros((165,1))
labeltest=np.concatenate((labeltest1,labeltest2),axis=0)
accuracy=accuracy_score(labeltest,predicted)
print("accuracy ",accuracy)
fpr, tpr, thresholds=roc_curve(labeltest, predicted, pos_label=0, sample_weight=None)
print("fpr= ",fpr)
print("tpr= ",tpr)
print("thresholds= ",thresholds)
plt.plot(tpr,fpr)
plt.title("roc curve")
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
print("auc= ",auc(tpr,fpr))
print("Matrice de confusion",confusion_matrix(labeltest, predicted, labels=None, sample_weight=None))


