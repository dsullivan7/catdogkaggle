from skimage.io import imread
from sklearn.naive_bayes import BernoulliNB
import sklearn.linear_model as lm
import os
import numpy as np

train = []
labels = []
print "starting to iterate through files"
for root, dirs, files in os.walk('./train/'):
    for name in files:
        train.append(imread(root+name))
        labels.append(name.split(".")[0])
print "completed file iteration"
rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)
print "starting the training"
rd.fit(train,labels)
print "training completed"

