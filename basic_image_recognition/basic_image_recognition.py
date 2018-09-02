# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:09:02 2018

@author: Yesman
"""

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
print(len(digits.data))

clf = svm.SVC(gamma=0.001, C=100)

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print('Prediction:',clf.predict(digits.data[-2]))

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

