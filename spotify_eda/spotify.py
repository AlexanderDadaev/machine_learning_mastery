# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 17:09:37 2018

@author: Yesman
"""

import pandas as pd #DataFrame, Series
import numpy as np #Scientific computing package - Array

from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier#, export_graphviz
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns

#import export_graphviz
#import pydotplus
import io
from scipy import misc

#%matplotlib inline

#Spotify Song Attributes EDA
#Import DataSet
#Explanatory data analysis (EDA) to visualize data and observe structure
#Train a classifier (Desicion Tree)
#Predict target using the trained classifier

#import the spotify music dataset; the aim is to predict whether the user likes the song or not
data = pd.read_csv('***/Spotify EDA/data.csv')
type(data)
data.describe()
data.head()
data.info()

#train test split our data
train, test = train_test_split(data, test_size=0.15)
print("Training size: {}; Test size: {}".format(len(train), len(test)))
train.shape

#visualize the data
pos_tempo = data[data['target'] == 1]['tempo']
neg_tempo = data[data['target'] == 0]['tempo']

#custom colour palette
red_blue = ['#1985FE', '#EF4836']
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style('white')

fig = plt.figure(figsize=(12,8))
plt.title("Song Tempo Like/Dislike Distribution")
pos_tempo.hist(alpha = 0.7, bins = 30, label='positive')
neg_tempo.hist(alpha = 0.7, bins = 30, label='negative')
plt.legend(loc='upper right')

###
pos_dance = data[data['target'] == 1]['danceability']
neg_dance = data[data['target'] == 0]['danceability']
fig2 = plt.figure(figsize=(15, 15))

#Danceability
ax3 = fig2.add_subplot(331)
ax3.set_xlabel('Danceability')
ax3.set_ylabel('Count')
ax3.set_title('Song Danceability Like Distirbution')
pos_dance.hist(alpha=0.5, bins=30)
ax4 = fig2.add_subplot(331)
neg_dance.hist(alpha=0.5, bins=30)

#Building a model
c = DecisionTreeClassifier(min_samples_split=100)
features = ["danceability","loudness","valence","energy","instrumentalness","acousticness","key","speechiness","duration_ms"]

X_train = train[features]
y_train = train["target"]

X_test = train[features]
y_test = train["target"]

dt = c.fit(X_train, y_train)

#def show_tree(tree, features, path):
#    f = io.StringIO()
#    export_graphviz(tree, out_file=f, feature_names=features)
#    pydotplus.graph_from_dot_data(f.getvalues()).write_png(path)
#    img = misc.imread(path)
#    plt.rcParams["figure.figsize"] = (20, 20)
#    plt.imshow(img)
#
#show_tree(dt, features, "dec_tree_01.png")

#make predictions
y_pred = c.predict(X_test)
print(y_pred)
    
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)*100
print("Accuracy using Decision Tree: ", round(score, 1), "%")    
    
    
    
    
