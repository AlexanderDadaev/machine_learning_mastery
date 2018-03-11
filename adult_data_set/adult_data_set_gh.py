# =============================================================================
# In this data set, the dependent variable is "target." 
# It is a binary classification problem. 
# We need to predict if the salary of a given person is less than or more than 50K. 
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#load the data
train  = pd.DataFrame.from_csv("/MachineLearning/University of California Irvine/Adult Data Set/train.csv")
test = pd.DataFrame.from_csv("/MachineLearning/University of California Irvine/Adult Data Set/test.csv")

original_data = pd.DataFrame.from_csv("/MachineLearning/University of California Irvine/Adult Data Set/train.csv")

#assign unique ID based on row number
id_loc = 0 #location of inserted column
train.insert(id_loc, 'NEW_ID', range(1, 1 + len(train)))
test.insert(id_loc, 'NEW_ID', range(1, 1 + len(test)))

#check the data set
train.info()
print("The train data has",train.shape)
print("The test data has",test.shape)
#Let us have a glimpse of the data set
train.head()

#Now, let's check the missing values (if present) in this data.
nans = train.shape[0] - train.dropna().shape[0]
print("%d rows have missing values in the train data" %nans)

nand = test.shape[0] - test.dropna().shape[0]
print("%d rows have missing values in the test data" %nand)

#We should be more curious to know which columns have missing values.
#only 3 columns have missing values
train.isnull().sum()

#Let's count the number of unique values from character variables.
cat = train.select_dtypes(include=['O'])
cat.apply(pd.Series.nunique)

#Since missing values are found in all 3 character variables, 
#let's impute these missing values with their respective modes.
#Education
train.workclass.value_counts(sort=True)
train.workclass.fillna('Private',inplace=True)

#Occupation
train.occupation.value_counts(sort=True)
train.occupation.fillna('Prof-specialty',inplace=True)

#Native Country
train['native.country'].value_counts(sort=True)
train['native.country'].fillna('United-States',inplace=True)

#Let's check again if there are any missing values left.
train.isnull().sum()

#Now, we'll check the target variable to investigate if this data is imbalanced or not. 
#check proportion of target variable
train.target.value_counts()/train.shape[0]

#We see that 75% of the data set belongs to <=50K class. 
#This means that even if we take a rough guess of target prediction as <=50K, 
#we'll get 75% accuracy. Isn't that amazing? 
#Let's create a cross tab of the target variable with education. 
#With this, we'll try to understand the influence of education on the target variable.
pd.crosstab(train.education, train.target,margins=True)/train.shape[0]

#encode all object type variables
for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))

#calculate the correlation and plot it
sns.heatmap(train.corr(), square=True)
plt.show()

#We see there is a high correlation between Education and Education-Num. 
#Let’s look at these columns

original_data[["education", "education.num"]].head(15)

# =============================================================================
# As you can see these two columns actually represent the same features, 
# but encoded as strings and as numbers. We don’t need the string representation, 
# so we can just delete this column. Note that it is a much better option to delete 
# the Education column as the Education-Num has the important property that the values 
# are ordered: the higher the number, the higher the education that person has. 
# This is a vaulable information a machine learning algorithm can use.
# =============================================================================

del train["education"]
del test["education"]

#So it seems that the data is mostly OK with the exception of Sex and Relationship, 
#which seems to be negatively correlated. Let’s explore that for a bit
original_data[["sex", "relationship"]].head(15)
#Yes. The data looks correlated, because for example Male and Husband are highly correlated values, 
#as well as Female and Wife. 
#There is no easy way to tackle this problem, so let’s carry on.

#Let's check the changes applied to the data set.
#print(train.head())
print(train.target.value_counts())

#Let's create a random forest model and check the model's accuracy.
y = train['target']

X = train.drop('target', axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

#train the RF classifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)
clf.fit(X_train,y_train)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                max_depth=6, max_features='auto', max_leaf_nodes=None,
#                min_impurity_split=1e-07, min_samples_leaf=1,
#                min_samples_split=2, min_weight_fraction_leaf=0.0,
#                n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
#                verbose=0, warm_start=False)
clf.predict(X_test)

#Now, let's make prediction on the test set and check the model's accuracy.
#make prediction and check model's accuracy
prediction_rf = clf.predict(X_test)
acc = accuracy_score(np.array(y_test),prediction_rf)
scores_cv = cross_val_score(clf, X, y, cv=10)
accuracy_cv = np.mean(scores_cv)
print('The accuracy of Random Forest is {}'.format(acc))
print('The cross_val accuracy of Random Forest is {}'.format(accuracy_cv))

#Making Predictions and Measuring their Accuracy using Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
prediction_lr = lr.predict(X_test)
scores_lr = cross_val_score(lr, X, y, cv=10)
accuracy_lr = np.mean(scores_lr)
print('The accuracy of Logistic Regression is {}'.format(accuracy_lr))




