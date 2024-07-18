# -*- coding: utf-8 -*-
"""Decision_Tree_RF.ipynb
https://colab.research.google.com/github/a-nagar/cs4375/blob/main/Decision_Tree_RF.ipynb

## Decision Tree Classification Model

We will use the built in Pima Indians Diabetes dataset. It is available as part of the SKlearn datasets. We have made it easily available for you.
"""

import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("https://an-utd-python.s3.us-west-1.amazonaws.com/pima-indians-diabetes.csv", header=None, names=col_names)

pima.head()

pima.shape

pima['label'].value_counts()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
predictions = clf.predict(X_test)
predicted_probas = clf.predict_proba(X_test)

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, predictions)) # predictions contain predicted values (derived from probability with 0.5 threshold)
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))

import scikitplot as skplt
import matplotlib.pyplot as plt


skplt.metrics.plot_confusion_matrix(y_test, predictions)
skplt.metrics.plot_roc(y_test, predicted_probas)
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()


X.columns

classes = y.unique()

import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names = feature_cols,class_names=['0','1'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph

"""# Excercise 1

Construct the following models on the same dataset:
- Bagging
- Random Forest
- Adaboost

Compare their performance and write a short paragraph on which one is the best. You are free to change the hyperparameters.

### Application on Cuisines Dataset
"""
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
bag = BaggingClassifier()
bag = bag.fit(X_train,y_train)
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train,y_train)
rfpredictions = rf.predict(X_test)
rfpredicted_probas = rf.predict_proba(X_test)
bagpredictions = bag.predict(X_test)
bagpredicted_probas = bag.predict_proba(X_test)
print(classification_report(y_test, bagpredictions)) # predictions contain predicted values (derived from probability with 0.5 threshold)
print('Bag Predicted labels: ', bagpredictions)
print('Bag Accuracy: ', accuracy_score(y_test, bagpredictions))
print(classification_report(y_test, rfpredictions)) # predictions contain predicted values (derived from probability with 0.5 threshold)
print('RF Predicted labels: ', rfpredictions)
print('RF Accuracy: ', accuracy_score(y_test, rfpredictions))
ada = AdaBoostClassifier(n_estimators=200)
ada = ada.fit(X_train, y_train)

# Predictions for AdaBoost
adapredictions = ada.predict(X_test)
adapredicted_probas = ada.predict_proba(X_test)
print(classification_report(y_test, adapredictions))
print('AdaBoost Predicted labels: ', adapredictions)
print('AdaBoost Accuracy: ', accuracy_score(y_test, adapredictions))

import pandas as pd
cuisines_df = pd.read_csv("https://an-utd-python.s3.us-west-1.amazonaws.com/cuisines.csv")
cuisines_df.head()

cuisines_label_df = cuisines_df['cuisine']
cuisines_label_df.head()

cuisines_label_df.value_counts()

type(cuisines_label_df)

cuisine_labels_distinct = cuisines_label_df.unique()

cuisine_labels_distinct

cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
cuisines_feature_df.head()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)

from matplotlib.pyplot import figure

figure(figsize=(12, 12), dpi=80)
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth = 5)
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=cuisines_feature_df.columns,
                     class_names=cuisine_labels_distinct,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph

from sklearn.metrics import accuracy_score, classification_report
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))

skplt.metrics.plot_confusion_matrix(y_test, y_pred)
skplt.metrics.plot_roc(y_test, y_probs)
skplt.metrics.plot_precision_recall_curve(y_test, y_probs)
plt.show()

rf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators = 100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))


"""# Exercise 2

The accuracy for this dataset is quite low. Can you try any other method that increases the accuracy. You can try either Random Forest or Adaboost. What do you notice?

# Parameter Grid Builder for Parameter Tuning
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

dt_pipe = Pipeline([('mms', MinMaxScaler()),
                     ('dt', DecisionTreeClassifier())])
params = [{'dt__max_depth': [3, 5, 7, 9],
         'dt__min_samples_leaf': [2, 3, 5]}]

gs_dt = GridSearchCV(dt_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5)
gs_dt.fit(cuisines_feature_df, cuisines_label_df)
print(gs_dt.best_params_)
# find best model score
print(gs_dt.score(cuisines_feature_df, cuisines_label_df))

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

rf = RandomForestClassifier()

params = {'max_depth': [5, 7, 9],
          'n_estimators': [50, 100, 200],
          'max_features': ['sqrt', 'log2']
          }

grid = GridSearchCV(rf, params, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(cuisines_feature_df, cuisines_label_df)

print(grid.best_params_)
# find best model score
print(grid.score(cuisines_feature_df, cuisines_label_df))
ada = AdaBoostClassifier()
params = {'n_estimators': [50, 100, 200],
          'learning_rate': [0.01, 0.1, 1.0]
         }

grid = GridSearchCV(ada, params, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(cuisines_feature_df, cuisines_label_df)

print(grid.best_params_)
print(grid.score(cuisines_feature_df, cuisines_label_df))

"""# Exercise 3

Try other combination of hyperparameters for Random Forest and AdaBoost models and check how good of an accuracy you can obtain.

# Regression Trees
"""

cars = pd.read_csv("https://an-vistra.s3.us-west-1.amazonaws.com/data/auto-mpg.csv")

cars.head()

from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

X = cars[['cyl', 'displ', 'hp', 'weight', 'accel', 'origin', 'size']]
y = cars['mpg']
le = preprocessing.LabelEncoder()
X['origin'] = le.fit_transform(X['origin'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0)

# fit the regressor with X and Y data
regressor.fit(X_train, y_train)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np
# finish
predictions = regressor.predict(X_test)

print("R2 square = ", r2_score(y_test, predictions))
print("MSE = ", mean_squared_error(y_test, predictions))
print("MAE = ", mean_absolute_error(y_test, predictions))
print("Explained variance score = ", explained_variance_score(y_test, predictions))

import graphviz
dot_data = tree.export_graphviz(regressor, out_file=None,
                     feature_names=X.columns,
                     class_names=y.unique(),
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph

from sklearn.ensemble import AdaBoostRegressor
ada_regressor = AdaBoostRegressor(n_estimators=200, random_state=0)
ada_regressor.fit(X_train, y_train)
ada_predictions = ada_regressor.predict(X_test)

print("AdaBoost R2 square = ", r2_score(y_test, ada_predictions))
print("AdaBoost MSE = ", mean_squared_error(y_test, ada_predictions))
print("AdaBoost MAE = ", mean_absolute_error(y_test, ada_predictions))
print("AdaBoost Explained variance score = ", explained_variance_score(y_test, ada_predictions))

dot_data = tree.export_graphviz(ada_regressor.estimators_[0], out_file=None,
                     feature_names=X.columns,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph