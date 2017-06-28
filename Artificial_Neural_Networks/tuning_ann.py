# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("modules have been imported")

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print("dataset has been read")
                
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# create dummy variables for geography
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Tuning the ANN
print("Tuning the ANN")

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
# KerasClassifier is used to enable combination of keras and scikit-learn
from keras.wrappers.scikit_learn import KerasClassifier
print('Imported the Keras libraries and classes')

# create keras classifier
print("create keras classifier")
def build_classifier(optimizer, units, dropout_rate):
  classifier = Sequential()
  classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
  classifier.add(Dropout(rate = dropout_rate))
  classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = 'relu'))
  classifier.add(Dropout(rate = dropout_rate))
  classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
  classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = { 'batch_size': [32],
               'epochs': [500],
               'optimizer': ['adam'],
               'units': [6, 15],
               'dropout_rate': [0.1, 0.2]
            }

print('starting grid search')
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('best parameters :')
print(best_parameters)
print('best accuracy :')
print(best_accuracy)

# mean: 0.858375
# variance: 0.0102018687014

#best parameters :
# {'optimizer': 'adam', 'epochs': 500, 'dropout_rate': 0.1, 'units': 15, 'batch_size': 32}
#best accuracy :
#0.8495