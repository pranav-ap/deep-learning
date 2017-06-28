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

print('dataset has been label encoded')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print('train test split done')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('feature scaling is done')

### Evaluating the ANN 

## Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
print('Imported the Keras libraries and classes')

print("Evaluating the ANN ")
# KerasClassifier is used to enable combination of keras and scikit-learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score 

# create keras classifier
print("create keras classifier")
def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
  classifier.add(Dropout(rate = 0.1))
  classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
  classifier.add(Dropout(rate = 0.1))
  classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
  classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  return classifier

# create a scikit-wrapped classifier 
print("create a scikit-wrapped classifier")
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 500)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('evaluation of ann:')
mean = accuracies.mean()
print('mean: ')
print(mean)
variance = accuracies.std()
print('variance: ')
print(variance)
