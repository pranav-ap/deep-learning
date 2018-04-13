import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
                
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# change word to number
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# create dummy variables for the geography column - change one column into multiple columns 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# prevent dummy variable trap
X = X[:, 1:] 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.2, 
    random_state = 0
    )

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# test data must undergo the same transformation for feature scaling
X_test = sc.transform(X_test)

## Tuning the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

# KerasClassifier is used to enable combination of keras and scikit-learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# create keras classifier
def build_classifier(optimizer, units, dropout_rate):
  classifier = Sequential()
  
  classifier.add(Dense(
      input_dim = 11,
      units = units, 
      kernel_initializer = 'random_uniform', 
      activation = 'relu'
      )
  )
  
  classifier.add(Dropout(rate = dropout_rate))
    
  classifier.add(Dense(
      units = units, 
      kernel_initializer = 'random_uniform', 
      activation = 'relu'
      )
  )
  
  classifier.add(Dropout(rate = dropout_rate))
  
  classifier.add(Dense(
      units = 1, 
      kernel_initializer = 'random_uniform', 
      activation = 'sigmoid'
      )
  )
  
  classifier.compile(
      optimizer = optimizer, 
      loss = 'binary_crossentropy', 
      metrics = ['accuracy']
  )
  
  # use the following line only to load weights and use the ann later
  # classifier.load_weights('ann_best_weights.hdf5')
  
  return classifier

# checkpoint
filepath = 'ann_best_weights.hdf5'

model_checkpoint = ModelCheckpoint(
    filepath, 
    monitor = 'val_acc', 
    verbose = 1, 
    save_best_only = True, 
    save_weights_only = True, 
    mode = 'max',
    period = 1
    )

callbacks_list = [model_checkpoint]

classifier = KerasClassifier(
    build_fn = build_classifier
    )

parameters = {
    'batch_size': [16, 32],
    'epochs': [500],
    'optimizer': ['adam'],
    'units': [6, 15],
    'dropout_rate': [0.1, 0.2]
    }

grid_search = GridSearchCV(
    estimator = classifier, 
    param_grid = parameters, 
    scoring = 'accuracy', 
    fit_params = {
        'callbacks': callbacks_list,
        'validation_data': (X_test, y_test)
        },
    cv = 10 
    )

grid_search_results = grid_search.fit(
    X_train, 
    y_train
    )

best_parameters = grid_search_results.best_params_
best_score = grid_search_results.best_score_
best_estimator = grid_search_results.best_estimator_

print('best parameters :')
print(best_parameters)

print('best score :')
print(best_score)

print('best estimator :')
print(best_estimator)

# best parameters : {'optimizer': 'adam', 'batch_size': 16, 'epochs': 500, 'dropout_rate': 0.1, 'units': 15}
# best accuracy : 0.854
