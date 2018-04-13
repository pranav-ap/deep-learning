# DATA PREPROCESSING
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range=20,
                                   featurewise_center=True,
                                   featurewise_std_normalization=True
                                   )

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/dataset/train',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary',
                                                 shuffle = True,
                                                 seed = 0
                                                 )

validation_set = validation_datagen.flow_from_directory('/dataset/valid',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            shuffle = True,
                                            seed = 0
                                            )

# SETUP CALLBACKS
from keras.callbacks import ModelCheckpoint
# checkpoint
filepath = 'cnn_best_weights.hdf5'

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

# BUILD THE CLASSIFIER
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_classifier(no_of_feature_detectors, feature_detector_dims, dropout_rate, optimizer):
  classifier = Sequential()

  # First Convolutional layer and Max pooling
  classifier.add(Conv2D(
      no_of_feature_detectors, # no of filters/ feature detectors
      feature_detector_dims, # its dimensions within (rows, cols)
      input_shape = (256, 256, 3), # transform all images to standard format (width, height, channel)
      activation = 'relu' # activation function used for more non linearity
      )
  )

  classifier.add(MaxPooling2D(pool_size = (2, 2)))

  classifier.add(Conv2D(
    no_of_feature_detectors * 2, 
    feature_detector_dims, 
    activation = 'relu'
    )
  )
  
  classifier.add(MaxPooling2D(pool_size = (2, 2)))

  classifier.add(Conv2D(
    no_of_feature_detectors * 2, 
    feature_detector_dims, 
    activation = 'relu'
    )
  )
  
  classifier.add(MaxPooling2D(pool_size = (2, 2)))

  classifier.add(Conv2D(
    no_of_feature_detectors * 4, 
    feature_detector_dims, 
    activation = 'relu'
    )
  )
  
  classifier.add(MaxPooling2D(pool_size = (2, 2)))

  # Flattening
  classifier.add(Flatten())

  # Full connection
  classifier.add(Dropout(dropout_rate))
  classifier.add(Dense(units = no_of_feature_detectors * 4, activation = 'relu'))

  classifier.add(Dropout(dropout_rate * 2))
  classifier.add(Dense(units = 1, activation = 'sigmoid'))

  classifier.compile(
      optimizer = optimizer, 
      loss = 'binary_crossentropy', 
      metrics = ['accuracy']
      )

  return classifier

# SETUP KERAS CLASSIFIER
# KerasClassifier is used to enable combination of keras and scikit-learn
from keras.wrappers.scikit_learn import KerasClassifier

classifier = KerasClassifier(
    build_fn = build_classifier
    )

# PERFORM GRID SEARCH
from sklearn.model_selection import GridSearchCV

parameters = {
    'batch_size': [16, 32],
    'epochs': [100],
    'steps_per_epoch': [8000],
    
    'optimizer': ['adam'],
    'dropout_rate': [0.2, 0.3],
    'no_of_feature_detectors': [32, 64],
    'feature_detector_dims': [(3, 3)]
    }

grid_search = GridSearchCV(
    estimator = classifier, 
    param_grid = parameters, 
    scoring = 'accuracy', 
    fit_params = {
        'callbacks': callbacks_list,
        'validation_data': validation_set,
        'validation_steps': 2000
        },
    cv = 10 
    )

grid_search_results = grid_search.fit(
  training_set
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

