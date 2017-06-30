# Convolutional Neural Network

# we will not tell what features to look for, only no of features

# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# no of filters/ feature detector/ feature maps - 32
# its dimensions within (rows, cols)
# input_shape - to transform all images to standard format (width, height, channel)
# activation function used for more non linearity
classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3 ,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3 ,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3 ,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dropout(0.6))
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(0.3))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

print('\n compiling the CNN');
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print('\n Fitting the CNN to dataset')
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/input/dataset/training_set',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/input/dataset/test_set',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 100,
                         validation_data = test_set,
                         validation_steps = 2000)

# Make single predictions
import numpy as np
from keras.preprocessing import image
print('\n Make single predictions: /input/dataset/test_set/cats/cat.4001.jpg')
test_image = image.load_img('/input/dataset/test_set/cats/cat.4001.jpg', target_size = (256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
  print('It is a dog')
else:
  print('It is a cat')


# loss: 0.0634 - acc: 0.9778 - val_loss: 0.2141 - val_acc: 0.9346

