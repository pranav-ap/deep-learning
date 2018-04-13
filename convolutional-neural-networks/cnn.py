from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

classifier = Sequential()

# First Convolutional layer and Max pooling
classifier.add(Conv2D(
    32, # no of filters/ feature detectors
    (3, 3), # its dimensions within (rows, cols)
    input_shape = (256, 256, 3), # transform all images to standard format (width, height, channel)
    activation = 'relu' # activation function used for more non linearity
    )
)

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3 ,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3 ,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3 ,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dropout(0.6))
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(0.3))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy']
    )

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

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set_generator = train_datagen.flow_from_directory('./dataset/training_set',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary',
                                                 shuffle = True,
                                                 seed = 0
                                                 )

test_set_generator = test_datagen.flow_from_directory('./dataset/test_set',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            shuffle = True,
                                            seed = 0
                                            )

classifier.fit_generator(
    training_set_generator,
    steps_per_epoch = 8000,
    epochs = 100,
    validation_data = test_set_generator,
    validation_steps = 2000
    )

# Make single predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/input/dataset/test_set/cats/cat.4001.jpg', target_size = (256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

if result[0][0] == 1:
  print('It is a dog')
else:
  print('It is a cat')
