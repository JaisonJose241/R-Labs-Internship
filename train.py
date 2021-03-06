'''

startDate:  05 - 06 -2020
Project  :  Gesture Recognition
Part     :  Training model, CNN architecture
Libraries:  Keras (tensorlow == 2.1)
Team ID  :  4

''' 

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
################################################################ Comment for no GPU-CUDA #################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#########################################################################################################################
# Step 1 - Building the CNN

# Initializing the CNN
model = Sequential()

# First convolution layer and pooling
model.add(Convolution2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# Second convolution layer and pooling
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# third convolution layer and pooling
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# Flattening the layers
model.add(Flatten())
# Adding a fully connected layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=6, activation='softmax')) 

# Compiling the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])           # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model

train_datagen = ImageDataGenerator( rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/TRAIN',
                                                 target_size=(128, 128),
                                                 batch_size=50,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/TEST',
                                            target_size=(128, 128),
                                            batch_size=50,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
model.fit_generator(
        training_set,
        steps_per_epoch=50, 
        epochs=150,
        validation_data=test_set,
        validation_steps=10)


# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

