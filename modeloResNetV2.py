from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
from keras.applications import ResNet50

width_shape = 224
height_shape = 224
num_classes = 3
epochs = 20
batch_size = 35

train_data_dir = 'D:/Dev/Python/DispositivosMoviles/images/train'
validation_data_dir = 'D:/Dev/Python/DispositivosMoviles/images/validation'

train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=preprocess_input)

valid_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    #save_to_dir='',
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    #save_to_dir='',
    class_mode='categorical')

nb_train_samples = 6400
nb_validation_samples = 2000

# Creación de modelo
model = Sequential()

inputShape = (height_shape, width_shape, 3)
model.add(Conv2D(32, (3, 3), input_shape=inputShape))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax', name='output'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model_history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_steps=nb_validation_samples // batch_size)

#Resnet50
image_input = Input(shape=(width_shape, height_shape, 3))

m_Resnet50 = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet')

m_Resnet50.summary()

last_layer = m_Resnet50.layers[-1].output

x = Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_model = Model(image_input, out)
custom_model.summary()

# freeze all the layers except the dense layers
for layer in custom_model.layers[:-6]:
    layer.trainable = False

custom_model.summary()

custom_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# train_generator = train_generator.reset()
#
# steps_per_epoch = len(train_generator)

model_history = custom_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=epochs,
    validation_steps=nb_validation_samples//batch_size)

custom_model.save("resnet508.h5")



