import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19

from keras import models
from keras import layers

from keras import Input

'''
    数据增强的单视觉模型 使用VGG19
'''

train_dir = 'data/train'
val_dir = 'data/val'

batch_size = 50

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode ='nearest'
)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size = batch_size,
    target_size = (224, 224),
    class_mode = 'binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    batch_size = batch_size,
    target_size = (224, 224),
    class_mode = 'binary'
)

# 构建模型
image_input = Input(shape=(224, 224, 3))
vgg_19 = VGG19(weights = 'imagenet', include_top = False, input_tensor = image_input)

flatten = layers.Flatten()(vgg_19.output)
dense_1 = layers.Dense(1024, activation='relu')(flatten)
drop_1 = layers.Dropout(0.5)(dense_1)
dense_2 = layers.Dense(512, activation='relu')(drop_1)
drop_2 = layers.Dropout(0.5)(dense_2)
dense_3 = layers.Dense(256, activation='relu')(drop_2)
drop_3 = layers.Dropout(0.5)(dense_3)

output = layers.Dense(1, activation='sigmoid')(drop_3)

model = models.Model(inputs = image_input, outputs = output)

# 冻结所有层
vgg_19.trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.fit_generator(
    train_generator,
    steps_per_epoch = 80,
    epochs = 10,
    validation_data = val_generator,
    validation_steps = 10
)
