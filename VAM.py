import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19

from keras import models
from keras import layers

from keras import Input

from keras.engine.topology import Layer
from keras import backend as K
'''
    添加注意力机制的单视觉模型
'''

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


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

# 加权
reshape = layers.Reshape((196, 512))(vgg_19.get_layer('block5_conv4').output)
att = AttentionLayer()(reshape)
# att_w = layers.Dense(512, activation='softmax')(att)
# a = layers.multiply([reshape, att_w]) # (49,512)
# def sum_w(args):
#     return K.sum(args, axis=-1) # axis值为-1或2，加权后求和得到(49,)的数据
# flatten_w = layers.Lambda(sum_w)(a)
# dense_1 = layers.Dense(128, activation='tanh')(flatten_w)

# flatten = layers.Flatten()(vgg_19.output)
dense_1 = layers.Dense(1024, activation='relu')(att)
drop_1 = layers.Dropout(0.5)(dense_1)
dense_2 = layers.Dense(512, activation='relu')(drop_1)
drop_2 = layers.Dropout(0.5)(dense_2)
dense_3 = layers.Dense(256, activation='relu')(drop_2)
drop_3 = layers.Dropout(0.5)(dense_3)

output = layers.Dense(1, activation='sigmoid')(drop_3)

model = models.Model(inputs = image_input, outputs = output)

model.summary()

# # 冻结所有层
# vgg_19.trainable = False

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# model.fit_generator(
#     train_generator,
#     steps_per_epoch = 80,
#     epochs = 10,
#     validation_data = val_generator,
#     validation_steps = 10
# )

