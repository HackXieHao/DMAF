import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19

from keras import models
from keras import layers

'''
    单视觉模型，使用VGG19提取图片特征
'''

train_dir = 'data/train'
val_dir = 'data/val'

data_gen = ImageDataGenerator(rescale = 1./255)
batch_size = 50

vgg_19 = VGG19(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

# 观察vgg_19的最终输出shape   (7, 7, 512)
# vgg_19.summary()

# 使用vgg_19进行特征抽取
def extract_features(dir, sample_count):
    features = np.zeros(shape = (sample_count, 7, 7, 512))
    labels = np.zeros(shape = (sample_count))

    generator = data_gen.flow_from_directory(
        dir,
        target_size = (224, 224),
        batch_size = batch_size,
        class_mode = 'binary'
    )

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg_19.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 4000)
val_features, val_labels = extract_features(val_dir, 500)

# flatten()
train_features = train_features.reshape((4000, 7 * 7 * 512))
val_features = val_features.reshape((500, 7 * 7 * 512))

model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_dim = 7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.fit(train_features, train_labels, epochs=50, batch_size=50, validation_data=(val_features, val_labels))