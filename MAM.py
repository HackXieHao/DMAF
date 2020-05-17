from keras import models
from keras import layers
from keras import Input, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping

from keras.engine.topology import Layer
import numpy as np
import json

from image_features_extract import MyImageDataExtractor

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


# 读取数据
target_file = 'desc.json'
fr = open(target_file, 'r', encoding='utf8')
desc_dict = json.load(fr)
desc = []
ids = []
label = []
for key, value in desc_dict.items():
    desc.append(value)
    label_and_id = key.split('_')
    label.append(int(label_and_id[0]))
    ids.append(label_and_id[1])

max_len = 100
max_words = 10000

# 构建单词索引
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(desc)
word_index = tokenizer.word_index

# 将文本转化为单词索引
sequences = tokenizer.texts_to_sequences(desc)

data = pad_sequences(sequences, maxlen = max_len)

label = np.asarray(label)

# 文本训练集验证集划分
train_num = 4000
val_num = 500
test_num = 500
train_text_features = data[:train_num]
y_train = label[:train_num]
val_text_features = data[train_num : train_num + val_num]
y_val = label[train_num : train_num + val_num]
test_text_features = data[train_num + val_num : train_num + val_num + test_num]
y_test = label[train_num + val_num : train_num + val_num + test_num]

# 构建模型
glove_dir = 'pre_data/glove.6B.100d.txt'

embedding_index = {}
f = open(glove_dir, 'r', encoding='utf8')
for line in f:
    values = line.split(' ')
    word = values[0]
    vector = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = vector
f.close()

# 构建一个可以加载到Embedding层的嵌入矩阵
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

text_input = Input(shape=(None,), dtype='int32')
embedded_text = layers.Embedding(max_words, embedding_dim, input_length = max_len)(text_input)

encoded_text = layers.LSTM(256, dropout = 0.5, recurrent_dropout = 0.5, return_sequences=True)(embedded_text)

att_dense = layers.TimeDistributed(layers.Dense(256))(encoded_text)
att_text = AttentionLayer()(att_dense)


# 图片

train_ids = ids[:train_num]
val_ids = ids[train_num : train_num + val_num]

file_path = "all_image_data"
# 图片特征抽取
train_image_features = MyImageDataExtractor(file_path, target_size=(224,224), ids=train_ids)
val_image_features = MyImageDataExtractor(file_path, target_size=(224,224), ids=val_ids)


# 构建模型
image_input = Input(shape=(224, 224, 3))
vgg_19 = VGG19(weights = 'imagenet', include_top = False, input_shape = image_input)


# 加权
reshape = layers.Reshape((196, 512))(vgg_19.get_layer('block5_conv4').output)
att_image = AttentionLayer()(reshape)

concatenated = layers.concatenate([att_text, att_image], axis = -1)
dense_1 = layers.Dense(1024, activation='relu')(concatenated)
drop_1 = layers.Dropout(0.5)(dense_1)
dense_2 = layers.Dense(512, activation='relu')(drop_1)
drop_2 = layers.Dropout(0.5)(dense_2)
dense_3 = layers.Dense(256, activation='relu')(drop_2)
output_layer = layers.Dense(1, activation='sigmoid')(dense_3)

model = Model(inputs = [text_input, image_input], outputs = output_layer)

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.fit([train_text_features, train_image_features], y_train, epochs=50, batch_size=64, 
    validation_data=([val_text_features, val_image_features], y_val))
