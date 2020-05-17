from keras import models
from keras import layers
from keras import Input, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import optimizers

from keras.callbacks import EarlyStopping

import numpy as np
import json

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

train_num = 4000
val_num = 500
test_num = 500
x_train = data[:train_num]
y_train = label[:train_num]
x_val = data[train_num : train_num + val_num]
y_val = label[train_num : train_num + val_num]
x_test = data[train_num + val_num : train_num + val_num + test_num]
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

encoded_text = layers.LSTM(256, dropout = 0.5, recurrent_dropout = 0.5)(embedded_text)

# 当不使用预训练的网络时，加入参数过大的正则项会导致模型过于简单，从而使得模型的训练效果不佳
# kernel_regularizer = regularizers.l2(0.01)
dense_1 = layers.Dense(1024, activation='relu', kernel_regularizer = regularizers.l2(0.01))(encoded_text)
dense_2 = layers.Dense(512, activation='relu', kernel_regularizer = regularizers.l2(0.01))(dense_1)
dense_3 = layers.Dense(256, activation='relu', kernel_regularizer = regularizers.l2(0.01))(dense_2)
output_layer = layers.Dense(1, activation='sigmoid')(dense_3)

model = Model(inputs = text_input, outputs = output_layer)

model.summary()

# 问题：收敛的很慢  最高只能到0.75左右
model.layers[1].set_weights([embedding_matrix])
model.layers[1].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 添加EarlyStopping
my_callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')]

model.fit(x_train, y_train, epochs = 100, batch_size = 128, callbacks=my_callbacks, validation_data=(x_val, y_val))
# model.fit(x_train, y_train, epochs = 100, batch_size = 128, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_loss:', test_loss)
print('test_acc:', test_acc) 