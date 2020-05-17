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

from image_features_extract import MyImageDataExtractor

# 读取数据
target_file = 'desc.json'
fr = open(target_file, 'r', encoding='utf8')
desc_dict = json.load(fr)
desc = []
ids = []
label = []
for key, value in desc_dict.items():
    label_and_id = key.split('_')
    label.append(int(label_and_id[0]))
    ids.append(label_and_id[1])
    desc.append(value)
print(ids[:5])
# print(desc)
# print(max(desc))

file_path = 'all_image_data'
gen = MyImageDataExtractor(file_path, target_size=(224,224), ids=ids[:10])
image_features = gen.generate()
print(image_features)
print(image_features.shape)