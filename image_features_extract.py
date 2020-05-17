from keras.applications.vgg19 import VGG19
from keras import models
from keras import layers
from keras.preprocessing import image

import numpy as np
import os
import json

import matplotlib.pyplot as plt

class MyImageDataExtractor:
        
    def __init__(self, file_path, target_size = None, ids = None, _max_example = 0, batch_size = 20):
        self.index= 0
        self.batch_size = batch_size
        self.num_of_examples = _max_example
        self.target_size = target_size
        self.file_path = file_path
        self.ids = ids
        self.load_images_labels()

    def load_images_labels(self):
        images_dict = {}
        # images = []
        for image_id in self.ids:
            image_name = image_id + '.jpg'
            img = image.load_img(os.path.join(self.file_path, image_name), target_size = self.target_size)
            # plt.imshow(img)
            # plt.show()
            img_tensor = image.img_to_array(img)
            img_tensor /= 255.

            images_dict[image_id] = img_tensor
        
        self.images_dict = images_dict

    def generate(self):
        image_features = []
        for id_index in self.ids:
            image_feature = self.images_dict.get(id_index)
            image_features.append(image_feature)
        image_features = np.array(image_features)
        return image_features