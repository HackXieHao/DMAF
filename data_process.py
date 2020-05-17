# 目的：要使得图片和文本的训练、验证、测试数据对应起来

import numpy as np
import os
import json

import shutil

'''
    文本数据处理
'''

base_dir = 'original_data/Flickr-m'
neg_desc_filename = 'neg_desc.json'
pos_desc_filename = 'pos_desc.json'

# 读取原始文本数据
def read_original_data(base_dir, file_name):
    text_desc_file = open(os.path.join(base_dir, file_name))
    content = text_desc_file.read()
    content = json.loads(content)
    text_desc_file.close()
    return content


def get_desc_data():
    neg_content_dict = read_original_data(base_dir, neg_desc_filename)
    pos_content_dict = read_original_data(base_dir, pos_desc_filename)

    # print(neg_content_dict)
    # print(type(neg_content_dict)) # dict

    # 合并数据
    desc_content_list = [neg for neg in neg_content_dict.values()] + [pos for pos in pos_content_dict.values()]
    neg_id_list = [neg for neg in neg_content_dict.keys()]
    pos_id_list = [pos for pos in pos_content_dict.keys()]
    desc_id_list = neg_id_list + pos_id_list

    desc_content_array = np.array(desc_content_list)
    desc_id_array = np.array(desc_id_list)

    # print(desc_id)
    return desc_content_array, desc_id_array, neg_id_list, pos_id_list

'''
    图片数据处理
    data
        train
            neg
            pos
        val
            neg
            pos
        test
            neg
            pos
'''

desc_content_array, desc_id_array, neg_id_list, pos_id_list = get_desc_data()
indices = np.arange(len(desc_id_array))
# print(indices)
np.random.shuffle(indices)
# print(indices)
desc_id_array = desc_id_array[indices]
desc_content_array = desc_content_array[indices]

# 图片复制
train_num = 4000
val_num = 500
test_num = 500

def get_desc_data4model():

    # 文本数据
    desc_data = list(desc_content_array)

    # 文本的标签
    desc_label = []
    for data_id in desc_id_array:
        if data_id in neg_id_list:
            desc_label.append(0)
        if data_id in pos_id_list:
            desc_label.append(1)
    

    print(len(desc_data))
    print(len(desc_label))
    # 存在文件中
    target_file = 'desc.json'
    desc_dict = {}
    
    for i, label in enumerate(desc_label):
        key = str(label) + '_' + str(desc_id_array[i])
        desc_dict[key] = desc_data[i]
    print(len(desc_dict))
    json_str = json.dumps(desc_dict, ensure_ascii=False, indent=4)
    with open(target_file, 'w', encoding='utf8') as fw:
        fw.write(json_str)


def get_image_data4model():

    # 用于模型的图片存放的文件夹
    taget_base_dir = 'data'
    os.mkdir(taget_base_dir)

    train_dir = os.path.join(taget_base_dir, 'train')
    os.mkdir(train_dir)
    val_dir = os.path.join(taget_base_dir, 'val')
    os.mkdir(val_dir)
    test_dir = os.path.join(taget_base_dir, 'test')
    os.mkdir(test_dir)


    train_neg_dir = os.path.join(train_dir, 'neg')
    os.mkdir(train_neg_dir)
    train_pos_dir = os.path.join(train_dir, 'pos')
    os.mkdir(train_pos_dir)

    val_neg_dir = os.path.join(val_dir, 'neg')
    os.mkdir(val_neg_dir)
    val_pos_dir = os.path.join(val_dir, 'pos')
    os.mkdir(val_pos_dir)

    test_neg_dir = os.path.join(test_dir, 'neg')
    os.mkdir(test_neg_dir)
    test_pos_dir = os.path.join(test_dir, 'pos')
    os.mkdir(test_pos_dir)



    for data_id in desc_id_array[:train_num]:
        image_name = '{}.jpg'.format(data_id)
        if data_id in neg_id_list:
            src = os.path.join(base_dir, 'Neg/{}'.format(image_name))
            dst = os.path.join(train_neg_dir, image_name)
        if data_id in pos_id_list:
            src = os.path.join(base_dir, 'Pos/{}'.format(image_name))
            dst = os.path.join(train_pos_dir, image_name)
        shutil.copyfile(src, dst)
            
        # print(image_name)

    for data_id in desc_id_array[train_num : train_num + val_num]:
        image_name = '{}.jpg'.format(data_id)
        if data_id in neg_id_list:
            src = os.path.join(base_dir, 'Neg/{}'.format(image_name))
            dst = os.path.join(val_neg_dir, image_name)
        if data_id in pos_id_list:
            src = os.path.join(base_dir, 'Pos/{}'.format(image_name))
            dst = os.path.join(val_pos_dir, image_name)
        shutil.copyfile(src, dst)

    for data_id in desc_id_array[train_num + val_num : train_num + val_num + test_num]:
        image_name = '{}.jpg'.format(data_id)
        if data_id in neg_id_list:
            src = os.path.join(base_dir, 'Neg/{}'.format(image_name))
            dst = os.path.join(test_neg_dir, image_name)
        if data_id in pos_id_list:
            src = os.path.join(base_dir, 'Pos/{}'.format(image_name))
            dst = os.path.join(test_pos_dir, image_name)
        shutil.copyfile(src, dst)

    
get_desc_data4model()
get_image_data4model()

