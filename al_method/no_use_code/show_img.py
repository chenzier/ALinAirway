import numpy as np
import torch
import os
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import from_numpy as from_numpy
from matplotlib.colors import ListedColormap
from active_learning_utils import process_images,show_all_2d_img_with_labels
import pickle

import sys
sys.path.append('../')  # 将上一层目录添加到模块搜索路径中
from func.model_arch2 import SegAirwayModel


Precrop_dataset_for_train_path = "/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_128"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path+"/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path+"/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
print(len(raw_case_name_list))

lidc_dataset_for_train_path='/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_128'
lidc_dataset_for_train_raw_path =lidc_dataset_for_train_path+"/image"
lidc_dataset_for_train_label_path = lidc_dataset_for_train_path+"/label"
lidc_raw_case_name_list = os.listdir(lidc_dataset_for_train_raw_path)
print(len(lidc_raw_case_name_list))

#执行这个文件必须在python 3.8版本下
import os
import numpy as np
import pickle
import edt

def save_obj(obj, name ):
    if name[-3:] != 'pkl':
        temp=name+'.pkl'
    else:
        temp=name
    with open(temp , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    if name[-3:] != 'pkl':
        temp=name+'.pkl'
    else:
        temp=name
    # print(temp)
    with open(temp, 'rb') as f:
        return pickle.load(f)
    
cluster_dict=load_obj('/home/wangc/now/NaviAirway/saved_objs/for_128_objs/indices.pkl')

from active_learning_utils import show_all_2d_img_with_labels
for key1 in cluster_dict:
    result=cluster_dict[key1]
    for key2 in result:
        a_class=result[key2]
        exact_sample,lidc_sample=[],[]
        for name in a_class:
            if name[:3] =='EXA':
                exact_sample.append(name)
            if name[:3] =='LID':
                lidc_sample.append(name)
        print(len(lidc_sample)+len(exact_sample)==len(a_class))
        out_folder='/home/wangc/now/NaviAirway/saved_picture/for_128_cluster'+f'cluster_{key1}/'+key2
        show_all_2d_img_with_labels(raw_img_path=Precrop_dataset_for_train_raw_path , output_folder=out_folder, img_num=200, 
                                        num_images_per_batch=16, slice_index=20, label_path=Precrop_dataset_for_train_label_path ,raw_img_list=exact_sample,file_name='EXACT09')
        show_all_2d_img_with_labels(raw_img_path=lidc_dataset_for_train_raw_path , output_folder=out_folder, img_num=200, 
                                        num_images_per_batch=16, slice_index=20, label_path=lidc_dataset_for_train_label_path ,raw_img_list=lidc_sample,file_name='LIDC')
