import numpy as np
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import from_numpy as from_numpy
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import sys
sys.path.append('../')  # 将上一层目录添加到模块搜索路径中
from al_method.active_learning_utils import process_images,visualize_and_return_indices,show_all_2d_img_with_labels,kmeans,save_obj,load_obj,load_partial_embeddings
import pickle
from func.model_arch2 import SegAirwayModel

import torch.utils.data as data_utils


import edt


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# 设置分布式环境
# local_rank = int(os.environ.get("LOCAL_RANK", 0))



crop_size=['128','256']
file_insert=crop_size[0]
Precrop_dataset_for_train_path = f"/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_{file_insert}"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path+"/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path+"/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
# print(len(raw_case_name_list))

lidc_dataset_for_train_path=f'/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_{file_insert}'
lidc_dataset_for_train_raw_path =lidc_dataset_for_train_path+"/image"
lidc_dataset_for_train_label_path = lidc_dataset_for_train_path+"/label"
lidc_raw_case_name_list = os.listdir(lidc_dataset_for_train_raw_path)
# print(len(lidc_raw_case_name_list))

# 从文件加载
# file_path1='/home/wangc/now/NaviAirway/saved_var/exact09_256_embeddings_data.pkl'
# file_path2='/home/wangc/now/NaviAirway/saved_var/lidc_256_embeddings_data.pkl'

file_path1='/home/wangc/now/NaviAirway/saved_var/exact09_128_op_segmentation_data.pkl'
file_path2='/home/wangc/now/NaviAirway/saved_var/lidc_128_op_segmentation_data.pkl'
test_names = ['LIDC_IDRI_0066', 'LIDC_IDRI_0328', 'LIDC_IDRI_0376',
'LIDC_IDRI_0441',  'LIDC_IDRI_0744', 'LIDC_IDRI_1004','EXACT09_CASE13',
'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']
exact_lidc_concatenated_array , merged_dict, merged_list = load_partial_embeddings(file_path1, file_path2, test_names=test_names)




data_shape = exact_lidc_concatenated_array.shape

X_t = exact_lidc_concatenated_array.reshape(data_shape[0], -1)
device2 = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
# device3 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')




batch_size = 50
N = X_t.shape[0]
uncertainy_dict = {}

for i in range(0, N, batch_size):
    X_t_batch = torch.from_numpy(X_t[i:i+batch_size]).float().to(device2)
    X_t_batch = X_t_batch.unsqueeze(1)  # Dimension becomes (batch_size, 1, 262144)
    

    X_log_X = X_t_batch * torch.log(X_t_batch)

# 沿着指定的维度求和
    uncertainty_batch = torch.sum(X_log_X, dim=(1, 2))
    print(uncertainty_batch.shape)
    # assert False
    # Update uncertainy_dict with batch results
    for j in range(batch_size):
        index = i + j
        if index < N:
            uncertainy_dict[merged_list[index]] = uncertainty_batch[j].cpu().numpy()
    print(f'epoch {i}')
print(len(uncertainy_dict))




file_path = '/home/wangc/now/NaviAirway/saved_var/entorpy_uncertainy.pkl'
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 保存到文件
with open(file_path, 'wb') as file:
    data_to_save = {'uncertainy_dict': uncertainy_dict}
    pickle.dump(data_to_save, file)

